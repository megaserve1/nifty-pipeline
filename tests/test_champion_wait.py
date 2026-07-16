"""
tests/test_champion_wait.py -- the champion must not crown a winner while a runner is still
on the track, and must not deadlock the only agent.

WHY THIS FILE EXISTS
    select_champion is queued at the same time as the three trainers, so it has to WAIT. that
    waiting is the whole risk:

      1. it waits by OCCUPYING AN AGENT. with one agent, if it gets picked up first, the three
         trainers are stuck behind it in the queue and can never start -- so it would poll for
         four hours for models it is itself preventing from running. it must notice and get out.

      2. when it gives up, it used to crown a champion out of whatever happened to be finished.
         a model that FAILED is a verdict -- fine, judge the survivors. a model that is STILL
         TRAINING is not a verdict, and the champion tag is what goes to production.

    none of this can be tested against real ClearML (it would take hours), so wait_for_models
    takes an injectable probe + clock. these tests drive it in microseconds.
"""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as C                                     # noqa: E402
from trainer.select_champion import wait_for_models, Deadlock   # noqa: E402


class FakeClock:
    """time, but we control it. sleep(30) jumps 30 seconds instead of taking them."""
    def __init__(self):
        self.t = 1_000_000.0
        self.slept = 0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.slept += 1


def script(*frames):
    """turn a list of {model: status} snapshots into a probe. the last frame repeats for ever."""
    state = {"i": 0}

    def probe():
        f = frames[min(state["i"], len(frames) - 1)]
        state["i"] += 1
        return {m: (st, st == "completed" and m not in f.get("_no_model", []), f"task_{m}")
                for m, st in f.items() if not m.startswith("_")}
    return probe


ALL = C.MODEL_TYPES          # ["random_forest", "xgboost", "catboost"]


# ---------------------------------------------------------------- the happy path
def test_returns_as_soon_as_all_three_are_done():
    clk = FakeClock()
    probe = script(
        {m: "in_progress" for m in ALL},
        {m: "in_progress" for m in ALL},
        {m: "completed" for m in ALL},
    )
    found, running, dead = wait_for_models("2", len(ALL), probe=probe,
                                           sleep=clk.sleep, now=clk.now)
    assert set(found) == set(ALL)
    assert running == [] and dead == []
    assert clk.slept == 2, "it must stop the moment they are all done, not keep polling"


def test_a_failed_model_is_a_verdict_and_we_judge_the_survivors():
    """a dead model is DEAD. we do not wait four hours for a corpse."""
    clk = FakeClock()
    probe = script({ALL[0]: "completed", ALL[1]: "failed", ALL[2]: "completed"})
    found, running, dead = wait_for_models("2", 3, probe=probe, sleep=clk.sleep, now=clk.now)
    assert set(found) == {ALL[0], ALL[2]}
    assert running == []
    assert dead == [f"{ALL[1]}(failed)"]
    assert clk.slept == 0, "nothing left to wait for -- it must return immediately"


def test_completed_but_no_model_artifact_counts_as_dead_not_as_a_candidate():
    """a task can finish 'completed' and still have saved nothing. it is not a candidate."""
    probe = script({ALL[0]: "completed", ALL[1]: "completed", ALL[2]: "completed",
                    "_no_model": [ALL[1]]})
    clk = FakeClock()
    found, running, dead = wait_for_models("2", 3, probe=probe, sleep=clk.sleep, now=clk.now)
    assert ALL[1] not in found
    assert dead == [f"{ALL[1]}(completed but saved NO model)"]


# ---------------------------------------------------------------- the timeout (#13)
def test_a_model_still_training_at_the_deadline_is_REPORTED_not_swallowed():
    """THE POINT OF #13.

    it used to return only `found` -- so a model that was still training at the deadline
    simply vanished from the answer, and main() crowned a champion from the rest without
    ever knowing one was missing. the unfinished model might have been the best.
    now it comes back in `running`, and main() refuses to crown.
    """
    clk = FakeClock()
    probe = script({ALL[0]: "completed", ALL[1]: "completed", ALL[2]: "in_progress"})
    found, running, dead = wait_for_models("2", 3, timeout_min=10, poll=30,
                                           probe=probe, sleep=clk.sleep, now=clk.now)
    assert set(found) == {ALL[0], ALL[1]}
    assert running == [f"{ALL[2]}(in_progress)"], "the unfinished model must be NAMED"
    assert clk.t - 1_000_000.0 > 10 * 60, "it must actually have waited the full timeout"


def test_the_deadline_is_honoured_and_it_does_not_poll_for_ever():
    clk = FakeClock()
    probe = script({m: "in_progress" for m in ALL})
    found, running, _ = wait_for_models("2", 3, timeout_min=60, poll=30,
                                        probe=probe, sleep=clk.sleep, now=clk.now)
    assert found == {}
    assert len(running) == 3
    assert clk.slept <= (60 * 60) // 30 + 1, "it must stop at the deadline, not spin"


# ---------------------------------------------------------------- the deadlock (the real bug)
def test_it_detects_that_IT_is_the_reason_nothing_is_running():
    """THE WORST CASE, and it is not the timeout.

    one agent. it picks up select_champion. the three trainers are behind it in the queue.
    the champion now waits for models that CANNOT START, because it is holding the only agent
    they need. before: four hours of polling, then "choosing from 0 models". a whole afternoon
    gone, and the pipeline looks broken for no visible reason.

    now: after 10 minutes with nothing even STARTED, it says so and exits, freeing the agent.
    """
    clk = FakeClock()
    probe = script({m: "queued" for m in ALL})       # queued for ever -- nobody can run them
    with pytest.raises(Deadlock) as e:
        wait_for_models("2", 3, timeout_min=240, poll=30, stall_min=10,
                        probe=probe, sleep=clk.sleep, now=clk.now)

    msg = str(e.value)
    assert "DEADLOCK" in msg
    assert "clearml-agent daemon" in msg, "it must tell him how to fix it"
    assert clk.t - 1_000_000.0 < 11 * 60, "it must give up in ~10 min, not four hours"


def test_it_is_NOT_a_deadlock_if_a_model_has_actually_started():
    """queued + in_progress means an agent IS working. that is just slow, not stuck."""
    clk = FakeClock()
    probe = script({ALL[0]: "in_progress", ALL[1]: "queued", ALL[2]: "queued"})
    found, running, _ = wait_for_models("2", 3, timeout_min=20, poll=30, stall_min=10,
                                        probe=probe, sleep=clk.sleep, now=clk.now)
    assert found == {}
    assert len(running) == 3        # it waited it out and timed out honestly, no Deadlock raised


def test_it_is_NOT_a_deadlock_if_one_model_has_already_finished():
    """something finished -> agents are clearly working -> we are not the blocker."""
    clk = FakeClock()
    probe = script({ALL[0]: "completed", ALL[1]: "queued", ALL[2]: "queued"})
    found, running, _ = wait_for_models("2", 3, timeout_min=20, poll=30, stall_min=10,
                                        probe=probe, sleep=clk.sleep, now=clk.now)
    assert set(found) == {ALL[0]}
    assert len(running) == 2
