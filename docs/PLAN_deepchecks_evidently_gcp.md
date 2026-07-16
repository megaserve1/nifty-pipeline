# The plan: Deepchecks, Evidently, GCP hosting

Written 2026-07-14. Three tools, three jobs, one rule that decides everything:
**the data never leaves our GCP.** Deepchecks and Evidently both run as plain Python
on our own machines — nothing is sent anywhere. That is why they were kept and
Fiddler/Arize were rejected.

The one-line division of labour:

| tool | job | when it runs |
|---|---|---|
| Deepchecks | the **gate** — refuse a bad dataset or a bad model BEFORE it ships | at publish, and after training |
| Evidently | the **watchman** — notice the world changing AFTER the model ships | nightly, once live |
| GCP | the **floor** it all stands on | always |

---

## 1. Deepchecks — the pre-deploy gate

### Why it earned its place
It is the only tool in the stack that checks for **leakage, overfit, and
beats-a-baseline** — the three ways a trading model lies to you. We already
found one lookahead leak by hand this month. The gate makes that hunt automatic.

### Where it sits — two gates, not one

```
build_dataset -> [GATE A: data]   -> publish_version -> train
train         -> [GATE B: model]  -> select_champion -> champion tag
```

**Gate A — the dataset gate.** Runs on `datasets/vN/` before `publish_version.py`
will publish it. Suites: `data_integrity` on the built parquet, and
`train_test_validation` on the exact split the trainer will use.
What it catches, in our terms:
- a feature column that is a near-copy of the label (leak — the guard's last line of defence)
- duplicate / conflicting rows, mixed dtypes, new-vs-old category mismatch
- train and test that do not look like the same market (a bad split, or a data hole)

**Gate B — the model gate.** Runs after the three trainers finish, before
`select_champion.py` crowns anything. Suite: `model_evaluation`, plus two custom
checks Deepchecks does not know about:
- **beats-the-baseline on trading_cost** — the model must beat "always say NO_TRADE"
  and a class-prior random guesser, on OUR metric, not accuracy
- **per-class recall floor** — recall(ENTRY_SUPER) and recall(EXIT_SUPER) must be > 0;
  a model blind to a class it will be paid to find is refused (this is the
  champion-BLIND rule, now enforced by a suite instead of one script)

### How it is wired (files, not hand-runs)
- `trainer/gate_data.py` — Gate A. Called by `publish_version.py` between the manifest
  check and the dvc push. **Refuses to publish on failure** — same pattern as the
  contract check: loud, early, no half-published state.
- `trainer/gate_model.py` — Gate B. Queued by `train.py` right after `queue_shap_for_me`
  (same pattern: queue it from inside the finished trainer, so the model provably exists).
  Writes PASS/FAIL as a task tag; `select_champion.py` refuses to crown a model whose
  gate task says FAIL.
- Reports land as ClearML artifacts → which means the HTML report bytes go to
  **our bucket** via `output_uri`, not to app.clear.ml.

### Thresholds to start with (tune after the first real run)
- feature-label correlation alarm: > 0.5 on any single feature (a real signal this
  strong does not exist in 1-minute futures; it is a leak)
- train/test drift (Deepchecks' own drift score): warn only, do not fail — 2020-covid
  is in train and that drift is REAL, not a bug
- overfit: train-test macro-F1 gap > 0.15 = fail
- baseline: trading_cost must be at least 5% better than the always-NO_TRADE baseline

### What NOT to do
Do not run all 5 suites. The other two need things we don't have (a production
stream for their monitoring suite) or duplicate Evidently. Three suites, two gates.

**My Call:** Deepchecks is the automation of the audit we just did by hand. Gate A
would have caught `fwd_ret_10` on its own. Build both gates before the first real
publish, not after.

---

## 2. Evidently — the live watchman

### Why it earned its place
Evidently **keeps history**. A spike that lasts one day is a flash crash; a shift
that holds for two weeks is a new regime. A tool without memory cannot tell those
apart, and the difference is exactly "do nothing" vs "retrain".

### The two-phase plan

**Phase 1 — now, before anything is live (half a day of work).**
Run Evidently once on v2: train slice vs test slice. Purpose: learn which of the
285 features are *naturally* unstable across time. Output: a per-feature reference
profile (JSON) stored next to the model bundle in the bucket. This kills the known
problem that a single global drift threshold (the 0.1 default) is nonsense for
markets — VIX drifting is Tuesday, `minute_of_day` drifting is a broken pipe.
Every feature gets its own yardstick, measured from our own history.

**Phase 2 — live (only after serving exists).**
- The serving loop already must log every feature vector it predicts on (that
  decision is made — it feeds the parity check). Evidently reads those logs.
- A nightly job, after market close, on the same VM before it stops:
  - **data drift**: today's feature vectors vs the reference profile, per-feature
    thresholds from Phase 1
  - **prediction drift**: the distribution of predicted classes vs the test-set
    distribution — this is the cheapest over-trading alarm we can build; if the
    model suddenly predicts ENTRY twice as often, something moved
  - **label drift**: added later, when the labels for day T arrive on T+1
- Reports: HTML + JSON to `gs://<bucket>/evidently/YYYY-MM-DD/`, one line into a
  ClearML task so the history is queryable.

### The alarm rule (write it down now, argue later)
- 1 day over threshold → note it, do nothing. Markets have days.
- 2 consecutive days over threshold on the SAME features → investigate.
- The FIRST investigation step is always the **parity check**, never retraining:
  if training-time features and live features disagree on the same minute, that is
  SKEW — the plumbing is broken and retraining would train on the broken plumbing.
  Drift is only drift after parity passes.

**My Call:** Phase 1 costs half a day and is useful even if we never go live —
it is a feature-stability report the feature team will want anyway. Do it with the
first real training run. Phase 2 waits for serving; do not build a watchman before
there is something to watch.

---

## 3. GCP hosting

### What is already settled (and not re-argued here)
Project `mega-ml`. Keyless ADC only — org policy blocks SA key files, so every
recipe that says `credentials_json` is dead on arrival. On a VM: attach a service
account with `roles/storage.objectAdmin`. `clearml.conf` must ALSO carry
`sdk.google.storage.project: "mega-ml"` or GCS calls fail with valid credentials.
app.clear.ml holds metadata only; all bytes go to our bucket via `default_output_uri`.

### Step 0 — before any machine is created
1. **Rotate the ClearML key.** The key printed in plaintext on 2026-07-08 is still
   live. New key from app.clear.ml → update `~/clearml.conf` on the laptop, the VM,
   and the remote PC. This is overdue.
2. **Set the production bucket.** `config.py` still says `demo-nifty-pipeline`.
   One line, but until it changes, every publish pushes production data into the
   demo bucket. Also update the dvc remote and `clearml.conf` output_uri.
3. `git init && dvc init` + first push (the repo/DVC bootstrap has still never run).

### The training machine
e2 was for the demo. For tree models (XGBoost/CatBoost are CPU-parallel):

| choice | spec | ₹/hr (asia-south1, approx) | when |
|---|---|---|---|
| `c2d-standard-8` | 8 vCPU / 32 GB | ~₹30 on-demand | the workhorse |
| the same, **Spot** | 8 vCPU / 32 GB | **~₹9–12** | same machine, 60–70% off |
| `e2-standard-4` | 4 vCPU / 16 GB | ~₹12 | fallback if quota blocks c2d |

**Use Spot for training agents.** A Spot VM can be reclaimed by Google at any
moment — which is normally a problem, but ClearML makes it a non-problem: the task
just goes back in the queue and the next agent run picks it up. Training is
restartable by construction. This is the single biggest cost lever available.
(Do NOT use Spot for the future live-serving VM — that one must not vanish mid-market.)

### The pattern: no always-on machines
Training is event-driven (a publish), not daily. So:

```
you publish (laptop)  ->  tasks land in the 'training' queue
you start the VM      ->  clearml-agent picks them up, one after another
queue empty           ->  the VM STOPS ITSELF
```

The self-stop is a 10-line systemd timer on the VM: every 5 minutes, ask the
ClearML API if the queue is empty AND no task is running; if idle 20 minutes,
`shutdown -h now`. This converts the "forgot the VM = ₹8,000/month" failure mode
into "forgot the VM = it turned itself off". I will write this script when the VM
is created — it is the one piece of this section that is code, not clicks.

RAM note: 285 features × 513k rows peaked around 3–4 GB per trainer today. Three
trainers in sequence on 32 GB is comfortable; three in PARALLEL needs the queue
kept at one agent per machine (which is already how clearml-agent works).

### Bucket layout (one bucket, four prefixes)

```
gs://<production-bucket>/final_pipeline/dvc     dvc objects (features, labels, datasets)
gs://<production-bucket>/clearml                task artifacts + models (output_uri)
gs://<production-bucket>/evidently              nightly reports, reference profiles
gs://<production-bucket>/serve-logs             live feature vectors (when serving exists)
```

### Money (the honest table)

| thing | cost |
|---|---|
| storage, everything above | ~₹5–20/month at current sizes |
| training burst (Spot c2d-8, 3 models ≈ 1–2 hr) | **~₹15–25 per training run** |
| the same if the VM is forgotten overnight | ~₹250 — which is why the self-stop exists |
| ClearML SaaS free tier | ₹0 (metadata only) |
| egress | ₹0 while everything stays in asia-south1 |

Budget alert at ₹2,000/month on the project as a backstop. Nothing in this plan
runs 24/7.

### What this plan deliberately does not cover
Live serving. Its blockers are unchanged and none of them is hosting: the feature
team must deliver **functions, not parquets**; the NO_TRADE weight question needs
its answer measured (`local_check` prints it); and the **parity script** must exist
before the first paper trade. Hosting for serving is a half-day problem once those
exist — a small on-demand VM on a market-hours schedule, already sketched.

**My Call:** rotate the key and set the bucket before anything else — both are
release-blockers and both are one-line fixes. Then Spot agents + self-stopping VM
makes training effectively free (~₹20/run). Deepchecks gates go in before the first
real publish; Evidently Phase 1 rides along with the first real training run.

---

## The order of work

1. Rotate ClearML key (laptop + VM + remote PC). *Overdue since 08 Jul.*
2. Production bucket into `config.py` + dvc remote + clearml.conf.
3. `git init && dvc init && dvc push` — the bootstrap that has never run.
4. `trainer/gate_data.py` (Gate A) wired into publish_version.
5. `trainer/gate_model.py` (Gate B) wired into train/select_champion.
6. Create the training VM (Spot c2d-standard-8) + the self-stop timer + agent daemon.
7. First real publish → training → gates → champion. Watch it end to end.
8. Evidently Phase 1 on that run; store the reference profile.
9. Then, and only then, the serving conversation.
