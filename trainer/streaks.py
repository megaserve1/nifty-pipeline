"""trainer/streaks.py -- drawdowns and losing streaks, per confusion-matrix cell.

WHAT A "DRAWDOWN" MEANS HERE
    it counts CLASSIFICATION CALLS, not money. walk the cell's own event stream in time
    order, +1 for a right call and -1 for a wrong one, and keep a running total. the
    drawdown is how far that total fell below its own running peak.

    a NO_TRADE prediction opens no position, so there is no rupee figure for most cells.
    a rupee drawdown belongs in the backtest, not here.

THE TWO KINDS OF ROW, AND WHY THEY DIFFER
    ERROR ROW (T != P) -- "how often does the model make THIS mistake"
        take every bar whose TRUE label is T, then keep ONLY two outcomes:
            predicted T -> right (+1)      predicted P -> wrong (-1)
        every other prediction is SKIPPED. it is a different mistake and belongs to a
        different row; counting it here would blame this cell for someone else's error.

    CLASS ROW (T == P) -- "can the model handle this class at all"
        take every bar whose TRUE label is T. nothing is skipped:
            predicted T -> right (+1)      anything else -> wrong (-1)

RIGHT-CENSORING
    if the running total never regains its peak before the data ends, the episode is still
    OPEN. its depth is a FLOOR, not a final figure. that is flagged rather than hidden --
    an unrecovered drawdown reported as if it were finished understates the risk.
"""
from __future__ import annotations

import numpy as np

# tier LETTERS only. the severity NUMBERS come from configs/severity_7class.json, which is
# also what the trainer's objective and the shap ranking read -- so a number cannot disagree
# with itself across the pipeline.
TIER_OF: dict[tuple[str, str], str] = {}
for _a, _b in [("ENTRY_SMALL", "NO_TRADE"), ("EXIT_SMALL", "NO_TRADE"), ("ENTRY_SUB", "NO_TRADE"),
               ("EXIT_SUB", "NO_TRADE"), ("ENTRY_SUPER", "NO_TRADE"), ("EXIT_SUPER", "NO_TRADE")]:
    TIER_OF[(_a, _b)] = "Tier E"        # MISSED OPPORTUNITY -- a real signal, model said nothing
for _a, _b in [("NO_TRADE", "ENTRY_SMALL"), ("NO_TRADE", "EXIT_SMALL"), ("NO_TRADE", "ENTRY_SUB"),
               ("NO_TRADE", "EXIT_SUB"), ("NO_TRADE", "ENTRY_SUPER"), ("NO_TRADE", "EXIT_SUPER")]:
    TIER_OF[(_a, _b)] = "Tier D"        # UNWANTED TRADE -- truth was sit out, model traded
for _a, _b in [("ENTRY_SUPER", "EXIT_SUPER"), ("EXIT_SUPER", "ENTRY_SUPER")]:
    TIER_OF[(_a, _b)] = "Tier A"        # FULL REVERSAL AT MAX SIZE
for _a, _b in [("ENTRY_SMALL", "EXIT_SMALL"), ("EXIT_SMALL", "ENTRY_SMALL"),
               ("ENTRY_SMALL", "EXIT_SUB"), ("ENTRY_SUB", "EXIT_SMALL"),
               ("EXIT_SMALL", "ENTRY_SUB"), ("EXIT_SUB", "ENTRY_SMALL"),
               ("ENTRY_SUB", "EXIT_SUB"), ("EXIT_SUB", "ENTRY_SUB")]:
    TIER_OF[(_a, _b)] = "Tier C"        # REVERSAL at medium or low size
for _a, _b in [("ENTRY_SMALL", "EXIT_SUPER"), ("ENTRY_SUPER", "EXIT_SMALL"),
               ("EXIT_SMALL", "ENTRY_SUPER"), ("EXIT_SUPER", "ENTRY_SMALL"),
               ("ENTRY_SUB", "EXIT_SUPER"), ("ENTRY_SUPER", "EXIT_SUB"),
               ("EXIT_SUB", "ENTRY_SUPER"), ("EXIT_SUPER", "ENTRY_SUB")]:
    TIER_OF[(_a, _b)] = "Tier B"        # REVERSAL involving one SUPER
for _a, _b in [("ENTRY_SUB", "ENTRY_SMALL"), ("EXIT_SUB", "EXIT_SMALL"),
               ("ENTRY_SUPER", "ENTRY_SUB"), ("ENTRY_SUPER", "ENTRY_SMALL"),
               ("EXIT_SUPER", "EXIT_SUB"), ("EXIT_SUPER", "EXIT_SMALL"),
               ("ENTRY_SMALL", "ENTRY_SUB"), ("EXIT_SMALL", "EXIT_SUB"),
               ("ENTRY_SUB", "ENTRY_SUPER"), ("EXIT_SUB", "EXIT_SUPER"),
               ("ENTRY_SMALL", "ENTRY_SUPER"), ("EXIT_SMALL", "EXIT_SUPER")]:
    TIER_OF[(_a, _b)] = "Tier F"        # RIGHT DIRECTION, WRONG SIZE

TIER_ORDER = ["Tier E", "Tier D", "Tier A", "Tier C", "Tier B", "Tier F"]

DEF_DD = ("# of correct classifications exceed or cover for wrong classifications "
          "in a given period")
DEF_PERIOD = ("DD-MM-YYYY to DD-MM-YYYY between date of previous right classification "
              "and next right classification which covered for all previous wrong "
              "classifications")
DEF_LS = "Continuous wrong classifications"


def losing_streaks(wrong: np.ndarray) -> list:
    """every run of consecutive WRONG as (length, first_idx, last_idx), longest first."""
    out, i, n = [], 0, len(wrong)
    while i < n:
        if wrong[i]:
            j = i
            while j + 1 < n and wrong[j + 1]:
                j += 1
            out.append((j - i + 1, i, j))
            i = j + 1
        else:
            i += 1
    out.sort(key=lambda r: (-r[0], r[1]))
    return out


def drawdowns(wrong: np.ndarray) -> list:
    """every fall of the running total below its running peak, as
    (depth, peak_idx, recovered_or_last_idx), deepest first.

    the total starts at a notional 0 before the first event, so the peak can be 0 with
    peak_idx -1; callers clamp that to the stream's first bar.
    """
    cum = np.cumsum(np.where(wrong, -1, 1))
    out = []
    peak, peak_i = 0, -1
    depth, start = 0, -1
    for i, v in enumerate(cum):
        if v > peak:
            if depth:
                out.append((depth, start, i))        # peak regained -> episode closes
            peak, peak_i, depth = v, i, 0
        elif v < peak:
            if depth == 0:
                start = peak_i
            depth = max(depth, peak - v)
    if depth:                                        # never regained -> right-censored
        out.append((depth, start, len(cum) - 1))
    out.sort(key=lambda r: (-r[0], r[1]))
    return out


def cell_stream(true_arr: np.ndarray, pred_arr: np.ndarray, T: str, P: str):
    """(row positions, wrong flags) for the cell (T, P). see the module docstring."""
    of_class = np.flatnonzero(true_arr == T)
    preds = pred_arr[of_class]
    if T == P:
        return of_class, preds != T
    keep = (preds == T) | (preds == P)
    return of_class[keep], preds[keep] == P


def cell_rows(true_arr, pred_arr, classes: list, sev: dict, times=None) -> list:
    """one record per (true, predicted) pair -- all 49, whether they occurred or not."""
    have_dates = times is not None

    def stamp(pos):
        import pandas as pd
        return (pd.Timestamp(times[pos]).strftime("%d-%m-%Y") if have_dates
                else f"row {int(pos):,}")

    out = []
    for T in classes:
        for P in classes:
            pos, wrong = cell_stream(true_arr, pred_arr, T, P)
            ls, dd = losing_streaks(wrong), drawdowns(wrong)
            cum = np.cumsum(np.where(wrong, -1, 1)) if len(wrong) else np.array([0])
            r = {"tier": TIER_OF.get((T, P)), "true": T, "pred": P,
                 "severity": sev.get((T, P), 0),
                 "n_occurrences": int(((true_arr == T) & (pred_arr == P)).sum()),
                 "n_events": len(wrong), "n_right": int((~wrong).sum()),
                 "n_wrong": int(wrong.sum()),
                 "wrong_rate_pct": round(float(wrong.mean()) * 100, 2) if len(wrong) else None,
                 "n_streaks": len(ls), "n_dd": len(dd),
                 "final_score": int(cum[-1]) if len(wrong) else 0}
            # is the deepest drawdown still OPEN when the data ends?
            r["dd1_recovered"] = ("" if not dd else
                                  ("no - data ended"
                                   if dd[0][2] == len(wrong) - 1 and cum[-1] < max(0, cum.max())
                                   else "yes"))
            for k in (1, 2, 3):
                if len(dd) >= k:
                    depth, a, b = dd[k - 1]
                    r[f"max_dd{k}"] = depth
                    r[f"dd{k}_period"] = f"{stamp(pos[max(a, 0)])} to {stamp(pos[b])}"
                else:
                    r[f"max_dd{k}"] = r[f"dd{k}_period"] = None
                if len(ls) >= k:
                    length, a, b = ls[k - 1]
                    r[f"max_streak{k}"] = length
                    r[f"streak{k}_period"] = f"{stamp(pos[a])} to {stamp(pos[b])}"
                else:
                    r[f"max_streak{k}"] = r[f"streak{k}_period"] = None
            out.append(r)
    return out
