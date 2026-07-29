"""Target-vs-predicted agreement statistics shown above the price chart."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AgreementSummary:
    """Comparison of the target and predicted label columns."""

    n_rows: int = 0
    n_target: int = 0
    n_predicted: int = 0
    n_compared: int = 0
    n_match: int = 0
    n_mismatch: int = 0
    accuracy: Optional[float] = None
    classes: list[str] = field(default_factory=list)
    target_counts: dict = field(default_factory=dict)
    predicted_counts: dict = field(default_factory=dict)

    @property
    def has_comparison(self) -> bool:
        return self.n_compared > 0


def label_classes(labels: pd.DataFrame, columns=("target", "predicted")) -> list[str]:
    """Sorted union of the distinct label values across ``columns``.

    Purely numeric classes sort numerically (so ``-1, 0, 1`` not ``-1, 0, 1``
    lexicographically ordered as ``"-1", "0", "1"`` by accident on wider sets).
    """
    values: set[str] = set()
    for col in columns:
        if col in labels.columns:
            values.update(labels[col].dropna().astype(str).unique().tolist())

    def sort_key(value: str):
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, value)

    return sorted(values, key=sort_key)


def comparison_mask(labels: pd.DataFrame) -> pd.Series:
    """Rows where both a target and a predicted label exist."""
    if "target" not in labels.columns or "predicted" not in labels.columns:
        return pd.Series(False, index=labels.index)
    return labels["target"].notna() & labels["predicted"].notna()


def mismatch_mask(labels: pd.DataFrame) -> pd.Series:
    """Rows where target and predicted are both present but disagree."""
    both = comparison_mask(labels)
    if not both.any():
        return both
    differ = labels["target"].astype("string") != labels["predicted"].astype("string")
    return both & differ


def summarise(labels: pd.DataFrame) -> AgreementSummary:
    """Aggregate counts/accuracy for the currently visible label rows."""
    summary = AgreementSummary(n_rows=int(len(labels)))
    if labels.empty:
        return summary

    if "target" in labels.columns:
        target = labels["target"].dropna().astype(str)
        summary.n_target = int(len(target))
        summary.target_counts = target.value_counts().to_dict()
    if "predicted" in labels.columns:
        predicted = labels["predicted"].dropna().astype(str)
        summary.n_predicted = int(len(predicted))
        summary.predicted_counts = predicted.value_counts().to_dict()

    both = comparison_mask(labels)
    summary.n_compared = int(both.sum())
    if summary.n_compared:
        mism = int(mismatch_mask(labels).sum())
        summary.n_mismatch = mism
        summary.n_match = summary.n_compared - mism
        summary.accuracy = summary.n_match / summary.n_compared

    summary.classes = label_classes(labels)
    return summary


def confusion_matrix(labels: pd.DataFrame, normalise: bool = False) -> pd.DataFrame:
    """Target (rows) x predicted (columns) counts over comparable rows.

    Both axes carry the full class union, so a class the model never predicts
    still shows up as an all-zero column instead of silently disappearing.
    """
    classes = label_classes(labels)
    frame = pd.DataFrame(
        np.zeros((len(classes), len(classes)), dtype=float if normalise else int),
        index=pd.Index(classes, name="target"),
        columns=pd.Index(classes, name="predicted"),
    )
    both = comparison_mask(labels)
    if not both.any() or not classes:
        return frame

    subset = labels.loc[both]
    counted = (
        subset.groupby(
            [subset["target"].astype(str), subset["predicted"].astype(str)]
        )
        .size()
        .unstack(fill_value=0)
    )
    counted = counted.reindex(index=classes, columns=classes, fill_value=0)
    frame.loc[:, :] = counted.to_numpy()

    if normalise:
        row_totals = frame.sum(axis=1).replace(0, np.nan)
        frame = frame.div(row_totals, axis=0).fillna(0.0)
    return frame


def per_class_report(labels: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall / F1 / support per class, plus a macro average row."""
    classes = label_classes(labels)
    both = comparison_mask(labels)
    rows = []
    subset = labels.loc[both]
    for cls in classes:
        actual = subset["target"].astype(str) == cls if len(subset) else pd.Series(dtype=bool)
        guess = subset["predicted"].astype(str) == cls if len(subset) else pd.Series(dtype=bool)
        tp = int((actual & guess).sum())
        fp = int((~actual & guess).sum())
        fn = int((actual & ~guess).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        if precision and recall and np.isfinite(precision) and np.isfinite(recall):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = np.nan
        rows.append(
            {
                "class": cls,
                "support (target)": int(actual.sum()) if len(subset) else 0,
                "predicted": int(guess.sum()) if len(subset) else 0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    report = pd.DataFrame(
        rows,
        columns=["class", "support (target)", "predicted", "precision", "recall", "f1"],
    )
    if not report.empty:
        macro = {
            "class": "macro avg",
            "support (target)": report["support (target)"].sum(),
            "predicted": report["predicted"].sum(),
            "precision": report["precision"].mean(skipna=True),
            "recall": report["recall"].mean(skipna=True),
            "f1": report["f1"].mean(skipna=True),
        }
        report = pd.concat([report, pd.DataFrame([macro])], ignore_index=True)
    return report
