"""Statistics for Stage 1 (object detection) zero-shot evaluation.

Implements the metrics battery described in the Methods:
- accuracy / precision / recall / F1 / confusion matrix, per class and pooled
- Wilson 95% confidence interval on accuracy
- binomial test of accuracy vs. chance
- two-proportion z-test comparing accuracy between two datasets (e.g. AVOS vs. hypospadias)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def n_positive_labels(self) -> int:
        return self.tp + self.fn


@dataclass
class ClassMetrics:
    class_name: str
    counts: ConfusionCounts
    accuracy: float
    precision: float
    recall: float
    f1: float
    wilson_ci: tuple[float, float]
    binomial_p_vs_chance: float


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    """y_true/y_pred are 0/1 arrays for a single class (presence/absence)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def binomial_test_vs_chance(successes: int, n: int, chance: float = 0.5, alternative: str = "greater") -> float:
    """Exact binomial test p-value for accuracy (or any proportion) vs. a chance level."""
    if n == 0:
        return float("nan")
    result = stats.binomtest(successes, n, chance, alternative=alternative)
    return float(result.pvalue)


def two_proportion_ztest(successes1: int, n1: int, successes2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test. Returns (z statistic, p-value)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = successes1 / n1, successes2 / n2
    p_pool = (successes1 + successes2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        # Both proportions identical (often both 0 or both 1): no evidence of a difference.
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return (float(z), float(p_value))


def class_metrics(class_name: str, y_true: np.ndarray, y_pred: np.ndarray, chance: float = 0.5) -> ClassMetrics:
    counts = confusion_counts(y_true, y_pred)
    n = counts.n
    accuracy = (counts.tp + counts.tn) / n if n else float("nan")
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else float("nan")
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
        else float("nan")
    )
    correct = counts.tp + counts.tn
    return ClassMetrics(
        class_name=class_name,
        counts=counts,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        wilson_ci=wilson_ci(correct, n),
        binomial_p_vs_chance=binomial_test_vs_chance(correct, n, chance=chance),
    )


@dataclass
class DatasetEvalResult:
    dataset_name: str
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)

    @property
    def pooled_counts(self) -> ConfusionCounts:
        tp = sum(m.counts.tp for m in self.per_class.values())
        fp = sum(m.counts.fp for m in self.per_class.values())
        tn = sum(m.counts.tn for m in self.per_class.values())
        fn = sum(m.counts.fn for m in self.per_class.values())
        return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)

    @property
    def pooled_accuracy(self) -> float:
        c = self.pooled_counts
        return (c.tp + c.tn) / c.n if c.n else float("nan")

    def pooled_wilson_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        c = self.pooled_counts
        return wilson_ci(c.tp + c.tn, c.n, confidence=confidence)

    def pooled_binomial_p_vs_chance(self, chance: float = 0.5) -> float:
        c = self.pooled_counts
        return binomial_test_vs_chance(c.tp + c.tn, c.n, chance=chance)


def evaluate_dataset(
    dataset_name: str,
    labels_by_class: dict[str, np.ndarray],
    predictions_by_class: dict[str, np.ndarray],
    chance: float = 0.5,
) -> DatasetEvalResult:
    """Compute per-class metrics for one dataset from aligned label/prediction arrays."""
    per_class: dict[str, ClassMetrics] = {}
    for cls, y_true in labels_by_class.items():
        if cls not in predictions_by_class:
            raise KeyError(f"no predictions for class '{cls}'")
        per_class[cls] = class_metrics(cls, y_true, predictions_by_class[cls], chance=chance)
    return DatasetEvalResult(dataset_name=dataset_name, per_class=per_class)


def compare_two_datasets(a: DatasetEvalResult, b: DatasetEvalResult) -> dict:
    """Pooled-accuracy two-proportion z-test between two dataset eval results
    (e.g. AVOS test set vs. hypospadias zero-shot eval), plus per-class comparisons
    for classes present in both.
    """
    ca, cb = a.pooled_counts, b.pooled_counts
    z, p = two_proportion_ztest(ca.tp + ca.tn, ca.n, cb.tp + cb.tn, cb.n)
    per_class = {}
    shared_classes = set(a.per_class) & set(b.per_class)
    for cls in sorted(shared_classes):
        ma, mb = a.per_class[cls], b.per_class[cls]
        cza, czb = ma.counts, mb.counts
        cz, pz = two_proportion_ztest(cza.tp + cza.tn, cza.n, czb.tp + czb.tn, czb.n)
        per_class[cls] = {"z": cz, "p_value": pz}
    return {
        "dataset_a": a.dataset_name,
        "dataset_b": b.dataset_name,
        "pooled": {"z": z, "p_value": p},
        "per_class": per_class,
    }
