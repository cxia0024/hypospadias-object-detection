import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage1_detection.metrics import (  # noqa: E402
    binomial_test_vs_chance,
    class_metrics,
    compare_two_datasets,
    confusion_counts,
    evaluate_dataset,
    two_proportion_ztest,
    wilson_ci,
)


def test_confusion_counts_basic():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    c = confusion_counts(y_true, y_pred)
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 1, 1)
    assert c.n == 5


def test_confusion_counts_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_counts(np.array([1, 0]), np.array([1, 0, 1]))


def test_wilson_ci_known_value():
    # 80/100 successes: classic textbook Wilson interval is approximately (0.712, 0.869)
    lo, hi = wilson_ci(80, 100)
    assert lo == pytest.approx(0.712, abs=0.01)
    assert hi == pytest.approx(0.869, abs=0.01)


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_ci_bounds_within_0_1():
    lo, hi = wilson_ci(3, 3)  # 100% success, small n -> interval should still be clamped to [0,1]
    assert 0.0 <= lo <= hi <= 1.0


def test_binomial_test_above_chance_is_significant():
    # 90/100 correct vs. chance 0.5 should be a very small p-value
    p = binomial_test_vs_chance(90, 100, chance=0.5, alternative="greater")
    assert p < 1e-10


def test_binomial_test_at_chance_is_not_significant():
    p = binomial_test_vs_chance(52, 100, chance=0.5, alternative="greater")
    assert p > 0.05


def test_two_proportion_ztest_identical_proportions():
    z, p = two_proportion_ztest(50, 100, 50, 100)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_two_proportion_ztest_detects_difference():
    # 95/100 vs 60/100 is a large, significant gap (e.g. AVOS accuracy vs. hypospadias zero-shot accuracy)
    z, p = two_proportion_ztest(95, 100, 60, 100)
    assert p < 0.001
    assert z > 0


def test_two_proportion_ztest_zero_n_returns_nan():
    z, p = two_proportion_ztest(0, 0, 5, 10)
    assert np.isnan(z) and np.isnan(p)


def test_class_metrics_perfect_classifier():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    m = class_metrics("bovie", y_true, y_pred)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.binomial_p_vs_chance < 1.0  # not deeply significant at n=4, but should compute without error


def test_class_metrics_no_positive_predictions():
    y_true = np.array([1, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    m = class_metrics("needle_driver", y_true, y_pred)
    assert m.accuracy == 0.75
    assert np.isnan(m.precision)  # no predicted positives -> precision undefined
    assert m.recall == 0.0


def test_evaluate_dataset_and_pooled_accuracy():
    labels = {
        "bovie": np.array([1, 1, 0, 0]),
        "forceps": np.array([1, 0, 0, 0]),
    }
    preds = {
        "bovie": np.array([1, 1, 0, 0]),  # 4/4 correct
        "forceps": np.array([1, 0, 0, 1]),  # 3/4 correct
    }
    result = evaluate_dataset("toy_dataset", labels, preds)
    assert result.per_class["bovie"].accuracy == 1.0
    assert result.per_class["forceps"].accuracy == 0.75
    assert result.pooled_accuracy == pytest.approx(7 / 8)


def test_compare_two_datasets_pooled_and_per_class():
    labels_a = {"bovie": np.ones(50)}
    preds_a = {"bovie": np.ones(50)}  # 100% accuracy
    labels_b = {"bovie": np.array([1] * 30 + [0] * 20)}
    preds_b = {"bovie": np.array([1] * 15 + [0] * 15 + [1] * 20)}  # 15/30 + 0/20 correct = 15/50 = 30%

    result_a = evaluate_dataset("avos_test", labels_a, preds_a)
    result_b = evaluate_dataset("hypospadias_eval", labels_b, preds_b)

    cmp = compare_two_datasets(result_a, result_b)
    assert cmp["dataset_a"] == "avos_test"
    assert cmp["dataset_b"] == "hypospadias_eval"
    assert cmp["pooled"]["p_value"] < 0.001  # 100% vs 30% accuracy is a stark, significant difference
    assert "bovie" in cmp["per_class"]
