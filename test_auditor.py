"""
Tests for the Demographic Bias Auditor
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.auditor import DemographicBiasAuditor


@pytest.fixture
def auditor():
    return DemographicBiasAuditor(
        demographic_parity_threshold=0.1,
        disparate_impact_threshold=0.8,
        equalized_odds_threshold=0.1,
    )


def test_perfect_fairness(auditor):
    """No bias when both groups have identical outcomes."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    report = auditor.audit(y_true, y_pred, groups, "group")

    assert report.demographic_parity_diff == 0.0
    assert report.disparate_impact_ratio == 1.0
    assert report.equalized_odds_diff == 0.0
    assert any("No fairness violations" in f for f in report.flags)


def test_demographic_parity_violation(auditor):
    """Flags when one group is predicted positive far more often."""
    # Group A: 90% predicted positive, Group B: 10% predicted positive
    y_true = np.ones(20, dtype=int)
    y_pred = np.array([1]*9 + [0] + [1] + [0]*9)
    groups = np.array(["A"]*10 + ["B"]*10)

    report = auditor.audit(y_true, y_pred, groups, "group")

    assert report.demographic_parity_diff > 0.1
    assert any("DEMOGRAPHIC PARITY" in f for f in report.flags)


def test_disparate_impact_violation(auditor):
    """Flags when selection rate ratio falls below 0.8 (80% rule)."""
    y_true = np.ones(20, dtype=int)
    # Group A: 100% selected, Group B: 50% selected → ratio = 0.5
    y_pred = np.array([1]*10 + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.array(["A"]*10 + ["B"]*10)

    report = auditor.audit(y_true, y_pred, groups, "group")

    assert report.disparate_impact_ratio < 0.8
    assert any("DISPARATE IMPACT" in f for f in report.flags)


def test_equalized_odds_violation(auditor):
    """Flags when TPR differs significantly across groups."""
    # Group A: good TPR, Group B: poor TPR (misses positives)
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # all positive
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # Group A: all correct
                       0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Group B: 50% missed
    groups = np.array(["A"]*10 + ["B"]*10)

    report = auditor.audit(y_true, y_pred, groups, "group")

    assert report.equalized_odds_diff > 0.1
    assert any("EQUALIZED ODDS" in f for f in report.flags)


def test_group_metrics_count(auditor):
    """Report contains one GroupMetrics entry per unique group."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    groups = np.array(["X", "Y", "Z", "X", "Y", "Z"])

    report = auditor.audit(y_true, y_pred, groups, "group")

    assert len(report.group_metrics) == 3


def test_to_csv(auditor, tmp_path):
    """CSV export runs without error and contains expected columns."""
    import pandas as pd

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    groups = np.array(["A", "A", "B", "B"])

    report = auditor.audit(y_true, y_pred, groups, "group")
    out_path = str(tmp_path / "test_report.csv")
    auditor.to_csv(report, out_path)

    df = pd.read_csv(out_path)
    assert "group_value" in df.columns
    assert "selection_rate" in df.columns
    assert "disparate_impact_ratio" in df.columns
    assert len(df) == 2
