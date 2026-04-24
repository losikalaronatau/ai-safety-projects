"""Tests for Fairness-Aware Model Evaluator."""
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.evaluator import FairnessEvaluator

@pytest.fixture
def ev(): return FairnessEvaluator(dp_threshold=0.1, eo_threshold=0.1, di_threshold=0.8, pp_threshold=0.1)

def balanced():
    y_true = np.array([1,0]*50); y_pred = np.array([1,0]*50)
    groups = np.array(["A"]*50 + ["B"]*50)
    return y_true, y_pred, groups

def biased():
    y_true = np.array([1]*100 + [1]*100)
    y_pred = np.array([1]*90 + [0]*10 + [1]*40 + [0]*60)
    groups = np.array(["A"]*100 + ["B"]*100)
    return y_true, y_pred, groups

def test_no_flags_balanced(ev):
    r = ev.evaluate(*balanced(), "g", "m")
    assert any("passed" in f for f in r.fairness_metrics.flags)

def test_dp_flagged(ev):
    r = ev.evaluate(*biased(), "g", "m")
    assert any("DEMOGRAPHIC PARITY" in f for f in r.fairness_metrics.flags)

def test_eo_flagged(ev):
    r = ev.evaluate(*biased(), "g", "m")
    assert any("EQUALIZED ODDS" in f for f in r.fairness_metrics.flags)

def test_group_count(ev):
    r = ev.evaluate(*balanced(), "g", "m")
    assert len(r.group_scorecards) == 2

def test_overall_accuracy(ev):
    r = ev.evaluate(*balanced(), "g", "m")
    assert r.overall_accuracy == 1.0

def test_threshold_analysis(ev):
    y_true, _, groups = balanced()
    y_prob = np.random.default_rng(0).uniform(0, 1, len(y_true))
    r = ev.evaluate(y_true, (y_prob>0.5).astype(int), groups, "g", "m", y_prob)
    assert len(r.threshold_analysis) > 0

def test_json_export(ev, tmp_path):
    import json
    r = ev.evaluate(*balanced(), "g", "m")
    out = str(tmp_path / "sc.json")
    ev.to_json(r, out)
    d = json.load(open(out))
    assert "group_scorecards" in d and "fairness_metrics" in d

def test_di_ratio_range(ev):
    r = ev.evaluate(*biased(), "g", "m")
    assert 0.0 <= r.fairness_metrics.disparate_impact_ratio <= 1.0
