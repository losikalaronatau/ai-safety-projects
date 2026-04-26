"""Tests for the Toxicity Classifier."""
import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.classifier import ToxicityClassifier

@pytest.fixture
def clf(): return ToxicityClassifier(threshold=0.2)

def test_safe_text(clf):
    r = clf.classify("I love reading books and learning new things.")
    assert not r.is_toxic
    assert r.severity == "SAFE"

def test_harassment_flagged(clf):
    r = clf.classify("Nobody likes you, you are worthless and pathetic.")
    assert r.is_toxic
    assert any(c.category == "HARASSMENT" for c in r.category_scores)

def test_threat_flagged(clf):
    r = clf.classify("I will find you and make you pay for what you did.")
    assert r.is_toxic
    assert any(c.category == "THREAT" for c in r.category_scores)

def test_self_harm_flagged(clf):
    r = clf.classify("I want to end my life, there is no reason to live.")
    assert r.is_toxic
    assert any(c.category == "SELF_HARM" for c in r.category_scores)

def test_misinformation_flagged(clf):
    r = clf.classify("Vaccines cause autism, the government is hiding this.")
    assert r.is_toxic
    assert any(c.category == "MISINFORMATION" for c in r.category_scores)

def test_score_range(clf):
    for text in ["hello world", "I will kill you", "vaccines cause autism"]:
        r = clf.classify(text)
        assert 0.0 <= r.overall_score <= 1.0

def test_batch_report_counts(clf):
    texts = [
        "Have a great day!",
        "Nobody likes you, you are worthless.",
        "I will find you and hurt you.",
    ]
    report = clf.classify_batch(texts)
    assert report.total_texts == 3
    assert report.toxic_count + report.safe_count == 3

def test_batch_toxicity_rate(clf):
    texts = ["Good morning!"] * 5 + ["I will kill you."] * 5
    report = clf.classify_batch(texts)
    assert report.toxicity_rate > 0.0

def test_json_export(clf, tmp_path):
    import json
    report = clf.classify_batch(["Hello!", "You are worthless."])
    out = str(tmp_path / "tox.json")
    clf.to_json(report, out)
    data = json.load(open(out))
    assert "results" in data
    assert "category_breakdown" in data

def test_empty_text(clf):
    r = clf.classify("")
    assert not r.is_toxic
    assert r.overall_score == 0.0

def test_primary_category_set_for_toxic(clf):
    r = clf.classify("I will find you and kill you.")
    assert r.is_toxic
    assert r.primary_category is not None

def test_high_risk_captured_in_batch(clf):
    texts = ["Nice day!", "I will kill you, you worthless piece of garbage."]
    report = clf.classify_batch(texts)
    assert len(report.high_risk_texts) >= 1
