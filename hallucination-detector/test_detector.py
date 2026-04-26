"""Tests for the Hallucination Detector."""
import sys, os, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.detector import HallucinationDetector, lexical_overlap, cosine_similarity, \
    entity_consistency_score, split_into_claims, tfidf_vectors

SOURCE = """The Eiffel Tower is a wrought-iron lattice tower in Paris, France.
It was designed by Gustave Eiffel and built between 1887 and 1889 for the World's Fair.
It stands 330 metres tall and attracts 7 million visitors per year."""

GROUNDED = """The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel
and built in 1889 for the World's Fair. The tower is 330 metres tall."""

HALLUCINATED = """The Eiffel Tower is in London, England. It was built by Napoleon in 1750.
The tower is 999 metres tall and cost $50 billion. It has 20 floors and was painted green."""

@pytest.fixture
def det(): return HallucinationDetector(grounding_threshold=0.25)

# ── Unit tests ────────────────────────────────────────────────────────────────

def test_lexical_overlap_identical():
    assert lexical_overlap("the cat sat on the mat", "the cat sat on the mat") == 1.0

def test_lexical_overlap_no_overlap():
    score = lexical_overlap("purple elephant dances", "quantum physics rocket ship")
    assert score < 0.1

def test_lexical_overlap_partial():
    score = lexical_overlap("Paris is a beautiful city", "Paris France tower city")
    assert 0 < score < 1.0

def test_entity_score_all_present():
    score = entity_consistency_score("Gustave Eiffel built the tower in Paris.", SOURCE)
    assert score == 1.0

def test_entity_score_fabricated():
    score = entity_consistency_score("Napoleon Bonaparte demolished it in 1750.", SOURCE)
    assert score < 1.0

def test_split_into_claims():
    text = "The sky is a beautiful shade of blue. Water is always wet and cold. Fire is dangerously hot."
    claims = split_into_claims(text)
    assert len(claims) == 3

def test_split_filters_short():
    claims = split_into_claims("Hi. The sky is very blue today indeed.")
    assert all(len(c) > 15 for c in claims)

# ── Integration tests ─────────────────────────────────────────────────────────

def test_grounded_response_low_risk(det):
    report = det.check(GROUNDED, SOURCE)
    assert report.risk_label in ("LOW", "MEDIUM")
    assert report.hallucination_rate < 0.5

def test_hallucinated_response_high_risk(det):
    report = det.check(HALLUCINATED, SOURCE)
    assert report.hallucinated_claims > 0
    assert report.risk_score > report.hallucination_rate * 0.3

def test_grounded_fewer_flags_than_hallucinated(det):
    r_good = det.check(GROUNDED, SOURCE)
    r_bad  = det.check(HALLUCINATED, SOURCE)
    assert r_good.hallucinated_claims <= r_bad.hallucinated_claims

def test_report_claim_count(det):
    report = det.check(GROUNDED, SOURCE)
    assert report.total_claims == len(split_into_claims(GROUNDED))

def test_risk_score_range(det):
    report = det.check(HALLUCINATED, SOURCE)
    assert 0.0 <= report.risk_score <= 1.0

def test_json_export(det, tmp_path):
    import json
    report = det.check(GROUNDED, SOURCE)
    out = str(tmp_path / "h.json")
    det.to_json(report, out)
    data = json.load(open(out))
    assert "claim_results" in data
    assert "risk_label" in data

def test_empty_response(det):
    report = det.check("", SOURCE)
    assert report.total_claims == 0
    assert report.risk_score == 0.0
