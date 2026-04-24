"""
Word Embedding Bias Detector
=============================
Measures gender and racial bias baked into word embeddings using:

  1. WEAT (Word Embedding Association Test) — Caliskan et al. 2017
     Measures how strongly target concepts (e.g. career vs family) associate
     with attribute groups (e.g. male vs female names/words).

  2. Direct Bias Score — Bolukbasi et al. 2016
     Measures how much occupation words project onto a gender direction.

  3. Analogy Probing
     Tests "man is to X as woman is to ?" style associations.

Works with any embedding dict {word: np.ndarray}.
Includes a lightweight mock embeddings builder for testing without
downloading large model files.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class WEATResult:
    test_name: str
    target_set_A: list[str]        # e.g. male names
    target_set_B: list[str]        # e.g. female names
    attribute_set_X: list[str]     # e.g. career words
    attribute_set_Y: list[str]     # e.g. family words
    effect_size: float             # Cohen's d — >0 means A associates with X
    p_value_approx: float
    flagged: bool
    interpretation: str = ""


@dataclass
class DirectBiasResult:
    word: str
    bias_score: float              # projection onto gender direction, -1 to 1
    direction: str                 # "male-leaning" or "female-leaning" or "neutral"


@dataclass
class DirectBiasReport:
    bias_direction_words: tuple[str, str]   # e.g. ("he", "she")
    words_tested: list[str]
    results: list[DirectBiasResult]
    mean_absolute_bias: float
    most_biased: list[DirectBiasResult]     # top 5 most biased words
    flagged: bool
    flag_message: str = ""


@dataclass
class AnalogyResult:
    query: str                     # e.g. "man : doctor :: woman : ?"
    top_predictions: list[str]
    expected_neutral: str          # what an unbiased model would say
    flagged: bool
    flag_message: str = ""


@dataclass
class EmbeddingBiasReport:
    embedding_name: str
    vocab_size: int
    weat_results: list[WEATResult] = field(default_factory=list)
    direct_bias_report: Optional[DirectBiasReport] = None
    analogy_results: list[AnalogyResult] = field(default_factory=list)
    total_flags: int = 0
    summary: list[str] = field(default_factory=list)


# ─── WEAT Test Definitions ────────────────────────────────────────────────────

WEAT_TESTS = {
    "gender_career_family": {
        "name": "Gender × Career vs Family",
        "target_A": ["brother", "male", "man", "boy", "son", "he", "his", "him"],
        "target_B": ["sister", "female", "woman", "girl", "daughter", "she", "her", "hers"],
        "attr_X":   ["executive", "management", "professional", "corporation", "salary",
                     "office", "business", "career"],
        "attr_Y":   ["home", "parents", "children", "family", "cousins", "marriage",
                     "wedding", "relatives"],
    },
    "gender_science_arts": {
        "name": "Gender × Science vs Arts",
        "target_A": ["brother", "male", "man", "boy", "son", "he", "his", "him"],
        "target_B": ["sister", "female", "woman", "girl", "daughter", "she", "her", "hers"],
        "attr_X":   ["science", "technology", "physics", "chemistry", "einstein",
                     "nasa", "experiment", "astronomy"],
        "attr_Y":   ["poetry", "art", "shakespeare", "dance", "literature",
                     "novel", "symphony", "drama"],
    },
    "race_pleasant_unpleasant": {
        "name": "Race × Pleasant vs Unpleasant",
        "target_A": ["adam", "chip", "harry", "josh", "roger", "alan", "frank", "ian"],
        "target_B": ["alonzo", "jamel", "lerone", "percell", "theo", "torrance", "darnell", "lamar"],
        "attr_X":   ["joy", "love", "peace", "wonderful", "pleasure", "beautiful", "lucky", "happy"],
        "attr_Y":   ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"],
    },
}

OCCUPATION_WORDS = [
    "nurse", "doctor", "engineer", "teacher", "programmer", "receptionist",
    "manager", "secretary", "pilot", "homemaker", "scientist", "librarian",
    "architect", "nanny", "surgeon", "babysitter", "ceo", "housekeeper",
    "analyst", "midwife", "professor", "janitor", "designer", "mechanic",
]


# ─── Core Detector ────────────────────────────────────────────────────────────

class EmbeddingBiasDetector:
    """
    Detects bias in word embeddings.

    Parameters
    ----------
    embeddings : dict mapping word -> np.ndarray
    embedding_name : display name
    weat_effect_size_threshold : flag if |effect_size| > this (default 0.5)
    direct_bias_threshold : flag if mean absolute bias > this (default 0.08)
    """

    def __init__(
        self,
        embeddings: dict,
        embedding_name: str = "embeddings",
        weat_effect_size_threshold: float = 0.5,
        direct_bias_threshold: float = 0.08,
    ):
        self.embeddings = {w.lower(): v for w, v in embeddings.items()}
        self.name = embedding_name
        self.weat_threshold = weat_effect_size_threshold
        self.direct_bias_threshold = direct_bias_threshold

    def run_all(self, occupation_words: Optional[list[str]] = None) -> EmbeddingBiasReport:
        """Run all bias tests and return a full report."""
        report = EmbeddingBiasReport(
            embedding_name=self.name,
            vocab_size=len(self.embeddings),
        )

        print(f"\n  Analyzing embeddings: {len(self.embeddings):,} words\n")

        # 1. WEAT tests
        print("  [1/3] Running WEAT association tests...")
        for key, test in WEAT_TESTS.items():
            result = self._run_weat(
                test["name"], test["target_A"], test["target_B"],
                test["attr_X"], test["attr_Y"]
            )
            report.weat_results.append(result)
            status = "⚠️  FLAGGED" if result.flagged else "✅  OK"
            print(f"         {test['name']}: effect_size={result.effect_size:.3f}  [{status}]")

        # 2. Direct Bias
        print("\n  [2/3] Running Direct Bias (gender direction)...")
        occ_words = occupation_words or OCCUPATION_WORDS
        db_report = self._run_direct_bias(occ_words)
        report.direct_bias_report = db_report
        status = "⚠️  FLAGGED" if db_report.flagged else "✅  OK"
        print(f"         Mean absolute bias: {db_report.mean_absolute_bias:.4f}  [{status}]")

        # 3. Analogy probing
        print("\n  [3/3] Running analogy probes...")
        analogy_queries = [
            ("man", "doctor",    "woman",  "doctor"),
            ("man", "engineer",  "woman",  "engineer"),
            ("man", "nurse",     "woman",  "nurse"),
            ("man", "ceo",       "woman",  "ceo"),
        ]
        for a, b, c, expected in analogy_queries:
            result = self._run_analogy(a, b, c, expected)
            if result:
                report.analogy_results.append(result)
                status = "⚠️  FLAGGED" if result.flagged else "✅  OK"
                print(f"         {result.query}  [{status}]")

        # Summary
        flags = (
            sum(1 for r in report.weat_results if r.flagged)
            + (1 if db_report.flagged else 0)
            + sum(1 for r in report.analogy_results if r.flagged)
        )
        report.total_flags = flags
        report.summary = self._build_summary(report)
        return report

    # ─── WEAT ─────────────────────────────────────────────────────────────────

    def _run_weat(self, name, A, B, X, Y) -> WEATResult:
        A = [w for w in A if w in self.embeddings]
        B = [w for w in B if w in self.embeddings]
        X = [w for w in X if w in self.embeddings]
        Y = [w for w in Y if w in self.embeddings]

        if not A or not B or not X or not Y:
            return WEATResult(name, A, B, X, Y, 0.0, 1.0, False, "insufficient vocabulary")

        def assoc(w, X, Y):
            vw = self.embeddings[w]
            return np.mean([self._cos(vw, self.embeddings[x]) for x in X]) - \
                   np.mean([self._cos(vw, self.embeddings[y]) for y in Y])

        s_A = np.array([assoc(w, X, Y) for w in A])
        s_B = np.array([assoc(w, X, Y) for w in B])

        diff = np.mean(s_A) - np.mean(s_B)
        pooled_std = np.std(np.concatenate([s_A, s_B]))
        effect_size = round(float(diff / pooled_std) if pooled_std > 0 else 0.0, 4)

        # Approximate p-value via permutation (1000 samples)
        combined = np.concatenate([s_A, s_B])
        n_A = len(s_A)
        extreme = 0
        for _ in range(1000):
            perm = np.random.permutation(combined)
            perm_diff = np.mean(perm[:n_A]) - np.mean(perm[n_A:])
            if abs(perm_diff) >= abs(diff):
                extreme += 1
        p_value = round((extreme + 1) / 1001, 4)

        flagged = abs(effect_size) > self.weat_threshold
        interp = (
            f"{'Strong' if abs(effect_size)>1 else 'Moderate'} association: "
            f"'{A[0]}'-like words → {'X' if effect_size>0 else 'Y'} attributes "
            f"(d={effect_size:.2f})"
            if flagged else f"No significant association (d={effect_size:.2f})"
        )
        return WEATResult(name, A, B, X, Y, effect_size, p_value, flagged, interp)

    # ─── Direct Bias ──────────────────────────────────────────────────────────

    def _run_direct_bias(self, words: list[str]) -> DirectBiasReport:
        gender_pairs = [("he","she"), ("man","woman"), ("boy","girl"), ("male","female")]
        valid_pairs = [(a, b) for a, b in gender_pairs
                       if a in self.embeddings and b in self.embeddings]

        if not valid_pairs:
            return DirectBiasReport(("he","she"), words, [], 0.0, [], False, "gender words not in vocab")

        # Gender direction = mean of (he-she, man-woman, ...)
        gender_dir = np.mean([
            self.embeddings[a] - self.embeddings[b]
            for a, b in valid_pairs
        ], axis=0)
        gender_dir = gender_dir / (np.linalg.norm(gender_dir) + 1e-10)

        results = []
        for w in words:
            if w not in self.embeddings:
                continue
            v = self.embeddings[w]
            proj = float(np.dot(v / (np.linalg.norm(v) + 1e-10), gender_dir))
            direction = "male-leaning" if proj > 0.05 else ("female-leaning" if proj < -0.05 else "neutral")
            results.append(DirectBiasResult(word=w, bias_score=round(proj, 4), direction=direction))

        if not results:
            return DirectBiasReport(valid_pairs[0], words, [], 0.0, [], False)

        mean_abs = round(float(np.mean([abs(r.bias_score) for r in results])), 4)
        top5 = sorted(results, key=lambda r: abs(r.bias_score), reverse=True)[:5]
        flagged = mean_abs > self.direct_bias_threshold
        msg = (
            f"⚠️  HIGH DIRECT BIAS: mean absolute projection onto gender direction = {mean_abs:.4f}"
            if flagged else ""
        )
        return DirectBiasReport(valid_pairs[0], words, results, mean_abs, top5, flagged, msg)

    # ─── Analogy ──────────────────────────────────────────────────────────────

    def _run_analogy(self, a: str, b: str, c: str, expected: str) -> Optional[AnalogyResult]:
        for w in [a, b, c]:
            if w not in self.embeddings:
                return None
        # d = b - a + c
        query_vec = self.embeddings[b] - self.embeddings[a] + self.embeddings[c]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        scores = {
            w: self._cos(query_vec, self.embeddings[w])
            for w in self.embeddings
            if w not in {a, b, c}
        }
        top = sorted(scores, key=scores.get, reverse=True)[:5]
        flagged = expected not in top
        msg = (
            f"⚠️  '{a} : {b} :: {c} : ?' → got {top[0]!r} instead of {expected!r}"
            if flagged else ""
        )
        return AnalogyResult(
            query=f"{a} : {b} :: {c} : ?",
            top_predictions=top,
            expected_neutral=expected,
            flagged=flagged,
            flag_message=msg,
        )

    # ─── Utils ────────────────────────────────────────────────────────────────

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _build_summary(self, report: EmbeddingBiasReport) -> list[str]:
        lines = []
        for r in report.weat_results:
            if r.flagged:
                lines.append(f"⚠️  WEAT [{r.test_name}]: {r.interpretation}")
        if report.direct_bias_report and report.direct_bias_report.flagged:
            lines.append(report.direct_bias_report.flag_message)
        for r in report.analogy_results:
            if r.flagged:
                lines.append(r.flag_message)
        if not lines:
            lines.append("✅  No significant embedding bias detected.")
        return lines

    # ─── Reporting ────────────────────────────────────────────────────────────

    def print_report(self, report: EmbeddingBiasReport) -> None:
        print("\n" + "=" * 70)
        print(f"  EMBEDDING BIAS REPORT — {report.embedding_name}")
        print("=" * 70)
        print(f"  Vocab Size   : {report.vocab_size:,}")
        print(f"  Total Flags  : {report.total_flags}")
        print()

        print("  ── WEAT Association Tests")
        for r in report.weat_results:
            status = "⚠️  FLAGGED" if r.flagged else "✅  OK"
            print(f"     {r.test_name:<35} d={r.effect_size:+.3f}  p≈{r.p_value_approx:.3f}  [{status}]")
            print(f"       {r.interpretation}")
        print()

        if report.direct_bias_report:
            db = report.direct_bias_report
            status = "⚠️  FLAGGED" if db.flagged else "✅  OK"
            print(f"  ── Direct Bias (gender direction)  [{status}]")
            print(f"     Mean absolute bias: {db.mean_absolute_bias:.4f}")
            print(f"     Most biased occupation words:")
            for r in db.most_biased:
                bar = "█" * int(abs(r.bias_score) * 40)
                sign = "+" if r.bias_score > 0 else "-"
                print(f"       {r.word:<15} {sign}{abs(r.bias_score):.4f}  {bar}  ({r.direction})")
        print()

        if report.analogy_results:
            print("  ── Analogy Probes")
            for r in report.analogy_results:
                status = "⚠️  FLAGGED" if r.flagged else "✅  OK"
                print(f"     {r.query:<30} → {r.top_predictions[0]!r:<15}  [{status}]")
        print()

        print("  ── Summary")
        for line in report.summary:
            print(f"     {line}")
        print("=" * 70 + "\n")

    def to_json(self, report: EmbeddingBiasReport, path: str) -> None:
        data = {
            "embedding_name": report.embedding_name,
            "vocab_size": report.vocab_size,
            "total_flags": report.total_flags,
            "summary": report.summary,
            "weat_results": [
                {"test_name": r.test_name, "effect_size": r.effect_size,
                 "p_value_approx": r.p_value_approx, "flagged": r.flagged,
                 "interpretation": r.interpretation}
                for r in report.weat_results
            ],
            "direct_bias": {
                "mean_absolute_bias": report.direct_bias_report.mean_absolute_bias,
                "flagged": report.direct_bias_report.flagged,
                "most_biased": [
                    {"word": r.word, "bias_score": r.bias_score, "direction": r.direction}
                    for r in report.direct_bias_report.most_biased
                ],
            } if report.direct_bias_report else None,
            "analogy_results": [
                {"query": r.query, "top_predictions": r.top_predictions,
                 "flagged": r.flagged, "flag_message": r.flag_message}
                for r in report.analogy_results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Report saved to: {path}")
