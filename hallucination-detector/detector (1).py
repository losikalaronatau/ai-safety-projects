"""
Hallucination Detector
======================
Detects when an LLM response contains claims not supported by a source document.

How it works:
  1. Splits the LLM response into individual claims (sentences)
  2. For each claim, checks if it is entailed by the source context using:
       - Lexical overlap (n-gram matching)
       - Semantic similarity (cosine similarity of TF-IDF vectors)
       - Entity consistency (named entities in claim must appear in source)
  3. Assigns each claim a grounding score (0–1)
  4. Flags claims below the threshold as potential hallucinations
  5. Returns an overall hallucination risk score for the response
"""

import re
import json
import math
import string
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ClaimResult:
    claim: str
    grounding_score: float       # 0.0 (hallucinated) → 1.0 (fully grounded)
    lexical_score: float         # n-gram overlap with source
    semantic_score: float        # TF-IDF cosine similarity
    entity_score: float          # entity consistency
    is_hallucination: bool
    flag_reason: str = ""


@dataclass
class HallucinationReport:
    response: str
    source: str
    total_claims: int
    hallucinated_claims: int
    grounded_claims: int
    hallucination_rate: float        # fraction of claims flagged
    risk_score: float                # 0.0 (safe) → 1.0 (high risk)
    risk_label: str                  # "LOW" / "MEDIUM" / "HIGH"
    claim_results: list[ClaimResult] = field(default_factory=list)
    flagged_claims: list[str] = field(default_factory=list)
    summary: str = ""


# ─── Text Utilities ───────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w]

def ngrams(tokens: list[str], n: int) -> set:
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def lexical_overlap(claim: str, source: str) -> float:
    """ROUGE-style n-gram overlap between claim and source."""
    c_tokens = tokenize(claim)
    s_tokens = tokenize(source)
    if not c_tokens:
        return 0.0

    scores = []
    for n in [1, 2]:
        c_ng = ngrams(c_tokens, n)
        s_ng = ngrams(s_tokens, n)
        if not c_ng:
            continue
        overlap = len(c_ng & s_ng) / len(c_ng)
        scores.append(overlap)

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def tfidf_vectors(docs: list[str]) -> list[dict]:
    """Compute TF-IDF vectors for a list of documents."""
    tokenized = [tokenize(d) for d in docs]
    # IDF
    n_docs = len(docs)
    df = Counter()
    for tokens in tokenized:
        for w in set(tokens):
            df[w] += 1
    idf = {w: math.log(n_docs / (1 + df[w])) for w in df}
    # TF-IDF
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {w: (tf[w]/total) * idf.get(w, 0) for w in tf}
        vectors.append(vec)
    return vectors

def cosine_similarity(v1: dict, v2: dict) -> float:
    """Cosine similarity between two TF-IDF sparse vectors."""
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[w] * v2[w] for w in common)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    return round(dot / (mag1 * mag2 + 1e-10), 4)

def extract_entities(text: str) -> set[str]:
    """
    Lightweight entity extraction — capitalized words/phrases not at sentence start.
    Not a full NER but effective for catching fabricated names, places, numbers, dates.
    """
    entities = set()

    # Numbers and years
    entities.update(re.findall(r'\b\d{4}\b', text))               # years
    entities.update(re.findall(r'\b\d+\.?\d*\s*%', text))         # percentages
    entities.update(re.findall(r'\$[\d,]+', text))                 # dollar amounts
    entities.update(re.findall(r'\b\d+[\.,]?\d+\b', text))        # numbers

    # Capitalized multi-word phrases (likely proper nouns)
    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for phrase in cap_phrases:
        if phrase not in {"The","This","These","That","Those","It","He","She","They",
                          "We","I","In","On","At","By","For","With","A","An"}:
            entities.add(phrase)

    return entities

def entity_consistency_score(claim: str, source: str) -> float:
    """
    Check what fraction of entities in the claim also appear in the source.
    Fabricated entities (names, dates, numbers) that aren't in the source → low score.
    """
    claim_entities = extract_entities(claim)
    if not claim_entities:
        return 1.0  # no checkable entities → neutral

    source_lower = source.lower()
    found = sum(1 for e in claim_entities if e.lower() in source_lower)
    return round(found / len(claim_entities), 4)


def split_into_claims(text: str) -> list[str]:
    """Split response text into individual claim sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


# ─── Core Detector ────────────────────────────────────────────────────────────

class HallucinationDetector:
    """
    Detects hallucinations in LLM responses by checking grounding against a source.

    Parameters
    ----------
    grounding_threshold : float
        Claims with grounding score below this are flagged (default 0.25).
    lexical_weight : float
        Weight for lexical overlap in grounding score (default 0.35).
    semantic_weight : float
        Weight for semantic similarity in grounding score (default 0.40).
    entity_weight : float
        Weight for entity consistency in grounding score (default 0.25).
    """

    def __init__(
        self,
        grounding_threshold: float = 0.25,
        lexical_weight: float = 0.35,
        semantic_weight: float = 0.40,
        entity_weight: float = 0.25,
    ):
        self.threshold = grounding_threshold
        self.lw = lexical_weight
        self.sw = semantic_weight
        self.ew = entity_weight

    def check(
        self,
        response: str,
        source: str,
    ) -> HallucinationReport:
        """
        Check an LLM response for hallucinations against a source document.

        Parameters
        ----------
        response : str — the LLM's generated response
        source   : str — the ground truth / reference document

        Returns
        -------
        HallucinationReport
        """
        claims = split_into_claims(response)
        if not claims:
            return HallucinationReport(
                response=response, source=source,
                total_claims=0, hallucinated_claims=0, grounded_claims=0,
                hallucination_rate=0.0, risk_score=0.0,
                risk_label="LOW", summary="No checkable claims found."
            )

        # Compute TF-IDF over source + all claims together for shared vocab
        all_docs = [source] + claims
        vectors  = tfidf_vectors(all_docs)
        source_vec = vectors[0]
        claim_vecs = vectors[1:]

        claim_results = []
        for i, claim in enumerate(claims):
            lex  = lexical_overlap(claim, source)
            sem  = cosine_similarity(claim_vecs[i], source_vec)
            ent  = entity_consistency_score(claim, source)

            score = round(self.lw*lex + self.sw*sem + self.ew*ent, 4)
            is_hall = score < self.threshold

            reason = ""
            if is_hall:
                parts = []
                if lex  < 0.1: parts.append("low lexical overlap")
                if sem  < 0.1: parts.append("semantically distant from source")
                if ent  < 0.5: parts.append("contains entities not in source")
                reason = "; ".join(parts) if parts else "below grounding threshold"

            claim_results.append(ClaimResult(
                claim=claim,
                grounding_score=score,
                lexical_score=lex,
                semantic_score=sem,
                entity_score=ent,
                is_hallucination=is_hall,
                flag_reason=reason,
            ))

        hallucinated = [r for r in claim_results if r.is_hallucination]
        grounded     = [r for r in claim_results if not r.is_hallucination]
        hall_rate    = round(len(hallucinated) / len(claims), 4)
        risk_score   = round(
            hall_rate * 0.6 +
            (1 - (sum(r.grounding_score for r in claim_results) / len(claims))) * 0.4,
            4
        )

        if risk_score < 0.3:
            risk_label = "LOW"
        elif risk_score < 0.6:
            risk_label = "MEDIUM"
        else:
            risk_label = "HIGH"

        summary = (
            f"{len(hallucinated)}/{len(claims)} claims flagged as potential hallucinations. "
            f"Risk: {risk_label} ({risk_score:.2f})"
        )

        return HallucinationReport(
            response=response,
            source=source,
            total_claims=len(claims),
            hallucinated_claims=len(hallucinated),
            grounded_claims=len(grounded),
            hallucination_rate=hall_rate,
            risk_score=risk_score,
            risk_label=risk_label,
            claim_results=claim_results,
            flagged_claims=[r.claim for r in hallucinated],
            summary=summary,
        )

    def print_report(self, report: HallucinationReport) -> None:
        risk_icon = {"LOW": "✅", "MEDIUM": "⚠️ ", "HIGH": "🚨"}[report.risk_label]
        print("\n" + "=" * 68)
        print(f"  HALLUCINATION DETECTION REPORT")
        print("=" * 68)
        print(f"  Risk Level     : {risk_icon}  {report.risk_label} (score={report.risk_score:.2f})")
        print(f"  Total Claims   : {report.total_claims}")
        print(f"  Hallucinated   : {report.hallucinated_claims}  ({report.hallucination_rate:.0%})")
        print(f"  Grounded       : {report.grounded_claims}")
        print()
        print("  Claim-by-Claim Analysis:")
        print(f"  {'#':<3} {'Grounded?':<12} {'Score':<7} {'Lex':<7} {'Sem':<7} {'Ent':<7}  Claim")
        print("  " + "─" * 80)
        for i, r in enumerate(report.claim_results, 1):
            status = "🚨 FLAGGED" if r.is_hallucination else "✅ OK     "
            snippet = r.claim[:60] + "..." if len(r.claim) > 60 else r.claim
            print(f"  {i:<3} {status:<12} {r.grounding_score:<7.3f} {r.lexical_score:<7.3f} "
                  f"{r.semantic_score:<7.3f} {r.entity_score:<7.3f}  {snippet}")
            if r.is_hallucination:
                print(f"       ↳ Reason: {r.flag_reason}")
        print()
        print(f"  Summary: {report.summary}")
        print("=" * 68 + "\n")

    def to_json(self, report: HallucinationReport, path: str) -> None:
        data = {
            "risk_label": report.risk_label,
            "risk_score": report.risk_score,
            "hallucination_rate": report.hallucination_rate,
            "total_claims": report.total_claims,
            "hallucinated_claims": report.hallucinated_claims,
            "grounded_claims": report.grounded_claims,
            "summary": report.summary,
            "flagged_claims": report.flagged_claims,
            "claim_results": [
                {"claim": r.claim, "grounding_score": r.grounding_score,
                 "lexical_score": r.lexical_score, "semantic_score": r.semantic_score,
                 "entity_score": r.entity_score, "is_hallucination": r.is_hallucination,
                 "flag_reason": r.flag_reason}
                for r in report.claim_results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Report saved to: {path}")
