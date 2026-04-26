"""
Toxicity Classifier
====================
Flags harmful, offensive, or dangerous text across multiple categories:

  - HATE_SPEECH     : attacks on race, gender, religion, sexuality
  - THREAT          : direct or implied threats of violence
  - HARASSMENT      : targeted bullying or personal attacks
  - PROFANITY       : explicit/vulgar language
  - SELF_HARM       : content promoting self-harm or suicide
  - MISINFORMATION  : known false claim patterns
  - SAFE            : no toxicity detected

Works with no external dependencies — pure Python.
Uses a weighted lexicon + pattern matching approach.
"""

import re
import json
import string
from dataclasses import dataclass, field
from typing import Optional


# ─── Categories ───────────────────────────────────────────────────────────────

CATEGORIES = [
    "HATE_SPEECH",
    "THREAT",
    "HARASSMENT",
    "PROFANITY",
    "SELF_HARM",
    "MISINFORMATION",
]

# ─── Lexicons ─────────────────────────────────────────────────────────────────

LEXICON = {
    "HATE_SPEECH": {
        "keywords": [
            "subhuman", "vermin", "inferior race", "go back to your country",
            "not fully human", "racial slur", "ethnic cleansing", "white supremacy",
            "racial inferiority", "religious extremist", "you people are",
            "they are all", "those people are", "your kind",
        ],
        "patterns": [
            r"\b(all|every)\s+(muslims?|jews?|christians?|blacks?|whites?|hispanics?|asians?)\s+(are|should|must|deserve)",
            r"\b(women|men|girls|boys)\s+(are|belong|should|can\'t|cannot)\s+(only|just|always|never)",
            r"born\s+(inferior|superior|less|better)\s+(than|to)",
        ],
        "weight": 1.0,
    },
    "THREAT": {
        "keywords": [
            "i will kill", "i'm going to kill", "you will die", "i'll hurt you",
            "watch your back", "you better run", "i know where you live",
            "you won't survive", "i will find you", "make you pay",
            "going to destroy you", "end your life",
        ],
        "patterns": [
            r"\b(i('ll| will| am going to)|we('ll| will))\s+(kill|hurt|destroy|attack|find|shoot)\s+(you|him|her|them)",
            r"\byou('re| are)\s+(dead|finished|done)\s+(when|if|after)",
            r"\b(bomb|shoot|attack|destroy)\s+(the|this|your)\s+\w+",
        ],
        "weight": 1.0,
    },
    "HARASSMENT": {
        "keywords": [
            "nobody likes you", "you are worthless", "you are pathetic",
            "kill yourself", "no one cares about you", "you are a loser",
            "you should be ashamed", "you disgust me", "you are garbage",
            "everyone hates you", "you are disgusting", "go away forever",
            "you don't deserve", "you are ugly",
        ],
        "patterns": [
            r"\byou\s+(are|r|re)\s+(worthless|pathetic|disgusting|garbage|trash|useless|stupid|idiot|moron)",
            r"\bnobody\s+(likes|wants|cares\s+about)\s+you",
            r"\byou\s+(should|deserve\s+to)\s+(die|suffer|disappear)",
        ],
        "weight": 0.9,
    },
    "PROFANITY": {
        "keywords": [
            "fuck", "shit", "asshole", "bitch", "cunt", "bastard",
            "damn", "crap", "piss", "dick", "cock", "pussy",
        ],
        "patterns": [
            r"\bf+u+c+k+\w*",
            r"\bs+h+i+t+\w*",
            r"\ba+s+s+h+o+l+e+\w*",
        ],
        "weight": 0.5,   # lower weight — profanity alone is less severe
    },
    "SELF_HARM": {
        "keywords": [
            "want to die", "kill myself", "end my life", "suicide",
            "cut myself", "hurt myself", "don't want to live",
            "better off dead", "no reason to live", "can't go on",
            "taking my own life", "self harm", "overdose on",
        ],
        "patterns": [
            r"\b(i\s+)?(want|plan|going)\s+to\s+(kill|end|take)\s+(my(self)?|my\s+life|it\s+all)",
            r"\b(no|nothing)\s+(reason|point|purpose)\s+(to|in)\s+(live|living|going\s+on)",
            r"\bhow\s+to\s+(commit\s+suicide|kill\s+myself|end\s+my\s+life|overdose)",
        ],
        "weight": 1.0,
    },
    "MISINFORMATION": {
        "keywords": [
            "vaccines cause autism", "5g causes cancer", "flat earth",
            "covid is a hoax", "moon landing was fake", "chemtrails",
            "microchips in vaccines", "bill gates population control",
            "election was stolen", "deep state controls",
            "climate change is a hoax", "qanon",
        ],
        "patterns": [
            r"vaccines?\s+(cause|causes|caused|giving|spread)\s+(autism|cancer|death|disease)",
            r"(covid|coronavirus)\s+(is\s+a\s+)?(hoax|fake|made\s+up|not\s+real)",
            r"(moon\s+landing|apollo)\s+(was\s+)?(fake|faked|staged|never\s+happened)",
        ],
        "weight": 0.7,
    },
}

# ─── Severity Mapping ─────────────────────────────────────────────────────────

SEVERITY_LEVELS = {
    (0.0, 0.2):  ("SAFE",     "✅"),
    (0.2, 0.4):  ("LOW",      "🟡"),
    (0.4, 0.65): ("MEDIUM",   "🟠"),
    (0.65, 0.85):("HIGH",     "🔴"),
    (0.85, 1.01):("CRITICAL", "🚨"),
}


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class CategoryScore:
    category: str
    score: float           # 0.0 → 1.0
    triggered_by: list[str]  # which keywords/patterns matched


@dataclass
class ToxicityResult:
    text: str
    is_toxic: bool
    overall_score: float
    severity: str
    severity_icon: str
    primary_category: Optional[str]
    category_scores: list[CategoryScore]
    triggered_by: list[str]
    explanation: str


@dataclass
class BatchReport:
    total_texts: int
    toxic_count: int
    safe_count: int
    toxicity_rate: float
    category_breakdown: dict
    results: list[ToxicityResult] = field(default_factory=list)
    high_risk_texts: list[str] = field(default_factory=list)


# ─── Core Classifier ──────────────────────────────────────────────────────────

class ToxicityClassifier:
    """
    Classifies text for toxicity across 6 categories.

    Parameters
    ----------
    threshold : float
        Overall score above which text is flagged as toxic (default 0.2).
    min_category_score : float
        Minimum score for a category to be reported (default 0.1).
    """

    def __init__(self, threshold: float = 0.2, min_category_score: float = 0.1):
        self.threshold = threshold
        self.min_category_score = min_category_score

    def classify(self, text: str) -> ToxicityResult:
        """Classify a single piece of text for toxicity."""
        text_lower = text.lower()
        clean = text_lower.translate(str.maketrans("", "", string.punctuation))

        category_scores = []
        all_triggers = []

        for category, config in LEXICON.items():
            score = 0.0
            triggers = []

            # Keyword matching
            for kw in config["keywords"]:
                if kw in text_lower:
                    score += 0.4
                    triggers.append(f'keyword: "{kw}"')

            # Pattern matching
            for pattern in config["patterns"]:
                match = re.search(pattern, text_lower)
                if match:
                    score += 0.5
                    triggers.append(f'pattern: "{match.group(0).strip()}"')

            # Normalize score to 0–1
            score = min(round(score * config["weight"], 4), 1.0)

            if score >= self.min_category_score:
                category_scores.append(CategoryScore(
                    category=category,
                    score=score,
                    triggered_by=triggers,
                ))
                all_triggers.extend(triggers)

        # Overall score = weighted max + average blend
        if category_scores:
            scores = [c.score for c in category_scores]
            overall = round(min(max(scores) * 0.7 + sum(scores)/len(scores) * 0.3, 1.0), 4)
        else:
            overall = 0.0

        # Severity
        severity, icon = "SAFE", "✅"
        for (lo, hi), (label, emoji) in SEVERITY_LEVELS.items():
            if lo <= overall < hi:
                severity, icon = label, emoji
                break

        # Primary category
        primary = max(category_scores, key=lambda c: c.score).category if category_scores else None

        is_toxic = overall >= self.threshold

        # Explanation
        if not is_toxic:
            explanation = "No significant toxicity detected."
        else:
            cats = ", ".join(c.category for c in sorted(category_scores, key=lambda c: -c.score)[:2])
            explanation = f"Detected {severity.lower()} toxicity. Primary concern: {cats}."

        return ToxicityResult(
            text=text,
            is_toxic=is_toxic,
            overall_score=overall,
            severity=severity,
            severity_icon=icon,
            primary_category=primary,
            category_scores=sorted(category_scores, key=lambda c: -c.score),
            triggered_by=all_triggers[:5],
            explanation=explanation,
        )

    def classify_batch(self, texts: list[str]) -> BatchReport:
        """Classify a list of texts and return a full batch report."""
        results = [self.classify(t) for t in texts]
        toxic   = [r for r in results if r.is_toxic]
        safe    = [r for r in results if not r.is_toxic]

        # Category breakdown
        cat_breakdown = {cat: 0 for cat in CATEGORIES}
        for r in toxic:
            if r.primary_category:
                cat_breakdown[r.primary_category] += 1

        return BatchReport(
            total_texts=len(results),
            toxic_count=len(toxic),
            safe_count=len(safe),
            toxicity_rate=round(len(toxic)/len(results), 4) if results else 0.0,
            category_breakdown=cat_breakdown,
            results=results,
            high_risk_texts=[r.text for r in results if r.severity in ("HIGH","CRITICAL")],
        )

    # ─── Reporting ────────────────────────────────────────────────────────────

    def print_result(self, result: ToxicityResult) -> None:
        print(f"\n  {result.severity_icon}  [{result.severity}] score={result.overall_score:.3f}  toxic={result.is_toxic}")
        print(f"  Text    : {result.text[:80]}{'...' if len(result.text)>80 else ''}")
        print(f"  Verdict : {result.explanation}")
        if result.category_scores:
            print(f"  Categories:")
            for cs in result.category_scores:
                bar = "█" * int(cs.score * 20)
                print(f"    {cs.category:<18} {cs.score:.3f}  {bar}")
        if result.triggered_by:
            print(f"  Triggered by: {result.triggered_by[0]}")

    def print_batch_report(self, report: BatchReport) -> None:
        print("\n" + "=" * 60)
        print("  TOXICITY BATCH REPORT")
        print("=" * 60)
        print(f"  Total Texts    : {report.total_texts}")
        print(f"  Toxic          : {report.toxic_count} ({report.toxicity_rate:.0%})")
        print(f"  Safe           : {report.safe_count}")
        print()
        print("  Category Breakdown:")
        for cat, count in report.category_breakdown.items():
            if count > 0:
                print(f"    {cat:<20} {count} texts")
        print()
        print("  Per-Text Results:")
        for r in report.results:
            flag = "🚨" if r.is_toxic else "✅"
            print(f"    {flag} [{r.severity:<8}] {r.overall_score:.3f}  {r.text[:55]}{'...' if len(r.text)>55 else ''}")
        print("=" * 60 + "\n")

    def to_json(self, report: BatchReport, path: str) -> None:
        data = {
            "total_texts": report.total_texts,
            "toxic_count": report.toxic_count,
            "toxicity_rate": report.toxicity_rate,
            "category_breakdown": report.category_breakdown,
            "high_risk_texts": report.high_risk_texts,
            "results": [
                {
                    "text": r.text,
                    "is_toxic": r.is_toxic,
                    "overall_score": r.overall_score,
                    "severity": r.severity,
                    "primary_category": r.primary_category,
                    "explanation": r.explanation,
                    "category_scores": [
                        {"category": c.category, "score": c.score}
                        for c in r.category_scores
                    ],
                }
                for r in report.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Report saved to: {path}")
