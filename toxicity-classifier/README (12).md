# ☢️ Toxicity Classifier

An AI safety tool that detects harmful, offensive, and dangerous text across 6 categories — with no external dependencies.

## Categories

| Category | Examples |
|---|---|
| **HATE_SPEECH** | Racial slurs, religious attacks, gender discrimination |
| **THREAT** | Direct or implied threats of violence |
| **HARASSMENT** | Bullying, personal attacks, targeted abuse |
| **PROFANITY** | Explicit or vulgar language |
| **SELF_HARM** | Content promoting suicide or self-harm |
| **MISINFORMATION** | Known false claims (anti-vax, flat earth, etc.) |

## Severity Levels

| Score | Level | |
|---|---|---|
| 0.0 – 0.2 | SAFE | ✅ |
| 0.2 – 0.4 | LOW | 🟡 |
| 0.4 – 0.65 | MEDIUM | 🟠 |
| 0.65 – 0.85 | HIGH | 🔴 |
| 0.85 – 1.0 | CRITICAL | 🚨 |

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
pytest tests/
```

## Usage

```python
from src.classifier import ToxicityClassifier

clf = ToxicityClassifier(threshold=0.2)

# Single text
result = clf.classify("I will find you and make you pay.")
print(result.severity)        # HIGH
print(result.primary_category) # THREAT

# Batch
report = clf.classify_batch(["Hello!", "You are worthless.", "Vaccines cause autism."])
clf.print_batch_report(report)
clf.to_json(report, "reports/toxicity_report.json")
```

## No External Dependencies
Pure Python standard library — no PyTorch, no transformers, no API keys needed.
