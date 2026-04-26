# 🔬 Hallucination Detector

An AI safety tool that checks whether an LLM's response is **grounded in a source document** — catching fabricated facts, invented statistics, and made-up entities before they cause harm.

## How It Works

Each sentence in the LLM response is scored against the source using three signals:

| Signal | Method | Weight |
|---|---|---|
| **Lexical Overlap** | ROUGE-style n-gram matching | 35% |
| **Semantic Similarity** | TF-IDF cosine similarity | 40% |
| **Entity Consistency** | Are names, dates, numbers in the source? | 25% |

Claims below the grounding threshold are flagged as potential hallucinations.

## Risk Levels

| Score | Label | Meaning |
|---|---|---|
| 0.0 – 0.3 | ✅ LOW | Response is well grounded |
| 0.3 – 0.6 | ⚠️ MEDIUM | Some unsupported claims |
| 0.6 – 1.0 | 🚨 HIGH | Heavily hallucinated |

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
pytest tests/
```

## Usage

```python
from src.detector import HallucinationDetector

detector = HallucinationDetector(grounding_threshold=0.25)

report = detector.check(
    response="The Eiffel Tower was built in London in 1750 by Napoleon.",
    source="The Eiffel Tower is in Paris, France. It was built in 1889 by Gustave Eiffel."
)

detector.print_report(report)
# Risk Level: 🚨 HIGH (score=0.82)
# 3/3 claims flagged as potential hallucinations.
```

## No External Dependencies

Works entirely with Python's standard library — no PyTorch, no transformers, no API keys needed.

## References

- [Survey of Hallucination in Natural Language Generation — Ji et al. 2022](https://arxiv.org/abs/2202.03629)
- [TruthfulQA — Lin et al. 2021](https://arxiv.org/abs/2109.07958)
