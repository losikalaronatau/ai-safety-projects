# 🔤 Word Embedding Bias Detector

Measures gender and racial bias encoded in word embeddings (Word2Vec, GloVe, fastText) using three complementary methods from the research literature.

## Methods

| Method | Paper | What It Measures |
|---|---|---|
| **WEAT** | Caliskan et al. 2017 | Association strength between demographic groups and concepts (career/family, science/arts) |
| **Direct Bias** | Bolukbasi et al. 2016 | How much occupation words project onto a gender axis |
| **Analogy Probing** | Mikolov et al. 2013 | Whether "man:doctor::woman:?" returns "doctor" or something stereotyped |

## Quickstart

```bash
pip install -r requirements.txt
python demo.py          # runs on synthetic biased embeddings
pytest tests/
```

## Use with Real Embeddings

```python
# GloVe (download from https://nlp.stanford.edu/projects/glove/)
from demo import load_glove
from src.detector import EmbeddingBiasDetector

embeddings = load_glove("glove.6B.100d.txt")
detector = EmbeddingBiasDetector(embeddings, "GloVe-6B-100d")
report = detector.run_all()
detector.print_report(report)
detector.to_json(report, "reports/glove_bias.json")
```

## Built-in WEAT Tests

| Test | Groups | Concepts |
|---|---|---|
| Gender × Career/Family | Male vs Female words | Career vs Family words |
| Gender × Science/Arts | Male vs Female words | Science vs Arts words |
| Race × Pleasant/Unpleasant | White-associated vs Black-associated names | Pleasant vs Unpleasant words |

## References

- [Semantics derived automatically from language corpora contain human-like biases — Caliskan et al. 2017](https://science.sciencemag.org/content/356/6334/183)
- [Man is to Computer Programmer as Woman is to Homemaker? — Bolukbasi et al. 2016](https://arxiv.org/abs/1607.06520)
