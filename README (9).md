# 🛡️ AI Safety Projects

A collection of simple but crucial AI safety tools focused on **bias detection and fairness** in machine learning systems. Each project is self-contained, well-tested, and ready to plug into real ML pipelines.

---

## Projects

### 1. 🔍 [Demographic Bias Auditor](./bias-auditor/)
**Audit any classifier's predictions for fairness violations across demographic groups.**

Post-deployment audit tool. Plug in your model's `y_true` and `y_pred` and get a full fairness breakdown.

| Metric | Description |
|---|---|
| Demographic Parity Difference | Gap in positive prediction rates |
| Disparate Impact Ratio | The "80% rule" from employment law |
| Equalized Odds Difference | Gap in TPR/FPR across groups |

```bash
cd bias-auditor && python demo.py
```

---

### 2. 🧪 [LLM Response Bias Tester](./llm-bias-tester/)
**Detect bias in LLMs by sending counterfactual prompt pairs — identical prompts with only demographic details swapped.**

Tests gender, race (name-based), and age bias across 7 built-in templates. Works with any LLM via a simple API wrapper.

```bash
cd llm-bias-tester && python demo.py --mock       # no API key needed
cd llm-bias-tester && python demo.py --suite gender
```

---

### 3. 🗂️ [Dataset Bias Scanner](./dataset-bias-scanner/)
**Scan training data BEFORE model training to catch bias at the source.**

Catches problems early — before a biased model is ever trained or deployed.

| Check | What It Finds |
|---|---|
| Representation Imbalance | Groups with too few samples |
| Label Bias | Target label correlated with sensitive attribute |
| Proxy Features | Non-sensitive features that encode sensitive ones (Cramér's V) |
| Intersectional Imbalance | Underrepresented group combinations |

```bash
cd dataset-bias-scanner && python demo.py
```

---

### 4. 🔤 [Word Embedding Bias Detector](./embedding-bias-detector/)
**Measure bias baked into word embeddings using methods from the research literature.**

| Method | Paper |
|---|---|
| WEAT (Word Embedding Association Test) | Caliskan et al., *Science* 2017 |
| Direct Bias Score | Bolukbasi et al., NeurIPS 2016 |
| Analogy Probing | Mikolov et al. 2013 |

```bash
cd embedding-bias-detector && python demo.py
# Swap in real GloVe embeddings for production use
```

---

### 5. 📊 [Fairness-Aware Model Evaluator](./fairness-model-evaluator/)
**Full fairness scorecard for any trained binary classifier — goes beyond accuracy.**

Produces a per-group breakdown of accuracy, precision, recall, F1, FPR, selection rate, and NPV, plus a threshold sweep showing how fairness metrics shift across classification thresholds.

```bash
cd fairness-model-evaluator && python demo.py
```

---

## Test Coverage

| Project | Tests | Status |
|---|---|---|
| bias-auditor | 6 | ✅ |
| llm-bias-tester | 12 | ✅ |
| dataset-bias-scanner | 7 | ✅ |
| embedding-bias-detector | 8 | ✅ |
| fairness-model-evaluator | 8 | ✅ |
| **Total** | **41** | ✅ **All passing** |

---

## Run All Tests

```bash
for project in bias-auditor llm-bias-tester dataset-bias-scanner embedding-bias-detector fairness-model-evaluator; do
  echo "\n── $project ──"
  cd $project && pip install -r requirements.txt -q && pytest tests/ -v && cd ..
done
```

---

## The AI Safety Pipeline

These tools map onto the full ML lifecycle:

```
Data Collection → Model Training → Model Evaluation → Deployment → LLM Usage
       │                                  │                            │
  [Scanner]                        [Evaluator]                  [LLM Tester]
  [Embedding                       [Auditor]
   Detector]
```

Use them together:
1. **Before training** → Dataset Bias Scanner + Embedding Bias Detector
2. **After training** → Fairness-Aware Model Evaluator
3. **In production** → Demographic Bias Auditor
4. **For LLMs** → LLM Response Bias Tester

---

## References

- Caliskan et al. (2017). *Semantics derived automatically from language corpora contain human-like biases.* Science.
- Bolukbasi et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker?* NeurIPS.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning.* NeurIPS.
- Barocas, Hardt & Narayanan. *Fairness and Machine Learning.* [fairmlbook.org](https://fairmlbook.org)
