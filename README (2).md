# 🔍 Demographic Bias Auditor

A simple but powerful AI safety tool that audits any binary classifier for **demographic bias and fairness violations**.

## Why This Matters

Biased ML models can cause real harm — unfair loan decisions, discriminatory hiring tools, or skewed healthcare predictions. This auditor gives you a clear, quantified picture of how your model treats different demographic groups.

---

## Fairness Metrics Covered

| Metric | What It Measures | Violation Threshold |
|---|---|---|
| **Demographic Parity Difference** | Gap in positive prediction rates between groups | > 0.1 |
| **Disparate Impact Ratio** | Ratio of selection rates (the "80% rule") | < 0.8 |
| **Equalized Odds Difference** | Gap in TPR/FPR between groups | > 0.1 |

---

## Project Structure

```
bias-auditor/
├── src/
│   └── auditor.py        # Core DemographicBiasAuditor class
├── tests/
│   └── test_auditor.py   # Pytest test suite
├── reports/              # CSV outputs (auto-created)
├── demo.py               # Demo on synthetic Adult Income-style data
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py

# Run tests
pytest tests/
```

---

## Usage

```python
from src.auditor import DemographicBiasAuditor
import numpy as np

auditor = DemographicBiasAuditor(
    demographic_parity_threshold=0.1,
    disparate_impact_threshold=0.8,
    equalized_odds_threshold=0.1,
)

# y_true, y_pred: your model's ground truth and predictions (0/1 arrays)
# sensitive_feature: demographic labels (e.g. "Male"/"Female")
report = auditor.audit(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_feature=gender_labels,
    feature_name="sex"
)

auditor.print_report(report)
auditor.to_csv(report, "reports/gender_bias.csv")
```

### Example Output

```
============================================================
  FAIRNESS AUDIT REPORT — 'sex'
============================================================
  Overall Accuracy     : 74.33%
  Overall Selection Rate: 31.17%

  Per-Group Metrics:
  Group           N       Actual+    Pred+      Acc      TPR      FPR      Sel.Rate   Precision
  -----------------------------------------------------------------------------------
  Female          193     44         16         78.24%   36.36%   13.27%   8.29%      100.00%
  Male            407     130        218        74.45%   95.38%   48.94%   53.56%     56.88%

  Aggregate Fairness Metrics:
  Demographic Parity Difference : 0.4527
  Equalized Odds Difference     : 0.5905
  Disparate Impact Ratio        : 0.1549

  Fairness Checks:
  ⚠️  DEMOGRAPHIC PARITY VIOLATION: difference=0.453 (threshold=0.1)
  ⚠️  DISPARATE IMPACT VIOLATION: ratio=0.155 (threshold=0.8, '80% rule')
  ⚠️  EQUALIZED ODDS VIOLATION: max_diff=0.591 (threshold=0.1)
============================================================
```

---

## Extending This Tool

- **Multiple sensitive features**: Audit multiple demographics in one pass
- **Intersectional bias**: E.g. Black women vs White men
- **Fairness-aware retraining**: Use flags to trigger resampling or reweighting
- **CI integration**: Fail your pipeline if fairness thresholds are breached

---

## References

- [Fairness and Machine Learning — Barocas, Hardt, Narayanan](https://fairmlbook.org/)
- [Disparate Impact (80% Rule) — EEOC Guidelines](https://www.eeoc.gov/)
- [Equalized Odds — Hardt et al. 2016](https://arxiv.org/abs/1610.02413)
