# 🗂️ Dataset Bias Scanner

Scans a training dataset **before model training** to surface bias at the source — catching problems early rather than auditing a deployed model.

## What It Checks

| Check | Description |
|---|---|
| **Representation Imbalance** | Are some groups severely under/over-represented? |
| **Label Bias** | Is the target label correlated with a sensitive attribute? |
| **Proxy Features** | Do non-sensitive features encode sensitive information (Cramér's V)? |
| **Intersectional Imbalance** | Are combinations of groups (e.g. Black women) nearly absent? |

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
pytest tests/
```

## Usage

```python
from src.scanner import DatasetBiasScanner

scanner = DatasetBiasScanner(
    imbalance_ratio_threshold=3.0,
    label_disparity_threshold=0.1,
    disparate_impact_threshold=0.8,
    proxy_correlation_threshold=0.3,
    intersectional_min_proportion=0.01,
)

report = scanner.scan(
    df,
    sensitive_features=["gender", "race"],
    label_column="hired",
    proxy_candidates=["department", "zip_code"],
)

scanner.print_report(report)
scanner.to_json(report, "reports/scan.json")
```
