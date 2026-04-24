"""
Dataset Bias Scanner
====================
Scans a training dataset BEFORE model training to detect:
  1. Representation imbalance — are groups under/over-represented?
  2. Label bias — is a sensitive attribute correlated with the target label?
  3. Feature proxy — do non-sensitive features act as proxies for sensitive ones?
  4. Intersectional imbalance — are combinations of groups severely underrepresented?
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RepresentationResult:
    feature: str
    group_counts: dict        # {group_value: count}
    group_proportions: dict   # {group_value: proportion}
    imbalance_ratio: float    # max_count / min_count
    flagged: bool
    flag_message: str = ""


@dataclass
class LabelBiasResult:
    sensitive_feature: str
    label_column: str
    group_positive_rates: dict   # {group: P(label=1 | group)}
    max_rate: float
    min_rate: float
    disparity: float
    disparate_impact_ratio: float
    flagged: bool
    flag_message: str = ""


@dataclass
class ProxyResult:
    sensitive_feature: str
    proxy_candidate: str
    correlation: float           # Cramér's V or point-biserial
    flagged: bool
    flag_message: str = ""


@dataclass
class IntersectionalResult:
    features: list[str]
    group_counts: dict           # {(val1, val2): count}
    min_group: tuple
    min_count: int
    min_proportion: float
    flagged: bool
    flag_message: str = ""


@dataclass
class DatasetScanReport:
    dataset_shape: tuple
    n_sensitive_features: int
    representation_results: list[RepresentationResult] = field(default_factory=list)
    label_bias_results: list[LabelBiasResult] = field(default_factory=list)
    proxy_results: list[ProxyResult] = field(default_factory=list)
    intersectional_results: list[IntersectionalResult] = field(default_factory=list)
    total_flags: int = 0
    summary: list[str] = field(default_factory=list)


# ─── Core Scanner ─────────────────────────────────────────────────────────────

class DatasetBiasScanner:
    """
    Scans a pandas DataFrame for bias before model training.

    Parameters
    ----------
    imbalance_ratio_threshold : float
        Flag if max_group_count / min_group_count exceeds this (default 3.0).
    label_disparity_threshold : float
        Flag if difference in positive label rates between groups exceeds this (default 0.1).
    disparate_impact_threshold : float
        Flag if min/max positive rate ratio falls below this (default 0.8).
    proxy_correlation_threshold : float
        Flag if Cramér's V between a feature and sensitive attr exceeds this (default 0.3).
    intersectional_min_proportion : float
        Flag if an intersectional group is below this fraction of the dataset (default 0.01).
    """

    def __init__(
        self,
        imbalance_ratio_threshold: float = 3.0,
        label_disparity_threshold: float = 0.1,
        disparate_impact_threshold: float = 0.8,
        proxy_correlation_threshold: float = 0.3,
        intersectional_min_proportion: float = 0.01,
    ):
        self.imbalance_threshold = imbalance_ratio_threshold
        self.label_disparity_threshold = label_disparity_threshold
        self.di_threshold = disparate_impact_threshold
        self.proxy_threshold = proxy_correlation_threshold
        self.intersectional_threshold = intersectional_min_proportion

    def scan(
        self,
        df: pd.DataFrame,
        sensitive_features: list[str],
        label_column: Optional[str] = None,
        proxy_candidates: Optional[list[str]] = None,
    ) -> DatasetScanReport:
        """
        Run a full dataset bias scan.

        Parameters
        ----------
        df : pd.DataFrame
        sensitive_features : list of column names to audit (e.g. ["gender", "race"])
        label_column : target column for label bias checks (binary 0/1)
        proxy_candidates : columns to check as potential proxies for sensitive features
        """
        report = DatasetScanReport(
            dataset_shape=df.shape,
            n_sensitive_features=len(sensitive_features),
        )

        print(f"\n  Scanning dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # 1. Representation
        print("  [1/4] Checking representation imbalance...")
        for feat in sensitive_features:
            result = self._check_representation(df, feat)
            report.representation_results.append(result)

        # 2. Label Bias
        if label_column:
            print("  [2/4] Checking label bias...")
            for feat in sensitive_features:
                result = self._check_label_bias(df, feat, label_column)
                report.label_bias_results.append(result)
        else:
            print("  [2/4] Skipping label bias (no label_column provided).")

        # 3. Proxy Features
        if proxy_candidates:
            print("  [3/4] Checking proxy features...")
            for feat in sensitive_features:
                for candidate in proxy_candidates:
                    if candidate != feat:
                        result = self._check_proxy(df, feat, candidate)
                        report.proxy_results.append(result)
        else:
            print("  [3/4] Skipping proxy check (no proxy_candidates provided).")

        # 4. Intersectional Imbalance
        if len(sensitive_features) >= 2:
            print("  [4/4] Checking intersectional imbalance...")
            for pair in combinations(sensitive_features, 2):
                result = self._check_intersectional(df, list(pair))
                report.intersectional_results.append(result)
        else:
            print("  [4/4] Skipping intersectional check (need ≥2 sensitive features).")

        # Summarize
        report.total_flags = sum([
            sum(1 for r in report.representation_results if r.flagged),
            sum(1 for r in report.label_bias_results if r.flagged),
            sum(1 for r in report.proxy_results if r.flagged),
            sum(1 for r in report.intersectional_results if r.flagged),
        ])
        report.summary = self._build_summary(report)
        return report

    # ─── Check Methods ────────────────────────────────────────────────────────

    def _check_representation(self, df: pd.DataFrame, feature: str) -> RepresentationResult:
        counts = df[feature].value_counts()
        proportions = (counts / len(df)).round(4)
        ratio = round(counts.max() / counts.min(), 2) if counts.min() > 0 else 999.0
        flagged = ratio > self.imbalance_threshold
        msg = (
            f"⚠️  IMBALANCE in '{feature}': ratio={ratio:.1f}x "
            f"(largest={counts.idxmax()}: {counts.max():,}, "
            f"smallest={counts.idxmin()}: {counts.min():,})"
            if flagged else ""
        )
        return RepresentationResult(
            feature=feature,
            group_counts=counts.to_dict(),
            group_proportions=proportions.to_dict(),
            imbalance_ratio=ratio,
            flagged=flagged,
            flag_message=msg,
        )

    def _check_label_bias(self, df: pd.DataFrame, feature: str, label: str) -> LabelBiasResult:
        group_rates = df.groupby(feature)[label].mean().round(4)
        max_rate = float(group_rates.max())
        min_rate = float(group_rates.min())
        disparity = round(max_rate - min_rate, 4)
        di_ratio = round(min_rate / max_rate, 4) if max_rate > 0 else 0.0

        flagged = disparity > self.label_disparity_threshold or di_ratio < self.di_threshold
        msg = ""
        if flagged:
            msg = (
                f"⚠️  LABEL BIAS in '{feature}' → '{label}': "
                f"disparity={disparity:.3f}, DI ratio={di_ratio:.3f} | "
                f"highest={group_rates.idxmax()} ({max_rate:.1%}), "
                f"lowest={group_rates.idxmin()} ({min_rate:.1%})"
            )
        return LabelBiasResult(
            sensitive_feature=feature,
            label_column=label,
            group_positive_rates=group_rates.to_dict(),
            max_rate=max_rate,
            min_rate=min_rate,
            disparity=disparity,
            disparate_impact_ratio=di_ratio,
            flagged=flagged,
            flag_message=msg,
        )

    def _check_proxy(self, df: pd.DataFrame, sensitive: str, candidate: str) -> ProxyResult:
        """Use Cramér's V to measure association between two categorical features."""
        v = self._cramers_v(df[sensitive], df[candidate])
        flagged = v > self.proxy_threshold
        msg = (
            f"⚠️  PROXY RISK: '{candidate}' correlates with '{sensitive}' "
            f"(Cramér's V={v:.3f}, threshold={self.proxy_threshold})"
            if flagged else ""
        )
        return ProxyResult(
            sensitive_feature=sensitive,
            proxy_candidate=candidate,
            correlation=round(v, 4),
            flagged=flagged,
            flag_message=msg,
        )

    def _check_intersectional(self, df: pd.DataFrame, features: list[str]) -> IntersectionalResult:
        group_counts = df.groupby(features).size()
        total = len(df)
        min_group = group_counts.idxmin()
        min_count = int(group_counts.min())
        min_proportion = round(min_count / total, 4)
        flagged = min_proportion < self.intersectional_threshold
        msg = (
            f"⚠️  INTERSECTIONAL UNDERREPRESENTATION [{' × '.join(features)}]: "
            f"group {min_group} has only {min_count} samples ({min_proportion:.1%} of dataset)"
            if flagged else ""
        )
        return IntersectionalResult(
            features=features,
            group_counts=group_counts.to_dict(),
            min_group=min_group if isinstance(min_group, tuple) else (min_group,),
            min_count=min_count,
            min_proportion=min_proportion,
            flagged=flagged,
            flag_message=msg,
        )

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Cramér's V: measure of association between two categorical variables."""
        confusion = pd.crosstab(x, y)
        n = confusion.sum().sum()
        r, k = confusion.shape
        chi2 = self._chi2_stat(confusion.values)
        phi2 = chi2 / n
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - (r - 1) ** 2 / (n - 1)
        k_corr = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_corr - 1, r_corr - 1)
        return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0

    def _chi2_stat(self, observed: np.ndarray) -> float:
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        total = observed.sum()
        expected = row_totals * col_totals / total
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.where(expected > 0, (observed - expected) ** 2 / expected, 0)
        return float(chi2.sum())

    # ─── Reporting ────────────────────────────────────────────────────────────

    def _build_summary(self, report: DatasetScanReport) -> list[str]:
        lines = []
        for r in report.representation_results:
            if r.flagged: lines.append(r.flag_message)
        for r in report.label_bias_results:
            if r.flagged: lines.append(r.flag_message)
        for r in report.proxy_results:
            if r.flagged: lines.append(r.flag_message)
        for r in report.intersectional_results:
            if r.flagged: lines.append(r.flag_message)
        if not lines:
            lines.append("✅  No bias flags detected in this dataset.")
        return lines

    def print_report(self, report: DatasetScanReport) -> None:
        print("\n" + "=" * 70)
        print("  DATASET BIAS SCAN REPORT")
        print("=" * 70)
        print(f"  Dataset Shape : {report.dataset_shape[0]:,} rows × {report.dataset_shape[1]} columns")
        print(f"  Total Flags   : {report.total_flags}")
        print()

        if report.representation_results:
            print("  ── Representation Imbalance")
            for r in report.representation_results:
                status = "⚠️  FLAGGED" if r.flagged else "✅  OK"
                print(f"     {r.feature:<20} ratio={r.imbalance_ratio:.1f}x  [{status}]")
                for grp, prop in r.group_proportions.items():
                    print(f"       {str(grp):<20} {r.group_counts[grp]:>6,} samples  ({prop:.1%})")
            print()

        if report.label_bias_results:
            print("  ── Label Bias")
            for r in report.label_bias_results:
                status = "⚠️  FLAGGED" if r.flagged else "✅  OK"
                print(f"     {r.sensitive_feature} → {r.label_column}  disparity={r.disparity:.3f}  DI={r.disparate_impact_ratio:.3f}  [{status}]")
                for grp, rate in r.group_positive_rates.items():
                    print(f"       {str(grp):<20} positive rate = {rate:.1%}")
            print()

        if report.proxy_results:
            print("  ── Proxy Feature Risk (Cramér's V)")
            for r in report.proxy_results:
                status = "⚠️  FLAGGED" if r.flagged else "✅  OK"
                print(f"     {r.proxy_candidate:<20} ↔ {r.sensitive_feature:<15} V={r.correlation:.3f}  [{status}]")
            print()

        if report.intersectional_results:
            print("  ── Intersectional Imbalance")
            for r in report.intersectional_results:
                status = "⚠️  FLAGGED" if r.flagged else "✅  OK"
                print(f"     {' × '.join(r.features):<30} min_group={r.min_group}  n={r.min_count}  ({r.min_proportion:.1%})  [{status}]")
            print()

        print("  ── Summary")
        for line in report.summary:
            print(f"     {line}")
        print("=" * 70 + "\n")

    def to_json(self, report: DatasetScanReport, path: str) -> None:
        def _safe(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, tuple): return list(obj)
            return str(obj)

        data = {
            "dataset_shape": list(report.dataset_shape),
            "total_flags": report.total_flags,
            "summary": report.summary,
            "representation": [
                {"feature": r.feature, "imbalance_ratio": r.imbalance_ratio,
                 "flagged": r.flagged, "group_proportions": {str(k): v for k, v in r.group_proportions.items()}}
                for r in report.representation_results
            ],
            "label_bias": [
                {"sensitive_feature": r.sensitive_feature, "label_column": r.label_column,
                 "disparity": r.disparity, "disparate_impact_ratio": r.disparate_impact_ratio,
                 "flagged": r.flagged, "group_positive_rates": {str(k): v for k, v in r.group_positive_rates.items()}}
                for r in report.label_bias_results
            ],
            "proxy_risks": [
                {"sensitive_feature": r.sensitive_feature, "proxy_candidate": r.proxy_candidate,
                 "correlation": r.correlation, "flagged": r.flagged}
                for r in report.proxy_results
            ],
            "intersectional": [
                {"features": r.features, "min_group": [_safe(x) for x in r.min_group],
                 "min_count": r.min_count, "min_proportion": r.min_proportion, "flagged": r.flagged}
                for r in report.intersectional_results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_safe)
        print(f"  Report saved to: {path}")
