"""
Demographic Bias Auditor
========================
Audits any binary classifier for fairness across demographic groups.
Metrics: Demographic Parity, Equalized Odds, Disparate Impact, and more.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroupMetrics:
    group_name: str
    group_value: str
    n_samples: int
    n_positive_actual: int
    n_positive_predicted: int
    true_positive_rate: float   # recall / sensitivity
    false_positive_rate: float
    accuracy: float
    selection_rate: float       # P(Y_hat=1 | group)
    precision: float


@dataclass
class FairnessReport:
    sensitive_feature: str
    overall_accuracy: float
    overall_selection_rate: float
    group_metrics: list[GroupMetrics] = field(default_factory=list)
    demographic_parity_diff: float = 0.0
    equalized_odds_diff: float = 0.0
    disparate_impact_ratio: float = 0.0
    flags: list[str] = field(default_factory=list)


class DemographicBiasAuditor:
    """
    Audits a classifier's predictions for bias across a sensitive demographic feature.

    Parameters
    ----------
    demographic_parity_threshold : float
        Max allowed difference in selection rates between groups (default 0.1).
    disparate_impact_threshold : float
        Min allowed ratio of selection rates — 0.8 is the "80% rule" (default 0.8).
    equalized_odds_threshold : float
        Max allowed difference in TPR/FPR between groups (default 0.1).
    """

    def __init__(
        self,
        demographic_parity_threshold: float = 0.1,
        disparate_impact_threshold: float = 0.8,
        equalized_odds_threshold: float = 0.1,
    ):
        self.dp_threshold = demographic_parity_threshold
        self.di_threshold = disparate_impact_threshold
        self.eo_threshold = equalized_odds_threshold

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray,
        feature_name: str = "sensitive_feature",
    ) -> FairnessReport:
        """
        Run a full fairness audit.

        Parameters
        ----------
        y_true : array-like of 0/1 — ground truth labels
        y_pred : array-like of 0/1 — model predictions
        sensitive_feature : array-like — demographic group labels
        feature_name : str — display name of the sensitive feature

        Returns
        -------
        FairnessReport
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_feature = np.array(sensitive_feature)

        groups = np.unique(sensitive_feature)
        group_metrics = []

        for group in groups:
            mask = sensitive_feature == group
            yt = y_true[mask]
            yp = y_pred[mask]

            n = len(yt)
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            tn = np.sum((yt == 0) & (yp == 0))
            fn = np.sum((yt == 1) & (yp == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            acc = (tp + tn) / n if n > 0 else 0.0
            selection_rate = np.mean(yp)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            group_metrics.append(GroupMetrics(
                group_name=feature_name,
                group_value=str(group),
                n_samples=n,
                n_positive_actual=int(np.sum(yt)),
                n_positive_predicted=int(np.sum(yp)),
                true_positive_rate=round(tpr, 4),
                false_positive_rate=round(fpr, 4),
                accuracy=round(acc, 4),
                selection_rate=round(float(selection_rate), 4),
                precision=round(precision, 4),
            ))

        # --- Fairness Metrics ---
        selection_rates = [gm.selection_rate for gm in group_metrics]
        tpr_values = [gm.true_positive_rate for gm in group_metrics]
        fpr_values = [gm.false_positive_rate for gm in group_metrics]

        dp_diff = round(max(selection_rates) - min(selection_rates), 4)
        eo_tpr_diff = round(max(tpr_values) - min(tpr_values), 4)
        eo_fpr_diff = round(max(fpr_values) - min(fpr_values), 4)
        eo_diff = round(max(eo_tpr_diff, eo_fpr_diff), 4)
        di_ratio = round(min(selection_rates) / max(selection_rates), 4) if max(selection_rates) > 0 else 0.0

        overall_acc = round(float(np.mean(y_true == y_pred)), 4)
        overall_sr = round(float(np.mean(y_pred)), 4)

        # --- Flag violations ---
        flags = []
        if dp_diff > self.dp_threshold:
            flags.append(
                f"⚠️  DEMOGRAPHIC PARITY VIOLATION: difference={dp_diff:.3f} (threshold={self.dp_threshold})"
            )
        if di_ratio < self.di_threshold:
            flags.append(
                f"⚠️  DISPARATE IMPACT VIOLATION: ratio={di_ratio:.3f} (threshold={self.di_threshold}, '80% rule')"
            )
        if eo_diff > self.eo_threshold:
            flags.append(
                f"⚠️  EQUALIZED ODDS VIOLATION: max_diff={eo_diff:.3f} (threshold={self.eo_threshold})"
            )
        if not flags:
            flags.append("✅  No fairness violations detected.")

        return FairnessReport(
            sensitive_feature=feature_name,
            overall_accuracy=overall_acc,
            overall_selection_rate=overall_sr,
            group_metrics=group_metrics,
            demographic_parity_diff=dp_diff,
            equalized_odds_diff=eo_diff,
            disparate_impact_ratio=di_ratio,
            flags=flags,
        )

    def print_report(self, report: FairnessReport) -> None:
        """Pretty-print the fairness report to console."""
        print("\n" + "=" * 60)
        print(f"  FAIRNESS AUDIT REPORT — '{report.sensitive_feature}'")
        print("=" * 60)
        print(f"  Overall Accuracy     : {report.overall_accuracy:.2%}")
        print(f"  Overall Selection Rate: {report.overall_selection_rate:.2%}")
        print()

        print("  Per-Group Metrics:")
        print(f"  {'Group':<15} {'N':>6} {'Actual+':<10} {'Pred+':<10} {'Acc':<8} {'TPR':<8} {'FPR':<8} {'Sel.Rate':<10} {'Precision'}")
        print("  " + "-" * 83)
        for gm in report.group_metrics:
            print(
                f"  {gm.group_value:<15} {gm.n_samples:>6} {gm.n_positive_actual:<10} "
                f"{gm.n_positive_predicted:<10} {gm.accuracy:<8.2%} {gm.true_positive_rate:<8.2%} "
                f"{gm.false_positive_rate:<8.2%} {gm.selection_rate:<10.2%} {gm.precision:.2%}"
            )

        print()
        print("  Aggregate Fairness Metrics:")
        print(f"  Demographic Parity Difference : {report.demographic_parity_diff:.4f}")
        print(f"  Equalized Odds Difference     : {report.equalized_odds_diff:.4f}")
        print(f"  Disparate Impact Ratio        : {report.disparate_impact_ratio:.4f}")
        print()
        print("  Fairness Checks:")
        for flag in report.flags:
            print(f"  {flag}")
        print("=" * 60 + "\n")

    def to_csv(self, report: FairnessReport, path: str) -> None:
        """Export the per-group metrics to a CSV file."""
        rows = []
        for gm in report.group_metrics:
            rows.append({
                "sensitive_feature": gm.group_name,
                "group_value": gm.group_value,
                "n_samples": gm.n_samples,
                "n_positive_actual": gm.n_positive_actual,
                "n_positive_predicted": gm.n_positive_predicted,
                "accuracy": gm.accuracy,
                "true_positive_rate": gm.true_positive_rate,
                "false_positive_rate": gm.false_positive_rate,
                "selection_rate": gm.selection_rate,
                "precision": gm.precision,
            })
        df = pd.DataFrame(rows)
        df["demographic_parity_diff"] = report.demographic_parity_diff
        df["equalized_odds_diff"] = report.equalized_odds_diff
        df["disparate_impact_ratio"] = report.disparate_impact_ratio
        df.to_csv(path, index=False)
        print(f"  Report saved to: {path}")
