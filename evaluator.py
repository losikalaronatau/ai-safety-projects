"""
Fairness-Aware Model Evaluator
================================
Produces a visual fairness scorecard for any trained binary classifier.
Combines standard ML metrics with fairness metrics across demographic groups.

Outputs:
  - Per-group scorecard (accuracy, precision, recall, F1, FPR, selection rate)
  - Fairness metric summary (DP diff, EO diff, DI ratio, predictive parity)
  - Threshold analysis — how fairness metrics shift as classification threshold changes
  - JSON + HTML report
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GroupScorecard:
    group: str
    n: int
    prevalence: float           # P(Y=1 | group)
    accuracy: float
    precision: float
    recall: float               # TPR
    f1: float
    fpr: float
    selection_rate: float       # P(Y_hat=1 | group)
    npv: float                  # negative predictive value


@dataclass
class FairnessMetrics:
    demographic_parity_diff: float
    equalized_odds_tpr_diff: float
    equalized_odds_fpr_diff: float
    predictive_parity_diff: float       # precision diff across groups
    disparate_impact_ratio: float
    individual_consistency: float       # placeholder — 1.0 if not computed
    flags: list[str] = field(default_factory=list)


@dataclass
class ThresholdPoint:
    threshold: float
    demographic_parity_diff: float
    equalized_odds_diff: float
    overall_accuracy: float


@dataclass
class ScorecardReport:
    sensitive_feature: str
    model_name: str
    overall_accuracy: float
    overall_f1: float
    group_scorecards: list[GroupScorecard]
    fairness_metrics: FairnessMetrics
    threshold_analysis: list[ThresholdPoint] = field(default_factory=list)


# ─── Core Evaluator ───────────────────────────────────────────────────────────

class FairnessEvaluator:
    """
    Evaluates a trained model's predictions with a fairness scorecard.

    Parameters
    ----------
    dp_threshold : float — demographic parity diff threshold (default 0.1)
    eo_threshold : float — equalized odds diff threshold (default 0.1)
    di_threshold : float — disparate impact ratio threshold (default 0.8)
    pp_threshold : float — predictive parity diff threshold (default 0.1)
    """

    def __init__(
        self,
        dp_threshold: float = 0.1,
        eo_threshold: float = 0.1,
        di_threshold: float = 0.8,
        pp_threshold: float = 0.1,
    ):
        self.dp_threshold = dp_threshold
        self.eo_threshold = eo_threshold
        self.di_threshold = di_threshold
        self.pp_threshold = pp_threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray,
        feature_name: str = "group",
        model_name: str = "model",
        y_prob: Optional[np.ndarray] = None,
    ) -> ScorecardReport:
        """
        Parameters
        ----------
        y_true        : ground truth labels (0/1)
        y_pred        : predicted labels (0/1)
        sensitive_feature : demographic labels
        feature_name  : display name of the sensitive attribute
        model_name    : display name of the model
        y_prob        : predicted probabilities (for threshold analysis)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_feature = np.array(sensitive_feature)

        groups = np.unique(sensitive_feature)
        scorecards = [self._group_scorecard(y_true, y_pred, sensitive_feature, g) for g in groups]

        overall_acc = round(float(np.mean(y_true == y_pred)), 4)
        overall_f1  = self._f1(y_true, y_pred)
        fairness    = self._compute_fairness(scorecards)

        threshold_analysis = []
        if y_prob is not None:
            threshold_analysis = self._threshold_sweep(y_true, y_prob, sensitive_feature)

        return ScorecardReport(
            sensitive_feature=feature_name,
            model_name=model_name,
            overall_accuracy=overall_acc,
            overall_f1=overall_f1,
            group_scorecards=scorecards,
            fairness_metrics=fairness,
            threshold_analysis=threshold_analysis,
        )

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _group_scorecard(self, y_true, y_pred, sensitive, group) -> GroupScorecard:
        mask = sensitive == group
        yt, yp = y_true[mask], y_pred[mask]
        n = len(yt)
        tp = int(np.sum((yt==1)&(yp==1)))
        fp = int(np.sum((yt==0)&(yp==1)))
        tn = int(np.sum((yt==0)&(yp==0)))
        fn = int(np.sum((yt==1)&(yp==0)))

        precision  = tp/(tp+fp)   if (tp+fp)>0  else 0.0
        recall     = tp/(tp+fn)   if (tp+fn)>0  else 0.0
        fpr        = fp/(fp+tn)   if (fp+tn)>0  else 0.0
        f1         = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        npv        = tn/(tn+fn)   if (tn+fn)>0  else 0.0
        acc        = (tp+tn)/n    if n>0         else 0.0

        return GroupScorecard(
            group=str(group), n=n,
            prevalence=round(float(np.mean(yt)), 4),
            accuracy=round(acc, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            fpr=round(fpr, 4),
            selection_rate=round(float(np.mean(yp)), 4),
            npv=round(npv, 4),
        )

    def _compute_fairness(self, scorecards: list[GroupScorecard]) -> FairnessMetrics:
        sr  = [s.selection_rate for s in scorecards]
        tpr = [s.recall         for s in scorecards]
        fpr = [s.fpr            for s in scorecards]
        pre = [s.precision      for s in scorecards]

        dp_diff  = round(max(sr)  - min(sr),  4)
        eo_tpr   = round(max(tpr) - min(tpr), 4)
        eo_fpr   = round(max(fpr) - min(fpr), 4)
        pp_diff  = round(max(pre) - min(pre), 4)
        di_ratio = round(min(sr)  / max(sr),  4) if max(sr) > 0 else 0.0

        flags = []
        if dp_diff  > self.dp_threshold:
            best  = scorecards[sr.index(max(sr))].group
            worst = scorecards[sr.index(min(sr))].group
            flags.append(f"⚠️  DEMOGRAPHIC PARITY: diff={dp_diff:.3f} — '{best}' selected {max(sr):.1%} vs '{worst}' {min(sr):.1%}")
        if max(eo_tpr, eo_fpr) > self.eo_threshold:
            flags.append(f"⚠️  EQUALIZED ODDS: TPR diff={eo_tpr:.3f}, FPR diff={eo_fpr:.3f}")
        if di_ratio < self.di_threshold:
            flags.append(f"⚠️  DISPARATE IMPACT: ratio={di_ratio:.3f} (below 0.8 '80% rule')")
        if pp_diff > self.pp_threshold:
            best  = scorecards[pre.index(max(pre))].group
            worst = scorecards[pre.index(min(pre))].group
            flags.append(f"⚠️  PREDICTIVE PARITY: precision diff={pp_diff:.3f} — '{best}' {max(pre):.1%} vs '{worst}' {min(pre):.1%}")
        if not flags:
            flags.append("✅  All fairness checks passed.")

        return FairnessMetrics(
            demographic_parity_diff=dp_diff,
            equalized_odds_tpr_diff=eo_tpr,
            equalized_odds_fpr_diff=eo_fpr,
            predictive_parity_diff=pp_diff,
            disparate_impact_ratio=di_ratio,
            individual_consistency=1.0,
            flags=flags,
        )

    def _threshold_sweep(self, y_true, y_prob, sensitive, thresholds=None) -> list[ThresholdPoint]:
        if thresholds is None:
            thresholds = np.arange(0.1, 0.91, 0.05)
        points = []
        groups = np.unique(sensitive)
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            sr_per_group = [float(np.mean(y_pred_t[sensitive==g])) for g in groups]
            tpr_per_group = []
            fpr_per_group = []
            for g in groups:
                mask = sensitive == g
                yt, yp = y_true[mask], y_pred_t[mask]
                tp = np.sum((yt==1)&(yp==1)); fn = np.sum((yt==1)&(yp==0))
                fp = np.sum((yt==0)&(yp==1)); tn = np.sum((yt==0)&(yp==0))
                tpr_per_group.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
                fpr_per_group.append(fp/(fp+tn) if (fp+tn)>0 else 0.0)
            dp = max(sr_per_group) - min(sr_per_group)
            eo = max(max(tpr_per_group)-min(tpr_per_group), max(fpr_per_group)-min(fpr_per_group))
            acc = float(np.mean(y_true == y_pred_t))
            points.append(ThresholdPoint(round(float(t),2), round(dp,4), round(eo,4), round(acc,4)))
        return points

    def _f1(self, y_true, y_pred) -> float:
        tp = np.sum((y_true==1)&(y_pred==1))
        fp = np.sum((y_true==0)&(y_pred==1))
        fn = np.sum((y_true==1)&(y_pred==0))
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        return round(2*p*r/(p+r) if (p+r)>0 else 0.0, 4)

    # ─── Reporting ────────────────────────────────────────────────────────────

    def print_scorecard(self, report: ScorecardReport) -> None:
        print("\n" + "=" * 72)
        print(f"  FAIRNESS SCORECARD — '{report.sensitive_feature}' | Model: {report.model_name}")
        print("=" * 72)
        print(f"  Overall Accuracy : {report.overall_accuracy:.2%}   Overall F1 : {report.overall_f1:.4f}")
        print()
        hdr = f"  {'Group':<14} {'N':>5} {'Prev':>6} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7} {'SelRate':>8} {'NPV':>7}"
        print(hdr)
        print("  " + "─" * 74)
        for s in report.group_scorecards:
            print(f"  {s.group:<14} {s.n:>5} {s.prevalence:>6.1%} {s.accuracy:>7.2%} "
                  f"{s.precision:>7.2%} {s.recall:>7.2%} {s.f1:>7.4f} "
                  f"{s.fpr:>7.2%} {s.selection_rate:>8.2%} {s.npv:>7.2%}")
        print()

        fm = report.fairness_metrics
        print("  Fairness Metrics:")
        print(f"    Demographic Parity Diff  : {fm.demographic_parity_diff:.4f}")
        print(f"    Equalized Odds (TPR diff): {fm.equalized_odds_tpr_diff:.4f}")
        print(f"    Equalized Odds (FPR diff): {fm.equalized_odds_fpr_diff:.4f}")
        print(f"    Predictive Parity Diff   : {fm.predictive_parity_diff:.4f}")
        print(f"    Disparate Impact Ratio   : {fm.disparate_impact_ratio:.4f}")
        print()
        print("  Flags:")
        for flag in fm.flags:
            print(f"    {flag}")

        if report.threshold_analysis:
            print()
            print("  Threshold Analysis (DP diff | EO diff | Accuracy):")
            print(f"  {'Thresh':>8} {'DP diff':>9} {'EO diff':>9} {'Accuracy':>10}")
            print("  " + "─" * 40)
            for pt in report.threshold_analysis[::2]:
                print(f"  {pt.threshold:>8.2f} {pt.demographic_parity_diff:>9.4f} "
                      f"{pt.equalized_odds_diff:>9.4f} {pt.overall_accuracy:>10.2%}")
        print("=" * 72 + "\n")

    def to_json(self, report: ScorecardReport, path: str) -> None:
        data = {
            "model_name": report.model_name,
            "sensitive_feature": report.sensitive_feature,
            "overall_accuracy": report.overall_accuracy,
            "overall_f1": report.overall_f1,
            "group_scorecards": [vars(s) for s in report.group_scorecards],
            "fairness_metrics": {
                "demographic_parity_diff": report.fairness_metrics.demographic_parity_diff,
                "equalized_odds_tpr_diff": report.fairness_metrics.equalized_odds_tpr_diff,
                "equalized_odds_fpr_diff": report.fairness_metrics.equalized_odds_fpr_diff,
                "predictive_parity_diff":  report.fairness_metrics.predictive_parity_diff,
                "disparate_impact_ratio":  report.fairness_metrics.disparate_impact_ratio,
                "flags": report.fairness_metrics.flags,
            },
            "threshold_analysis": [vars(p) for p in report.threshold_analysis],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Report saved to: {path}")
