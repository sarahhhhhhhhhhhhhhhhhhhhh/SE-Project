"""HRAgentInterceptor: audit entry point for the decision calls of an HR agent.

Provides the entry point through which the decision calls of an actual HR agent
are sent for auditing purposes, so the system can be used as a plugin without a
complete re-design of the host application.

Two usage patterns:
  1. Direct:     HRAgentInterceptor(scorer).intercept(df, "faircv", target=...)
  2. Decorator:  @audited(scorer, threshold=0.8) around a plain function; the
                 host application never imports our classes.

The scorer is duck-typed: anything with predict_kaggle and/or predict_faircv
(optionally a generic predict). BaselineScorer, ImprovedScorer and
BERTidRAAFLiteScorer all work with no special-casing.

FairnessAuditor usage (single-frame API, unchanged; compare() is not used):
  fairness_measure(df, group_col, basis)            -> min/max ratio in [0, 1]
  bootstrap_confidence_interval(df, metric_fn, cluster_col=..., **kwargs)
                                                    -> (point, lower, upper)
Accuracy ratio uses basis="accuracy". The selection-rate basis name is resolved
at first use from FAIRCV_BASIS_CANDIDATES, so a different spelling in your
FairnessAuditor only needs that tuple edited.
"""
from __future__ import annotations

import functools
import logging

import pandas as pd

logger = logging.getLogger(__name__)

VALID_DATASET_TYPES = ("kaggle", "faircv")

GROUP_COL = "intersectional_group"
CLUSTER_COL = "original_id"
KAGGLE_BASIS = "accuracy"
FAIRCV_BASIS_CANDIDATES = ("selection_rate", "selection", "selection_rate_ratio")


def _default_auditor():
    """Import FairnessAuditor lazily so importing this module is cheap."""
    from fairness_auditor import FairnessAuditor
    return FairnessAuditor()


class HRAgentInterceptor:
    def __init__(self, scorer, fairness_threshold=0.8, auditor=None):
        """scorer: any object with predict_kaggle and/or predict_faircv
        (optionally predict). `auditor` is an optional FairnessAuditor-like
        object (mainly for tests); defaults to FairnessAuditor()."""
        has = lambda name: callable(getattr(scorer, name, None))  # noqa: E731
        if not (has("predict_kaggle") or has("predict_faircv") or has("predict")):
            raise TypeError(
                f"{type(scorer).__name__} cannot be intercepted: it must define "
                "predict_kaggle and/or predict_faircv (optionally predict)."
            )
        self.scorer = scorer
        self.scorer_name = type(scorer).__name__
        self.fairness_threshold = fairness_threshold
        self._auditor = auditor
        self._faircv_basis = None  # resolved lazily

    # ------------------------------------------------------------------ utils
    @property
    def auditor(self):
        if self._auditor is None:
            self._auditor = _default_auditor()
        return self._auditor

    def _predict(self, resumes_df, dataset_type):
        name = f"predict_{dataset_type}"
        fn = getattr(self.scorer, name, None)
        if not callable(fn):
            fn = getattr(self.scorer, "predict", None)
        if not callable(fn):
            raise TypeError(
                f"{self.scorer_name} has no {name} (nor a generic predict) "
                f"needed for dataset_type='{dataset_type}'."
            )
        return fn(resumes_df)

    def _resolve_basis(self, preds, dataset_type):
        """Return (basis, point_value). Kaggle -> accuracy. FairCV -> first
        selection-rate basis name the auditor accepts (cached)."""
        if dataset_type == "kaggle":
            return KAGGLE_BASIS, self.auditor.fairness_measure(preds, GROUP_COL, basis=KAGGLE_BASIS)
        candidates = (self._faircv_basis,) if self._faircv_basis else FAIRCV_BASIS_CANDIDATES
        last_err = None
        for basis in candidates:
            try:
                value = self.auditor.fairness_measure(preds, GROUP_COL, basis=basis)
            except ValueError as e:  # unknown basis name -> try the next spelling
                last_err = e
                continue
            self._faircv_basis = basis
            return basis, value
        raise ValueError(
            f"FairnessAuditor rejected every selection-rate basis name tried "
            f"{FAIRCV_BASIS_CANDIDATES}: {last_err}. Edit FAIRCV_BASIS_CANDIDATES."
        )

    def _audit_frame(self, preds, dataset_type):
        basis, value = self._resolve_basis(preds, dataset_type)
        ci = self.auditor.bootstrap_confidence_interval(
            preds, self.auditor.fairness_measure,
            cluster_col=CLUSTER_COL, group_col=GROUP_COL, basis=basis,
        )  # (point, lower, upper)
        return float(value), ci, bool(value < self.fairness_threshold)

    # ------------------------------------------------------------- public API
    def intercept(self, resumes_df, dataset_type, target=None) -> dict:
        if dataset_type not in VALID_DATASET_TYPES:
            raise ValueError(
                f"dataset_type must be one of {VALID_DATASET_TYPES}, got {dataset_type!r}"
            )

        preds = self._predict(resumes_df, dataset_type)

        if dataset_type == "kaggle":
            if "true_label" not in preds.columns:
                raise ValueError(
                    "Kaggle auditing uses the accuracy ratio and needs a "
                    "'true_label' column in the predictions; it is missing."
                )
            value, ci, flagged = self._audit_frame(preds, "kaggle")
            return self._result(preds, value, flagged, dataset_type, target, ci)

        # faircv
        if target is not None:
            sub = self._filter_target(preds, target)
            value, ci, flagged = self._audit_frame(sub, "faircv")
            return self._result(sub, value, flagged, dataset_type, target, ci)

        if "target" in preds.columns:
            targets = list(pd.unique(preds["target"]))
            measures, flags, cis = {}, {}, {}
            for t in targets:
                sub = preds[preds["target"] == t]
                measures[t], cis[t], flags[t] = self._audit_frame(sub, "faircv")
            return self._result(preds, measures, flags, dataset_type, None, cis)

        # single-target scorer (e.g. BaselineScorer): no target column
        value, ci, flagged = self._audit_frame(preds, "faircv")
        return self._result(preds, value, flagged, dataset_type, None, ci)

    @staticmethod
    def _filter_target(preds, target):
        if "target" not in preds.columns:
            return preds  # scorer is already single-target
        sub = preds[preds["target"] == target]
        if sub.empty:
            raise ValueError(
                f"target {target!r} not found; available: "
                f"{sorted(map(str, pd.unique(preds['target'])))}"
            )
        return sub

    def _result(self, preds, measure, flagged, dataset_type, target, ci):
        return {
            "predictions": preds,
            "fairness_measure": measure,
            "flagged": flagged,
            "dataset_type": dataset_type,
            "target": target,
            "fairness_ci": ci,
            "scorer": self.scorer_name,
        }


def _any_flagged(audit) -> bool:
    flagged = audit["flagged"]
    return any(flagged.values()) if isinstance(flagged, dict) else bool(flagged)


def audited(scorer, threshold=0.8, dataset_type="faircv", target=None):
    """Decorator for a plain `mock_hr_agent(resumes)` function.

    The wrapped function's return value is passed through unchanged. The audit
    runs on the same input via HRAgentInterceptor.intercept() (identical code
    path) and is stored on `wrapper.last_audit`.
    """
    interceptor = HRAgentInterceptor(scorer, fairness_threshold=threshold)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(resumes, *args, **kwargs):
            result = func(resumes, *args, **kwargs)
            audit = interceptor.intercept(resumes, dataset_type, target=target)
            wrapper.last_audit = audit
            if _any_flagged(audit):
                logger.warning(
                    "Fairness audit FLAGGED %s (scorer=%s, threshold=%s): %s",
                    func.__name__, audit["scorer"], threshold, audit["fairness_measure"],
                )
            return result

        wrapper.last_audit = None
        wrapper.interceptor = interceptor
        return wrapper

    return decorator
