"""Tests for hr_agent_interceptor. Stub scorers and a stub auditor; no GPU,
no project modules needed."""
import pandas as pd
import pytest

from hr_agent_interceptor import HRAgentInterceptor, audited

TARGETS = ["gender", "ethnicity", "age"]


class StubAuditor:
    """Mirrors the real FairnessAuditor single-frame API:
    fairness_measure(df, group_col, basis) and
    bootstrap_confidence_interval(df, metric_fn, cluster_col=..., **kwargs)
    -> (point, lower, upper). `valid_selection` is the selection-rate basis
    spelling it accepts (others raise ValueError, like _check_basis)."""

    def __init__(self, valid_selection="selection_rate"):
        self.valid_selection = valid_selection

    def fairness_measure(self, df, group_col, basis="accuracy"):
        if basis == "accuracy":
            per = (df["predicted_label"] == df["true_label"]).groupby(df[group_col]).mean()
        elif basis == self.valid_selection:
            per = df.groupby(group_col)["predicted_label"].mean()
        else:
            raise ValueError(f"unknown basis {basis!r}")
        return float(per.min() / per.max()) if per.max() > 0 else 1.0

    def bootstrap_confidence_interval(self, df, metric_fn, n_iterations=1000,
                                      confidence=0.9, cluster_col="original_id",
                                      seed=42, **kwargs):
        m = float(metric_fn(df, **kwargs))
        return m, m - 0.01, m + 0.01


def make_frame(rate_a, rate_b, n=20, with_true=True):
    """Two groups of n rows; selection rates rate_a / rate_b."""
    rows = []
    for g, rate in (("A", rate_a), ("B", rate_b)):
        k = int(round(rate * n))
        for i in range(n):
            lab = 1 if i < k else 0
            rows.append({
                "original_id": f"{g}{i}", "resume_id": f"{g}{i}_v",
                "intersectional_group": g,
                "predicted_label": lab, "prob_positive": float(lab),
                **({"true_label": 1 if i % 2 == 0 else 0} if with_true else {}),
            })
    return pd.DataFrame(rows)


class FaircvStub:
    def __init__(self, rate_a, rate_b):
        self.rate_a, self.rate_b = rate_a, rate_b

    def predict_faircv(self, df):
        return make_frame(self.rate_a, self.rate_b)


class KaggleStub:
    def __init__(self, with_true=True):
        self.with_true = with_true

    def predict_kaggle(self, df):
        return make_frame(0.5, 0.5, with_true=self.with_true)


class ImprovedShapeStub:
    """Mimics ImprovedScorer / BERTidRAAFLiteScorer: all three targets, a
    `target` column, and different fairness per target."""
    RATES = {"gender": (1.0, 0.5), "ethnicity": (0.5, 0.5), "age": (0.8, 0.8)}

    def predict_faircv(self, df):
        parts = []
        for t, (a, b) in self.RATES.items():
            f = make_frame(a, b)
            f["target"] = t
            parts.append(f)
        return pd.concat(parts, ignore_index=True)


def interceptor(scorer, thr=0.8):
    return HRAgentInterceptor(scorer, fairness_threshold=thr, auditor=StubAuditor())


DF = pd.DataFrame({"x": [1]})


def test_flags_true_below_threshold():
    res = interceptor(FaircvStub(1.0, 0.5)).intercept(DF, "faircv")  # ratio 0.5
    assert res["flagged"] is True
    assert res["fairness_measure"] == pytest.approx(0.5)


def test_flags_false_above_threshold():
    res = interceptor(FaircvStub(1.0, 0.95)).intercept(DF, "faircv")
    assert res["flagged"] is False


def _assert_same(a, b):
    assert set(a) == set(b)
    pd.testing.assert_frame_equal(a["predictions"], b["predictions"])
    for k in a:
        if k != "predictions":
            assert a[k] == b[k], k


def test_decorator_last_audit_equals_direct_intercept():
    scorer = FaircvStub(1.0, 0.5)

    @audited(scorer, threshold=0.8, dataset_type="faircv")
    def mock_hr_agent(resumes):
        return ["decision"]

    # use the stub auditor on the wrapper's interceptor so no real class is needed
    mock_hr_agent.interceptor._auditor = StubAuditor()
    out = mock_hr_agent(DF)
    assert out == ["decision"]                      # passthrough unchanged
    assert mock_hr_agent.__name__ == "mock_hr_agent"  # functools.wraps
    _assert_same(mock_hr_agent.last_audit, interceptor(scorer).intercept(DF, "faircv"))


def test_decorator_logs_warning_when_flagged(caplog):
    @audited(FaircvStub(1.0, 0.5))
    def agent(resumes):
        return 1

    agent.interceptor._auditor = StubAuditor()
    with caplog.at_level("WARNING"):
        agent(DF)
    assert any("FLAGGED" in r.message for r in caplog.records)


def test_object_without_predict_methods_raises_typeerror():
    class Nope:
        pass

    with pytest.raises(TypeError, match="predict_kaggle|predict_faircv"):
        HRAgentInterceptor(Nope())


def test_unknown_dataset_type_raises_valueerror():
    with pytest.raises(ValueError):
        interceptor(FaircvStub(1, 1)).intercept(DF, "linkedin")


def test_kaggle_without_true_label_raises_valueerror():
    with pytest.raises(ValueError, match="true_label"):
        interceptor(KaggleStub(with_true=False)).intercept(DF, "kaggle")


def test_kaggle_with_true_label_works():
    res = interceptor(KaggleStub()).intercept(DF, "kaggle")
    assert res["dataset_type"] == "kaggle"
    assert 0 <= res["fairness_measure"] <= 1


def test_multi_target_stub_audited_per_target_when_none():
    res = interceptor(ImprovedShapeStub()).intercept(DF, "faircv")
    assert set(res["fairness_measure"]) == set(TARGETS)
    assert res["flagged"] == {"gender": True, "ethnicity": False, "age": False}
    assert set(res["fairness_ci"]) == set(TARGETS)


def test_multi_target_stub_only_requested_target():
    res = interceptor(ImprovedShapeStub()).intercept(DF, "faircv", target="gender")
    assert res["target"] == "gender"
    assert res["flagged"] is True
    assert set(res["predictions"]["target"]) == {"gender"}


def test_unknown_target_raises_valueerror():
    with pytest.raises(ValueError, match="not found"):
        interceptor(ImprovedShapeStub()).intercept(DF, "faircv", target="height")


def test_ci_is_point_lower_upper_tuple():
    res = interceptor(FaircvStub(1.0, 0.5)).intercept(DF, "faircv")
    point, lo, hi = res["fairness_ci"]
    assert point == pytest.approx(res["fairness_measure"])
    assert lo <= point <= hi


def test_selection_basis_spelling_is_resolved():
    ic = HRAgentInterceptor(FaircvStub(1.0, 0.5), 0.8,
                            auditor=StubAuditor(valid_selection="selection"))
    res = ic.intercept(DF, "faircv")
    assert res["fairness_measure"] == pytest.approx(0.5)
    assert ic._faircv_basis == "selection"


def test_no_accepted_selection_basis_raises_clear_valueerror():
    ic = HRAgentInterceptor(FaircvStub(1.0, 0.5), 0.8,
                            auditor=StubAuditor(valid_selection="weird_name"))
    with pytest.raises(ValueError, match="FAIRCV_BASIS_CANDIDATES"):
        ic.intercept(DF, "faircv")


def test_scorer_key_equals_stub_class_name():
    assert interceptor(FaircvStub(1, 1)).intercept(DF, "faircv")["scorer"] == "FaircvStub"
    assert interceptor(ImprovedShapeStub()).intercept(DF, "faircv")["scorer"] == "ImprovedShapeStub"
    assert interceptor(FaircvStub(1, 1)).scorer_name == "FaircvStub"
