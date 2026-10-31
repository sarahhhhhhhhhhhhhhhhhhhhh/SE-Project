import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "code"), HERE):
    sys.path.insert(0, os.path.abspath(p))

from fairness_auditor import FairnessAuditor, FAIRCV_TARGETS  # noqa: E402

GROUPS = ["WM", "WF", "BM", "BF"]


# ------------------------------------------------------------------ fixtures
def make_models(n_models=3, n_profiles=40, seed=0):
    """Synthetic prediction frames sharing identical test rows."""
    base = np.random.default_rng(seed)
    kag_true = base.integers(0, 5, n_profiles)
    kag_grp = base.choice(GROUPS, n_profiles)
    fc_true = {t: base.integers(0, 2, (n_profiles, 4)) for t in FAIRCV_TARGETS}
    native = pd.DataFrame({
        "resume_id": [f"p{i}_{g}" for i in range(n_profiles) for g in GROUPS],
        "gender_label": [("M" if i % 2 else "F") for i in range(n_profiles) for _ in GROUPS],
        "ethnicity_label": [("A" if i % 3 else "B") for i in range(n_profiles) for _ in GROUPS],
    })
    frames = []
    for m in range(n_models):
        rng = np.random.default_rng(100 + m)
        rows = []
        flip = rng.random(n_profiles) < 0.3
        pred = np.where(flip, rng.integers(0, 5, n_profiles), kag_true)
        for i in range(n_profiles):
            rows.append(dict(dataset="kaggle", target="category_label", resume_id=f"k{i}",
                             original_id=f"k{i}", intersectional_group=kag_grp[i],
                             true_label=str(kag_true[i]), predicted_label=str(pred[i])))
        for t in FAIRCV_TARGETS:
            for i in range(n_profiles):
                for j, g in enumerate(GROUPS):
                    p = int(rng.random() < (0.5 + 0.1 * m * (j % 2)))
                    rows.append(dict(dataset="faircv", target=t, resume_id=f"p{i}_{g}",
                                     original_id=f"p{i}", intersectional_group=g,
                                     true_label=int(fc_true[t][i, j]), predicted_label=p,
                                     prob_positive=0.5))
        frames.append(pd.DataFrame(rows))
    return frames, native


@pytest.fixture
def aud():
    return FairnessAuditor(n_iterations=60, confidence=0.90, seed=42)


# --------------------------------------------------------- point measures
def test_accuracy_by_group_with_empty_group(aud):
    df = pd.DataFrame({
        "g": pd.Categorical(["a", "a", "b"], categories=["a", "b", "c"]),
        "true_label": [1, 0, 2], "predicted_label": [1.0, 1.0, 2.0],
    })
    acc = aud.accuracy_by_group(df, "g")
    assert acc["a"] == 0.5 and acc["b"] == 1.0
    assert np.isnan(acc["c"])  # empty group -> NaN, no error


def test_fairness_measure_all_zero_group_and_all_zero_dataset(aud):
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "true_label": [1, 1, 1, 1],
                       "predicted_label": [1, 1, 0, 0]})
    assert aud.fairness_measure(df, "g", "selection_rate") == 0.0  # min 0 / max 1
    zero = df.assign(predicted_label=0)
    assert aud.fairness_measure(zero, "g", "selection_rate") == 1.0  # every group 0
    wrong = df.assign(true_label=9)
    assert aud.fairness_measure(wrong, "g", "accuracy") == 1.0
    with pytest.raises(ValueError):
        aud.fairness_measure(df, "g", "nonsense")


def test_fairness_measure_ignores_nan_groups(aud):
    df = pd.DataFrame({
        "g": pd.Categorical(["a", "b"], categories=["a", "b", "c"]),
        "true_label": [1, 1], "predicted_label": [1, 1]})
    assert aud.fairness_measure(df, "g", "accuracy") == 1.0


# ------------------------------------------------------------- bootstrap
def _acc(d):
    return float((d["true_label"] == d["predicted_label"]).mean())


def _simple_df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"original_id": np.arange(n),
                         "true_label": rng.integers(0, 2, n),
                         "predicted_label": rng.integers(0, 2, n)})


def test_bootstrap_ci_width_stabilises_with_iterations(aud):
    df = _simple_df()

    def widths(n_it):
        out = []
        for s in range(8):
            _, lo, hi = aud.bootstrap_confidence_interval(df, _acc, n_iterations=n_it, seed=s)
            out.append(hi - lo)
        return np.array(out)

    small, large = widths(50), widths(1000)
    assert large.mean() > 0
    assert large.std() <= small.std()          # estimate of the width gets steadier
    assert abs(large.mean() - small.mean()) < 0.5 * large.mean()


def test_bootstrap_point_inside_interval_and_row_fallback(aud):
    df = _simple_df()
    pt, lo, hi = aud.bootstrap_confidence_interval(df, _acc, n_iterations=300, cluster_col=None)
    assert lo <= pt <= hi
    pt2, lo2, hi2 = aud.bootstrap_confidence_interval(df, _acc, n_iterations=100, cluster_col="missing")
    assert lo2 <= pt2 <= hi2


def test_cluster_bootstrap_never_splits_a_profile(aud):
    rng = np.random.default_rng(0)
    sizes = {i: int(rng.integers(1, 5)) for i in range(30)}
    df = pd.DataFrame({"original_id": [i for i, s in sizes.items() for _ in range(s)]})
    df["true_label"] = 1
    df["predicted_label"] = 1
    seen = []

    def check(d):
        for oid, cnt in Counter(d["original_id"]).items():
            assert cnt % sizes[oid] == 0, f"profile {oid} was split"
        seen.append(len(d))
        return 1.0

    aud.bootstrap_confidence_interval(df, check, n_iterations=100)
    assert len(seen) == 101  # point estimate + 100 resamples


def test_is_significant_gap():
    assert FairnessAuditor.is_significant_gap((0.1, 0.2), (0.3, 0.4)) is True
    assert FairnessAuditor.is_significant_gap((0.3, 0.4), (0.1, 0.2)) is True
    assert FairnessAuditor.is_significant_gap((0.1, 0.35), (0.3, 0.4)) is False
    assert FairnessAuditor.is_significant_gap((0.1, 0.3), (0.3, 0.4)) is False  # touching
    assert FairnessAuditor.is_significant_gap((0.5, 0.1, 0.2), (0.35, 0.3, 0.4)) is True
    assert FairnessAuditor.is_significant_gap((np.nan, np.nan), (0.3, 0.4)) is False


# ----------------------------------------------------------------- paired
def test_paired_difference_zero_for_identical_inputs(aud):
    (df, _, _), _ = make_models()
    sub = df[(df.dataset == "faircv") & (df.target == "score_blind")]
    d, lo, hi = aud.paired_fairness_difference(sub, sub.copy(), "intersectional_group",
                                               "selection_rate", n_iterations=100)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0, abs=1e-12)


def test_paired_difference_detects_a_real_shift(aud):
    n = 400
    ids = np.repeat(np.arange(n), 2)
    g = np.tile(["a", "b"], n)
    fair = pd.DataFrame({"original_id": ids, "g": g, "true_label": 1, "predicted_label": 1})
    unfair = fair.assign(predicted_label=np.where(g == "b", 0, 1))
    unfair.loc[unfair.index % 10 == 1, "predicted_label"] = 1
    d, lo, hi = aud.paired_fairness_difference(unfair, fair, "g", "selection_rate", n_iterations=200)
    assert d > 0 and lo > 0  # "to" (fair) is significantly fairer


# ------------------------------------------------- compare / pairwise
def test_compare_three_models(aud):
    frames, native = make_models(3)
    names = ["baseline", "bertidraaf_lite", "improved"]
    res = aud.compare(dict(zip(names, frames)),
                      ["intersectional_group", "gender_label", "ethnicity_label"], native)
    assert list(res["algorithm"].unique()) == names
    sizes = res.groupby("algorithm").size()
    assert sizes.nunique() == 1 and sizes.iloc[0] == 1 + 3 * 3
    assert set(res["metric_basis"]) == {"accuracy", "selection_rate"}
    k = res[res.dataset == "kaggle"]
    assert (k["group_attribute"] == "intersectional_group").all()
    assert (k["metric_basis"] == "accuracy").all()
    assert ((res.fairness_ci_low <= res.fairness_measure + 1e-9) |
            res.fairness_ci_low.isna()).all()
    assert res[["accuracy", "fairness_measure"]].apply(lambda c: c.between(0, 1)).all().all()


def test_compare_two_models_and_missing_dataset(aud, capsys):
    frames, native = make_models(2)
    res = aud.compare({"a": frames[0], "b": frames[1]}, ["intersectional_group"], native)
    assert set(res["algorithm"]) == {"a", "b"}
    only_kaggle = frames[1][frames[1].dataset == "kaggle"]
    res2 = aud.compare({"a": frames[0], "b": only_kaggle}, ["intersectional_group"])
    assert "no rows for dataset 'faircv'" in capsys.readouterr().out
    assert (res2[res2.algorithm == "b"].dataset == "kaggle").all()
    single = aud.compare({"only": frames[0]}, ["intersectional_group"])
    assert set(single["algorithm"]) == {"only"}


def test_pairwise_three_models(aud):
    frames, native = make_models(3)
    models = dict(zip(["baseline", "bertidraaf_lite", "improved"], frames))
    cols = ["intersectional_group", "gender_label", "ethnicity_label"]
    comp = aud.compare(models, cols, native)
    per_model = len(comp) // 3
    for cdf in (None, comp):
        pw = aud.pairwise_comparison(models, None, cols, native, comparison_df=cdf)
        assert len(pw) == 3 * per_model
        assert pw.groupby(["model_from", "model_to"]).ngroups == 3
        assert np.allclose(pw.fairness_diff, pw.fairness_to - pw.fairness_from)
        assert set(pw.direction) <= {"higher", "equal", "lower"}
        excl = ~((pw.fairness_diff_low <= 0) & (pw.fairness_diff_high >= 0))
        assert (pw.significant_paired == excl).all()
        assert ((pw.direction == "equal") == ~pw.significant_paired).all()


def test_pairwise_skips_pairs_with_missing_models(aud):
    frames, native = make_models(2)
    pw = aud.pairwise_comparison({"baseline": frames[0], "improved": frames[1]},
                                 group_cols=["intersectional_group"], native_labels_df=native)
    assert set(zip(pw.model_from, pw.model_to)) == {("baseline", "improved")}


# ---------------------------------------------------------- validation
def test_validate_same_test_rows(aud, capsys):
    frames, _ = make_models(3)
    models = dict(zip(["baseline", "bertidraaf_lite", "improved"], frames))
    aud.validate_same_test_rows(models)  # ok
    bad = frames[2][frames[2].original_id != "p3"]
    models["improved"] = bad
    with pytest.raises(ValueError, match="identical test rows"):
        aud.validate_same_test_rows(models)
    assert "MISMATCH" in capsys.readouterr().out
    renamed = frames[2].assign(original_id=frames[2].original_id.replace({"p3": "zzz"}))
    models["improved"] = renamed
    with pytest.raises(ValueError):
        aud.validate_same_test_rows(models)


def test_inspect_native_labels(aud):
    frames, native = make_models(1)
    info = aud.inspect_native_labels(frames[0], native)
    assert set(info.column) == {"gender_label", "ethnicity_label"}
    assert info.interpretation.str.startswith("profile-level").all()
