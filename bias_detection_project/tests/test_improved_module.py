"""Run from the code/ directory:  pytest -q test_improved_module.py"""
import os
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ibs_calculator import IBSCalculator, tokenize_terms  # noqa: E402


# ----------------------------------------------------------- IBSCalculator
class TestIBS:
    def setup_method(self):
        self.ibs = IBSCalculator()

    def test_term_in_only_one_group_is_maximally_biased(self):
        rates = {"white_male": 0.8, "white_female": 0.0, "black_male": 0.0, "black_female": 0.0}
        assert self.ibs.compute_ibs(rates) == 0.0

    def test_identical_presence_across_groups_is_neutral(self):
        rates = {"white_male": 0.4, "white_female": 0.4, "black_male": 0.4, "black_female": 0.4}
        assert self.ibs.compute_ibs(rates) == 1.0

    def test_absent_everywhere_is_neutral_no_divide_by_zero(self):
        assert self.ibs.compute_ibs({"a": 0.0, "b": 0.0}) == 1.0
        assert self.ibs.compute_ibs({}) == 1.0

    def test_min_max_is_joint_across_all_groups(self):
        assert self.ibs.compute_ibs({"a": 0.5, "b": 0.4, "c": 0.1}) == pytest.approx(0.2)

    def test_build_lookup(self):
        by_group = {
            "g1": ["john python sql", "john java"],
            "g2": ["lakisha python sql", "lakisha java"],
        }
        lk = self.ibs.build_ibs_lookup(by_group, ["john", "lakisha", "python", "java", "unseen"])
        assert lk["john"] == 0.0 and lk["lakisha"] == 0.0
        assert lk["python"] == 1.0 and lk["java"] == 1.0
        assert lk["unseen"] == 1.0


# ------------------------------------------------- ImprovedScorer smoke test
class FakeEngine:
    """Deterministic stand-in for EmbeddingEngine so the test is offline and fast."""
    hidden_size = 16
    device = "cpu"
    max_length = 512

    @staticmethod
    def _vec(key):
        return np.random.RandomState(zlib.crc32(key.encode()) % (2**31)).randn(16).astype(np.float32)

    def embed(self, texts, batch_size=16):
        return np.stack([self._vec(t) for t in texts])

    def embed_weighted(self, texts, term_weights, batch_size=16, oov_weight=0.0, chunk_long=True):
        cls = self.embed(texts)
        rare = np.zeros_like(cls)
        for i, t in enumerate(texts):
            ws = [(term_weights.get(w, oov_weight), self._vec(w)) for w in tokenize_terms(t)]
            tot = sum(w for w, _ in ws)
            if tot > 0:
                rare[i] = sum(w * v for w, v in ws) / tot
        return cls, rare


def test_improved_scorer_fit_one_epoch_smoke():
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    from improved_scorer import ImprovedScorer

    rows = []
    cats = ["engineer", "chef", "teacher"]
    for i in range(12):
        for grp, name in [("white_male", "john"), ("black_female", "lakisha")]:
            kaggle = i % 2 == 0
            rows.append(dict(
                resume_id=f"{i}_{grp}", original_id=i, intersectional_group=grp,
                counterfactual_text=f"{name} experienced {cats[i % 3]} with skills {i}",
                category_label=cats[i % 3] if kaggle else None,
                suitability_score=None if kaggle else (0.9 if i % 4 == 1 else 0.1),
                source_dataset="kaggle" if kaggle else "faircvdb",
                split="train",
            ))
    df = pd.DataFrame(rows)

    scorer = ImprovedScorer(engine=FakeEngine(), epochs=1, min_df=1, train_batch_size=8,
                            faircv_targets=["suitability_score"],
                            faircv_thresholds={"suitability_score": 0.5})
    scorer.fit(df)

    assert len(scorer.history_) == 1
    assert np.isfinite(scorer.history_[0]["loss"])
    assert 0.0 <= scorer.history_[0]["mean_gate"] <= 1.0

    pk = scorer.predict_kaggle(scorer.kaggle_rows(df))
    assert len(pk) == len(scorer.kaggle_rows(df))
    assert set(pk["predicted_label"]) <= set(cats)
    assert {"true_label", "confidence", "resume_id"} <= set(pk.columns)
    pf = scorer.predict_faircv(scorer.faircv_rows(df))
    assert list(pf["target"].unique()) == ["suitability_score"]
    assert pf["prob_positive"].between(0, 1).all()
    assert set(pf["predicted_label"]) <= {0, 1}
    assert pf["true_label"].notna().all()
    assert pf["fusion_gate"].between(0, 1).all()

    rep = scorer.fair_representation("john experienced chef", scorer.ibs_lookup_)
    assert rep.shape == (16,) and np.isfinite(rep).all()


# ------------------------------------------------------------ comparison.py
def test_auc_rank_and_faircv_report():
    pd = pytest.importorskip("pandas")
    from comparison import auc_rank, faircv_report
    assert auc_rank(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_rank(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5

    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    out = pd.DataFrame({
        "original_id": [1, 1, 2, 2], "intersectional_group": ["a", "b", "a", "b"],
        "true_label": [1, 1, 0, 0], "predicted_label": [1, 1, 0, 1],
        "prob_positive": [sig(2.0), sig(1.0), sig(-1.0), sig(0.5)]})
    r = faircv_report(out)
    assert r["accuracy"] == 0.75
    assert r["spread_mean"] == pytest.approx((1.0 + 1.5) / 2)
    assert r["spread_max"] == pytest.approx(1.5)
    assert r["flip_rate"] == 0.5          # original_id 2 flips (0 vs 1)
    assert r["pos_rate_by_group"] == {"a": 0.5, "b": 1.0}


def test_paired_delta():
    pd = pytest.importorskip("pandas")
    from comparison import paired_delta, verdict
    base = pd.DataFrame({"resume_id": list("abcd"), "original_id": [1, 1, 2, 2],
                         "true_label": [1, 1, 0, 0], "predicted_label": [1, 0, 0, 1]})
    impr = pd.DataFrame({"resume_id": list("abcd"),
                         "true_label": [1, 1, 0, 0], "predicted_label": [1, 1, 0, 1]})
    d = paired_delta(base, impr)
    assert d["baseline"] == 0.5 and d["improved"] == 0.75 and d["delta"] == pytest.approx(0.25)
    assert d["n_clusters"] == 2 and d["n_rows"] == 4
    assert verdict({"delta": -0.01, "se": 0.01}) == "NOT better than baseline"


def test_improved_scorer_finetune_smoke():
    """End-to-end path (backbone + gate + heads) with a tiny fake backbone: runs 1 epoch."""
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    from improved_scorer import ImprovedScorer

    class FakeFTEngine(FakeEngine):
        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.nn.Linear(16, 16).to(self.device)

        def set_train(self, flag):
            self.model.train(flag)

        def train_forward(self, batch, term_weights, oov_weight=0.0):
            cls, rare = FakeEngine.embed_weighted(self, batch, term_weights, oov_weight=oov_weight)
            x = torch.tensor(np.stack([cls, rare]), device=self.device)
            return self.model(x[0]), self.model(x[1])

    rows = []
    cats = ["engineer", "chef", "teacher"]
    for i in range(12):
        for grp, name in [("white_male", "john"), ("black_female", "lakisha")]:
            kaggle = i % 2 == 0
            rows.append(dict(
                resume_id=f"{i}_{grp}", original_id=i, intersectional_group=grp,
                counterfactual_text=f"{name} experienced {cats[i % 3]} with skills {i}",
                category_label=cats[i % 3] if kaggle else None,
                suitability_score=None if kaggle else (0.9 if i % 4 == 1 else 0.1),
                source_dataset="kaggle" if kaggle else "faircvdb"))
    df = pd.DataFrame(rows)
    sc = ImprovedScorer(engine=FakeFTEngine(), epochs=1, min_df=1, fine_tune=True,
                        ft_batch_faircv=4, ft_batch_kaggle=4,
                        faircv_targets=["suitability_score"], faircv_thresholds={"suitability_score": 0.5})
    sc.fit(df, val_df=df)
    assert len(sc.history_) == 1 and np.isfinite(sc.history_[0]["loss"])
    pk = sc.predict_kaggle(sc.kaggle_rows(df))
    assert set(pk["predicted_label"]) <= set(cats)
    pf = sc.predict_faircv(sc.faircv_rows(df))
    assert pf["prob_positive"].between(0, 1).all()


def test_faircv_only_aux_reg_and_calibration_smoke():
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    from improved_scorer import ImprovedScorer

    class Eng(FakeEngine):
        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.nn.Linear(16, 16).to(self.device)

        def set_train(self, flag):
            self.model.train(flag)

        def train_forward(self, batch, term_weights, oov_weight=0.0):
            cls, rare = FakeEngine.embed_weighted(self, batch, term_weights, oov_weight=oov_weight)
            x = torch.tensor(np.stack([cls, rare]), device=self.device)
            return self.model(x[0]), self.model(x[1])

    rows = []
    for i in range(12):
        for grp, name in [("white_male", "john"), ("black_female", "lakisha")]:
            rows.append(dict(resume_id=f"{i}_{grp}", original_id=i, intersectional_group=grp,
                             counterfactual_text=f"{name} candidate skills {i}",
                             category_label=None, suitability_score=0.2 + 0.05 * i,
                             source_dataset="faircvdb"))
    df = pd.DataFrame(rows)
    sc = ImprovedScorer(engine=Eng(), epochs=1, min_df=1, fine_tune=True, faircv_aux_reg=True,
                        ft_batch_faircv=4, faircv_targets=["suitability_score"],
                        faircv_thresholds={"suitability_score": 0.5})
    sc.fit(df, val_df=df)
    assert np.isfinite(sc.history_[0]["loss"])
    cuts = sc.calibrate_decision_thresholds(df)
    assert 0.3 <= cuts["suitability_score"] <= 0.7
    pf = sc.predict_faircv(df)
    assert set(pf["predicted_label"]) <= {0, 1}
