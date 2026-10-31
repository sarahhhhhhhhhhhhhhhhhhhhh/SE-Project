"""Offline, CPU, fast tests for bertidraaf_lite_scorer (stub engine; no transformers / downloads)."""
import importlib
import inspect
import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from ibs_calculator import IBSCalculator, tokenize_terms          # noqa: E402
from improved_scorer import ImprovedScorer                          # noqa: E402
from bertidraaf_lite_scorer import (BERTidRAAFLiteScorer, BERTidRAAFPaperScorer, IDFCalculator,   # noqa: E402
                                    assert_matched, build_matched_pair, signature_drift)

TARGETS = ["score_blind", "score_biased_gender", "score_biased_ethnicity"]
THR = {t: 0.5 for t in TARGETS}
KW = dict(faircv_targets=TARGETS, faircv_thresholds=THR, epochs=2, train_batch_size=16,
          embed_batch_size=4, min_df=2, gate_hidden=8, seed=7, time_budget_minutes=5)


# ------------------------------------------------------------------------------ stub engine
class StubEngine:
    """Deterministic stand-in for EmbeddingEngine. Tokens = alphanumeric runs OR single punctuation chars
    (as the WordPiece pre-tokenizer would split them); weight = term_weights.get(tok.lower(), oov_weight)."""
    _TOK = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

    def __init__(self, hidden_size=8, max_length=64, model_name="stub"):
        self.hidden_size, self.max_length, self.model_name = hidden_size, max_length, model_name
        self.device = "cpu"
        self.oov_seen, self.train_flags = [], []

    @staticmethod
    def _vec(tok, h):
        return np.random.RandomState(zlib.crc32(tok.encode())).randn(h).astype(np.float32)

    def set_train(self, flag):
        self.train_flags.append(flag)

    def _pool(self, texts, tw, oov):
        H = self.hidden_size
        ctx, rare = np.zeros((len(texts), H), np.float32), np.zeros((len(texts), H), np.float32)
        for i, t in enumerate(texts):
            toks = [k.lower() for k in self._TOK.findall(str(t))]
            if not toks:
                continue
            V = np.stack([self._vec(k, H) for k in toks])
            w = np.array([tw.get(k, oov) for k in toks], dtype=np.float64)
            ctx[i] = V.mean(0)
            rare[i] = ((w[:, None] * V).sum(0) / max(w.sum(), 1e-8)).astype(np.float32)
        return ctx, rare

    def embed_weighted(self, texts, term_weights, batch_size=16, oov_weight=0.0, chunk_long=True):
        self.oov_seen.append(oov_weight)
        return self._pool(texts, term_weights, oov_weight)

    def train_forward(self, batch, term_weights, oov_weight=0.0):
        import torch
        self.oov_seen.append(oov_weight)
        c, r = self._pool(batch, term_weights, oov_weight)
        return torch.tensor(c), torch.tensor(r)


# ------------------------------------------------------------------------------------ data
def make_df(n=60, seed=0):
    rng = np.random.RandomState(seed)
    groups = ["g0", "g1", "g2", "g3"]
    common = ["experience", "team", "project", "manager", "python", "sql", "data", "report"]
    rows = []
    for src in ("kaggle", "faircv"):
        for i in range(n):
            g = groups[i % 4]
            words = list(rng.choice(common, 6)) + [f"{g}word", "c++", "node.js,"]
            if i % 7 == 0:
                words.append(f"uniq{src}{i}")              # guarantees df == 1 terms exist
            row = dict(resume_id=f"{src}{i}", original_id=f"{src}{i // 4}", intersectional_group=g,
                       source_dataset=src, counterfactual_text=" ".join(words),
                       split="train" if i < int(0.6 * n) else ("val" if i < int(0.8 * n) else "test"))
            if src == "kaggle":
                row["category_label"] = ["A", "B", "C"][i % 3]
                row.update({t: np.nan for t in TARGETS})
            else:
                row["category_label"] = np.nan
                row.update({t: float(rng.rand()) for t in TARGETS})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def parts():
    df = make_df()
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def fit_control(parts, **over):
    tr, va, _ = parts
    ctl = BERTidRAAFLiteScorer(engine=StubEngine(), **{**KW, **over})
    return ctl.fit(tr, val_df=va)


def expected_token_median(df_counts, n):
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df_counts.items()}
    expanded = sorted(v for t, v in idf.items() for _ in range(df_counts[t]))
    return expanded[math.ceil(len(expanded) / 2) - 1]


# ------------------------------------------------------------------------ 1. IDF formula
def test_idf_formula_toy_and_train_only(parts):
    groups = {"a": ["python sql python", "Java SQL"], "b": ["python excel Node.JS"]}
    calc = IDFCalculator()
    vocab = calc.build_vocab([t for v in groups.values() for t in v], min_df=5, max_vocab=2)  # ignored
    assert set(vocab) == {"python", "sql", "java", "excel", "node", "js"}
    lk = calc.build_ibs_lookup(groups, vocab=["python"])                                       # ignored
    assert calc.n_docs_ == 3 and set(lk) == set(vocab)
    assert lk["python"] == pytest.approx(math.log(4 / 3) + 1)      # df=2 (counted once per document)
    assert lk["sql"] == pytest.approx(math.log(4 / 3) + 1)
    assert lk["java"] == pytest.approx(math.log(4 / 2) + 1)        # df=1
    assert lk["node"] == pytest.approx(math.log(4 / 2) + 1)        # same tokenizer as IBS: "Node.JS" -> node, js
    assert calc.unseen_weight_ == pytest.approx(math.log(3 + 1) + 1)

    tr, va, te = parts
    a = fit_control(parts)
    va2 = va.copy()
    va2.iloc[0, va2.columns.get_loc("counterfactual_text")] += " zebraquux"
    b = fit_control((tr, va2, te))
    assert "zebraquux" not in b.idf_lookup_
    assert a.idf_lookup_ == b.idf_lookup_ and b.ibs_calc.n_docs_ == len(tr)


# ------------------------------------------------------ 2. coverage, OOV weight and override
def test_lookup_covers_training_terms_and_oov_default(parts):
    tr = parts[0]
    ctl = fit_control(parts)
    cnt = {}
    for t in tr["counterfactual_text"]:
        for u in set(tokenize_terms(t)):
            cnt[u] = cnt.get(u, 0) + 1
    assert set(ctl.idf_lookup_) == set(cnt)                         # min_df=2 is ignored
    assert any(c == 1 for c in cnt.values())
    assert ctl.idf_lookup_ is ctl.ibs_lookup_ and ctl.weight_mode == "idf"
    n = len(tr)
    assert ctl.ibs_calc.unseen_weight_ == pytest.approx(math.log(n + 1) + 1)   # Eq.1 df=0 value, reported
    want = expected_token_median(cnt, n)
    assert ctl.oov_weight == pytest.approx(want)
    assert ctl.engine.oov_seen and all(o == pytest.approx(want) for o in ctl.engine.oov_seen)  # engine only sees it
    assert want < max(ctl.idf_lookup_.values()) < ctl.ibs_calc.unseen_weight_  # NOT the max-IDF value
    over = fit_control(parts, oov_weight=0.25)
    assert over.oov_weight == 0.25 and over.engine.oov_seen and set(over.engine.oov_seen) == {0.25}


# ----------------------------------------------------------------------- 3. signature
def test_constructor_signature_matches_improved_minus_weight_mode():
    base = [(p.name, p.kind, p.default) for n, p in inspect.signature(ImprovedScorer.__init__).parameters.items()
            if n != "weight_mode"]
    ctl = [(p.name, p.kind, p.default) for p in inspect.signature(BERTidRAAFLiteScorer.__init__).parameters.values()]
    assert base == ctl
    assert signature_drift() == []
    assert "weight_mode" not in inspect.signature(BERTidRAAFLiteScorer.__init__).parameters


# ------------------------------------------------------------------ 4. IBS-vs-IDF is the only difference
def test_equivalence_with_parity_when_lookup_is_ibs(parts, monkeypatch):
    tr, va, te = parts
    ctl = BERTidRAAFLiteScorer(engine=StubEngine(), oov_weight=1.0, **KW)    # parity's OOV weight
    monkeypatch.setattr(ctl, "ibs_calc", IBSCalculator())                    # lookup == IBS lookup
    imp = ImprovedScorer(engine=StubEngine(), weight_mode="parity", **KW)
    ctl.fit(tr, val_df=va)
    imp.fit(tr, val_df=va)
    assert ctl.idf_lookup_ == imp.ibs_lookup_
    pd.testing.assert_frame_equal(ctl.predict_faircv(te[te.source_dataset == "faircv"]),
                                  imp.predict_faircv(te[te.source_dataset == "faircv"]), check_exact=True)
    pd.testing.assert_frame_equal(ctl.predict_kaggle(te[te.source_dataset == "kaggle"]),
                                  imp.predict_kaggle(te[te.source_dataset == "kaggle"]), check_exact=True)
    assert ctl.decision_thr_ == imp.decision_thr_ == {}
    c1 = ctl.calibrate_decision_thresholds(va[va.source_dataset == "faircv"])
    c2 = imp.calibrate_decision_thresholds(va[va.source_dataset == "faircv"])
    assert c1 == c2


# ----------------------------------------------------------------------- 5. output schema
def test_output_schema_identical_to_improved(parts):
    tr, va, te = parts
    ctl = fit_control(parts)
    imp = ImprovedScorer(engine=StubEngine(), weight_mode="parity", **KW).fit(tr, val_df=va)
    tf, tk = te[te.source_dataset == "faircv"], te[te.source_dataset == "kaggle"]
    for fc, fi in ((ctl.predict_faircv(tf), imp.predict_faircv(tf)),
                   (ctl.predict_kaggle(tk), imp.predict_kaggle(tk))):
        assert list(fc.columns) == list(fi.columns)
        assert fc.dtypes.equals(fi.dtypes)
        assert len(fc) == len(fi)
        assert fc["resume_id"].tolist() == fi["resume_id"].tolist()
    f = ctl.predict_faircv(tf)
    assert f.columns[0] == "target" and f["target"].unique().tolist() == TARGETS   # stacked, leading 'target'
    assert f["resume_id"].tolist() == tf["resume_id"].tolist() * 3
    assert ctl.predict_kaggle(tk)["resume_id"].tolist() == tk["resume_id"].tolist()


# ------------------------------------------------------------------ 6. matched-pair builder
def test_build_matched_pair_engines_and_mismatch_detection():
    made = []

    def factory():
        made.append(StubEngine())
        return made[-1]

    imp, ctl = build_matched_pair(dict(KW, fine_tune=True), "parity", factory, verbose=False)
    assert len(made) == 2 and imp.engine is not ctl.engine

    imp, ctl = build_matched_pair(dict(KW, fine_tune=False), "disparity", factory, verbose=False)
    assert len(made) == 3 and imp.engine is ctl.engine
    assert imp.weight_mode == "disparity" and ctl.weight_mode == "idf"

    ctl.epochs += 1
    with pytest.raises(AssertionError, match="epochs"):
        assert_matched(imp, ctl)
    ctl.epochs -= 1
    ctl.thresholds_["score_blind"] = 0.9
    with pytest.raises(AssertionError, match="thresholds_"):
        assert_matched(imp, ctl)
    ctl.thresholds_["score_blind"] = 0.5
    ctl.surprise_attr = 1
    with pytest.raises(AssertionError, match="only one scorer"):
        assert_matched(imp, ctl)

    a, b = StubEngine(), StubEngine()
    ft_i = ImprovedScorer(engine=a, weight_mode="parity", fine_tune=True, **KW)
    ft_c = BERTidRAAFLiteScorer(engine=a, fine_tune=True, **KW)               # SAME engine object
    with pytest.raises(AssertionError, match="share ONE engine"):
        assert_matched(ft_i, ft_c)
    for bad in ({"weight_mode": "parity"}, {"oov_weight": 1.0}, {"engine": a}, {"not_an_arg": 1}):
        with pytest.raises(ValueError):
            build_matched_pair(dict(KW, **bad), "parity", factory, verbose=False)
    with pytest.raises(ValueError, match="same object twice"):
        build_matched_pair(dict(KW, fine_tune=True), "parity", lambda: a, verbose=False)


# ------------------------------------------------------------------ 7. HRAgentInterceptor smoke
def _load_interceptor():
    """The interceptor module was not part of the pasted sources; try common names, skip if absent."""
    for mod_name in ("hr_agent_interceptor", "interceptor", "hr_agent", "agent_interceptor", "hr_interceptor"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "HRAgentInterceptor") and hasattr(mod, "simple_measure"):
            return mod.HRAgentInterceptor, mod.simple_measure
    pytest.skip("HRAgentInterceptor / simple_measure not importable from code/ (module name unknown)")


@pytest.mark.parametrize("kind", ["faircv", "kaggle"])
def test_hr_agent_interceptor_runs(parts, kind):
    Interceptor, simple_measure = _load_interceptor()
    te = parts[2]
    df = te[te.source_dataset == kind]
    ctl = fit_control(parts)
    if "measure_fn" in inspect.signature(Interceptor.__init__).parameters:
        icp, extra = Interceptor(ctl, measure_fn=simple_measure), {}
    else:
        icp, extra = Interceptor(ctl), {"measure_fn": simple_measure}
    out = icp.intercept(df, kind, **extra)
    assert out is not None


# ------------------------------------------------------------------ paper-literal variant
def test_paper_scorer_uses_max_idf_oov(parts):
    tr, va, _ = parts
    ctl = BERTidRAAFPaperScorer(engine=StubEngine(), **KW).fit(tr, val_df=va)
    assert ctl.oov_weight == pytest.approx(math.log(len(tr) + 1) + 1)
    assert all(o == pytest.approx(ctl.oov_weight) for o in ctl.engine.oov_seen)
    assert ctl.oov_weight > max(ctl.idf_lookup_.values())
    imp, c2 = build_matched_pair(KW, "parity", lambda: StubEngine(), verbose=False,
                                 control_cls=BERTidRAAFPaperScorer)
    assert isinstance(c2, BERTidRAAFPaperScorer)


# ------------------------------------------------------------------ v2 options
def test_label_smoothing_changes_training_and_zero_is_v1(parts):
    tr, va, te = parts
    tf = te[te.source_dataset == "kaggle"]
    base = fit_control(parts).predict_kaggle(tf)
    zero = fit_control(parts, label_smoothing=0.0).predict_kaggle(tf)
    pd.testing.assert_frame_equal(base, zero, check_exact=True)          # default == v1 behaviour
    smooth = fit_control(parts, label_smoothing=0.1)
    assert smooth.label_smoothing == 0.1
    assert np.isfinite(smooth.history_[-1]["loss"])
    assert not np.array_equal(base["confidence"].values, smooth.predict_kaggle(tf)["confidence"].values)
    with pytest.raises(ValueError):
        BERTidRAAFLiteScorer(engine=StubEngine(), label_smoothing=1.0, **KW)
    with pytest.raises(ValueError):
        BERTidRAAFLiteScorer(engine=StubEngine(), swa_last_epochs=KW["epochs"] + 1, **KW)
