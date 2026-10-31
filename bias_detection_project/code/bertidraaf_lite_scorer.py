"""
bertidraaf_lite_scorer.py  --  CONTROL for the IBS comparison

BERTidRAAFLiteScorer(ImprovedScorer) is identical to ImprovedScorer in everything (backbone, feature
extraction, gate, heads, joint Kaggle + FairCV training, thresholds, epoch selection, decision-threshold
calibration, predict output) EXCEPT where the per-term weights come from:

    ImprovedScorer            : IBSCalculator  -> weight = IBS ("parity") or 1 - IBS ("disparity")
    BERTidRAAFLiteScorer      : IDFCalculator  -> weight = idf(u) = log((N+1)/(df(u)+1)) + 1   (paper Eq. 1)
                                                  N, df from the TRAINING split only.

Nothing is copied from ImprovedScorer: fit()/predict_*()/calibrate_*() are inherited. The only hooks used are
the two methods fit() already calls on `self.ibs_calc` (build_vocab, build_ibs_lookup) and the existing
`_term_weights` method (override: ~6 lines, resolves the OOV weight).

DELIBERATE DIFFERENCES (see also the note delivered with this file)
 vs the BERTidRAAF paper
   * Backbone/fusion are the "lite" ones of ImprovedScorer (DistilBERT, post-BERT weighted pooling, gate).
   * OOV weight: the paper's df=0 value is log(N+1)+1 (the maximum IDF). That value is NOT used by default,
     because EmbeddingEngine hands `oov_weight` to every word-level token that is not a lookup key, and
     punctuation/symbols (each its own "word" for the WordPiece pre-tokenizer) are never lookup keys. With a
     maximal oov_weight punctuation would dominate the pooled vector. Default here (option 1, agreed):
     oov_weight = document-frequency-weighted median of the training IDFs, i.e. the IDF of the median
     training TOKEN. (A median over term TYPES would be near the maximum because the vocabulary is
     dominated by rare terms, which would reintroduce the problem.) Pass oov_weight=... to override.
     [CLS]/[SEP]/padding have word_id None in the engine and keep weight 0 regardless.
   * Natural log is used in Eq. 1.
   * "Training documents" = every training row (Kaggle + FairCV, all counterfactual name variants), i.e. the
     same rows ImprovedScorer.fit() uses to build its IBS lookup.
 vs ImprovedScorer
   * IDF lookup covers ALL training terms (df >= 1) and ignores min_df / max_vocab, so "unseen" means
     "unseen in training" as in the paper. IBS needs min_df for stable group statistics; IDF does not.
     (min_df / max_vocab are still accepted and stored so the constructor is identical.)
   * (v2) accepts the base class's label_smoothing / swa_last_epochs options; requires improved_scorer v2.
   * weight_mode is fixed internally (attribute value "idf") and is not a constructor argument.
   * fit() prints "IBS lookup ... mean IBS" for this class too; for the control read those numbers as IDF
     statistics (cosmetic only).
"""
from __future__ import annotations

import inspect
import math
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from embedding_engine import EmbeddingEngine
from ibs_calculator import IBSCalculator, tokenize_terms
from improved_scorer import ImprovedScorer

__all__ = ["IDFCalculator", "BERTidRAAFLiteScorer", "build_matched_pair", "assert_matched",
           "config_table", "signature_drift", "BERTidRAAFPaperScorer"]


# =============================================================================== IDF
class IDFCalculator:
    """Drop-in for IBSCalculator as far as ImprovedScorer.fit() is concerned.

    idf(u) = log((N + 1) / (df(u) + 1)) + 1, N = #training documents, df = document frequency.
    Terms are produced by ibs_calculator.tokenize_terms (the tokenizer IBSCalculator uses for its counts).
    """

    def __init__(self):
        self.n_docs_: int = 0
        self.df_: Dict[str, int] = {}
        self.unseen_weight_: float = 1.0       # Eq. 1 with df = 0  ->  log(N+1) + 1
        self.token_median_idf_: float = 1.0    # IDF of the median training TOKEN (df-weighted)
        self.type_median_idf_: float = 1.0     # median over term types (reported only)

    @staticmethod
    def idf(n_docs: int, df: int) -> float:
        return math.log((n_docs + 1.0) / (df + 1.0)) + 1.0

    def build_vocab(self, texts, min_df: int = 5, max_vocab: int = 30000) -> List[str]:
        """All training terms with df >= 1. min_df / max_vocab are accepted for interface compatibility and
        deliberately ignored (see module docstring)."""
        return IBSCalculator().build_vocab(texts, min_df=1, max_vocab=sys.maxsize)

    def build_ibs_lookup(self, resumes_by_group: Dict[str, List[str]],
                         vocab: Optional[List[str]] = None) -> Dict[str, float]:
        """{term: idf} for EVERY term in the training documents. `vocab` is ignored (kept for the call
        signature fit() uses). Groups are flattened: only the document collection matters for IDF."""
        df: Counter = Counter()
        n = 0
        for texts in resumes_by_group.values():
            for t in texts:
                df.update(set(tokenize_terms(t)))
                n += 1
        self.n_docs_, self.df_ = n, dict(df)
        self.unseen_weight_ = self.idf(n, 0)
        lookup = {term: self.idf(n, c) for term, c in df.items()}
        if lookup:
            terms = list(df)
            idf_v = np.array([lookup[t] for t in terms], dtype=np.float64)
            df_v = np.array([df[t] for t in terms], dtype=np.float64)
            order = np.argsort(idf_v, kind="stable")
            cum = np.cumsum(df_v[order])
            k = int(np.searchsorted(cum, cum[-1] / 2.0, side="left"))
            self.token_median_idf_ = float(idf_v[order][k])
            self.type_median_idf_ = float(np.median(idf_v))
        else:
            self.token_median_idf_ = self.type_median_idf_ = self.unseen_weight_
        return lookup


# ====================================================================== the control
class BERTidRAAFLiteScorer(ImprovedScorer):
    """ImprovedScorer with training-only IDF term weights. Constructor = ImprovedScorer's minus weight_mode.

    NOTE: the explicit argument list below is intentionally a copy of ImprovedScorer.__init__'s (a *args /
    **kwargs pass-through would break the "identical signature" guarantee); signature_drift() and the test
    suite fail loudly if the two ever diverge.
    """

    def __init__(self, engine: Optional[EmbeddingEngine] = None,
                 model_name: str = "distilbert-base-uncased",
                 faircv_targets: Optional[List[str]] = None,
                 faircv_thresholds: Optional[Dict[str, float]] = None,
                 epochs: int = 4, lr: float = 1e-3, train_batch_size: int = 64,
                 embed_batch_size: int = 16,
                 min_df: int = 5, max_vocab: int = 30000, oov_weight: Optional[float] = None,
                 restore_best: bool = True,
                 fine_tune: bool = False, backbone_lr: float = 3e-5,
                 ft_batch_faircv: int = 32, ft_batch_kaggle: int = 16,
                 chunk_long: Optional[bool] = None,
                 faircv_aux_reg: bool = False, aux_reg_weight: float = 1.0,
                 gate_hidden: int = 128, dropout: float = 0.1,
                 time_budget_minutes: float = 20.0, seed: int = 42,
                 label_smoothing: float = 0.0, swa_last_epochs: int = 0):
        super().__init__(engine=engine, model_name=model_name,
                         weight_mode="parity",   # base class passes the lookup through unchanged
                         faircv_targets=faircv_targets, faircv_thresholds=faircv_thresholds,
                         epochs=epochs, lr=lr, train_batch_size=train_batch_size,
                         embed_batch_size=embed_batch_size,
                         min_df=min_df, max_vocab=max_vocab, oov_weight=oov_weight,
                         restore_best=restore_best,
                         fine_tune=fine_tune, backbone_lr=backbone_lr,
                         ft_batch_faircv=ft_batch_faircv, ft_batch_kaggle=ft_batch_kaggle,
                         chunk_long=chunk_long,
                         faircv_aux_reg=faircv_aux_reg, aux_reg_weight=aux_reg_weight,
                         gate_hidden=gate_hidden, dropout=dropout,
                         time_budget_minutes=time_budget_minutes, seed=seed,
                         label_smoothing=label_smoothing, swa_last_epochs=swa_last_epochs)
        self.weight_mode = "idf"
        self.ibs_calc = IDFCalculator()
        self._oov_explicit = oov_weight is not None
        # None until the lookup exists; _term_weights() resolves it before any engine call.
        self.oov_weight = oov_weight

    # alias requested in the spec; a property so it can never go stale
    @property
    def idf_lookup_(self) -> Dict[str, float]:
        return self.ibs_lookup_

    @idf_lookup_.setter
    def idf_lookup_(self, value: Dict[str, float]):
        self.ibs_lookup_ = value

    _OOV_LABEL = "token-median IDF"

    def _auto_oov_weight(self) -> float:
        m = getattr(self.ibs_calc, "token_median_idf_", None)
        if m is None:
            raise RuntimeError("No IDF statistics available for the default oov_weight; "
                               "pass oov_weight=... explicitly when replacing ibs_calc.")
        return float(m)

    def _term_weights(self, ibs_lookup: Dict[str, float]) -> Dict[str, float]:
        # Hook: every code path (frozen features, fine-tuning, predict) calls this after the lookup is
        # built and BEFORE it reads self.oov_weight, so the default OOV weight is resolved here.
        if not self._oov_explicit:
            w = self._auto_oov_weight()
            if self.oov_weight != w:
                self.oov_weight = w
                c = self.ibs_calc
                print(f"IDF: N={c.n_docs_} docs, {len(ibs_lookup)} terms | oov_weight = {w:.3f} [{self._OOV_LABEL}] "
                      f"(token-median {c.token_median_idf_:.3f}, type-median {c.type_median_idf_:.3f}, "
                      f"Eq.1 df=0 value {c.unseen_weight_:.3f})")
        return dict(ibs_lookup)          # weight = idf, unchanged


class BERTidRAAFPaperScorer(BERTidRAAFLiteScorer):
    """PAPER-LITERAL variant: every token that is not a lookup key (unseen words AND punctuation/symbols,
    which the engine also hands `oov_weight`) gets Eq. 1 with df = 0, i.e. log(N+1)+1 = the maximum IDF.
    This is the setting the agreed default deliberately avoids (punctuation then outweighs every real word
    in the pooled vector), so treat it as a clearly-labelled variant, not as the fair control. An explicit
    oov_weight=... still wins. Same constructor, same everything else."""

    _OOV_LABEL = "paper Eq.1 df=0 = max IDF"

    def _auto_oov_weight(self) -> float:
        return float(self.ibs_calc.unseen_weight_)


# ============================================================ equal-standing helpers
_DESIGN_DIFF = ("weight_mode", "oov_weight", "ibs_calc")          # may differ by design
_FIT_STATE = ("ibs_lookup_", "model", "history_", "classes_", "decision_thr_", "fit_seconds_")


def signature_drift() -> List[str]:
    """Differences between BERTidRAAFLiteScorer.__init__ and ImprovedScorer.__init__ minus weight_mode."""
    base = [p for n, p in inspect.signature(ImprovedScorer.__init__).parameters.items() if n != "weight_mode"]
    ctl = list(inspect.signature(BERTidRAAFLiteScorer.__init__).parameters.values())
    if [p.name for p in base] != [p.name for p in ctl]:
        return [f"parameter names/order differ: {[p.name for p in base]} vs {[p.name for p in ctl]}"]
    out = []
    for b, c in zip(base, ctl):
        if b.kind != c.kind or not _same(b.default, c.default) or str(b.annotation) != str(c.annotation):
            out.append(f"{b.name}: {b.kind}/{b.default!r}/{b.annotation} vs {c.kind}/{c.default!r}/{c.annotation}")
    return out


def _same(x, y) -> bool:
    try:
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return True
        return bool(x == y)
    except Exception:
        return False


def _public_attrs(s) -> dict:
    return {k: v for k, v in vars(s).items() if not k.startswith("_")}


def assert_matched(improved: ImprovedScorer, control: BERTidRAAFLiteScorer) -> None:
    """Raise AssertionError unless every attribute except weight_mode / oov_weight / ibs_calc (and the
    engine object itself and fit-state) is equal, and the engine rules hold."""
    a, b = _public_attrs(improved), _public_attrs(control)
    skip = set(_DESIGN_DIFF) | {"engine"} | set(_FIT_STATE)
    problems = []
    for k in sorted((set(a) ^ set(b)) - skip):
        problems.append(f"attribute '{k}' exists on only one scorer")
    for k in sorted((set(a) & set(b)) - skip):
        if not _same(a[k], b[k]):
            problems.append(f"{k}: improved={a[k]!r} control={b[k]!r}")
    ei, ec = improved.engine, control.engine
    if (ei is None) != (ec is None):
        problems.append("engine: one scorer has an engine, the other does not")
    elif ei is not None:
        if improved.fine_tune and ei is ec:
            problems.append("fine_tune=True but both scorers share ONE engine (the second would start from "
                            "the first one's fine-tuned backbone)")
        for attr in ("model_name", "max_length", "hidden_size"):
            if not _same(getattr(ei, attr, None), getattr(ec, attr, None)):
                problems.append(f"engine.{attr}: {getattr(ei, attr, None)!r} vs {getattr(ec, attr, None)!r}")
    drift = signature_drift()
    problems += [f"constructor drift: {d}" for d in drift]
    if problems:
        raise AssertionError("Matched pair is NOT matched:\n  " + "\n  ".join(problems))


def _fmt(k: str, v) -> str:
    if k == "engine":
        return "None (lazy)" if v is None else f"{type(v).__name__}@{id(v) % 10**6:06d}"
    if k == "ibs_calc":
        return type(v).__name__
    if k == "oov_weight" and v is None:
        return "auto: token-median IDF (set at fit)"
    return repr(v)


def config_table(improved: ImprovedScorer, control: BERTidRAAFLiteScorer) -> pd.DataFrame:
    a, b = _public_attrs(improved), _public_attrs(control)
    rows = []
    for k in sorted((set(a) | set(b)) - set(_FIT_STATE)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if k == "engine":
            status = "same" if va is vb else "separate engines"
        elif _same(va, vb):
            status = "same"
        else:
            status = "BY DESIGN" if k in _DESIGN_DIFF else "MISMATCH"
        rows.append({"attribute": k, "improved": _fmt(k, va), "control": _fmt(k, vb), "status": status})
    return pd.DataFrame(rows)


def build_matched_pair(common_kwargs: dict, ibs_weight_mode: str,
                       engine_factory: Optional[Callable[[], object]] = None,
                       verbose: bool = True,
                       control_cls: type = BERTidRAAFLiteScorer) -> Tuple[ImprovedScorer, BERTidRAAFLiteScorer]:
    """(improved, control) built from the SAME kwargs; they differ only in weight source.

    common_kwargs: the improved cell's constructor kwargs WITHOUT weight_mode / engine / oov_weight.
    fine_tune=True  -> each scorer gets its own engine_factory() engine (a shared engine would let the
                       second scorer start from the first one's fine-tuned backbone).
    fine_tune=False -> one engine, shared.
    engine_factory=None -> both scorers lazily build their own engines.
    control_cls: BERTidRAAFLiteScorer (default, agreed OOV) or BERTidRAAFPaperScorer (paper-literal OOV).
    """
    ck = dict(common_kwargs)
    forbidden = [k for k in ("weight_mode", "engine", "oov_weight") if k in ck]
    if forbidden:
        raise ValueError(f"common_kwargs must not contain {forbidden}: they are the weighting variable "
                         f"(weight_mode/oov_weight) or are handled by engine_factory (engine).")
    if ibs_weight_mode not in ("disparity", "parity"):
        raise ValueError("ibs_weight_mode must be 'disparity' or 'parity'")
    valid = set(inspect.signature(ImprovedScorer.__init__).parameters) - {"self", "weight_mode"}
    unknown = sorted(set(ck) - valid)
    if unknown:
        raise ValueError(f"Unknown ImprovedScorer constructor arguments: {unknown}")
    drift = signature_drift()
    if drift:
        raise AssertionError(f"BERTidRAAFLiteScorer signature drifted from ImprovedScorer: {drift}")

    fine_tune = bool(ck.get("fine_tune", inspect.signature(ImprovedScorer.__init__).parameters["fine_tune"].default))
    if engine_factory is None:
        eng_i = eng_c = None
    elif fine_tune:
        eng_i, eng_c = engine_factory(), engine_factory()
        if eng_i is eng_c:
            raise ValueError("engine_factory returned the same object twice; with fine_tune=True it must "
                             "build a fresh engine on every call.")
    else:
        eng_i = eng_c = engine_factory()

    improved = ImprovedScorer(engine=eng_i, weight_mode=ibs_weight_mode, **ck)
    control = control_cls(engine=eng_c, **ck)
    assert_matched(improved, control)
    if verbose:
        print(config_table(improved, control).to_string(index=False))
    return improved, control
