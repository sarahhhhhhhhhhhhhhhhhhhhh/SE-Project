"""FairnessAuditor: accuracy and fairness measures with cluster-bootstrap CIs.

It receives the prediction frames of the scorer modules (baseline,
bertidraaf_lite, improved) and calculates accuracy and fairness measures for
the various attributes, reported with bootstrap confidence intervals.

Conventions
-----------
* Kaggle (24-class)  -> fairness = min/max of per-group ACCURACY.
* FairCV (binary)    -> fairness = min/max of per-group SELECTION RATE
  (predicted-positive rate). Two of the three FairCV targets carry deliberately
  biased labels, so accuracy against them is not evidence of fairness.
* FairCV profiles (original_id) appear once per name group, so rows are not
  independent: the bootstrap resamples whole original_id values.

Expected prediction columns: dataset, target, resume_id, original_id,
intersectional_group, true_label, predicted_label (+ prob_positive for FairCV).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

KAGGLE_TARGETS = ("category_label",)
FAIRCV_TARGETS = ("score_blind", "score_biased_gender", "score_biased_ethnicity")
NATIVE_COLS = ("gender_label", "ethnicity_label")
DEFAULT_GROUP_COLS = ["intersectional_group", "gender_label", "ethnicity_label"]
DEFAULT_PAIRS = [
    ("baseline", "bertidraaf_lite"),
    ("bertidraaf_lite", "improved"),
    ("baseline", "improved"),
]
COMPARE_COLUMNS = [
    "algorithm", "dataset", "target", "group_attribute", "metric_basis",
    "accuracy", "accuracy_ci_low", "accuracy_ci_high",
    "fairness_measure", "fairness_ci_low", "fairness_ci_high",
    "n_rows", "n_profiles",
]
PAIRED_COLUMNS = [
    "dataset", "target", "group_attribute", "metric_basis",
    "model_from", "model_to", "fairness_from", "fairness_to",
    "fairness_diff", "fairness_diff_low", "fairness_diff_high",
    "significant_paired", "significant_non_overlap", "direction",
]
_CHUNK = 250  # bootstrap iterations processed per matrix product


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _as_str(s: pd.Series) -> pd.Series:
    """Label -> string, so 1, 1.0 and True all compare equal as '1'."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int).astype(str)
    if pd.api.types.is_float_dtype(s):
        isint = s.notna() & (s == np.floor(s))
        out = s.astype(str)
        out[isint] = s[isint].astype(np.int64).astype(str)
        return out
    if s.dtype == object:
        s = s.map(lambda v: int(v) if isinstance(v, (bool, np.bool_)) else v)
    return s.astype(str)


def _correct(df: pd.DataFrame) -> np.ndarray:
    return (_as_str(df["predicted_label"]) == _as_str(df["true_label"])).to_numpy()


def _pred_int(df: pd.DataFrame) -> np.ndarray:
    s = df["predicted_label"]
    if not pd.api.types.is_numeric_dtype(s):
        s = s.map(lambda v: {"true": 1, "false": 0}.get(str(v).strip().lower(), v))
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()


def _levels(series: pd.Series) -> pd.Index:
    """Deterministic sorted unique non-null values (same order for every model)."""
    vals = pd.unique(series.dropna().astype(object))
    return pd.Index(sorted(vals, key=str), dtype=object)


def _rates(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _ratio_rows(rates: np.ndarray) -> np.ndarray:
    """Row-wise min/max over non-NaN groups. All-zero -> 1.0; no groups -> NaN."""
    rates = np.atleast_2d(rates)
    out = np.full(rates.shape[0], np.nan)
    if rates.shape[1] == 0:
        return out
    valid = ~np.isnan(rates)
    any_valid = valid.any(axis=1)
    mn = np.where(valid, rates, np.inf).min(axis=1)
    mx = np.where(valid, rates, -np.inf).max(axis=1)
    pos = any_valid & (mx > 0)
    out[pos] = mn[pos] / mx[pos]
    out[any_valid & ~(mx > 0)] = 1.0
    return out


def _ci(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.nanpercentile(values, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def _weights(n_clusters: int, n_iterations: int, seed: int):
    """Yield (m x C) matrices of resampling multiplicities, chunk by chunk.

    Same seed + same number of clusters -> identical draws, so every model (and
    both models of a pair) is resampled on the same original_ids.
    """
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_iterations:
        m = min(_CHUNK, n_iterations - done)
        idx = rng.integers(0, n_clusters, size=(m, n_clusters))
        W = np.empty((m, n_clusters))
        for i in range(m):
            W[i] = np.bincount(idx[i], minlength=n_clusters)
        yield W
        done += m


def _tables(df, group_col, basis, cluster_levels, group_levels, cluster_col):
    """Per-cluster x per-group count tables: N (rows), A (correct), K (basis numerator)."""
    C, G = len(cluster_levels), len(group_levels)
    if cluster_col is not None and cluster_col in df.columns:
        c = cluster_levels.get_indexer(pd.Index(df[cluster_col].astype(object)))
    else:
        c = np.arange(len(df))
    g = group_levels.get_indexer(pd.Index(df[group_col].astype(object)))
    keep = (c >= 0) & (g >= 0)
    flat = c[keep] * G + g[keep]
    size = C * G
    N = np.bincount(flat, minlength=size).reshape(C, G).astype(float)
    A = np.bincount(flat, weights=_correct(df)[keep].astype(float), minlength=size).reshape(C, G)
    if basis == "accuracy":
        K = A
    else:
        K = np.bincount(flat, weights=_pred_int(df)[keep].astype(float), minlength=size).reshape(C, G)
    return N, A, K


def _fair_from(W, N, K):
    return _ratio_rows(_rates(W @ K, W @ N))


def _check_basis(basis: str) -> None:
    if basis not in ("accuracy", "selection_rate"):
        raise ValueError(f"basis must be 'accuracy' or 'selection_rate', got {basis!r}")


# --------------------------------------------------------------------------
class FairnessAuditor:
    """Accuracy / fairness measures with cluster-bootstrap confidence intervals."""

    def __init__(self, n_iterations: int = 1000, confidence: float = 0.90,
                 seed: int = 42, cluster_col: Optional[str] = "original_id"):
        self.n_iterations = n_iterations
        self.confidence = confidence
        self.seed = seed
        self.cluster_col = cluster_col

    # ---- 1-3: point measures ------------------------------------------------
    def accuracy_by_group(self, df: pd.DataFrame, group_col: str) -> pd.Series:
        """Per-group accuracy (labels compared as strings). Empty groups -> NaN."""
        tmp = pd.DataFrame({
            "g": df[group_col].reset_index(drop=True),
            "v": _correct(df).astype(float),
        })
        return tmp.groupby("g", observed=False)["v"].mean().rename("accuracy").rename_axis(group_col)

    def selection_rate_by_group(self, df: pd.DataFrame, group_col: str) -> pd.Series:
        """Per-group mean of predicted_label cast to int (FairCV binary only)."""
        tmp = pd.DataFrame({
            "g": df[group_col].reset_index(drop=True),
            "v": _pred_int(df).astype(float),
        })
        return tmp.groupby("g", observed=False)["v"].mean().rename("selection_rate").rename_axis(group_col)

    def fairness_measure(self, df: pd.DataFrame, group_col: str, basis: str = "accuracy") -> float:
        """min/max ratio of per-group accuracy or selection rate, in [0, 1]."""
        _check_basis(basis)
        rates = (self.accuracy_by_group(df, group_col) if basis == "accuracy"
                 else self.selection_rate_by_group(df, group_col))
        return float(_ratio_rows(rates.to_numpy(dtype=float)[None, :])[0])

    # ---- 4: generic cluster bootstrap ----------------------------------------
    def bootstrap_confidence_interval(self, df: pd.DataFrame, metric_fn: Callable,
                                      n_iterations: int = 1000, confidence: float = 0.90,
                                      cluster_col: Optional[str] = "original_id",
                                      seed: int = 42, **kwargs) -> Tuple[float, float, float]:
        """Cluster bootstrap of ``metric_fn(df, **kwargs)``.

        Whole ``cluster_col`` values are resampled with replacement and all of a
        profile's rows stay together. Falls back to row resampling when
        ``cluster_col`` is None or not a column. Returns (point, lower, upper).
        """
        df = df.reset_index(drop=True)
        point = float(metric_fn(df, **kwargs))
        n = len(df)
        if n == 0:
            return point, float("nan"), float("nan")
        if cluster_col is not None and cluster_col in df.columns:
            codes = pd.factorize(df[cluster_col])[0].astype(np.int64)
            missing = codes < 0  # NaN ids: each row its own cluster
            if missing.any():
                codes[missing] = codes.max() + 1 + np.arange(int(missing.sum()))
        else:
            codes = np.arange(n)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes)
        starts = np.cumsum(counts) - counts
        C = len(counts)
        rng = np.random.default_rng(seed)
        vals = np.empty(n_iterations)
        for i in range(n_iterations):
            s = rng.integers(0, C, size=C)
            lens = counts[s]
            cum = np.cumsum(lens) - lens
            off = np.repeat(starts[s] - cum, lens) + np.arange(lens.sum())
            vals[i] = metric_fn(df.iloc[order[off]], **kwargs)
        lo, hi = _ci(vals, confidence)
        return point, lo, hi

    # ---- fast path used by compare / pairwise ---------------------------------
    def _stats(self, df: pd.DataFrame, group_col: str, basis: str) -> Optional[dict]:
        """Accuracy + fairness point estimates and CIs from one vectorised bootstrap."""
        _check_basis(basis)
        df = df[df[group_col].notna()].reset_index(drop=True)
        if len(df) == 0:
            return None
        use_cluster = self.cluster_col is not None and self.cluster_col in df.columns
        cl = _levels(df[self.cluster_col]) if use_cluster else pd.RangeIndex(len(df))
        gl = _levels(df[group_col])
        N, A, K = _tables(df, group_col, basis, cl, gl, self.cluster_col if use_cluster else None)
        C = len(cl)
        ones = np.ones((1, C))
        acc_pt = float(A.sum() / N.sum())
        fair_pt = float(_fair_from(ones, N, K)[0])
        acc_s, fair_s = [], []
        for W in _weights(C, self.n_iterations, self.seed):
            tn, ta = W @ N, W @ A
            tot = tn.sum(axis=1)
            acc_s.append(np.divide(ta.sum(axis=1), tot, out=np.full(len(tot), np.nan), where=tot > 0))
            fair_s.append(_fair_from(W, N, K))
        a_lo, a_hi = _ci(np.concatenate(acc_s), self.confidence)
        f_lo, f_hi = _ci(np.concatenate(fair_s), self.confidence)
        return dict(accuracy=acc_pt, accuracy_ci_low=a_lo, accuracy_ci_high=a_hi,
                    fairness_measure=fair_pt, fairness_ci_low=f_lo, fairness_ci_high=f_hi,
                    n_rows=int(len(df)), n_profiles=int(C))

    # ---- slicing -------------------------------------------------------------
    @staticmethod
    def _normalize_target(df: pd.DataFrame) -> pd.DataFrame:
        """Kaggle has a single target; some prediction files leave it blank.
        Fill blank/NaN Kaggle targets with 'category_label' so groupbys don't drop them."""
        if df is None or "dataset" not in df.columns:
            return df
        df = df.copy()
        if "target" not in df.columns:
            df["target"] = np.nan
        blank = df["target"].isna() | (df["target"].astype(str).str.strip() == "")
        mask = (df["dataset"] == "kaggle") & blank
        if mask.any():
            df["target"] = df["target"].astype(object)
            df.loc[mask, "target"] = KAGGLE_TARGETS[0]
        return df

    @staticmethod
    def _basis_for(dataset: str) -> str:
        return "accuracy" if dataset == "kaggle" else "selection_rate"

    @staticmethod
    def _attach_native(df: pd.DataFrame, native: Optional[pd.DataFrame]) -> pd.DataFrame:
        if native is None or "resume_id" not in df.columns or "resume_id" not in native.columns:
            return df
        cols = [c for c in NATIVE_COLS if c in native.columns and c not in df.columns]
        if not cols:
            return df
        nat = native.drop_duplicates("resume_id").copy()
        nat.index = nat["resume_id"].astype(str)
        df = df.copy()
        key = df["resume_id"].astype(str)
        for c in cols:
            df[c] = key.map(nat[c])
        return df

    def _slices(self, df: pd.DataFrame, group_cols: Sequence[str], label: str) -> Dict[tuple, pd.DataFrame]:
        """{(dataset, target, group_col): rows}, printing a note for anything skipped."""
        out: Dict[tuple, pd.DataFrame] = {}
        if df is None or len(df) == 0:
            print(f"[{label}] empty prediction frame; skipping.")
            return out
        df = self._normalize_target(df)
        warned = set()
        for ds in ("kaggle", "faircv"):
            dsdf = df[df["dataset"] == ds]
            if dsdf.empty:
                print(f"[{label}] no rows for dataset '{ds}'; skipping.")
                continue
            targets = (list(FAIRCV_TARGETS) if ds == "faircv"
                       else sorted(dsdf["target"].dropna().unique()))
            for tgt in targets:
                tdf = dsdf[dsdf["target"] == tgt]
                if tdf.empty:
                    print(f"[{label}] no rows for {ds}/{tgt}; skipping.")
                    continue
                for gcol in group_cols:
                    if ds == "kaggle" and gcol in NATIVE_COLS:
                        continue
                    if gcol not in tdf.columns or tdf[gcol].notna().sum() == 0:
                        if (ds, gcol) not in warned:
                            print(f"[{label}] group column '{gcol}' missing/empty for {ds}; skipping it.")
                            warned.add((ds, gcol))
                        continue
                    out[(ds, tgt, gcol)] = tdf
        return out

    # ---- 5: compare ------------------------------------------------------------
    def compare(self, model_dfs: Dict[str, pd.DataFrame], group_cols: Sequence[str],
                native_labels_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Tidy table of accuracy and fairness (with CIs) per model/dataset/target/attribute.

        Does not validate that models share test rows; call
        ``validate_same_test_rows`` first. The same seed is used for every model.
        """
        rows = []
        for algo, df in model_dfs.items():
            df = self._attach_native(df, native_labels_df)
            for (ds, tgt, gcol), sub in self._slices(df, group_cols, algo).items():
                basis = self._basis_for(ds)
                st = self._stats(sub, gcol, basis)
                if st is None:
                    continue
                rows.append(dict(algorithm=algo, dataset=ds, target=tgt,
                                 group_attribute=gcol, metric_basis=basis, **st))
        return pd.DataFrame(rows, columns=COMPARE_COLUMNS)

    # ---- 6: interval comparison -------------------------------------------------
    @staticmethod
    def is_significant_gap(ci_a: Sequence[float], ci_b: Sequence[float]) -> bool:
        """True if the two intervals do not overlap.

        Each interval is (low, high); a (point, low, high) triple is accepted too.
        NaN intervals are never significant.
        """
        a = tuple(ci_a)[-2:]
        b = tuple(ci_b)[-2:]
        if any(pd.isna(v) for v in (*a, *b)):
            return False
        return bool(a[1] < b[0] or b[1] < a[0])

    # ---- 7: paired bootstrap ------------------------------------------------------
    def paired_fairness_difference(self, df_a: pd.DataFrame, df_b: pd.DataFrame,
                                   group_col: str, basis: str, n_iterations: int = 1000,
                                   confidence: float = 0.90, seed: int = 42,
                                   cluster_col: str = "original_id") -> Tuple[float, float, float]:
        """Cluster bootstrap of fairness(df_b) - fairness(df_a) on shared resamples.

        df_a is the "from" model, df_b the "to" model. Returns (diff, lower, upper).
        The difference is significant when the interval excludes 0.
        """
        _check_basis(basis)
        a = df_a[df_a[group_col].notna()].reset_index(drop=True)
        b = df_b[df_b[group_col].notna()].reset_index(drop=True)
        for d in (a, b):
            if cluster_col not in d.columns:
                raise ValueError(f"paired bootstrap needs cluster column '{cluster_col}'")
        cl = _levels(pd.concat([a[cluster_col], b[cluster_col]], ignore_index=True))
        gl = _levels(pd.concat([a[group_col], b[group_col]], ignore_index=True))
        Na, _, Ka = _tables(a, group_col, basis, cl, gl, cluster_col)
        Nb, _, Kb = _tables(b, group_col, basis, cl, gl, cluster_col)
        C = len(cl)
        ones = np.ones((1, C))
        diff = float(_fair_from(ones, Nb, Kb)[0] - _fair_from(ones, Na, Ka)[0])
        draws = []
        for W in _weights(C, n_iterations, seed):
            draws.append(_fair_from(W, Nb, Kb) - _fair_from(W, Na, Ka))
        lo, hi = _ci(np.concatenate(draws), confidence)
        return diff, lo, hi

    # ---- 8: pairwise comparison -----------------------------------------------------
    def pairwise_comparison(self, model_dfs: Dict[str, pd.DataFrame],
                            pairs: Optional[List[Tuple[str, str]]] = None,
                            group_cols: Optional[Sequence[str]] = None,
                            native_labels_df: Optional[pd.DataFrame] = None,
                            comparison_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """One row per pair x dataset x target x group attribute (see PAIRED_COLUMNS)."""
        pairs = list(DEFAULT_PAIRS if pairs is None else pairs)
        group_cols = list(DEFAULT_GROUP_COLS if group_cols is None else group_cols)
        frames = {k: self._attach_native(v, native_labels_df) for k, v in model_dfs.items()}
        slices = {k: self._slices(v, group_cols, k) for k, v in frames.items()}

        lookup = {}
        if comparison_df is not None and len(comparison_df):
            for r in comparison_df.itertuples(index=False):
                lookup[(r.algorithm, r.dataset, r.target, r.group_attribute)] = (
                    r.fairness_measure, r.fairness_ci_low, r.fairness_ci_high)

        def fair_ci(name, key, sub, basis):
            k = (name, *key)
            if k not in lookup:
                st = self._stats(sub, key[2], basis)
                lookup[k] = (st["fairness_measure"], st["fairness_ci_low"], st["fairness_ci_high"])
            return lookup[k]

        rows = []
        for frm, to in pairs:
            if frm not in frames or to not in frames:
                print(f"[pairwise] pair ({frm}, {to}) skipped: model not provided.")
                continue
            for key, sub_a in slices[frm].items():
                if key not in slices[to]:
                    print(f"[pairwise] {frm}->{to}: {key} missing for '{to}'; skipping.")
                    continue
                ds, tgt, gcol = key
                basis = self._basis_for(ds)
                sub_b = slices[to][key]
                fa, a_lo, a_hi = fair_ci(frm, key, sub_a, basis)
                fb, b_lo, b_hi = fair_ci(to, key, sub_b, basis)
                _, lo, hi = self.paired_fairness_difference(
                    sub_a, sub_b, gcol, basis, n_iterations=self.n_iterations,
                    confidence=self.confidence, seed=self.seed,
                    cluster_col=self.cluster_col or "original_id")
                diff = fb - fa
                includes_zero = bool(lo <= 0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else True
                if includes_zero:
                    direction = "equal"
                else:
                    direction = "higher" if diff > 0 else "lower"
                rows.append(dict(
                    dataset=ds, target=tgt, group_attribute=gcol, metric_basis=basis,
                    model_from=frm, model_to=to, fairness_from=fa, fairness_to=fb,
                    fairness_diff=diff, fairness_diff_low=lo, fairness_diff_high=hi,
                    significant_paired=not includes_zero,
                    significant_non_overlap=self.is_significant_gap((a_lo, a_hi), (b_lo, b_hi)),
                    direction=direction))
        return pd.DataFrame(rows, columns=PAIRED_COLUMNS)

    # ---- validation helpers ------------------------------------------------------------
    @staticmethod
    def validate_same_test_rows(model_dfs: Dict[str, pd.DataFrame], id_col: str = "original_id",
                                verbose: bool = True) -> None:
        """Raise ValueError unless every model has the same (dataset, target, id) set
        and the same row count per (dataset, target)."""
        sets, counts = {}, {}
        for name, df in model_dfs.items():
            df = FairnessAuditor._normalize_target(df)
            missing = [c for c in ("dataset", "target", id_col) if c not in df.columns]
            if missing:
                raise ValueError(f"model '{name}' is missing columns {missing}")
            keys = df[["dataset", "target", id_col]].dropna().astype(str)
            sets[name] = set(keys.drop_duplicates().itertuples(index=False, name=None))
            counts[name] = df.groupby(["dataset", "target"]).size().to_dict()
        names = list(model_dfs)
        ref = names[0]
        problems = []
        for other in names[1:]:
            only_ref, only_other = sets[ref] - sets[other], sets[other] - sets[ref]
            if only_ref or only_other:
                problems.append(f"'{ref}' vs '{other}': {len(only_ref)} combos only in '{ref}', "
                                f"{len(only_other)} only in '{other}'")
                if verbose:
                    print(f"MISMATCH {ref} vs {other}")
                    print(f"  only in {ref}  (first 10): {sorted(only_ref)[:10]}")
                    print(f"  only in {other} (first 10): {sorted(only_other)[:10]}")
            if counts[ref] != counts[other]:
                problems.append(f"'{ref}' vs '{other}': row counts per (dataset, target) differ "
                                f"({counts[ref]} vs {counts[other]})")
        if problems:
            raise ValueError("Models were not scored on identical test rows, so a comparison "
                             "is invalid: " + "; ".join(problems))
        if verbose:
            print(f"OK: all {len(names)} models share the same {len(sets[ref]):,} "
                  f"(dataset, target, {id_col}) combinations and row counts.")

    @staticmethod
    def inspect_native_labels(pred_df: pd.DataFrame, native_df: pd.DataFrame,
                              label_cols: Sequence[str] = NATIVE_COLS) -> pd.DataFrame:
        """Do native labels describe the original profile or the swapped name?

        profile-level: constant within every original_id (varies across profiles).
        name-level: constant within every intersectional_group (follows the swapped name).
        """
        fc = pred_df[pred_df["dataset"] == "faircv"][["resume_id", "original_id", "intersectional_group"]]
        fc = fc.drop_duplicates("resume_id").copy()
        nat = native_df.drop_duplicates("resume_id").copy()
        nat.index = nat["resume_id"].astype(str)
        out = []
        for col in label_cols:
            if col not in nat.columns:
                out.append(dict(column=col, interpretation="column not found in native labels"))
                continue
            m = fc.copy()
            m[col] = m["resume_id"].astype(str).map(nat[col])
            per_profile = m.groupby("original_id")[col].nunique()
            per_group = m.groupby("intersectional_group")[col].nunique()
            vary = float((per_profile > 1).mean()) if len(per_profile) else float("nan")
            if vary == 0:
                interp = "profile-level (describes the ORIGINAL profile; constant across name swaps)"
            elif (per_group <= 1).all():
                interp = "name-level (follows the SWAPPED name; just a coarsening of intersectional_group)"
            else:
                interp = "mixed (varies within profiles and within name groups)"
            out.append(dict(column=col, share_missing=float(m[col].isna().mean()),
                            n_distinct=int(m[col].nunique()),
                            share_profiles_varying=vary, interpretation=interp))
        return pd.DataFrame(out)
