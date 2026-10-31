"""
comparison.py -- shared split, shared FairCV metrics, baseline-vs-improved table.

Deliberately torch-free (numpy/pandas only) so both scorers can import it.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd


# --------------------------------------------------------------- shared split
def attach_shared_split(df: pd.DataFrame, path: str, id_col: str = "resume_id") -> pd.DataFrame:
    """Overwrite df['split'] with the assignments the BASELINE used.

    We refuse to create a new split here: a silently different split would make
    the baseline/improved comparison meaningless.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the baseline cell first - it writes data/processed/splits.csv.")
    sp = pd.read_csv(path)[[id_col, "split"]].drop_duplicates(id_col)
    out = df.drop(columns=["split"], errors="ignore").merge(sp, on=id_col, how="left")
    n_missing = out["split"].isna().sum()
    if n_missing:
        raise ValueError(f"{n_missing} rows have no split assignment in {path}.")
    return out


# ------------------------------------------------------------------- metrics
def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC via rank statistic (ties averaged). No sklearn needed."""
    y = np.asarray(y).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def faircv_report(out: pd.DataFrame, group_col: str = "intersectional_group",
                  id_col: str = "original_id") -> Dict:
    """Metrics for ONE FairCV target from a baseline-schema frame (true_label,
    predicted_label, prob_positive, original_id, intersectional_group).

    Mirrors the baseline cell's inline code exactly:
      log-odds = log(p/(1-p)) with p clipped to [1e-9, 1-1e-9]
      spread   = per original_id, max-min log-odds across name variants
      flip     = share of original_ids with >1 distinct predicted_label
      pos rate = mean predicted_label per intersectional_group
    Use this SAME function on both models' outputs.
    """
    y = out["true_label"].astype(int).values
    yh = out["predicted_label"].astype(int).values
    p = out["prob_positive"].astype(float).clip(1e-9, 1 - 1e-9)
    lg = pd.Series(np.log(p / (1 - p)).to_numpy(), index=out.index)
    g = lg.groupby(out[id_col])
    spread = g.max() - g.min()
    flip = out.groupby(id_col)["predicted_label"].nunique().gt(1)
    pos = out.groupby(group_col)["predicted_label"].apply(lambda x: x.astype(int).mean())
    return {
        "accuracy": float((y == yh).mean()),
        "auc": auc_rank(y, out["prob_positive"].astype(float).values),
        "spread_mean": float(spread.mean()),
        "spread_max": float(spread.max()),
        "flip_rate": float(flip.mean()),
        "pos_rate_by_group": pos.round(4).to_dict(),
        "n": int(len(y)),
    }


def pos_rate_gap(d: Dict) -> float:
    v = list(d.get("pos_rate_by_group", {}).values())
    return float(max(v) - min(v)) if v else float("nan")


def compare(baseline: Dict[str, Dict], improved: Dict[str, Dict]) -> pd.DataFrame:
    """Side-by-side table: rows = task/metric, cols = baseline, improved, delta."""
    rows = []
    for task, b in baseline.items():
        if task not in improved:
            continue
        i = improved[task]
        for m in ("accuracy", "auc", "spread_mean", "spread_max", "flip_rate", "pos_rate_gap"):
            if m == "pos_rate_gap":
                bv, iv = pos_rate_gap(b), pos_rate_gap(i)
            else:
                bv, iv = b.get(m, np.nan), i.get(m, np.nan)
            if np.isnan(bv) and np.isnan(iv):
                continue
            rows.append({"task": task, "metric": m, "baseline": bv, "improved": iv, "delta": iv - bv})
    return pd.DataFrame(rows)


def _correct(df: pd.DataFrame) -> np.ndarray:
    t, p = df["true_label"], df["predicted_label"]
    tn, pn = pd.to_numeric(t, errors="coerce"), pd.to_numeric(p, errors="coerce")
    if tn.notna().all() and pn.notna().all():
        return (tn.values == pn.values).astype(float)
    return (t.astype(str).values == p.astype(str).values).astype(float)


def paired_delta(base: pd.DataFrame, impr: pd.DataFrame, key=("resume_id",),
                 id_col: str = "original_id") -> Dict:
    """Paired accuracy difference (improved - baseline) on identical rows, with a standard error
    clustered by original_id (the 4 name variants of one profile are not independent)."""
    key = list(key)
    b = base[key + [id_col]].copy()
    b["cb"] = _correct(base)
    i = impr[key].copy()
    i["ci"] = _correct(impr)
    m = b.merge(i, on=key, validate="1:1")
    per = m.groupby(id_col)[["cb", "ci"]].mean()
    diff = per["ci"] - per["cb"]
    n = len(diff)
    se_i = float(per["ci"].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"baseline": float(per["cb"].mean()), "improved": float(per["ci"].mean()), "se_improved": se_i,
            "delta": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "n_clusters": int(n), "n_rows": int(len(m))}


def verdict(d: Dict) -> str:
    if d["delta"] > 1.96 * d["se"]:
        return "BETTER than baseline (statistically significant)"
    if d["delta"] > 0:
        return "numerically higher than baseline, but within noise (NOT a significant improvement)"
    return "NOT better than baseline"


def placement(d: Dict, paper: float) -> str:
    """Where the improved accuracy sits relative to baseline and a published figure, with a 95% CI.
    Descriptive only - nothing here is used to tune or select a model."""
    acc, se = d["improved"], d["se_improved"]
    lo, hi = acc - 1.96 * se, acc + 1.96 * se
    ci = f"95% CI [{lo:.4f}, {hi:.4f}]"
    if acc <= d["baseline"]:
        return f"at or below the baseline ({ci})"
    if acc < paper:
        return f"between baseline ({d['baseline']:.4f}) and paper ({paper:.4f}) ({ci})"
    if lo <= paper <= hi:
        return (f"numerically above the paper's {paper:.4f}, but the paper value is inside the {ci}: "
                f"statistically indistinguishable from it")
    return (f"ABOVE the paper's {paper:.4f} even allowing for noise ({ci}) - verify there is no leakage/"
            f"duplicates and that the datasets/splits/metric are comparable before claiming this")


def text_ceiling(train: pd.DataFrame, test: pd.DataFrame, targets, thr: Dict[str, float], name_terms,
                 text_col: str = "counterfactual_text", id_col: str = "original_id") -> pd.DataFrame:
    """Best achievable accuracy for ANY text-only model when the text has few distinct 'keys'.

    Key = the text with name tokens removed. For every key, predict the training-majority label
    (a lookup table = the optimal classifier if keys repeat). One row per profile (the 4 name variants
    share text and label). `train_insample_ceiling` is an optimistic upper bound for any model;
    `lookup_test_acc` is the honest test number for the lookup model.
    """
    from ibs_calculator import tokenize_terms

    def key(t):
        return " ".join(w for w in tokenize_terms(t) if w not in name_terms)

    tr = train.drop_duplicates(id_col).copy()
    te = test.drop_duplicates(id_col).copy()
    tr["_k"], te["_k"] = tr[text_col].map(key), te[text_col].map(key)
    rows = []
    for t in targets:
        ytr = (pd.to_numeric(tr[t]) >= thr[t]).astype(int).values
        yte = (pd.to_numeric(te[t]) >= thr[t]).astype(int).values
        g = pd.DataFrame({"k": tr["_k"].values, "y": ytr}).groupby("k")["y"].agg(["mean", "size"])
        maj = (g["mean"] >= 0.5).astype(int)
        insample = float((np.maximum(g["mean"], 1 - g["mean"]) * g["size"]).sum() / g["size"].sum())
        pred = te["_k"].map(maj)
        seen = pred.notna()
        pred = pred.fillna(int(ytr.mean() >= 0.5)).astype(int).values
        rows.append({"target": t, "train_profiles": len(tr), "unique_text_keys": int(len(g)),
                     "test_key_coverage": float(seen.mean()), "lookup_test_acc": float((pred == yte).mean()),
                     "train_insample_ceiling": insample})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- seed ensembling (FairCV)
def average_predictions(frames) -> pd.DataFrame:
    """Average prob_positive across models' predict_faircv() frames (same rows, same order)."""
    base = frames[0].copy()
    for f in frames[1:]:
        assert (f["resume_id"].values == base["resume_id"].values).all() and \
               (f["target"].values == base["target"].values).all(), "frames are not row-aligned"
    p = np.mean([f["prob_positive"].values for f in frames], axis=0)
    base["prob_positive"] = p
    base["confidence"] = np.maximum(p, 1 - p)
    base["predicted_label"] = (p >= 0.5).astype(int)
    if "fusion_gate" in base.columns:
        base["fusion_gate"] = np.mean([f["fusion_gate"].values for f in frames], axis=0)
    return base


def pick_cutoffs(val_pred: pd.DataFrame, targets, grid=None) -> Dict[str, float]:
    """Per-target probability cut-off maximising accuracy on VALIDATION predictions (ties -> nearest 0.5)."""
    grid = np.round(np.arange(0.30, 0.701, 0.02), 2) if grid is None else grid
    cuts = {}
    for t in targets:
        o = val_pred[val_pred["target"] == t]
        ok = o["true_label"].notna().values
        y, p = o["true_label"].astype(int).values[ok], o["prob_positive"].values[ok]
        accs = np.array([((p >= c).astype(int) == y).mean() for c in grid])
        cands = grid[accs >= accs.max() - 1e-12]
        cuts[t] = float(cands[np.argmin(np.abs(cands - 0.5))])
    return cuts


def apply_cutoffs(pred: pd.DataFrame, cuts: Dict[str, float]) -> pd.DataFrame:
    out = pred.copy()
    out["predicted_label"] = [int(p >= cuts.get(t, 0.5)) for p, t in zip(out["prob_positive"], out["target"])]
    return out
