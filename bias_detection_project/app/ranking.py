"""CSV validation, preprocessing, name-swap, scoring, ranking and fairness - all from existing project code."""
import re
import numpy as np
import pandas as pd

import model_adapter as ma  # sets sys.path to project code/

MAX_ROWS = 2000
MIN_WORDS = 30
GROUP_ATTRS = [("intersectional_group", "Intersectional (gender x name group)"),
               ("gender_group", "Gender (name-based proxy)"),
               ("name_group", "Name group (name-based proxy)")]
FAIRNESS_THRESHOLD = 0.8


# ------------------------------------------------------------------ validation
def read_csv(file) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
    raise ValueError("Could not decode the file. Save it as UTF-8 CSV.")


def validate(df: pd.DataFrame):
    """Returns (clean_df, issues[(level, message)]). Errors mean the run cannot proceed."""
    issues = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    if "resume_text" not in low:
        return df, [("error", f"Missing required column 'resume_text'. Found: {list(df.columns)}")]
    df = df.rename(columns={low["resume_text"]: "resume_text"})
    for opt in ("resume_id", "name"):
        if opt in low:
            df = df.rename(columns={low[opt]: opt})
    if "name" not in df.columns:
        issues.append(("warning", "No 'name' column: candidate names cannot be removed from the text "
                                  "before scoring, so a name in the text may influence the result."))
        df["name"] = ""
    if "resume_id" not in df.columns:
        df["resume_id"] = [f"R{i+1:04d}" for i in range(len(df))]
        issues.append(("info", "No 'resume_id' column: IDs were generated."))
    df["resume_id"] = df["resume_id"].astype(str)
    if len(df) == 0:
        return df, [("error", "The file has no rows.")]
    if len(df) > MAX_ROWS:
        return df, [("error", f"{len(df)} rows exceeds the limit of {MAX_ROWS}. Split the file.")]
    dup = df["resume_id"].duplicated(keep="first")
    if dup.any():
        issues.append(("warning", f"{int(dup.sum())} duplicate resume_id value(s) dropped (first kept)."))
        df = df[~dup]
    empty = df["resume_text"].isna() | (df["resume_text"].astype(str).str.strip() == "")
    if empty.any():
        issues.append(("warning", f"{int(empty.sum())} row(s) with empty resume_text were removed: "
                                  f"{', '.join(df.loc[empty, 'resume_id'].head(5))}"))
        df = df[~empty]
    short = df["resume_text"].astype(str).str.split().str.len() < MIN_WORDS
    if short.any():
        issues.append(("warning", f"{int(short.sum())} resume(s) have fewer than {MIN_WORDS} words; "
                                  "scores for very short text are unreliable."))
    if len(df) < 5:
        return df, issues + [("error", "Need at least 5 valid resumes to rank and audit.")]
    return df.reset_index(drop=True), issues


# ------------------------------------------------------------------ preprocessing
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(text: str, name: str) -> str:
    """Remove the candidate's own name, emails and phone numbers so the name-swap audit is the only
    name signal the model sees."""
    t = _EMAIL.sub(" ", str(text))
    t = _PHONE.sub(" ", t)
    name = str(name or "").strip()
    if name and name.lower() != "nan":
        t = re.sub(re.escape(name), " ", t, flags=re.I)
        for part in re.split(r"\s+", name):
            if len(part) >= 3:
                t = re.sub(rf"\b{re.escape(part)}\b", " ", t, flags=re.I)
    return t


def build_variants(df: pd.DataFrame) -> pd.DataFrame:
    """redact -> Preprocessor.clean -> NameSwapAuditor (same order as the training data)."""
    from preprocessor import Preprocessor
    from name_swap_auditor import NameSwapAuditor
    pp = Preprocessor()
    base = df[["resume_id"]].copy()
    base["clean_text"] = [pp.clean(redact(t, n)) for t, n in zip(df["resume_text"], df["name"])]
    return NameSwapAuditor(seed=42).process_dataset(base, text_col="clean_text")


# ------------------------------------------------------------------ scoring
def score_variants(model_key: str, long_df: pd.DataFrame, target: str, max_length=None) -> pd.DataFrame:
    probs, classes = ma.kaggle_probabilities(model_key, long_df["counterfactual_text"].tolist(), max_length)
    if target not in classes:
        raise ValueError(f"Category '{target}' not in model classes.")
    out = long_df[["resume_id", "original_id", "intersectional_group", "name_group", "gender_group"]].copy()
    out["score"] = probs[:, classes.index(target)]
    return out


def skill_match(text: str, required: list, preferred: list):
    t = " " + re.sub(r"[^a-z0-9+#. ]", " ", str(text).lower()) + " "
    hit = lambda s: f" {s.lower()} " in t  # noqa: E731
    req = [s for s in required if hit(s)]
    pre = [s for s in preferred if hit(s)]
    return req, pre


def rank(df: pd.DataFrame, scored: pd.DataFrame, required: list, preferred: list,
         min_years: float, top_k: int, blend: float) -> pd.DataFrame:
    """One row per candidate. model_score = mean P(target) over the 4 name variants."""
    g = scored.groupby("original_id")["score"]
    s = pd.DataFrame({"model_score": g.mean(), "score_spread": g.max() - g.min()})
    out = df.set_index("resume_id").join(s)
    sm = [skill_match(t, required, preferred) for t in out["resume_text"]]
    out["required_skills_matched"] = ["; ".join(a) for a, _ in sm]
    out["preferred_skills_matched"] = ["; ".join(b) for _, b in sm]
    out["required_skill_pct"] = [len(a) / len(required) if required else np.nan for a, _ in sm]
    if "years_experience" in out.columns and min_years:
        out["meets_min_experience"] = pd.to_numeric(out["years_experience"], errors="coerce") >= min_years
    rs = out["required_skill_pct"].fillna(0)
    out["final_score"] = (1 - blend) * out["model_score"] + blend * rs
    out = out.sort_values("final_score", ascending=False).reset_index()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["shortlisted"] = out["rank"] <= top_k
    return out.drop(columns=["resume_text"])


# ------------------------------------------------------------------ fairness
def fairness(scored: pd.DataFrame, top_k: int, n_boot: int = 300) -> dict:
    """Selection-rate fairness of the shortlist across name-swap groups.

    A single score cut-off is taken from the POOLED variants (top_k candidates' worth of rows, i.e.
    top_k * n_groups rows), so groups can end up with different selection rates. Per-group
    shortlisting would force every rate to be equal and show nothing.
    """
    from fairness_auditor import FairnessAuditor
    n_groups = scored["intersectional_group"].nunique()
    n_cand = scored["original_id"].nunique()
    top_k = min(top_k, n_cand)
    cutoff = float(np.sort(scored["score"].to_numpy())[::-1][top_k * n_groups - 1])
    f = scored.copy()
    f["predicted_label"] = (f["score"] >= cutoff).astype(int)
    f["true_label"] = f["predicted_label"]  # unused by the selection-rate basis
    aud = FairnessAuditor(n_iterations=n_boot)
    rows, tables = [], {}
    for col, label in GROUP_ATTRS:
        ratio = aud.fairness_measure(f, col, basis="selection_rate")
        _, lo, hi = aud.bootstrap_confidence_interval(
            f, aud.fairness_measure, n_iterations=n_boot, confidence=0.90,
            cluster_col="original_id", group_col=col, basis="selection_rate")
        rows.append({"attribute": label, "fairness_ratio": ratio, "ci_low": lo, "ci_high": hi,
                     "flagged": bool(ratio < FAIRNESS_THRESHOLD)})
        t = f.groupby(col).agg(selection_rate=("predicted_label", "mean"), mean_score=("score", "mean"),
                               n_rows=("score", "size")).reset_index().rename(columns={col: "group"})
        t.insert(0, "attribute", label)
        tables[col] = t
    sel = f.groupby("original_id")["predicted_label"].sum()
    unstable = sel[(sel > 0) & (sel < n_groups)].rename("variants_shortlisted").reset_index()
    rng = f.groupby("original_id")["score"].agg(lambda x: x.max() - x.min()).rename("score_spread")
    unstable = unstable.merge(rng, on="original_id").sort_values("score_spread", ascending=False)
    unstable = unstable.rename(columns={"original_id": "resume_id"})
    return {"top_k": top_k, "n_candidates": n_cand, "n_groups": n_groups, "cutoff": cutoff,
            "threshold": FAIRNESS_THRESHOLD, "summary": pd.DataFrame(rows),
            "groups": pd.concat(tables.values(), ignore_index=True), "unstable": unstable,
            "n_unstable": int(len(unstable)), "mean_spread": float(rng.mean())}


# ------------------------------------------------------------------ orchestration (full vs fast)
def build_nameless(df: pd.DataFrame) -> pd.DataFrame:
    """One name-free text per resume (1 model pass each instead of 4)."""
    from preprocessor import Preprocessor
    pp = Preprocessor()
    out = pd.DataFrame({"resume_id": df["resume_id"].values})
    out["original_id"] = out["resume_id"]
    out["intersectional_group"] = out["name_group"] = out["gender_group"] = "nameless"
    out["counterfactual_text"] = [pp.clean(redact(t, n)) for t, n in zip(df["resume_text"], df["name"])]
    return out


def analyse(key, df, target, required, preferred, min_years, top_k, blend,
            fast=False, sample_n=40, max_len=None, cache=None):
    """Returns dict(ranked, fair, scored). cache (dict) shares text variants across models."""
    cache = {} if cache is None else cache
    if not fast:
        if "full" not in cache:
            cache["full"] = build_variants(df)
        scored = score_variants(key, cache["full"], target, max_len)
        return dict(ranked=rank(df, scored, required, preferred, min_years, top_k, blend),
                    fair=fairness(scored, top_k), scored=scored)
    if "nameless" not in cache:
        cache["nameless"] = build_nameless(df)
    sc = score_variants(key, cache["nameless"], target, max_len)
    ranked = rank(df, sc, required, preferred, min_years, top_k, blend).drop(columns=["score_spread"])
    n_s = min(sample_n, len(df))
    sample = df.sample(n_s, random_state=42)
    if "sample" not in cache:
        cache["sample"] = build_variants(sample)
    scs = score_variants(key, cache["sample"], target, max_len)
    k = max(1, round(top_k * n_s / len(df)))
    fair = fairness(scs, k)
    fair["note"] = (f"FAST MODE: ranking used name-free text; the fairness audit ran on a random sample of "
                    f"{n_s} of {len(df)} candidates (shortlist scaled to top {k}). Ratios from small samples are noisy.")
    return dict(ranked=ranked, fair=fair, scored=scs)
