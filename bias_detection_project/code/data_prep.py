"""data_prep.py -- the same data preparation as the baseline cell (labels from the npy, shared split,
optional verbal-score scrub), packaged so a cell can reuse it. Torch-free."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from baseline_scorer import strip_suitability_phrase
from comparison import attach_shared_split


def load_prepared(project_root: str, apply_scrub: bool = True):
    proc = project_root + "data/processed/"
    raw_path = project_root + "data/raw/faircvdb/FairCVdb.npy"
    df = pd.read_csv(proc + "counterfactual_resumes.csv", low_memory=False)
    src = df["source_dataset"].astype(str).str.lower()
    kaggle = df[src.str.contains("kaggle")].copy().drop(columns=["split"], errors="ignore")
    fc = df[src.str.contains("faircv")].copy()

    d = np.load(raw_path, allow_pickle=True)
    raw = d.item() if (d.dtype == object and d.shape == ()) else d

    def labels(split):
        n = len(raw[f"Profiles {split}"])
        return pd.DataFrame({
            "original_id": [f"faircv_{split.lower()}_{i}" for i in range(n)],
            "score_blind": raw[f"Blind Labels {split}"],
            "score_biased_gender": raw[f"Biased Labels {split} (Gender)"],
            "score_biased_ethnicity": raw[f"Biased Labels {split} (Ethnicity)"]})

    lab = pd.concat([labels("Train"), labels("Test")], ignore_index=True)
    fc = fc.drop(columns=["suitability_score", "split"], errors="ignore").merge(
        lab, on="original_id", how="left", validate="m:1")
    assert fc["score_blind"].notna().all(), "original_id values did not match the npy rows"

    all_df = pd.concat([kaggle, fc], ignore_index=True)
    all_df = attach_shared_split(all_df, proc + "splits.csv")     # the BASELINE's split
    is_fc = all_df["source_dataset"].astype(str).str.lower().str.contains("faircv")
    if apply_scrub:
        all_df.loc[is_fc, "counterfactual_text"] = (
            all_df.loc[is_fc, "counterfactual_text"].map(strip_suitability_phrase))
    thr = json.load(open(proc + "faircv_thresholds.json"))
    tr, te = all_df[all_df.split == "train"], all_df[all_df.split == "test"]
    assert not set(tr.original_id) & set(te.original_id), "train/test share original_ids"
    return all_df, thr
