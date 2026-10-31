"""Information-ceiling check for the FairCV text (pandas only, no GPU)."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from ibs_calculator import IBSCalculator, tokenize_terms


def name_tokens(train_df, text_col="counterfactual_text", min_df=5):
    """Terms with IBS < 1 on the TRAIN split (same rule ImprovedScorer uses): name tokens in counterfactual data."""
    calc = IBSCalculator()
    texts = train_df[text_col].astype(str).tolist()
    vocab = calc.build_vocab(texts, min_df=min_df, max_vocab=30000)
    groups = defaultdict(list)
    for g, t in zip(train_df["intersectional_group"], texts):
        groups[g].append(t)
    lk = calc.build_ibs_lookup(groups, vocab)
    return {t for t, v in lk.items() if v < 1.0}


def name_free_key(texts, names):
    return [" ".join(w for w in tokenize_terms(t) if w not in names) for t in texts.astype(str)]


def ceiling_table(all_df, thr, targets, text_col="counterfactual_text"):
    """Lookup-table classifier on the NAME-FREE text: majority label per distinct text, learned on train,
    scored on test. It uses only what the text says, so a text model that cannot beat it is information-limited."""
    f = all_df[all_df["source_dataset"].astype(str).str.lower().str.contains("faircv")]
    tr, te = f[f.split == "train"], f[f.split == "test"]
    names = name_tokens(tr, text_col)
    ktr, kte = name_free_key(tr[text_col], names), name_free_key(te[text_col], names)
    seen = set(ktr)
    cov = float(np.mean([k in seen for k in kte]))
    print(f"name tokens removed: {len(names)} | distinct name-free texts: train {len(seen)}, test {len(set(kte))} "
          f"| test rows whose text occurs in train: {cov:.2%}")
    rows = []
    for t in targets:
        ytr = (pd.to_numeric(tr[t], errors="coerce").values >= thr[t]).astype(int)
        yte = (pd.to_numeric(te[t], errors="coerce").values >= thr[t]).astype(int)
        g = pd.DataFrame({"k": ktr, "y": ytr}).groupby("k")["y"].agg(["mean", "size"])
        fallback = int(ytr.mean() >= 0.5)
        pred = np.array([int(g.at[k, "mean"] >= 0.5) if k in g.index else fallback for k in kte])
        in_sample = float((np.maximum(g["mean"], 1 - g["mean"]) * g["size"]).sum() / g["size"].sum())
        rows.append({"target": t, "table_test_acc": float((pred == yte).mean()), "in_sample_purity": in_sample})
    return pd.DataFrame(rows).set_index("target")
