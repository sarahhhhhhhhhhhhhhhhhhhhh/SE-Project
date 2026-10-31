import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_scorer import BaselineScorer, ensure_split, split_kaggle, split_faircv  # noqa

GROUPS = ["female_groupA", "female_groupB", "male_groupA", "male_groupB"]
NAMES = ["Emily Ryan", "Lakisha Jones", "Greg Kelly", "Jamal Booker"]
KAGGLE_WORDS = {"IT": "python django flask sql",
                "HEALTH": "nurse patient ward clinical",
                "TRADES": "welder steel weld fabrication"}


def _variants(oid, text, **fields):
    return [dict(resume_id=f"{oid}__{g}", original_id=oid, intersectional_group=g,
                 counterfactual_text=f"{NAMES[j]} {text}", **fields)
            for j, g in enumerate(GROUPS)]


@pytest.fixture
def kaggle_df():
    rows = []
    for cat, words in KAGGLE_WORDS.items():
        for i in range(20):
            rows += _variants(f"{cat}{i}", f"{words} uniq{cat}{i}",
                              category_label=cat, source_dataset="kaggle",
                              suitability_score=np.nan)
    return pd.DataFrame(rows)


@pytest.fixture
def faircv_df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(100):
        high = i % 2 == 0
        score = rng.uniform(0.7, 1.0) if high else rng.uniform(0.0, 0.3)
        text = ("expert leadership senior" if high else "novice entry junior") + f" u{i}"
        rows += _variants(f"F{i}", text, suitability_score=score,
                          source_dataset="faircvdb", category_label=None,
                          gender_label=i % 2, ethnicity_label=i % 3)
    return pd.DataFrame(rows)


def _scorer():
    return BaselineScorer(min_df=1, ngram_range=(1, 1))


def test_split_is_grouped_and_roughly_80_10_10(kaggle_df):
    out = split_kaggle(kaggle_df)
    assert (out.groupby("original_id")["split"].nunique() == 1).all()
    frac = out.drop_duplicates("original_id")["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.8, abs=0.05)
    assert frac["val"] == pytest.approx(0.1, abs=0.05)
    assert frac["test"] == pytest.approx(0.1, abs=0.05)


def test_predict_kaggle(kaggle_df):
    df = split_kaggle(kaggle_df)
    s = _scorer()
    s.fit(df[df.split == "train"])
    pred = s.predict_kaggle(df)
    test = df[df.split == "test"]
    assert list(pred.columns) == ["resume_id", "original_id", "intersectional_group",
                                  "source_dataset", "true_label", "predicted_label",
                                  "confidence"]
    assert set(pred.resume_id) == set(test.resume_id)
    assert not set(pred.original_id) & set(df[df.split == "train"].original_id)  # no leakage
    assert pred.confidence.between(0, 1).all()
    assert (pred.true_label == pred.predicted_label).mean() > 0.9
    assert set(pred.predicted_label) <= set(KAGGLE_WORDS)


def test_predict_faircv(faircv_df):
    df = split_faircv(faircv_df)
    s = _scorer()
    s.fit(df[df.split == "train"])
    pred = s.predict_faircv(df)
    assert list(pred.columns) == ["resume_id", "original_id", "intersectional_group",
                                  "source_dataset", "true_label", "predicted_label",
                                  "confidence", "p_positive"]
    assert set(pred.true_label) <= {0, 1} and set(pred.predicted_label) <= {0, 1}
    assert pred.p_positive.between(0, 1).all()
    assert (pred.true_label == pred.predicted_label).mean() > 0.85
    # threshold is the median over TRAIN originals only
    train_scores = df[df.split == "train"].drop_duplicates("original_id").suitability_score
    assert s.faircv_threshold_ == pytest.approx(train_scores.median())


def test_existing_split_column_is_respected(kaggle_df):
    ids = kaggle_df.original_id.unique()
    mapping = {o: ("test" if i % 10 == 0 else "val" if i % 10 == 1 else "training")
               for i, o in enumerate(ids)}
    kaggle_df["split"] = kaggle_df.original_id.map(mapping)
    s = _scorer()
    s.fit(kaggle_df[kaggle_df.split == "training"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = s.predict_kaggle(kaggle_df)
    # must NOT have created a new split
    assert not [w for w in caught if "split" in str(w.message).lower()]
    expected = {o for o, sp in mapping.items() if sp == "test"}
    assert set(pred.original_id) == expected


def test_leaky_split_column_rejected(kaggle_df):
    kaggle_df["split"] = ["train", "test"] * (len(kaggle_df) // 2)  # splits variants apart
    with pytest.raises(ValueError):
        ensure_split(kaggle_df)


def test_predict_before_fit_raises(kaggle_df):
    with pytest.raises(RuntimeError):
        _scorer().predict_kaggle(split_kaggle(kaggle_df))
