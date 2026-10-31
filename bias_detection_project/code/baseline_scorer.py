"""
baseline_scorer.py  --  Module 4: BaselineScorer

Implements the original tf-idf scoring system so the improved scorer has a
baseline to compare with. It uses the known system (scikit-learn's
TfidfVectorizer + LogisticRegression) instead of building tf-idf from scratch.

Design notes
------------
* fit(train_df) fits ONE TfidfVectorizer on the training split's text only
  (no val/test text ever touches the vocabulary or IDF weights).
* predict_kaggle / predict_faircv lazily train a LogisticRegression on the
  tf-idf features of the training rows, then predict on whatever df is passed.
  - Kaggle task:  multi-class, target = category_label (24 classes).
  - FairCVdb task: binary, target = binarized suitability_score.
* BINARIZATION RULE (FairCVdb): label = 1 ("high suitability") if
  suitability_score >= MEDIAN of the TRAINING split's scores, else 0.
  The median is computed from training rows only (no test leakage) and stored
  in `faircv_threshold_`. NOTE: this is a placeholder definition -- if the
  BERTidRAAF paper defines the binary target differently, change
  `binarize_suitability` so both scorers use the same target.

Splitting (important for comparing against ImprovedScorer)
----------------------------------------------------------
`ensure_split` respects an existing `split` column (values train/val/test,
with common aliases normalised). If there is NO `split` column, it CREATES a
stratified 80/10/10 split, and says so with a warning. Two details matter:
  1. Splitting is done at the `original_id` level, so every counterfactual
     variant of one resume lands in the SAME split. Splitting on rows would
     leak near-identical resumes from train into test and inflate accuracy.
  2. The split is deterministic (fixed seed). Persist it (the Colab cell saves
     data/processed/splits.csv) and reuse it in ImprovedScorer.
"""
from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test",
}


# --------------------------------------------------------------------------
# Target-leakage scrubber (FairCVdb text)
# --------------------------------------------------------------------------
# FairCV counterfactual_text ends with a verbal label such as
# "... and high current availability reflecting a low overall suitability score",
# which IS the target. It must be removed before ANY model (tf-idf baseline,
# ImprovedScorer, BERTidRAAF) sees the text, or accuracy is trivially ~100%.
_SCORE_PHRASE = re.compile(
    r"[\s,]*\breflecting\s+(?:an?\s+)?[\w\- ]{0,30}?\boverall\s+suitability\s+score\b\.?",
    flags=re.IGNORECASE,
)


def strip_suitability_phrase(text: str) -> str:
    """Remove the 'reflecting a <level> overall suitability score' phrase."""
    return _SCORE_PHRASE.sub("", str(text)).strip()


# --------------------------------------------------------------------------
# Splitting helper
# --------------------------------------------------------------------------
def ensure_split(
    df: pd.DataFrame,
    strata: pd.Series,
    group_col: str = "original_id",
    split_col: str = "split",
    seed: int = 42,
) -> tuple[pd.DataFrame, bool]:
    """Return (df_with_split, created).

    If `split_col` exists it is used as-is (normalised to train/val/test) and
    created=False. Otherwise a stratified 80/10/10 split is CREATED here,
    grouped by `group_col` (falls back to resume_id), and created=True.
    `strata` is a Series aligned with df.index giving the label to stratify on.
    """
    df = df.copy()

    if split_col in df.columns:
        norm = df[split_col].astype(str).str.strip().str.lower().map(_SPLIT_ALIASES)
        if norm.isna().any():
            bad = df.loc[norm.isna(), split_col].unique()[:5]
            raise ValueError(f"Unrecognised values in '{split_col}': {list(bad)}")
        df[split_col] = norm
        return df, False

    warnings.warn(
        f"No '{split_col}' column found: creating a stratified 80/10/10 split "
        f"grouped by '{group_col}' (seed={seed}). Reuse this same split in "
        "ImprovedScorer so the comparison is consistent."
    )
    if group_col not in df.columns:
        group_col = "resume_id"

    units = (
        pd.DataFrame({"g": df[group_col], "s": strata})
        .groupby("g", sort=True)["s"].first()
    )
    ids, strat = units.index.to_numpy(), units.to_numpy()

    def _split(a, s, test_size):
        try:
            return train_test_split(a, s, test_size=test_size,
                                    stratify=s, random_state=seed)
        except ValueError:  # a class too small to stratify
            warnings.warn("Stratification failed (rare class); splitting unstratified.")
            return train_test_split(a, s, test_size=test_size, random_state=seed)

    train_ids, temp_ids, _, temp_s = _split(ids, strat, 0.2)
    val_ids, test_ids, _, _ = _split(temp_ids, temp_s, 0.5)

    mapping = {**{i: "train" for i in train_ids},
               **{i: "val" for i in val_ids},
               **{i: "test" for i in test_ids}}
    df[split_col] = df[group_col].map(mapping)
    return df, True


# --------------------------------------------------------------------------
# BaselineScorer
# --------------------------------------------------------------------------
class BaselineScorer:
    def __init__(
        self,
        tfidf_kwargs: Optional[dict] = None,
        C: float = 1.0,
        random_state: int = 42,
        faircv_threshold: Optional[float] = None,
    ):
        # Defaults are deliberately plain; override via tfidf_kwargs if needed.
        self.tfidf_kwargs = {
            "lowercase": True, "stop_words": "english",
            "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True,
            **(tfidf_kwargs or {}),
        }
        self.C = C
        # If given, this FIXED cut-off defines the FairCV binary target (use the
        # same value in every scorer). If None, the TRAIN-split median is used,
        # which can drift when the split is grouped by profile.
        self._fixed_threshold = faircv_threshold
        self.random_state = random_state

        self.text_col: str = "counterfactual_text"
        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.faircv_threshold_: Optional[float] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._X_train = None
        self._kaggle_clf: Optional[LogisticRegression] = None
        self._faircv_clf: Optional[LogisticRegression] = None

    # ---- 1. fit ----------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, text_col: str = "counterfactual_text") -> None:
        """Fit a TfidfVectorizer on the training split's text only."""
        self.text_col = text_col
        self._train_df = train_df.reset_index(drop=True)
        self.vectorizer_ = TfidfVectorizer(**self.tfidf_kwargs)
        self._X_train = self.vectorizer_.fit_transform(
            self._train_df[text_col].fillna("").astype(str)
        )
        # Invalidate any classifiers trained on a previous fit.
        self._kaggle_clf = None
        self._faircv_clf = None
        self.faircv_threshold_ = None

    def _check_fitted(self) -> None:
        if self.vectorizer_ is None:
            raise RuntimeError("BaselineScorer is not fitted: call fit(train_df) first.")

    def _transform(self, df: pd.DataFrame):
        return self.vectorizer_.transform(df[self.text_col].fillna("").astype(str))

    def _new_clf(self) -> LogisticRegression:
        return LogisticRegression(max_iter=1000, C=self.C,
                                  random_state=self.random_state)

    # ---- 2. Kaggle 24-class ---------------------------------------------
    def predict_kaggle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict category_label. Returns resume_id, [original_id],
        intersectional_group, true_label, predicted_label, confidence
        (confidence = probability of the predicted class)."""
        self._check_fitted()
        if self._kaggle_clf is None:
            mask = self._train_df["category_label"].notna().to_numpy()
            if not mask.any():
                raise ValueError("No training rows with a category_label.")
            clf = self._new_clf()
            clf.fit(self._X_train[mask], self._train_df.loc[mask, "category_label"])
            self._kaggle_clf = clf

        proba = self._kaggle_clf.predict_proba(self._transform(df))
        idx = proba.argmax(axis=1)
        out = self._base_frame(df)
        out["true_label"] = df["category_label"].to_numpy()
        out["predicted_label"] = self._kaggle_clf.classes_[idx]
        out["confidence"] = proba.max(axis=1)
        return out

    # ---- 3. FairCVdb binary ---------------------------------------------
    @staticmethod
    def binarize_suitability(scores: pd.Series, threshold: float) -> pd.Series:
        """1 if suitability_score >= threshold (train-set median), else 0.
        Missing scores stay missing (<NA>)."""
        s = pd.to_numeric(scores, errors="coerce")
        out = (s >= threshold).astype("Int64")
        out[s.isna()] = pd.NA
        return out

    def predict_faircv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict binarized suitability (median split on the training scores).
        Same columns as predict_kaggle, plus prob_positive = P(label == 1)."""
        self._check_fitted()
        if self._faircv_clf is None:
            scores = pd.to_numeric(self._train_df["suitability_score"], errors="coerce")
            mask = scores.notna().to_numpy()
            if not mask.any():
                raise ValueError("No training rows with a suitability_score.")
            # MEDIAN SPLIT on the training split only (see module docstring).
            self.faircv_threshold_ = (float(self._fixed_threshold)
                                      if self._fixed_threshold is not None
                                      else float(scores[mask].median()))
            y = self.binarize_suitability(scores[mask], self.faircv_threshold_).astype(int)
            if y.nunique() < 2:
                raise ValueError(
                    "Median split produced a single class (many tied scores?). "
                    "Choose a different threshold in binarize_suitability."
                )
            clf = self._new_clf()
            clf.fit(self._X_train[mask], y)
            self._faircv_clf = clf

        proba = self._faircv_clf.predict_proba(self._transform(df))
        idx = proba.argmax(axis=1)
        classes = self._faircv_clf.classes_
        out = self._base_frame(df)
        out["true_label"] = self.binarize_suitability(
            df["suitability_score"].reset_index(drop=True), self.faircv_threshold_
        )
        out["predicted_label"] = classes[idx]
        out["confidence"] = proba.max(axis=1)
        out["prob_positive"] = proba[:, list(classes).index(1)]
        return out

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("resume_id", "original_id", "intersectional_group")
                if c in df.columns]
        return df[cols].reset_index(drop=True).copy()
