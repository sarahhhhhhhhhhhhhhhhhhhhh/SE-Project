"""
resume_ingestor.py

Ingests two heterogeneous HR datasets (Kaggle 24-class resume dataset and
FairCVdb) into a single unified DataFrame schema for downstream bias
detection processing.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Positional decode maps for FairCVdb's 'Profiles' arrays.
ETHNICITY_MAP: Dict[int, str] = {0: "East Asian", 1: "Sub-Saharan African and South Asian", 2: "Caucasian"}
GENDER_MAP: Dict[int, str] = {0: "Male", 1: "Female"}
OCCUPATION_MAP: Dict[int, str] = {
    0: "nurse",
    1: "surgeon",
    2: "physician",
    3: "journalist",
    4: "photographer",
    5: "filmmaker",
    6: "teacher",
    7: "professor",
    8: "attorney",
    9: "accountant",
}

# Expected dataset sizes, used only for a soft validation warning.
EXPECTED_KAGGLE_ROWS = 2_484
EXPECTED_FAIRCV_TRAIN_ROWS = 19_200
EXPECTED_FAIRCV_TEST_ROWS = 4_800

UNIFIED_COLUMNS: List[str] = [
    "resume_id",
    "raw_text",
    "category_label",
    "bio_text",
    "name",
    "gender_label",
    "ethnicity_label",
    "occupation_label",
    "suitability_score",
    "blind_score",
    "biased_score_gender",
    "biased_score_ethnicity",
    "source_dataset",
]


class ResumeIngestor:
    """Ingests Kaggle resume data and FairCVdb into one common schema."""

    def ingest_kaggle(self, csv_path: str) -> pd.DataFrame:
        """Load the Kaggle 24-class Resume dataset CSV.

        Args:
            csv_path: Path to Resume.csv (columns: ID, Resume_str,
                Resume_html, Category).

        Returns:
            DataFrame with columns resume_id, raw_text, category_label,
            source_dataset.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            ValueError: If expected columns are missing.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Kaggle Resume.csv not found at: {csv_path}. "
                "Confirm the dataset is placed under "
                "PROJECT_ROOT/data/raw/kaggle/Resume/Resume.csv."
            )

        df = pd.read_csv(csv_path)

        required_cols = {"ID", "Resume_str", "Category"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Resume.csv is missing expected columns: {missing_cols}"
            )

        out = pd.DataFrame(
            {
                "resume_id": df["ID"],
                "raw_text": df["Resume_str"],
                "category_label": df["Category"],
                "source_dataset": "kaggle_resume",
            }
        )

        n_malformed = out["raw_text"].isna().sum()
        if n_malformed:
            print(
                f"[WARNING] {n_malformed} Kaggle rows have missing "
                "Resume_str text."
            )

        return out

    def ingest_faircv(self, npy_path: str) -> pd.DataFrame:
        """Load FairCVdb.npy and unify its Train/Test splits.

        Args:
            npy_path: Path to FairCVdb.npy.

        Returns:
            DataFrame with the FairCVdb-specific unified schema (see
            module docstring / UNIFIED_COLUMNS).

        Raises:
            FileNotFoundError: If npy_path does not exist.
            KeyError: If expected keys are missing from the pickled dict.
        """
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(
                f"FairCVdb.npy not found at: {npy_path}. "
                "Confirm the dataset is placed under "
                "PROJECT_ROOT/data/raw/faircvdb/FairCVdb.npy."
            )

        data = np.load(npy_path, allow_pickle=True).item()

        required_keys = {
            "Profiles Train",
            "Bios Train",
            "Names Train",
            "Blind Labels Train",
            "Biased Labels Train (Gender)",
            "Biased Labels Train (Ethnicity)",
            "Profiles Test",
            "Bios Test",
            "Names Test",
            "Blind Labels Test",
            "Biased Labels Test (Gender)",
            "Biased Labels Test (Ethnicity)",
        }
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise KeyError(
                f"FairCVdb.npy is missing expected keys: {missing_keys}"
            )

        frames = []
        for split in ("Train", "Test"):
            frames.append(self._decode_faircv_split(data, split))

        combined = pd.concat(frames, ignore_index=True)

        n_train = len(data["Profiles Train"])
        n_test = len(data["Profiles Test"])
        total_expected = EXPECTED_FAIRCV_TRAIN_ROWS + EXPECTED_FAIRCV_TEST_ROWS
        if n_train != EXPECTED_FAIRCV_TRAIN_ROWS or n_test != EXPECTED_FAIRCV_TEST_ROWS:
            print(
                f"[WARNING] FairCVdb split sizes (train={n_train}, "
                f"test={n_test}) differ from expected "
                f"(train={EXPECTED_FAIRCV_TRAIN_ROWS}, "
                f"test={EXPECTED_FAIRCV_TEST_ROWS}); total expected "
                f"{total_expected}."
            )

        return combined

    def _decode_faircv_split(
        self, data: Dict[str, np.ndarray], split: str
    ) -> pd.DataFrame:
        """Decode one FairCVdb split ('Train' or 'Test') into the unified schema."""
        profiles = np.asarray(data[f"Profiles {split}"])
        bios = data[f"Bios {split}"]
        names = data[f"Names {split}"]
        blind = data[f"Blind Labels {split}"]
        biased_gender = data[f"Biased Labels {split} (Gender)"]
        biased_ethnicity = data[f"Biased Labels {split} (Ethnicity)"]

        n_rows = profiles.shape[0]
        prefix = "faircv_train" if split == "Train" else "faircv_test"

        records = []
        for i in range(n_rows):
            row = profiles[i]

            ethnicity_code = int(row[0])
            gender_code = int(row[1])
            occupation_code = int(row[2])
            suitability = float(row[3])
            education = row[4]
            experience = row[5]
            recommendation = row[6]
            availability = row[7]
            language_proficiency = row[8:11]

            ethnicity_label = ETHNICITY_MAP.get(ethnicity_code, "Unknown")
            gender_label = GENDER_MAP.get(gender_code, "Unknown")
            occupation_label = OCCUPATION_MAP.get(occupation_code, "Unknown")

            raw_text = (
                f"Occupation: {occupation_label}; "
                f"Suitability: {suitability}; "
                f"Education: {education}; "
                f"Experience: {experience}; "
                f"Recommendation: {recommendation}; "
                f"Availability: {availability}; "
                f"Language proficiency: {list(language_proficiency)}"
            )

            records.append(
                {
                    "resume_id": f"{prefix}_{i}",
                    "raw_text": raw_text,
                    "category_label": np.nan,
                    "bio_text": bios[i],
                    "name": names[i],
                    "gender_label": gender_label,
                    "ethnicity_label": ethnicity_label,
                    "occupation_label": occupation_label,
                    "suitability_score": suitability,
                    "blind_score": blind[i],
                    "biased_score_gender": biased_gender[i],
                    "biased_score_ethnicity": biased_ethnicity[i],
                    "source_dataset": "faircv",
                }
            )

        return pd.DataFrame(records)

    def ingest_all(
        self, kaggle_csv_path: str, faircv_npy_path: str
    ) -> pd.DataFrame:
        """Ingest both datasets and concatenate into one unified schema.

        Args:
            kaggle_csv_path: Path to Resume.csv.
            faircv_npy_path: Path to FairCVdb.npy.

        Returns:
            A single DataFrame with UNIFIED_COLUMNS, NaN-filled where a
            field doesn't apply to a given source.
        """
        kaggle_df = self.ingest_kaggle(kaggle_csv_path)
        faircv_df = self.ingest_faircv(faircv_npy_path)

        # Reindex both to the full unified schema so concat aligns cleanly.
        kaggle_full = kaggle_df.reindex(columns=UNIFIED_COLUMNS)
        faircv_full = faircv_df.reindex(columns=UNIFIED_COLUMNS)

        unified = pd.concat([kaggle_full, faircv_full], ignore_index=True)

        n_kaggle = len(kaggle_df)
        n_faircv = len(faircv_df)
        expected_faircv_total = (
            EXPECTED_FAIRCV_TRAIN_ROWS + EXPECTED_FAIRCV_TEST_ROWS
        )

        if n_kaggle != EXPECTED_KAGGLE_ROWS:
            print(
                f"[WARNING] Kaggle row count {n_kaggle} does not match "
                f"expected {EXPECTED_KAGGLE_ROWS}. Dataset version may "
                "have drifted."
            )
        if n_faircv != expected_faircv_total:
            print(
                f"[WARNING] FairCVdb total row count {n_faircv} does not "
                f"match expected {expected_faircv_total}. Dataset version "
                "may have drifted."
            )

        return unified
