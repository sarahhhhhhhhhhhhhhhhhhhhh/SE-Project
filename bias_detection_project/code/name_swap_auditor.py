"""
NameSwapAuditor: counterfactual resume generation via name substitution.

Each resume is duplicated once per intersectional group (gender x name group,
e.g. "female_groupB"), with only the name changed. Any difference in a model's
output between variants of the same resume can then be attributed to the name.
No real personal data is used, and resumes with no native demographic label
(e.g. the Kaggle 24-class set) receive a proxy label by construction.

FairCVdb note
-------------
For FairCVdb rows, this module creates an ADDITIONAL demographic label
(`intersectional_group`) alongside the dataset's own native gender/ethnicity
labels. Both are carried through side by side, so the FairnessAuditor module
can use either and compare them.

Interpretation note: the name-swap label describes the *variant*, not the
original applicant. It measures a model's sensitivity to names; it is not a
claim about the real candidate's demographics.
"""
import hashlib
import random
from typing import Optional

import pandas as pd

from name_lists import NAME_GROUPS, GENDER_GROUPS

PLACEHOLDER = "FULL_NAME_HERE"


class NameSwapAuditor:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.name_groups = list(NAME_GROUPS.keys())
        self.gender_groups = list(GENDER_GROUPS)
        # Order: gender-major, e.g. female_groupA, female_groupB, male_groupA, ...
        self.labels = [
            self.build_intersectional_label(n, g)
            for g in self.gender_groups
            for n in self.name_groups
        ]

    # ------------------------------------------------------------------ #
    def insert_placeholder(self, text: str) -> str:
        """Prepend PLACEHOLDER as an opening header line. Idempotent."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            text = ""
        if PLACEHOLDER in text:
            return text
        return f"{PLACEHOLDER}\n\n{text}" if text else PLACEHOLDER

    # ------------------------------------------------------------------ #
    def build_intersectional_label(self, name_group: str, gender_group: str) -> str:
        """Combine both dimensions, e.g. ("groupB", "female") -> "female_groupB"."""
        if name_group not in NAME_GROUPS:
            raise ValueError(f"Unknown name group: {name_group!r}")
        if gender_group not in GENDER_GROUPS:
            raise ValueError(f"Unknown gender group: {gender_group!r}")
        return f"{gender_group}_{name_group}"

    # ------------------------------------------------------------------ #
    def _pick_name(self, name_group: str, gender_group: str, key: str) -> str:
        """Deterministic name choice per (key, group, gender) so runs reproduce."""
        grp = NAME_GROUPS[name_group]
        rng = random.Random(f"{self.seed}|{key}|{name_group}|{gender_group}")
        first = rng.choice(grp["first_names"][gender_group])
        last = rng.choice(grp["last_names"])
        return f"{first} {last}"

    def generate_counterfactuals(
        self, resume_text: str, key: Optional[str] = None
    ) -> dict[str, str]:
        """
        Return {intersectional_label: resume_text_with_name} for every
        gender x name-group combination. Everything except the name is
        identical across variants.

        `key` seeds name selection (process_dataset passes the resume's id).
        If omitted it is derived from the resume text.
        """
        with_ph = self.insert_placeholder(resume_text)
        if key is None:
            key = hashlib.md5(with_ph.encode("utf-8")).hexdigest()[:12]
        out = {}
        for gender in self.gender_groups:
            for ng in self.name_groups:
                label = self.build_intersectional_label(ng, gender)
                name = self._pick_name(ng, gender, str(key))
                out[label] = with_ph.replace(PLACEHOLDER, name)
        return out

    # ------------------------------------------------------------------ #
    def process_dataset(
        self, df: pd.DataFrame, text_col: str = "clean_text"
    ) -> pd.DataFrame:
        """
        Long-format expansion: every resume x every intersectional group.

        Output columns (first): resume_id, original_id, intersectional_group,
        name_group, gender_group, name_used, counterfactual_text; followed by
        all original columns unchanged. If the input has a `resume_id` column
        it becomes `original_id`, and `resume_id` is replaced by a unique
        per-variant id ("<original_id>__<label>"). Otherwise the row position
        is used as original_id.

        FairCVdb rows keep their native gender/ethnicity columns, so both
        label sets are available downstream (see module docstring).
        """
        base = df.reset_index(drop=True).copy()
        if "resume_id" in base.columns:
            original_ids = base["resume_id"].astype(str)
        else:
            original_ids = pd.Series(base.index.astype(str), index=base.index)

        cf_list = [
            self.generate_counterfactuals(t, key=oid)
            for t, oid in zip(base[text_col].tolist(), original_ids.tolist())
        ]

        pieces = []
        for gender in self.gender_groups:
            for ng in self.name_groups:
                label = self.build_intersectional_label(ng, gender)
                piece = base.copy()
                piece["original_id"] = original_ids
                piece["resume_id"] = original_ids + "__" + label
                piece["intersectional_group"] = label
                piece["name_group"] = ng
                piece["gender_group"] = gender
                piece["counterfactual_text"] = [cf[label] for cf in cf_list]
                piece["name_used"] = [
                    cf[label].split("\n", 1)[0] if cf[label] else ""
                    for cf in cf_list
                ]
                pieces.append(piece)

        long_df = pd.concat(pieces).sort_index(kind="stable").reset_index(drop=True)

        front = ["resume_id", "original_id", "intersectional_group", "name_group",
                 "gender_group", "name_used", "counterfactual_text"]
        rest = [c for c in long_df.columns if c not in front]
        return long_df[front + rest]
