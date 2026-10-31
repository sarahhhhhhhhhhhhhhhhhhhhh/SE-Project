"""
preprocessor.py

Cleans and tokenizes free-text resume data, and converts FairCVdb's
structured records into free-text biography sentences so both data
sources share one text representation before scoring.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd


# Small built-in stopword list (no NLTK dependency).
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "as", "by", "from", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "i", "his",
    "her", "their", "our", "your", "my", "will", "would", "can", "could",
    "should", "has", "have", "had", "do", "does", "did", "not", "no",
    "into", "about", "than", "which", "who", "whom", "such",
}

# Common resume boilerplate phrases to strip during cleaning.
BOILERPLATE_PHRASES = [
    "references available upon request",
    "references furnished upon request",
    "curriculum vitae",
    "to whom it may concern",
    "objective statement",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class Preprocessor:
    """Cleans, tokenizes, and normalizes resume/biography text."""

    def clean(self, text: str) -> str:
        """Lowercase, strip punctuation (except intra-word hyphens),
        collapse whitespace, and remove common resume boilerplate.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text. Returns an empty string for None/NaN/empty input.
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        cleaned = str(text).lower()

        for phrase in BOILERPLATE_PHRASES:
            cleaned = cleaned.replace(phrase, " ")

        # Keep letters, digits, whitespace, and intra-word hyphens.
        cleaned = re.sub(r"(?<=\w)-(?=\w)", "\x00HYPHEN\x00", cleaned)
        cleaned = re.sub(r"[^\w\s\x00]", " ", cleaned)
        cleaned = cleaned.replace("\x00HYPHEN\x00", "-")

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text on word boundaries and remove stopwords.

        Args:
            text: Input text (ideally already cleaned).

        Returns:
            List of lowercase tokens with stopwords removed. Empty list
            for empty/None input.
        """
        if not text:
            return []

        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def _parse_faircv_raw_text(self, raw_text: str) -> Dict[str, str]:
        """Parse the 'Key: value; Key: value; ...' string produced by
        ResumeIngestor for FairCVdb rows into a field dict.

        Args:
            raw_text: The semicolon-delimited structured record string.

        Returns:
            Dict mapping field name -> raw string value. Missing/blank
            fields are simply absent from the dict.
        """
        fields: Dict[str, str] = {}
        if not raw_text or not isinstance(raw_text, str):
            return fields

        for chunk in raw_text.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            key, _, value = chunk.partition(":")
            key = key.strip()
            value = value.strip()
            if key and value:
                fields[key] = value

        return fields

    def _describe_score(self, value_str: str, low: str, mid: str, high: str) -> Optional[str]:
        """Convert a 0-1 continuous score string into a qualitative phrase.

        Args:
            value_str: The score as a string (e.g. "0.82").
            low: Phrase to use for scores < 0.33.
            mid: Phrase to use for scores in [0.33, 0.66].
            high: Phrase to use for scores > 0.66.

        Returns:
            The qualitative phrase, or None if value_str isn't parseable.
        """
        try:
            value = float(value_str)
        except (ValueError, TypeError):
            return None

        if value < 0.33:
            return low
        if value <= 0.66:
            return mid
        return high

    def align_faircv_record(self, row: pd.Series) -> str:
        """Convert a FairCVdb structured record into a free-text biography
        sentence, matching the free-text shape of the Kaggle resumes.

        Args:
            row: A row (pandas Series) from the ingested DataFrame, whose
                `raw_text` field holds the semicolon-delimited structured
                record produced by ResumeIngestor for faircv rows.

        Returns:
            A short biography sentence. Missing fields are skipped
            gracefully rather than inserting "None" placeholders. Returns
            an empty string if no usable fields are found.
        """
        raw_text = row.get("raw_text", "") if hasattr(row, "get") else row["raw_text"]
        fields = self._parse_faircv_raw_text(raw_text)

        # No parseable fields at all -> nothing to describe.
        if not fields:
            return ""

        clauses: List[str] = []

        occupation = fields.get("Occupation")
        if occupation and occupation.lower() != "unknown":
            clauses.append(f"Candidate for a {occupation} position")
        else:
            clauses.append("Candidate")

        experience_desc = self._describe_score(
            fields.get("Experience", ""),
            low="limited professional experience",
            mid="a moderate amount of professional experience",
            high="extensive professional experience",
        )
        if experience_desc:
            clauses.append(f"with {experience_desc}")

        education_desc = self._describe_score(
            fields.get("Education", ""),
            low="a basic level of formal education",
            mid="an undergraduate level of education",
            high="a graduate level of education",
        )
        if education_desc:
            clauses.append(f"and {education_desc}")

        recommendation = fields.get("Recommendation")
        if recommendation is not None:
            rec_lower = recommendation.strip().lower()
            if rec_lower in {"1", "1.0", "true", "yes"}:
                clauses.append("holding a positive recommendation letter")
            elif rec_lower in {"0", "0.0", "false", "no"}:
                clauses.append("without a recommendation letter on file")

        availability_desc = self._describe_score(
            fields.get("Availability", ""),
            low="limited current availability",
            mid="moderate current availability",
            high="high current availability",
        )
        if availability_desc:
            clauses.append(f"and {availability_desc}")

        suitability_desc = self._describe_score(
            fields.get("Suitability", ""),
            low="a low overall suitability score",
            mid="a moderate overall suitability score",
            high="a strong overall suitability score",
        )
        if suitability_desc:
            clauses.append(f"reflecting {suitability_desc}")

        sentence = " ".join(clauses).strip()
        sentence = re.sub(r"\s+", " ", sentence)
        if not sentence.endswith("."):
            sentence += "."
        return sentence[0].upper() + sentence[1:] if sentence else sentence

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning, FairCVdb-to-text alignment, and tokenization to
        an ingested DataFrame.

        Args:
            df: DataFrame produced by ResumeIngestor.ingest_all(), containing
                at least `raw_text` and `source_dataset` columns.

        Returns:
            A copy of df with two new columns: `clean_text` and `tokens`.
        """
        result = df.copy()

        is_faircv = result["source_dataset"] == "faircv"

        # For faircv rows, first convert the structured record to a
        # free-text biography sentence; for all other rows, use raw_text
        # as-is. Then clean everything uniformly.
        pre_clean_text = result["raw_text"].copy()
        if is_faircv.any():
            pre_clean_text.loc[is_faircv] = result.loc[is_faircv].apply(
                self.align_faircv_record, axis=1
            )

        result["clean_text"] = pre_clean_text.apply(self.clean)
        result["tokens"] = result["clean_text"].apply(self.tokenize)

        return result
