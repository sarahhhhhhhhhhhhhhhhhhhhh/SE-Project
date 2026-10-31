"""
Tests for preprocessor.py: clean(), tokenize(), and align_faircv_record().
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), ".."),
)

from preprocessor import Preprocessor  # noqa: E402


@pytest.fixture
def preprocessor() -> Preprocessor:
    """Provides a fresh Preprocessor instance for each test."""
    return Preprocessor()


class TestClean:
    """Tests for Preprocessor.clean()."""

    def test_removes_boilerplate_phrase(self, preprocessor: Preprocessor) -> None:
        text = "Experienced engineer. References available upon request."
        result = preprocessor.clean(text)
        assert "references available upon request" not in result

    def test_lowercases_and_strips_punctuation_but_keeps_hyphens(
        self, preprocessor: Preprocessor
    ) -> None:
        text = "Senior Full-Stack Engineer, Python/Java expert!!"
        result = preprocessor.clean(text)
        assert result == result.lower()
        assert "full-stack" in result
        assert "," not in result
        assert "!" not in result
        assert "/" not in result

    def test_collapses_whitespace(self, preprocessor: Preprocessor) -> None:
        text = "Too    many\n\nspaces\tand tabs"
        result = preprocessor.clean(text)
        assert "  " not in result

    def test_empty_and_none_input(self, preprocessor: Preprocessor) -> None:
        assert preprocessor.clean("") == ""
        assert preprocessor.clean(None) == ""


class TestTokenize:
    """Tests for Preprocessor.tokenize()."""

    def test_empty_string_returns_empty_list(self, preprocessor: Preprocessor) -> None:
        assert preprocessor.tokenize("") == []

    def test_none_input_returns_empty_list(self, preprocessor: Preprocessor) -> None:
        assert preprocessor.tokenize(None) == []  # type: ignore[arg-type]

    def test_removes_stopwords(self, preprocessor: Preprocessor) -> None:
        text = "the candidate is a strong engineer with the skills"
        tokens = preprocessor.tokenize(text)
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "with" not in tokens
        assert "candidate" in tokens
        assert "engineer" in tokens

    def test_preserves_hyphenated_words(self, preprocessor: Preprocessor) -> None:
        text = "full-stack developer"
        tokens = preprocessor.tokenize(text)
        assert "full-stack" in tokens


class TestAlignFaircvRecord:
    """Tests for Preprocessor.align_faircv_record()."""

    def test_full_record_produces_sentence(self, preprocessor: Preprocessor) -> None:
        raw_text = (
            "Occupation: nurse; Suitability: 0.8; Education: 0.7; "
            "Experience: 0.6; Recommendation: 1; Availability: 0.9; "
            "Language proficiency: [0.5, 0.6, 0.7]"
        )
        row = pd.Series({"raw_text": raw_text, "source_dataset": "faircv"})
        result = preprocessor.align_faircv_record(row)

        assert "nurse" in result
        assert result.endswith(".")
        assert "None" not in result

    def test_record_missing_several_fields(self, preprocessor: Preprocessor) -> None:
        raw_text = "Occupation: teacher; Suitability: 0.9"
        row = pd.Series({"raw_text": raw_text, "source_dataset": "faircv"})
        result = preprocessor.align_faircv_record(row)

        assert "teacher" in result
        assert "None" not in result
        assert "none" not in result.lower()
        assert len(result) > 0

    def test_completely_empty_record_returns_empty_string(
        self, preprocessor: Preprocessor
    ) -> None:
        row = pd.Series({"raw_text": "", "source_dataset": "faircv"})
        result = preprocessor.align_faircv_record(row)
        assert result == ""

    def test_malformed_raw_text_does_not_crash(self, preprocessor: Preprocessor) -> None:
        row = pd.Series({"raw_text": "not a valid key value format", "source_dataset": "faircv"})
        result = preprocessor.align_faircv_record(row)
        assert isinstance(result, str)
