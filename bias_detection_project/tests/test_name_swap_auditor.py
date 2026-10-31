import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from name_swap_auditor import NameSwapAuditor, PLACEHOLDER  # noqa: E402
from name_lists import NAME_GROUPS  # noqa: E402


@pytest.fixture
def auditor():
    return NameSwapAuditor(seed=0)


def n_labels(a):
    return len(a.labels)


def test_resume_without_existing_name(auditor):
    text = "experienced java developer with 5 years in backend systems"
    assert auditor.insert_placeholder(text).startswith(PLACEHOLDER)

    cfs = auditor.generate_counterfactuals(text)
    assert set(cfs) == {"female_groupA", "female_groupB",
                        "male_groupA", "male_groupB"}
    for label, out in cfs.items():
        assert PLACEHOLDER not in out
        name, body = out.split("\n\n", 1)
        assert body == text  # everything except the name is identical
        gender, group = label.split("_")
        first, last = name.split(" ", 1)
        assert first in NAME_GROUPS[group]["first_names"][gender]
        assert last in NAME_GROUPS[group]["last_names"]


def test_empty_string(auditor):
    assert auditor.insert_placeholder("") == PLACEHOLDER
    cfs = auditor.generate_counterfactuals("")
    assert len(cfs) == n_labels(auditor)
    assert all(v.strip() and PLACEHOLDER not in v for v in cfs.values())


def test_insert_placeholder_idempotent(auditor):
    once = auditor.insert_placeholder("some text")
    assert auditor.insert_placeholder(once) == once


def test_deterministic(auditor):
    t = "data analyst"
    assert auditor.generate_counterfactuals(t) == auditor.generate_counterfactuals(t)


def test_intersectional_label(auditor):
    assert auditor.build_intersectional_label("groupB", "female") == "female_groupB"
    with pytest.raises(ValueError):
        auditor.build_intersectional_label("groupZ", "female")
    with pytest.raises(ValueError):
        auditor.build_intersectional_label("groupA", "other")


def test_process_dataset_row_multiplication(auditor):
    df = pd.DataFrame({
        "resume_id": ["r1", "r2", "r3"],
        "clean_text": ["python developer", "", "nurse with icu experience"],
        "category_label": ["IT", "HEALTH", "HEALTH"],
        "suitability_score": [0.9, 0.1, 0.5],
        "source_dataset": ["kaggle", "faircvdb", "faircvdb"],
    })
    out = auditor.process_dataset(df)

    k = n_labels(auditor)
    assert len(out) == len(df) * k
    assert out["resume_id"].is_unique
    assert set(out["original_id"]) == {"r1", "r2", "r3"}
    assert (out.groupby("original_id")["intersectional_group"].nunique() == k).all()
    assert out["intersectional_group"].value_counts().nunique() == 1
    for col in ["resume_id", "original_id", "intersectional_group",
                "counterfactual_text", "category_label",
                "suitability_score", "source_dataset", "clean_text"]:
        assert col in out.columns
    # original columns carried through unchanged
    r1 = out[out["original_id"] == "r1"]
    assert (r1["category_label"] == "IT").all()
    assert (r1["clean_text"] == "python developer").all()


def test_process_dataset_without_resume_id(auditor):
    df = pd.DataFrame({"clean_text": ["a", "b"]})
    out = auditor.process_dataset(df)
    assert len(out) == 2 * n_labels(auditor)
    assert set(out["original_id"]) == {"0", "1"}
