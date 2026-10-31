
"""Run in Colab after run_pipeline(). Prints PASS/FAIL and unexpected observations."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

TARGETS = (
    "score_blind",
    "score_biased_gender",
    "score_biased_ethnicity"
)

NAMES = (
    "baseline",
    "bertidraaf_lite",
    "improved"
)

PAIRS = (
    ("baseline", "bertidraaf_lite"),
    ("bertidraaf_lite", "improved"),
    ("baseline", "improved")
)


def validate_pipeline(
    project_root="/content/drive/MyDrive/bias_detection_project/",
    pipeline_result=None
):
    root = Path(project_root)
    sys.path.insert(0, str(root / "code"))

    from comparison import paired_delta, verdict, placement
    from bertidraaf_lite_scorer import (
        build_matched_pair,
        signature_drift,
        assert_matched
    )
    from embedding_engine import EmbeddingEngine
    from data_prep import load_prepared

    proc = root / "data/processed"
    reports = root / "reports"

    checks = {}
    unexpected = []

    def check(label, condition, details=""):
        checks[label] = bool(condition)
        print(
            f"{'PASS' if condition else 'FAIL'} | {label}"
            + (f" | {details}" if details else "")
        )

    def read(name):
        return pd.read_csv(
            name,
            low_memory=False,
            dtype={"resume_id": str, "original_id": str}
        )

    try:
        ingest = read(proc / "ingested_resumes.csv")
        cf = read(proc / "counterfactual_resumes.csv")

        prepared, thr = load_prepared(
            str(root) + "/",
            apply_scrub=True
        )

        models = {
            n: read(proc / (n + "_predictions.csv"))
            for n in NAMES
        }

        comp = read(reports / "fairness_comparison.csv")
        paired = read(reports / "fairness_paired_differences.csv")

    except Exception as e:
        check("required inputs readable", False, repr(e))
        print(
            "OVERALL FAIL | failed: required inputs readable "
            "| UNEXPECTED: none"
        )
        return {
            "passed": False,
            "checks": checks,
            "unexpected": unexpected
        }

    # ============================================================
    # CHECK 1: ROW COUNTS, SPLITS AND PREDICTION ALIGNMENT
    # ============================================================

    counts = (
        ingest.source_dataset.astype(str)
        .str.lower()
        .str.contains("faircv")
        .value_counts()
    )

    nfair = int(counts.get(True, 0))
    nkag = int(counts.get(False, 0))

    print(
        f"INGEST profiles: total={len(ingest)}, "
        f"FairCV={nfair}, Kaggle={nkag}; "
        f"counterfactual rows={len(cf)}"
    )

    def scale_ok(actual, expected):
        return (
            actual > 0
            and abs(actual - expected) / expected < 0.01
        )

    check(
        "1 ingestion size",
        scale_ok(len(ingest), 26484)
        and nfair == 24000
        and scale_ok(nkag, 2484),
        f"expected ~26481-26484; actual {len(ingest)}"
    )

    check(
        "1 expansion size",
        scale_ok(len(cf), 4 * len(ingest))
        and cf.original_id.nunique() == len(ingest),
        f"actual {len(cf)}, expected {4 * len(ingest)}"
    )

    fc = prepared[
        prepared.source_dataset.astype(str)
        .str.contains("faircv", case=False)
    ]

    splits = (
        fc.groupby("split")
        .original_id.nunique()
        .to_dict()
    )

    check(
        "1 native FairCV profile splits",
        splits == {
            "train": 17280,
            "val": 1920,
            "test": 4800
        },
        str(splits)
    )

    def key(d):
        return set(
            map(
                tuple,
                d[
                    [
                        "dataset",
                        "target",
                        "original_id",
                        "resume_id"
                    ]
                ]
                .fillna("")
                .astype(str)
                .values
            )
        )

    expected = key(models["baseline"])

    ref_counts = (
        models["baseline"]
        .groupby(["dataset", "target"], dropna=False)
        .size()
        .to_dict()
    )

    match = all(
        key(d) == expected
        and d.groupby(
            ["dataset", "target"],
            dropna=False
        ).size().to_dict() == ref_counts
        and len(d) == len(models["baseline"])
        and not d[
            ["dataset", "target", "resume_id"]
        ].duplicated().any()
        for d in models.values()
    )

    check(
        "1 identical prediction keys and target row counts",
        match,
        str({n: len(d) for n, d in models.items()})
    )

    # ============================================================
    # CHECK 2-3: NaNs AND FAIRNESS RANGE
    # ============================================================

    for n, d in models.items():
        check(
            f"2 {n}: no prediction/label NaNs",
            d[
                ["predicted_label", "true_label"]
            ].notna().all().all()
        )

        check(
            f"3 {n}: all prediction files nonempty",
            len(d) > 0
        )

    check(
        "2 fairness_measure no NaNs",
        comp.fairness_measure.notna().all()
    )

    check(
        "3 fairness_measure in [0,1]",
        pd.to_numeric(
            comp.fairness_measure,
            errors="coerce"
        ).between(0, 1).all()
    )

    # ============================================================
    # CHECK 4: CONFIDENCE INTERVALS
    # ============================================================

    def intervals(df, triplets):
        errs = []

        for point, lo, hi in triplets:
            a = df[
                [point, lo, hi]
            ].apply(pd.to_numeric, errors="coerce")

            bad = (
                a.isna().any(axis=1)
                | (a[lo] > a[point] + 1e-12)
                | (a[point] > a[hi] + 1e-12)
            )

            errs.append(
                (point, int(bad.sum()))
            )

        return (
            all(n == 0 for _, n in errs),
            errs
        )

    ok, details = intervals(
        comp,
        [
            (
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high"
            ),
            (
                "fairness_measure",
                "fairness_ci_low",
                "fairness_ci_high"
            )
        ]
    )

    check(
        "4 point confidence intervals enclose estimates",
        ok,
        str(details)
    )

    ok, details = intervals(
        paired,
        [
            (
                "fairness_diff",
                "fairness_diff_low",
                "fairness_diff_high"
            )
        ]
    )

    check(
        "4 paired confidence intervals enclose differences",
        ok,
        str(details)
    )

    # ============================================================
    # CHECK 5: ACCURACY AND PAIRED COMPARISONS
    # ============================================================

    def norm(s):
        return (
            s.astype(str)
            .str.replace(r"\.0$", "", regex=True)
        )

    def acc(d):
        if not len(d):
            return float("nan")

        return float(
            (
                norm(d.true_label)
                == norm(d.predicted_label)
            ).mean()
        )

    models_accuracy = {
        n: d.assign(
            target=np.where(
                d.dataset == "kaggle",
                "category_label",
                d.target
            )
        )
        for n, d in models.items()
    }

    tasks = [
        ("kaggle", "category_label"),
        ("faircv", "score_blind"),
        ("faircv", "score_biased_gender"),
        ("faircv", "score_biased_ethnicity"),
        ("faircv", None)
    ]

    for ds, tgt in tasks:

        subs = {
            n: (
                d[d.dataset == ds]
                if tgt is None
                else d[
                    (d.dataset == ds)
                    & (d.target == tgt)
                ]
            )
            for n, d in models_accuracy.items()
        }

        a = {
            n: acc(subs[n])
            for n in NAMES
        }

        print(
            f"ACCURACY {ds}/{tgt or 'pooled'}: "
            + " | ".join(
                f"{n}={v:.4f}"
                for n, v in a.items()
            )
        )

        check(
            f"5 accuracy calculable {ds}/{tgt or 'pooled'}",
            all(np.isfinite(v) for v in a.values())
        )

        for lhs, rhs in PAIRS:
            try:
                d = paired_delta(
                    subs[lhs],
                    subs[rhs],
                    key=("resume_id", "target")
                )

                pl = (
                    placement(d, 0.8517)
                    if ds == "faircv"
                    else "paper benchmark not supplied"
                )

                print(
                    f"  {lhs} vs {rhs}: "
                    f"paired_delta={d['delta']:+.4f}; "
                    f"{verdict(d)}; "
                    f"placement={pl}"
                )

                check(
                    f"5 paired alignment "
                    f"{ds}/{tgt or 'pooled'} "
                    f"{lhs}->{rhs}",
                    d["n_rows"]
                    == len(subs[lhs])
                    == len(subs[rhs])
                )

            except Exception as e:
                check(
                    f"5 paired alignment "
                    f"{ds}/{tgt or 'pooled'} "
                    f"{lhs}->{rhs}",
                    False,
                    repr(e)
                )

        if (
            a["baseline"] > a["bertidraaf_lite"] + 0.005
            or a["baseline"] > a["improved"] + 0.005
        ):
            unexpected.append(
                f"{ds}/{tgt or 'pooled'}: "
                "UNEXPECTED: investigate before writing up "
                "(baseline wins)"
            )

        if (
            a["improved"]
            < a["bertidraaf_lite"] - 0.005
        ):
            unexpected.append(
                f"{ds}/{tgt or 'pooled'}: "
                "UNEXPECTED: IBS lowers accuracy vs "
                "the IDF control, investigate before writing up"
            )

        for lhs, rhs in PAIRS:
            if abs(a[rhs] - a[lhs]) < 0.005:
                print(
                    f"  PARITY (<0.5 points): "
                    f"{lhs} vs {rhs}"
                )

        for n, v in a.items():
            if v >= 0.99:
                unexpected.append(
                    f"{ds}/{tgt or 'pooled'} {n}: "
                    "WARNING near-perfect accuracy >= 0.99; "
                    "investigate leakage"
                )

    print(
        "Kaggle improved predictions are a locked earlier run; "
        "the bertidraaf_lite Kaggle predictions are a single "
        "fresh run (run-to-run spread ~0.5 points), "
        "so treat Kaggle gaps under ~0.5 points as parity."
    )

    # ============================================================
    # CHECK 6: OUTPUT FILES AND ALGORITHM NAMES
    # ============================================================

    required = [
        reports / "fairness_report.html",
        reports / "fairness_comparison.csv",
        reports / "fairness_paired_differences.csv",
        proc / "bertidraaf_lite_predictions.csv"
    ]

    check(
        "6 required reports and control CSV exist and are nonempty",
        all(
            p.is_file() and p.stat().st_size > 0
            for p in required
        )
    )

    check(
        "6 algorithms exactly match ladder",
        set(comp.algorithm) == set(NAMES),
        str(sorted(set(comp.algorithm)))
    )

    # ============================================================
    # CHECK 7: LEAKAGE GUARDS
    # ============================================================

    tr = set(
        prepared.loc[
            prepared.split == "train",
            "original_id"
        ]
    )

    te = set(
        prepared.loc[
            prepared.split == "test",
            "original_id"
        ]
    )

    check(
        "7 train/test original_id disjoint",
        not bool(tr & te)
    )

    thrfile = proc / "faircv_thresholds.json"

    stored = (
        json.loads(thrfile.read_text())
        if thrfile.is_file()
        else {}
    )

    fc_native_train = (
        fc[fc.split == "train"]
        .drop_duplicates("original_id")
    )

    recomputed = {
        t: float(
            pd.to_numeric(
                fc_native_train[t]
            ).median()
        )
        for t in TARGETS
    }

    check(
        "7 thresholds exist and agree with native training labels",
        all(
            t in stored
            and abs(stored[t] - recomputed[t]) < 1e-10
            for t in TARGETS
        ),
        f"saved={stored} recomputed={recomputed}"
    )

    lock = (
        proc / "improved_predictions_kaggle_locked.csv"
    )

    if lock.exists():
        locked = read(lock)

        actual = (
            models["improved"][
                models["improved"].dataset == "kaggle"
            ]
            .reset_index(drop=True)
        )

        check(
            "7 improved Kaggle predictions exactly locked",
            set(locked.columns) == set(actual.columns)
            and actual.equals(
                locked[
                    actual.columns
                ].reset_index(drop=True)
            )
        )

    else:
        check(
            "7 improved Kaggle predictions exactly locked",
            False,
            "locked file absent"
        )

    # ============================================================
    # CHECK 8: MATCHED PAIR CONFIGURATION
    # ============================================================

    try:
        common = (
            pipeline_result or {}
        ).get(
            "common_kwargs",
            dict(
                epochs=6,
                fine_tune=True,
                faircv_targets=list(TARGETS),
                faircv_thresholds=stored,
                time_budget_minutes=20,
                seed=42
            )
        )

        imp, control = build_matched_pair(
            common,
            "parity",
            engine_factory=lambda: EmbeddingEngine()
        )

        drift = signature_drift()

        assert_matched(imp, control)

        check(
            "8 signature_drift empty and assert_matched passes",
            not bool(drift),
            str(drift)
        )

        check(
            "8 control weight_mode idf",
            control.weight_mode == "idf"
        )

        check(
            "8 default OOV is automatic token-median IDF",
            control.oov_weight is None
            and not control._oov_explicit,
            "fitted IDF: N / token-median weight "
            "must be confirmed from the training log"
        )

        del imp, control

    except Exception as e:
        check(
            "8 matched configuration guard",
            False,
            repr(e)
        )

    # Verify fitted IDF metadata rather than guessing
    # its OOV value from an unfitted scorer.

    meta_path = (
        proc / "bertidraaf_lite_fit_metadata.json"
    )

    meta = (
        json.loads(meta_path.read_text())
        if meta_path.is_file()
        else {}
    )

    evidence_ok = (
        meta.get("weight_mode") == "idf"
        and meta.get("n_docs", 0) > 0
        and meta.get("oov_explicit") is False
        and abs(
            meta.get("oov_weight", float("nan"))
            - meta.get(
                "token_median_idf",
                float("inf")
            )
        ) < 1e-8
    )

    check(
        "8 fitted OOV equals training-token median IDF",
        evidence_ok,
        str(meta)
    )

    # ============================================================
    # FINAL VALIDATION SUMMARY
    # ============================================================

    bad = [
        k for k, v in checks.items()
        if not v
    ]

    print(
        "OVERALL "
        + ("PASS" if not bad else "FAIL")
        + " | failed: "
        + (", ".join(bad) if bad else "none")
        + " | UNEXPECTED: "
        + (
            "; ".join(unexpected)
            if unexpected
            else "none"
        )
    )

    return {
        "passed": not bad,
        "checks": checks,
        "unexpected": unexpected
    }
