"""
report_dashboard.py - ReportDashboard
=====================================

Turns the numerical output of the FairnessAuditor into a plain-language report
that a non-technical reader can follow.

Inputs (both are produced by FairnessAuditor.compare / .pairwise_comparison)
    comparison_df : algorithm, dataset, target, group_attribute, metric_basis,
                    accuracy, accuracy_ci_low/high, fairness_measure,
                    fairness_ci_low/high, n_rows, n_profiles
    paired_df     : dataset, target, group_attribute, metric_basis, model_from,
                    model_to, fairness_from, fairness_to, fairness_diff,
                    fairness_diff_low/high, significant_paired,
                    significant_non_overlap, direction        (optional)

Models, in ladder order
    baseline         - simple word-counting model (TF-IDF)
    bertidraaf_lite  - AI language model, plain IDF weights (the control)
    improved         - AI language model, bias-aware IBS weights

Robustness
    * paired_df missing/invalid  -> "clearly" falls back to non-overlapping ranges
    * only two algorithms present -> shows what exists, unavailable pairs are skipped
    * blank Kaggle `target`       -> treated as 'category_label'
    * nothing usable              -> a friendly message instead of a traceback

Usage
    from report_dashboard import ReportDashboard
    dash = ReportDashboard()
    dash.render_dashboard(comparison_df, paired_df)
    dash.export_html_report(comparison_df, ".../fairness_report.html", paired_df)
"""
import base64
import datetime as _dt
import html as _html
import io
import math
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

import pandas as pd

# --------------------------------------------------------------------------- constants
ALGORITHM_ORDER = ("baseline", "bertidraaf_lite", "improved")
KAGGLE_TARGET = "category_label"
FAIRCV_TARGETS = ("score_blind", "score_biased_gender", "score_biased_ethnicity")
BIASED_TARGETS = ("score_biased_gender", "score_biased_ethnicity")
GROUP_ORDER = ("intersectional_group", "gender_label", "ethnicity_label")
DATASET_ORDER = ("kaggle", "faircv")

KEY_PAIR = ("bertidraaf_lite", "improved")           # effect of bias-aware weighting alone
PAIR_ORDER = [KEY_PAIR, ("baseline", "bertidraaf_lite"), ("baseline", "improved")]

SMALL_DIFF = 0.02    # fairness gaps below this get the "run-to-run variation" caution

_FRIENDLY_TARGET = {
    "category_label": "job category",
    "score_blind": "scores that ignore gender and ethnicity (unbiased)",
    "score_biased_gender": "scores deliberately biased by gender",
    "score_biased_ethnicity": "scores deliberately biased by ethnicity",
}
_FRIENDLY_ALGORITHM = {
    "baseline": "the simple word-counting model",
    "bertidraaf_lite": "the AI language model without bias-aware weighting",
    "improved": "the AI language model with bias-aware weighting",
}
_SHORT_ALGORITHM = {
    "baseline": "Simple",
    "bertidraaf_lite": "AI model",
    "improved": "AI model + bias-aware",
}
_FRIENDLY_GROUP = {
    "intersectional_group": "combined gender-and-ethnicity groups",
    "gender_label": "gender groups",
    "ethnicity_label": "ethnic groups",
}
_GROUP_MENU = {
    "intersectional_group": "Gender and ethnicity combined",
    "gender_label": "Gender",
    "ethnicity_label": "Ethnicity",
}
_DATASET_LABEL = {"kaggle": "Kaggle", "faircv": "FairCVdb"}
_ALGO_COLOR = {"baseline": "#8c929c", "bertidraaf_lite": "#4c78a8", "improved": "#2e9e5b"}

_COMPARE_MENU = [
    ("AI model without vs with bias-aware weighting", ("bertidraaf_lite", "improved")),
    ("Simple model vs AI model without bias-aware weighting", ("baseline", "bertidraaf_lite")),
    ("Simple model vs AI model with bias-aware weighting", ("baseline", "improved")),
]

_NUMERIC_COLS = ["accuracy", "accuracy_ci_low", "accuracy_ci_high", "fairness_measure",
                 "fairness_ci_low", "fairness_ci_high", "n_rows", "n_profiles"]
_REQUIRED_COMPARISON = ["algorithm", "dataset", "group_attribute", "fairness_measure"]
_REQUIRED_PAIRED = ["dataset", "group_attribute", "model_from", "model_to",
                    "fairness_diff_low", "fairness_diff_high"]


# --------------------------------------------------------------------------- small helpers
def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _get(row, key, default=None):
    """row may be a dict or a pandas Series; missing keys give `default`."""
    if row is None:
        return default
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _esc(text) -> str:
    return _html.escape(str(text), quote=True)


def _sorted_by(values, preferred) -> list:
    values = list(dict.fromkeys(values))
    return [v for v in preferred if v in values] + sorted(v for v in values if v not in preferred)


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(text)).strip("-").lower()


# =========================================================================== the class
class ReportDashboard:
    """Plain-language dashboard and static HTML report for the fairness results."""

    def __init__(self):
        self._index: Dict[tuple, Dict[str, dict]] = {}   # (dataset, target, group) -> {algo: row}
        self._lookup: Dict[tuple, dict] = {}              # (ds, tgt, grp, from, to) -> paired row
        self._has_paired = False

    # ------------------------------------------------------------------ 1-2. wording
    @staticmethod
    def friendly_target(target) -> str:
        if _isnan(target) or target is None:
            return "this target"
        return _FRIENDLY_TARGET.get(str(target), str(target).replace("_", " "))

    @staticmethod
    def friendly_algorithm(algorithm) -> str:
        return _FRIENDLY_ALGORITHM.get(str(algorithm), str(algorithm))

    @staticmethod
    def short_algorithm(algorithm) -> str:
        return _SHORT_ALGORITHM.get(str(algorithm), str(algorithm))

    @staticmethod
    def friendly_group(group_attribute) -> str:
        return _FRIENDLY_GROUP.get(str(group_attribute), str(group_attribute).replace("_", " "))

    # ------------------------------------------------------------------ 3. summary text
    @staticmethod
    def _verdict(row_from, row_to, paired_row) -> str:
        """'higher' | 'equal' | 'lower' | 'unknown' (unknown is reported like 'equal')."""
        if paired_row is not None:
            lo = _num(_get(paired_row, "fairness_diff_low"))
            hi = _num(_get(paired_row, "fairness_diff_high"))
            if not (math.isnan(lo) or math.isnan(hi)):
                if lo > 0:
                    return "higher"
                if hi < 0:
                    return "lower"
                return "equal"
        a_lo, a_hi = _num(_get(row_from, "fairness_ci_low")), _num(_get(row_from, "fairness_ci_high"))
        b_lo, b_hi = _num(_get(row_to, "fairness_ci_low")), _num(_get(row_to, "fairness_ci_high"))
        if any(math.isnan(v) for v in (a_lo, a_hi, b_lo, b_hi)):
            return "unknown"
        if b_lo > a_hi:
            return "higher"
        if b_hi < a_lo:
            return "lower"
        return "equal"

    @staticmethod
    def _fmt_scores(fa: float, fb: float) -> Tuple[str, str]:
        for decimals in (2, 3, 4):           # add decimals so a tiny gap isn't shown as "0.91 vs 0.91"
            sa, sb = f"{fa:.{decimals}f}", f"{fb:.{decimals}f}"
            if sa != sb or fa == fb:
                break
        return sa, sb

    def plain_language_summary(self, row_from, row_to, paired_row=None) -> str:
        """One jargon-free paragraph comparing any two models.

        row_from / row_to : rows of the comparison table (dict or Series)
        paired_row        : matching row of the paired table, if available
        """
        algo_a, algo_b = _get(row_from, "algorithm"), _get(row_to, "algorithm")
        fa, fb = _num(_get(row_from, "fairness_measure")), _num(_get(row_to, "fairness_measure"))
        name_a, name_b = self.friendly_algorithm(algo_a), self.friendly_algorithm(algo_b)
        target = _get(row_to, "target", _get(row_from, "target"))
        group = self.friendly_group(_get(row_to, "group_attribute", _get(row_from, "group_attribute")))

        if math.isnan(fa) or math.isnan(fb):
            return (f"{_cap(name_a)} and {name_b} cannot be compared here because a fairness "
                    f"score is missing for at least one of them.")

        verdict = self._verdict(row_from, row_to, paired_row)
        s_a, s_b = self._fmt_scores(fa, fb)

        parts = []
        if verdict == "higher":
            parts.append(f"When predicting {self.friendly_target(target)} and comparing {group}, "
                         f"{name_b} is clearly fairer than {name_a}; it treats the different "
                         f"groups more evenly.")
        elif verdict == "lower":
            parts.append(f"When predicting {self.friendly_target(target)} and comparing {group}, "
                         f"{name_b} is clearly less fair than {name_a}; it treats the different "
                         f"groups less evenly.")
        else:
            parts.append(f"When predicting {self.friendly_target(target)} and comparing {group}, "
                         f"{name_a} and {name_b} are about equally fair; any difference in how "
                         f"evenly they treat the different groups is too small to be sure it is real.")
            if verdict == "unknown":
                parts.append("We could not work out a reliable range for this comparison, so "
                             "please treat any difference with care.")

        parts.append(f"On how evenly the model treats different groups, {name_a} gets {s_a} out "
                     f"of 1.0 (higher is fairer) and {name_b} gets {s_b} out of 1.0 (higher is fairer).")

        if str(target) in BIASED_TARGETS:
            parts.append("Keep in mind that the models were taught from deliberately biased "
                         "labels, so matching those labels is not a sign of fairness.")

        pair = frozenset((str(algo_a), str(algo_b)))
        if pair == frozenset(KEY_PAIR):
            parts.append("This comparison shows the effect of the bias-aware weighting alone, "
                         "because both use the same AI language model.")
        elif pair == frozenset(("baseline", "bertidraaf_lite")):
            parts.append("This comparison shows the effect of switching to the AI language model alone.")
        elif pair == frozenset(("baseline", "improved")):
            parts.append("This comparison shows the combined effect of both changes: the AI "
                         "language model and the bias-aware weighting.")

        if verdict in ("equal", "unknown") or abs(fb - fa) < SMALL_DIFF:
            parts.append("Caution: this gap is small enough that it could be normal run-to-run variation.")

        return " ".join(parts)

    # ------------------------------------------------------------------ data preparation
    def _prepare(self, comparison_df, paired_df) -> bool:
        """Clean both tables and build lookups. Returns False if nothing usable."""
        self._index, self._lookup, self._has_paired = {}, {}, False
        if comparison_df is None or not hasattr(comparison_df, "columns") or len(comparison_df) == 0:
            return False
        comp = comparison_df.copy()
        if "target" not in comp.columns:
            comp["target"] = ""
        if any(c not in comp.columns for c in _REQUIRED_COMPARISON):
            return False
        for c in _NUMERIC_COLS:
            comp[c] = pd.to_numeric(comp[c], errors="coerce") if c in comp.columns else float("nan")
        comp = comp.dropna(subset=["algorithm", "dataset", "group_attribute"])
        comp["algorithm"] = comp["algorithm"].astype(str).str.strip()
        comp["dataset"] = comp["dataset"].astype(str).str.strip().str.lower()
        comp["target"] = self._clean_target(comp["target"], comp["dataset"])
        for rec in comp.to_dict("records"):
            key = (rec["dataset"], rec["target"], str(rec["group_attribute"]))
            self._index.setdefault(key, {}).setdefault(rec["algorithm"], rec)
        if not self._index:
            return False

        if (paired_df is not None and hasattr(paired_df, "columns") and len(paired_df)
                and all(c in paired_df.columns or c == "target" for c in _REQUIRED_PAIRED)):
            p = paired_df.copy()
            if "target" not in p.columns:
                p["target"] = ""
            for c in ("fairness_diff_low", "fairness_diff_high"):
                p[c] = pd.to_numeric(p[c], errors="coerce")
            p["dataset"] = p["dataset"].astype(str).str.strip().str.lower()
            p["target"] = self._clean_target(p["target"], p["dataset"])
            for rec in p.to_dict("records"):
                key = (rec["dataset"], rec["target"], str(rec["group_attribute"]),
                       str(rec["model_from"]), str(rec["model_to"]))
                self._lookup.setdefault(key, rec)
            self._has_paired = bool(self._lookup)
        return True

    @staticmethod
    def _clean_target(target: pd.Series, dataset: pd.Series) -> pd.Series:
        t = target.where(target.notna(), "").astype(str).str.strip()
        t = t.where(t.str.lower() != "nan", "")
        return t.where(~((dataset == "kaggle") & (t == "")), KAGGLE_TARGET)

    def _datasets(self) -> list:
        return _sorted_by({k[0] for k in self._index}, DATASET_ORDER)

    def _targets(self, ds) -> list:
        return _sorted_by({k[1] for k in self._index if k[0] == ds}, (KAGGLE_TARGET,) + FAIRCV_TARGETS)

    def _groups(self, ds, tgt) -> list:
        return _sorted_by({k[2] for k in self._index if k[0] == ds and k[1] == tgt}, GROUP_ORDER)

    def _pair_summary(self, rows: Dict[str, dict], key: tuple, pair: Tuple[str, str]) -> Optional[str]:
        """Summary for `pair`, or None if either model has no results for this combination."""
        a, b = pair
        if a not in rows or b not in rows:
            return None
        return self.plain_language_summary(rows[a], rows[b], self._lookup.get((*key, a, b)))

    @staticmethod
    def _missing_pair_message(rows: Dict[str, dict], pair: Tuple[str, str]) -> str:
        missing = [ReportDashboard.friendly_algorithm(x) for x in pair if x not in rows]
        return ("This comparison is not available here because the results for "
                + " and for ".join(missing) + " are not in the results file.")

    # ------------------------------------------------------------------ charts
    def _build_chart_png(self, rows: Dict[str, dict], dataset: str, target: str,
                         highlight: Optional[Tuple[str, str]] = None) -> bytes:
        """Fairness chart (all models present) next to a small accuracy chart, as PNG bytes."""
        from matplotlib.figure import Figure

        algos = _sorted_by(rows, ALGORITHM_ORDER)
        fig = Figure(figsize=(10, 4.2), dpi=110)
        ax_f, ax_a = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})

        def draw(ax, value_col, lo_col, hi_col, title, ylabel):
            vals = [_num(rows[a].get(value_col)) for a in algos]
            heights = [0.0 if math.isnan(v) else v for v in vals]
            lows, highs = [], []
            for a, v in zip(algos, vals):
                lo, hi = _num(rows[a].get(lo_col)), _num(rows[a].get(hi_col))
                if math.isnan(v) or math.isnan(lo) or math.isnan(hi):
                    lows.append(0.0), highs.append(0.0)
                else:
                    lows.append(max(v - lo, 0.0)), highs.append(max(hi - v, 0.0))
            xs = list(range(len(algos)))
            bars = ax.bar(xs, heights, width=0.62,
                          color=[_ALGO_COLOR.get(a, "#999999") for a in algos],
                          edgecolor=["black" if (highlight and a in highlight) else "none" for a in algos],
                          linewidth=2.0, zorder=2)
            ax.errorbar(xs, heights, yerr=[lows, highs], fmt="none", ecolor="black",
                        capsize=5, lw=1.3, zorder=3)
            labels = []
            for a, v in zip(algos, vals):
                shown = "n/a" if math.isnan(v) else f"{v:.2f}"
                labels.append(textwrap.fill(self.short_algorithm(a), 12) + f"\n{shown}")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
            ax.grid(axis="y", alpha=0.3, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            return bars

        draw(ax_f, "fairness_measure", "fairness_ci_low", "fairness_ci_high",
             "How evenly the model treats different groups",
             "Fairness score (higher is fairer)")
        if dataset == "faircv":
            draw(ax_a, "accuracy", "accuracy_ci_low", "accuracy_ci_high",
                 "Agreement with these scores", "Share in agreement")
        else:
            draw(ax_a, "accuracy", "accuracy_ci_low", "accuracy_ci_high",
                 "Accuracy", "Share of correct answers")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    # ------------------------------------------------------------------ html pieces
    @staticmethod
    def _card_html(title: str, text: str, highlight: bool = False) -> str:
        if highlight:
            style = ("background:#eaf6ee;border:1px solid #9fd3b0;border-left:6px solid #2e9e5b;"
                     "color:#1b2a21;")
        else:
            style = "background:#f5f6f8;border:1px solid #dcdfe4;border-left:6px solid #b8bec8;color:#222;"
        return (f'<div style="{style}border-radius:6px;padding:10px 14px;margin:8px 0;'
                f'font-family:sans-serif;font-size:14px;line-height:1.5;">'
                f'<div style="font-weight:700;margin-bottom:4px;">{_esc(title)}</div>'
                f'<div>{_esc(text)}</div></div>')

    @staticmethod
    def _notice_html(message: str) -> str:
        return (f'<div style="background:#fff8e5;border:1px solid #f0d98a;color:#4a3b00;'
                f'border-radius:6px;padding:10px 14px;margin:8px 0;font-family:sans-serif;'
                f'font-size:14px;">{_esc(message)}</div>')

    # ------------------------------------------------------------------ 4. dashboard
    def render_dashboard(self, comparison_df, paired_df=None) -> None:
        """Interactive ipywidgets dashboard (call inside Colab / Jupyter)."""
        try:
            import ipywidgets as w
            from IPython.display import HTML, Image, clear_output, display
        except ImportError:
            print("The dashboard needs ipywidgets and IPython (both are pre-installed in Colab).")
            return

        if not self._prepare(comparison_df, paired_df):
            display(HTML(self._notice_html(
                "There are no fairness results to show. Please check that the comparison "
                "file exists and is not empty.")))
            return

        style = {"description_width": "initial"}
        dd_data = w.Dropdown(options=[(_DATASET_LABEL.get(d, d), d) for d in self._datasets()],
                             description="Dataset:", style=style, layout=w.Layout(width="260px"))
        dd_target = w.Dropdown(options=[("", "")], description="Target:", style=style,
                               layout=w.Layout(width="480px"))
        dd_group = w.Dropdown(options=[("", "")], description="Group:", style=style,
                              layout=w.Layout(width="300px"))
        dd_cmp = w.Dropdown(options=_COMPARE_MENU, value=_COMPARE_MENU[0][1], description="Compare:",
                            style=style, layout=w.Layout(width="520px"))
        out = w.Output()
        busy = {"on": False}

        def sync_targets():
            ds = dd_data.value
            if ds == "kaggle":
                dd_target.options = [(_cap(self.friendly_target(KAGGLE_TARGET)), KAGGLE_TARGET)]
                dd_target.value = KAGGLE_TARGET
                dd_target.disabled = True
                return
            targets = self._targets(ds)
            keep = dd_target.value if dd_target.value in targets else None
            dd_target.options = [(_cap(self.friendly_target(t)), t) for t in targets]
            if keep is not None:
                dd_target.value = keep
            dd_target.disabled = len(targets) <= 1

        def sync_groups():
            groups = self._groups(dd_data.value, dd_target.value)
            keep = dd_group.value if dd_group.value in groups else None
            dd_group.options = [(_GROUP_MENU.get(g, g), g) for g in groups] or [("(none)", "")]
            if keep is not None:
                dd_group.value = keep

        def draw():
            with out:
                clear_output(wait=True)
                try:
                    key = (dd_data.value, dd_target.value, dd_group.value)
                    rows = self._index.get(key)
                    if not rows:
                        display(HTML(self._notice_html("There are no results for this combination.")))
                        return
                    pair = dd_cmp.value
                    text = self._pair_summary(rows, key, pair)
                    if text is None:
                        display(HTML(self._notice_html(self._missing_pair_message(rows, pair))))
                    else:
                        display(HTML(self._card_html("In plain words", text, highlight=True)))
                    display(Image(data=self._build_chart_png(rows, key[0], key[1], highlight=pair)))
                    foot = ("The thin black lines show the range of plausible values given the size "
                            "of the test set. The outlined bars are the two models being compared.")
                    if not self._has_paired:
                        foot += (" (No paired-comparison file was found, so 'clearly' here means "
                                 "the two ranges do not overlap.)")
                    display(HTML(f'<div style="color:#666;font-family:sans-serif;font-size:12px;">'
                                 f'{_esc(foot)}</div>'))
                except Exception as exc:  # never show a traceback to a non-technical reader
                    display(HTML(self._notice_html(
                        f"Sorry, this view could not be drawn ({type(exc).__name__}: {exc}).")))

        def on_dataset(_=None):
            if busy["on"]:
                return
            busy["on"] = True
            try:
                sync_targets()
                sync_groups()
            finally:
                busy["on"] = False
            draw()

        def on_target(_=None):
            if busy["on"]:
                return
            busy["on"] = True
            try:
                sync_groups()
            finally:
                busy["on"] = False
            draw()

        def on_other(_=None):
            if not busy["on"]:
                draw()

        busy["on"] = True
        sync_targets()
        sync_groups()
        busy["on"] = False
        dd_data.observe(on_dataset, names="value")
        dd_target.observe(on_target, names="value")
        dd_group.observe(on_other, names="value")
        dd_cmp.observe(on_other, names="value")

        header = w.HTML(
            "<h3 style='font-family:sans-serif;margin:0 0 4px 0;'>Fairness dashboard</h3>"
            "<div style='font-family:sans-serif;font-size:13px;color:#555;'>Pick a dataset, a "
            "target and a group, then choose which two models to compare. Fairness scores run from "
            "0 to 1; higher means the model treats different groups more evenly.</div>")
        controls = w.VBox([w.HBox([dd_data, dd_target], layout=w.Layout(flex_flow="row wrap")),
                           w.HBox([dd_group, dd_cmp], layout=w.Layout(flex_flow="row wrap"))])
        display(w.VBox([header, controls, out]))
        draw()

    # ------------------------------------------------------------------ 5. static HTML export
    _CSS = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#222;
         max-width:980px;margin:24px auto;padding:0 16px;line-height:1.5;background:#fff}
    h1{margin-bottom:2px} h2{margin-top:40px;border-bottom:2px solid #dcdfe4;padding-bottom:4px}
    h3{margin-top:28px;color:#333} h4{margin-top:22px;margin-bottom:6px;color:#444}
    .sub{color:#666;font-size:13px}
    .how{background:#eef3fb;border:1px solid #c5d5ee;border-radius:8px;padding:12px 18px;margin:18px 0}
    .how h2{margin:0 0 6px 0;border:0;font-size:18px} .how ul{margin:6px 0 6px 18px;padding:0}
    .chart{max-width:100%;height:auto;border:1px solid #e3e5e9;border-radius:6px;margin:6px 0}
    .toc{font-size:14px;columns:2;margin:10px 0} .toc a{color:#2a5db0;text-decoration:none}
    .foot{color:#666;font-size:12.5px;border-top:1px solid #dcdfe4;margin-top:40px;padding-top:10px}
    """

    def _how_to_read_html(self) -> str:
        return """
        <div class="how"><h2>How to read this report</h2>
        <p>Each chart gives every model a <b>fairness score between 0 and 1</b>. A score of 1.0 means
        the model treats all groups evenly; lower scores mean some groups do worse than others.
        The thin black lines show the range of plausible values given the size of the test set.
        When a summary says <b>"clearly"</b>, the gap between two models is bigger than that normal wobble.
        The smaller chart on the right shows accuracy (Kaggle) or, for FairCVdb, how closely the model
        agrees with the scores it was given. That is <i>not</i> a measure of fairness.</p>
        <p><b>The three models, from simplest to most advanced:</b></p>
        <ul>
          <li><b>Simple</b>: the simple word-counting model.</li>
          <li><b>AI model</b>: an AI language model without bias-aware weighting. This is the control.</li>
          <li><b>AI model + bias-aware</b>: the same AI language model, with bias-aware weighting added.</li>
        </ul>
        <p><b>The datasets:</b> Kaggle asks the models to predict a job category. FairCVdb has three
        kinds of scores: one that ignores gender and ethnicity, and two that are deliberately biased
        by gender or by ethnicity.</p>
        <p>For each result, the first box (highlighted in green) shows the effect of the bias-aware
        weighting on its own. The other boxes show the effect of the AI language model alone and
        of both changes together.</p></div>"""

    def export_html_report(self, comparison_df, output_path: str, paired_df=None) -> str:
        """Write a self-contained HTML report (inline CSS, base64 PNG charts); return its path."""
        ok = self._prepare(comparison_df, paired_df)
        body: List[str] = []
        toc: List[str] = []

        if not ok:
            body.append(self._notice_html("There were no fairness results to put in this report."))
        else:
            for ds in self._datasets():
                body.append(f'<h2 id="{_slug(ds)}">{_esc(_DATASET_LABEL.get(ds, ds))}</h2>')
                for tgt in self._targets(ds):
                    body.append(f"<h3>{_esc(_cap(self.friendly_target(tgt)))}</h3>")
                    for grp in self._groups(ds, tgt):
                        key = (ds, tgt, grp)
                        rows = self._index[key]
                        anchor = _slug(f"{ds}-{tgt}-{grp}")
                        label = f"{_DATASET_LABEL.get(ds, ds)}: {_cap(self.friendly_target(tgt))}, {_GROUP_MENU.get(grp, grp)}"
                        toc.append(f'<a href="#{anchor}">{_esc(label)}</a><br>')
                        body.append(f'<h4 id="{anchor}">Groups compared: {_esc(_GROUP_MENU.get(grp, grp))}</h4>')
                        try:
                            png = base64.b64encode(self._build_chart_png(rows, ds, tgt, highlight=KEY_PAIR)).decode("ascii")
                            body.append(f'<img class="chart" alt="Fairness and accuracy chart" '
                                        f'src="data:image/png;base64,{png}">')
                        except Exception as exc:
                            body.append(self._notice_html(f"The chart could not be drawn ({type(exc).__name__}: {exc})."))
                        cards = [
                            (KEY_PAIR, "Effect of the bias-aware weighting", True),
                            (PAIR_ORDER[1], "Effect of switching to the AI language model", False),
                            (PAIR_ORDER[2], "Both changes together", False),
                        ]
                        shown = 0
                        for pair, title, hl in cards:
                            text = self._pair_summary(rows, key, pair)
                            if text is not None:
                                body.append(self._card_html(title, text, highlight=hl))
                                shown += 1
                        if not shown:
                            body.append(self._notice_html(
                                "Only one model has results for this combination, so there is nothing to compare."))

        notes = ["The ranges on the charts reflect test-set sampling only, and each AI model was "
                 "trained once; a different training run could give slightly different results."]
        if ok and not self._has_paired:
            notes.append("No paired-comparison file was available, so 'clearly' was decided by "
                         "whether the two models' ranges overlap.")
        stamp = _dt.datetime.now().strftime("%d %B %Y, %H:%M")
        page = (
            '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Fairness report</title><style>{self._CSS}</style></head><body>"
            f'<h1>Fairness report</h1><div class="sub">Generated {_esc(stamp)}</div>'
            f"{self._how_to_read_html()}"
            + (f'<div class="toc"><b>Contents</b><br>{"".join(toc)}</div>' if toc else "")
            + "".join(body)
            + '<div class="foot">' + "<br>".join(_esc(n) for n in notes) + "</div></body></html>"
        )
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(page)
        return output_path
