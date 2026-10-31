"""Downloadable reports: ranking CSV, fairness CSV, self-contained fairness HTML."""
import html
from datetime import datetime
import pandas as pd


def ranking_csv(ranked: pd.DataFrame) -> bytes:
    return ranked.to_csv(index=False).encode("utf-8")


def fairness_csv(fair: dict) -> bytes:
    a = fair["summary"].assign(section="summary")
    b = fair["groups"].assign(section="groups")
    return pd.concat([a, b], ignore_index=True).to_csv(index=False).encode("utf-8")


def _tbl(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.select_dtypes("float"):
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return d.to_html(index=False, escape=True, border=0, classes="t")


def fairness_html(fair: dict, model_label: str, job: dict) -> bytes:
    flagged = fair["summary"]["flagged"].any()
    verdict = ("FLAGGED: at least one group comparison is below the "
               f"{fair['threshold']:.0%} fairness threshold." if flagged else
               f"No group comparison fell below the {fair['threshold']:.0%} threshold.")
    note = f"<p><b>{html.escape(fair['note'])}</b></p>" if fair.get("note") else ""
    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}"
           "table.t{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}.t th,.t td{border-bottom:1px solid #ddd;"
           "padding:6px 10px;text-align:left}.t th{background:#f3f3f3}.flag{padding:10px 14px;border-radius:6px;"
           "background:BG}small{color:#555}").replace("BG", "#fde2e2" if flagged else "#e2f5e6")
    body = f"""<h1>Fairness report</h1>
<p><b>Model:</b> {html.escape(model_label)}<br><b>Job:</b> {html.escape(str(job.get('title','')))} |
<b>Target category:</b> {html.escape(str(job.get('category','')))}<br>
<b>Candidates:</b> {fair['n_candidates']} | <b>Shortlist size:</b> {fair['top_k']} |
<b>Generated:</b> {datetime.now():%Y-%m-%d %H:%M}</p>
<p class="flag">{html.escape(verdict)}</p>
{note}
<h2>How to read this</h2>
<p>Each resume was scored four times with only the name changed (gender x name group, from
Bertrand &amp; Mullainathan name lists). The shortlist uses one pooled score cut-off ({fair['cutoff']:.4f}).
The fairness ratio is the lowest group shortlist rate divided by the highest (1.0 = identical).
Intervals are 90% cluster-bootstrap intervals. Groups are name-based proxies, not real demographics.</p>
<h2>Fairness ratios</h2>{_tbl(fair['summary'])}
<h2>Per-group shortlist rate</h2>{_tbl(fair['groups'])}
<h2>Unstable candidates</h2>
<p>{fair['n_unstable']} candidate(s) are shortlisted under some names but not others. Mean score spread
across name variants: {fair['mean_spread']:.4f}.</p>{_tbl(fair['unstable'].head(50))}
<p><small>Limitations: the models were trained on Kaggle resumes (24 occupations); ratios from a small
candidate pool are noisy; this is an audit aid, not a legal compliance determination.</small></p>"""
    return f"<!doctype html><meta charset=utf-8><title>Fairness report</title><style>{css}</style>{body}".encode()
