"""Resume ranking + fairness audit interface (desktop window via desktop.py, or: streamlit run app/app.py)"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import auth, errors, model_adapter as ma, ranking as rk, reports  # noqa: E402

st.set_page_config(page_title="Resume Ranking & Fairness Audit", page_icon="⚖️", layout="wide")
auth.init_db()
S = st.session_state
S.setdefault("issues", [])
S.setdefault("results", {})
S.setdefault("user", None)
S.setdefault("job", {})

INDIGO, VIOLET, GREEN, RED, GREY = "#4F46E5", "#7C3AED", "#16A34A", "#DC2626", "#6B7280"
st.markdown("""<style>
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0}
.stApp {background:#F4F6FB}
.block-container {padding-top:1.2rem; max-width:1280px}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#1E1B4B 0%,#4338CA 100%)}
[data-testid="stSidebar"] * {color:#fff !important}
[data-testid="stSidebar"] .stButton>button {background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.3)}
.brand {font-size:1.25rem; font-weight:800; letter-spacing:.2px; margin-bottom:.2rem}
.userpill {background:rgba(255,255,255,.16); border-radius:999px; padding:6px 14px; display:inline-block; margin:.4rem 0 1rem}
.hero {background:linear-gradient(120deg,#4F46E5,#7C3AED 60%,#EC4899); color:#fff; padding:26px 32px;
       border-radius:18px; margin-bottom:18px; box-shadow:0 12px 30px rgba(79,70,229,.28)}
.hero h1 {margin:0; font-size:1.9rem; color:#fff; padding:0} .hero p {margin:.35rem 0 0; opacity:.92; font-size:1rem}
.steps {display:flex; gap:10px; margin:0 0 14px; flex-wrap:wrap}
.step {background:#fff; border:1px solid #E5E7EB; border-radius:999px; padding:6px 16px; font-size:.88rem; color:#374151}
.step b {background:#4F46E5; color:#fff; border-radius:50%; padding:1px 8px; margin-right:8px}
div[data-testid="stMetric"] {background:#fff; border:1px solid #E5E7EB; border-radius:14px; padding:14px 18px;
       box-shadow:0 2px 10px rgba(17,24,39,.05)}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px; background:#fff; border-color:#E5E7EB;
       box-shadow:0 2px 10px rgba(17,24,39,.04)}
.stButton>button, .stDownloadButton>button {border-radius:10px; font-weight:600; padding:.55rem 1.1rem}
.stButton>button[kind="primary"] {background:linear-gradient(90deg,#4F46E5,#7C3AED); border:0; color:#fff}
.stTabs [data-baseweb="tab"] {font-weight:600}
.fcard {background:#fff; border-radius:16px; padding:18px 20px; border:1px solid #E5E7EB; border-left:8px solid var(--c);
       box-shadow:0 2px 10px rgba(17,24,39,.05)}
.fcard .big {font-size:2.2rem; font-weight:800; color:var(--c); line-height:1.1}
.fcard .lab {color:#4B5563; font-size:.9rem} .fcard .sub {color:#6B7280; font-size:.8rem; margin-top:4px}
h2, h3 {color:#1F2937}
</style>""", unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def steps(active=1):
    items = ["Upload resumes", "Describe the job", "Choose model", "Review results"]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><b>{i+1}</b>{t}</div>' for i, t in enumerate(items)) + "</div>", unsafe_allow_html=True)


def issue(level, where, msg, exc=None):
    errors.flag(S["issues"], level, where, msg, exc)


def fcard(label, value, ok, sub):
    c = GREEN if ok else RED
    st.markdown(f'<div class="fcard" style="--c:{c}"><div class="lab">{label}</div>'
                f'<div class="big">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ login
def login_view():
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        hero("⚖️ Fair Hiring Assistant", "Rank candidates and check the ranking for bias, in one place.")
        with st.container(border=True):
            t1, t2 = st.tabs(["🔑 Log in", "✨ Create account"])
            with t1:
                u = st.text_input("Username", key="lu")
                p = st.text_input("Password", type="password", key="lp")
                if st.button("Log in", type="primary", width="stretch"):
                    if auth.verify(u, p):
                        S["user"] = u.strip().lower()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            with t2:
                u = st.text_input("Choose a username", key="cu")
                e = st.text_input("Email", key="ce")
                p = st.text_input("Password (min 8 characters)", type="password", key="cp")
                c = st.text_input("Confirm password", type="password", key="cc")
                if st.button("Create account", type="primary", width="stretch"):
                    ok, msg = auth.create_account(u, e, p, c)
                    (st.success if ok else st.error)(msg)


if not S["user"]:
    login_view()
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand">⚖️ Fair Hiring Assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="userpill">👤 {S["user"]}</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["📋 Rank candidates", "⚖️ Fairness report", "🛠 Issues & diagnostics"],
                    label_visibility="collapsed")
    n_err = sum(i["level"] == "error" for i in S["issues"])
    if n_err:
        st.error(f"{n_err} error(s) logged")
    st.divider()
    if st.button("Log out", width="stretch"):
        S["user"] = None
        st.rerun()


def label_col(rd):
    return rd["name"].where(rd["name"].astype(str).str.strip().ne("") if "name" in rd else False,
                            rd["resume_id"]) if "name" in rd else rd["resume_id"]


# ------------------------------------------------------------------ rank page
if page.endswith("Rank candidates"):
    hero("Rank candidates", "Upload resumes, describe the role, and let the model order the candidates.")
    steps()
    df = None
    with st.container(border=True):
        st.subheader("1 · Upload resumes")
        up = st.file_uploader("CSV file. Required column: resume_text. Recommended: name, resume_id, email, "
                              "years_experience.", type=["csv"])
        if up:
            try:
                raw = rk.read_csv(up)
                df, iss = rk.validate(raw)
                for lvl, msg in iss:
                    issue(lvl, "csv upload", msg)
                    {"error": st.error, "warning": st.warning, "info": st.info}[lvl](msg)
                if any(l == "error" for l, _ in iss):
                    df = None
                else:
                    st.success(f"✅ {len(df)} resumes ready")
                    st.dataframe(df.drop(columns=["resume_text"]).head(8), width="stretch", hide_index=True)
            except Exception as e:  # noqa: BLE001
                issue("error", "csv upload", f"Could not read the CSV: {e}", e)
                st.error(f"Could not read the CSV: {e}")

    with st.container(border=True):
        st.subheader("2 · Job details")
        c1, c2 = st.columns(2)
        title = c1.text_input("Job title", S["job"].get("title", ""))
        jd = c1.text_area("Job description", S["job"].get("description", ""), height=140)
        idx = ma.CLASSES.index(S["job"].get("category", "INFORMATION-TECHNOLOGY"))
        cat = c1.selectbox("Job category (what the model scores against)", ma.CLASSES, index=idx)
        if c1.button("✨ Suggest category from description") and jd.strip():
            try:
                g, p = ma.suggest_category(jd)
                st.info(f"Suggested: **{g}** ({p:.0%} confidence)")
            except Exception as e:  # noqa: BLE001
                issue("error", "suggest category", str(e), e)
                st.error(f"Could not suggest a category: {e}")
        req = c2.text_input("Required skills (comma-separated)", S["job"].get("req", ""))
        pre = c2.text_input("Preferred skills (comma-separated)", S["job"].get("pre", ""))
        min_years = c2.number_input("Minimum years of experience", 0, 40, int(S["job"].get("min_years", 0)))
        top_k = c2.number_input("Shortlist size (top K)", 1, 500, int(S["job"].get("top_k", 20)))
        blend = c2.slider("Weight of required-skill match in final score (0 = model only)", 0.0, 1.0, 0.0, 0.05)

    with st.container(border=True):
        st.subheader("3 · Choose model")
        status = ma.artifact_status()
        run_all = st.checkbox("🔀 Run all three models and compare")
        cols = st.columns(3)
        chosen = []
        for col, (k, m) in zip(cols, ma.MODELS.items()):
            ok = status[k][0]
            with col.container(border=True):
                on = st.checkbox(m["label"], value=run_all and ok, disabled=run_all or not ok, key=f"m_{k}")
                st.caption("Ready" if ok else f":red[Unavailable: {status[k][1]}]")
            if (on or run_all) and ok:
                chosen.append(k)
        cuda = ma.has_cuda()
        if not cuda:
            st.warning("No NVIDIA GPU detected: neural models run on CPU. Fast mode is recommended.")
        mode = st.radio("Speed", ["⚡ Fast (rank on name-free text, audit a sample)",
                                  "🔬 Full (every candidate × 4 name variants)"], index=1 if cuda else 0,
                        horizontal=True)
        fast = mode.startswith("⚡")
        c1, c2 = st.columns(2)
        sample_n = c1.number_input("Fairness-audit sample size (Fast mode)", 10, 500, 40, disabled=not fast)
        max_len = c2.selectbox("Max tokens per resume (lower = faster; training used 512)",
                               [512, 384, 256, 128], index=0 if cuda else 2)
        if df is not None:
            n_pass = (len(df) + min(int(sample_n), len(df)) * 4) if fast else len(df) * 4
            st.caption(f"{n_pass} text passes per neural model. Speed on your PC is not benchmarked; "
                       "time a small file first.")
        go = st.button("🚀 Run ranking", type="primary", disabled=df is None or not chosen)

    if go:
        S["job"] = dict(title=title, description=jd, req=req, pre=pre, min_years=min_years,
                        top_k=top_k, category=cat)
        required = [s.strip() for s in req.split(",") if s.strip()]
        preferred = [s.strip() for s in pre.split(",") if s.strip()]
        S["results"] = {}
        cache = {}
        for k in chosen:
            label = ma.MODELS[k]["label"]
            with st.status(f"Running {label} ...", expanded=False) as stt:
                try:
                    S["results"][k] = rk.analyse(k, df, cat, required, preferred, min_years, int(top_k),
                                                 blend, fast, int(sample_n), int(max_len), cache)
                    stt.update(label=f"✅ {label}: done", state="complete")
                    if S["results"][k]["fair"]["summary"]["flagged"].any():
                        issue("warning", label, "Fairness ratio below 0.8 for at least one group comparison.")
                except Exception as e:  # noqa: BLE001
                    issue("error", label, f"{type(e).__name__}: {e}", e)
                    stt.update(label=f"❌ {label}: FAILED - {e}", state="error")

    res = S["results"]
    if res:
        st.markdown("## Results")
        tabs = st.tabs([ma.MODELS[k]["label"] for k in res] + (["🔀 Model comparison"] if len(res) > 1 else []))
        for t, (k, r) in zip(tabs, res.items()):
            with t:
                rd = r["ranked"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Candidates", len(rd))
                m2.metric("Shortlisted", int(rd["shortlisted"].sum()))
                m3.metric("Top score", f"{rd['final_score'].max():.3f}")
                if "score_spread" in rd:
                    m4.metric("Avg name sensitivity", f"{rd['score_spread'].mean():.3f}",
                              help="Average score change when only the name changes. Lower is fairer.")
                top = rd.head(15).copy()
                top["label"] = (top["name"].astype(str).replace("", pd.NA).fillna(top["resume_id"])
                                if "name" in top else top["resume_id"])
                top["label"] = top["label"].astype(str) + "  (#" + top["rank"].astype(str) + ")"
                ch = alt.Chart(top).mark_bar(cornerRadiusEnd=5, color=INDIGO).encode(
                    x=alt.X("final_score:Q", title="Final score"),
                    y=alt.Y("label:N", sort=None, title=None),
                    tooltip=["rank", "resume_id", alt.Tooltip("final_score:Q", format=".3f")]
                ).properties(height=420, title="Top 15 candidates")
                st.altair_chart(ch, width="stretch")
                cfg = {"final_score": st.column_config.ProgressColumn("final score", min_value=0, max_value=1, format="%.3f"),
                       "model_score": st.column_config.ProgressColumn("model score", min_value=0, max_value=1, format="%.3f"),
                       "shortlisted": st.column_config.CheckboxColumn("shortlisted")}
                st.dataframe(rd, width="stretch", hide_index=True, column_config=cfg)
                st.download_button("⬇ Download ranking report (CSV)", reports.ranking_csv(rd),
                                   f"ranking_{k}.csv", "text/csv", key=f"dl_{k}", type="primary")
        if len(res) > 1:
            with tabs[-1]:
                first = next(iter(res))
                comp = res[first]["ranked"][["resume_id"] + (["name"] if "name" in res[first]["ranked"] else [])]
                for k, r in res.items():
                    comp = comp.merge(r["ranked"][["resume_id", "rank"]].rename(columns={"rank": f"rank_{k}"}),
                                      on="resume_id")
                cols_r = [c for c in comp if c.startswith("rank_")]
                comp["rank_disagreement"] = comp[cols_r].max(axis=1) - comp[cols_r].min(axis=1)
                st.caption("Rank disagreement = best rank minus worst rank across the selected models.")
                st.dataframe(comp.sort_values(cols_r[0]), width="stretch", hide_index=True)
                st.download_button("⬇ Download comparison (CSV)", comp.to_csv(index=False).encode(),
                                   "model_comparison.csv", "text/csv")

# ------------------------------------------------------------------ fairness page
elif page.endswith("Fairness report"):
    hero("Fairness report", "Does changing only the candidate's name change who gets shortlisted?")
    res = S["results"]
    if not res:
        st.info("Run a ranking first (📋 Rank candidates).")
    else:
        k = st.selectbox("Model", list(res), format_func=lambda x: ma.MODELS[x]["label"])
        f = res[k]["fair"]
        if f.get("note"):
            st.info(f["note"])
        if f["summary"]["flagged"].any():
            st.error(f"⚠ Flagged: a ratio is below {f['threshold']:.0%} for at least one comparison.")
        else:
            st.success(f"✅ No comparison fell below the {f['threshold']:.0%} threshold.")
        cols = st.columns(len(f["summary"]))
        for c, row in zip(cols, f["summary"].itertuples()):
            with c:
                fcard(row.attribute, f"{row.fairness_ratio:.2f}", not row.flagged,
                      f"90% CI {row.ci_low:.2f} – {row.ci_high:.2f}  ·  1.00 = perfectly equal")
        st.write("")
        g = f["groups"]
        g = g[g["attribute"].str.startswith("Intersectional")]
        bars = alt.Chart(g).mark_bar(cornerRadiusEnd=6, color=VIOLET).encode(
            x=alt.X("group:N", title=None, sort=None), y=alt.Y("selection_rate:Q", title="Shortlist rate"),
            tooltip=["group", alt.Tooltip("selection_rate:Q", format=".1%"), alt.Tooltip("mean_score:Q", format=".3f")]
        ).properties(height=300, title="Shortlist rate by name group (should be about equal)")
        st.altair_chart(bars, width="stretch")
        with st.expander("All group tables"):
            st.dataframe(f["groups"], hide_index=True, width="stretch")
        st.subheader(f"Unstable candidates ({f['n_unstable']})")
        st.caption("Shortlisted under some names but not others. Mean score spread across variants: "
                   f"{f['mean_spread']:.4f}")
        st.dataframe(f["unstable"].head(100), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Fairness report (HTML)", reports.fairness_html(f, ma.MODELS[k]["label"], S["job"]),
                           f"fairness_{k}.html", "text/html", type="primary")
        c2.download_button("⬇ Fairness tables (CSV)", reports.fairness_csv(f), f"fairness_{k}.csv", "text/csv")
        st.caption("Groups are name-based proxies, not real demographics. Small pools give noisy ratios.")

# ------------------------------------------------------------------ diagnostics
else:
    hero("Issues & diagnostics", "Checks that everything is installed, and a log of anything that went wrong.")
    with st.container(border=True):
        st.write(f"**Project root:** `{ma.ROOT}`  (override with env var BIAS_PROJECT_ROOT)")
        st.write(ma.cuda_info())
        for k, (ok, msg) in ma.artifact_status().items():
            (st.success if ok else st.error)(f"{k}: {msg}")
    with st.container(border=True):
        st.subheader("Logged issues (this session)")
        if S["issues"]:
            st.dataframe(pd.DataFrame(S["issues"])[["time", "level", "where", "message"]],
                         hide_index=True, width="stretch")
            with st.expander("Technical details"):
                for i in S["issues"]:
                    if i["detail"]:
                        st.code(i["detail"])
        else:
            st.write("None 🎉")
        st.caption("Full log: app/logs/app.log")