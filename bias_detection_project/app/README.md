# Resume Ranking & Fairness Audit (Streamlit)

Place this `app/` folder inside `bias_detection_project/` (next to `code/` and `artifacts/`).
Copy `code/`, `artifacts/` (not `data/raw`) from Drive to your laptop.

```
pip install -r app/requirements.txt      # install the CUDA build of torch first if you have an NVIDIA GPU
streamlit run app/app.py
```
Different location? `set BIAS_PROJECT_ROOT=C:\path\to\bias_detection_project` first.
The first neural run downloads distilbert-base-uncased (needs internet once).

CSV: required `resume_text`; recommended `name`, `resume_id`, `email`, `years_experience`
(see synthetic_resumes.csv). Max 2000 rows.

Notes: login is a basic demo (SQLite + scrypt). Ranking = mean P(selected job category) over four
name-swapped copies of each resume. Fairness = pooled-cutoff shortlist rate ratio across name groups.
