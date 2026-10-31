# HR Bias Detection Layer

A bias detection and mitigation layer for HR AI agents that screen resumes. It scores candidates with three models, audits each one for gender and ethnicity bias, and gives recruiters a ranked, downloadable report with a fairness view.

Built on the BERTidRAAF resume-classification paper (Haddad & Omri, 2026), extended with a **fairness audit on FairCVdb**, which the original paper does not report.

## Features

- Recruiter app: login / create account, CSV resume upload, job description + skills input
- Three selectable models: run one at a time or all three on the same upload
  - `tf-idf`: TF-IDF + logistic regression baseline
  - `bertidraaf_lite`: DistilBERT features + IDF term weights + learned gate
  - `improved`: BERTidRAAF-lite with the Intersectional Bias Score (IBS) weighting
- Ranked candidate list (best to worst), downloadable as CSV
- Fairness view and downloadable fairness report
- Error flagging for bad or missing rows in uploads
- Counterfactual name-swap auditing (Bertrand & Mullainathan, 2004)
- Bootstrap confidence intervals (1000 iterations, 90% CI, seed 42)

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 (the `.bat` scripts); Linux/macOS work with the manual steps |
| Python | 3.10 or newer <<confirm your version>>, added to PATH |
| RAM | 16 GB recommended |
| GPU | Optional. The default install is CPU-only; the `improved` model is faster with an NVIDIA GPU + CUDA PyTorch |
| Disk | ~3 GB free (dependencies + model weights) |

## Project Structure

```
bias_detection_project/
├── SE_project.ipynb            # Google Colab notebook: data prep, training, auditing, reports
├── app/                        # Recruiter interface (Streamlit + pywebview)
│   └── logs/                   # Application logs (created at runtime)
├── artifacts/                  # Trained models (download separately, see below)
│   ├── baseline/
│   │   ├── kaggle/
│   │   ├── score_blind/
│   │   ├── score_biased_gender/
│   │   └── score_biased_ethnicity/
│   ├── bertidraaf_lite/
│   └── improved/
├── code/                       # Core pipeline modules
│   ├── name_lists.py
│   ├── name_swap_auditor.py
│   ├── baseline_scorer.py
│   ├── bertidraaf_lite_scorer.py
│   ├── improved_scorer.py
│   ├── fairness_auditor.py
│   ├── report_dashboard.py
│   ├── hr_agent_interceptor.py
│   ├── model_artifacts.py
│   ├── comparison.py
│   ├── data_prep.py
│   └── tests/                  # Unit tests for the modules above
├── data/
│   ├── raw/
│   │   ├── kaggle/             # Kaggle 24-class Resume dataset (data/ + Resume/)
│   │   └── faircvdb/           # FairCVdb files
│   └── processed/              # Preprocessed resumes, counterfactuals, predictions
├── reports/                    # fairness_comparison.csv, fairness_report.html
├── tests/                      # Additional tests
├── sample_data/
│   ├── synthetic_resumes.csv       # 150 synthetic resumes for testing
│   └── synthetic_job_posting.json  # Sample Data Engineer job posting
├── install.bat                 # One-time setup (Windows)
├── run.bat                     # Launch the app (Windows)
├── requirements.txt
├── LICENSE
└── README.md
```

`venv/` is created by `install.bat` and is not part of the repository.

| Folder / file | What it holds |
|---|---|
| `SE_project.ipynb` | The Google Colab notebook on which everything was run: preprocessing, name-swap auditing, training all three models, fairness auditing, report generation, and saving the model artifacts |
| `app/` | The recruiter-facing interface (login, upload, model selection, rankings, fairness view) |
| `code/` | All scoring, auditing and reporting logic. The notebook and the app both import from here |
| `artifacts/` | Trained model files. Each model folder has its saved weights/manifest, loaded through `code/model_artifacts.py` |
| `data/raw/` | Original datasets (not included in this repo) |
| `data/processed/` | Pipeline outputs such as `preprocessed_resumes.csv`, `counterfactual_resumes.csv`, and prediction files |
| `reports/` | Generated fairness comparison tables and HTML reports |
| `sample_data/` | Synthetic resume CSV and job posting for testing the app |
| `tests/`, `code/tests/` | pytest suites |

## Quick Start (Windows)

This runs the recruiter app on your own machine using the already-trained models. You do **not** need the notebook for this.

### 1. Clone the repository

```bat
git clone https://github.com/<<your-username>>/<<repo-name>>.git
cd <<repo-name>>
```

### 2. Download the trained model files

The model weights are 250 MB+ each, so they are not stored in the normal repo.
Download them from **<<Releases page / Git LFS / Drive link>>** and extract them into `artifacts/` so the layout matches the Project Structure above.

If you use Git LFS: run `git lfs install` before cloning, then `git lfs pull`.

### 3. Run `install.bat` (first time only)

```bat
@echo off
setlocal
echo === HR Bias Detection: installing ===

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ and tick "Add to PATH".
    pause
    exit /b 1
)

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

python -m pip install --upgrade pip

REM Default: CPU-only PyTorch.
REM For NVIDIA GPU, replace the next line with:
REM   pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

echo === Install complete. Run run.bat to start the app. ===
pause
```

### 4. Run `run.bat` to launch the app

```bat
@echo off
setlocal
call venv\Scripts\activate.bat
echo Starting HR Bias Detection app...
python app\main.py
pause
```

> `app\main.py` is a placeholder. Use your real entry-point file. <<confirm>>

The app opens in its own desktop window (pywebview). If no window appears, open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

## Manual Setup (any OS)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt

python app/main.py
# or, browser only:
streamlit run app/main.py
```

## Using the App

1. **Create an account / log in.**
2. **Upload resumes** as a CSV file (format below).
3. **Enter the job description and required skills**, or load a job posting JSON file.
4. **Pick a model**: `tf-idf`, `bertidraaf_lite`, `improved`, or **Run all three**. All models use the same uploaded CSV.
5. **View the ranking** (best to worst) and download it as CSV.
6. **Open the Fairness view** to see per-group results and download the fairness report.
7. Rows that could not be processed are flagged with the reason.

**Quick test:** after launching, upload `sample_data/synthetic_resumes.csv`, load `sample_data/synthetic_job_posting.json`, choose **Run all three**, and check that a ranked list and a fairness view appear.

### Resume CSV format

Upload a `.csv` file with one resume per row. Columns:

| Column | Example | Notes |
|---|---|---|
| `resume_id` | `R0002` | Unique ID per resume |
| `name` | `Mateo Jackson` | Candidate name |
| `email` | `mateo.jackson2@example.com` | Candidate email |
| `category` | `INFORMATION-TECHNOLOGY` | Job category of the resume |
| `years_experience` | `11` | Whole number |
| `education` | `MCA` | Highest qualification |
| `skills` | `pandas; git; python; data modeling; airflow` | Separated by semicolons |
| `resume_text` | `Mateo Jackson\nProfessional summary: ...` | Full plain-text resume. **Scored by all three models** |

<<Confirm which columns are required and which are optional, e.g. "`resume_id` and `resume_text` are required; the rest are optional".>>

Rows with a missing ID or empty `resume_text` are flagged and skipped instead of stopping the run.

A ready-to-use example is in `sample_data/synthetic_resumes.csv` (150 resumes across 5 categories). It contains synthetic data only: the names and emails are fake.

### Job posting format

The job description can be typed into the app or loaded from a JSON file. Example (`sample_data/synthetic_job_posting.json`):

```json
{
  "title": "Data Engineer",
  "department": "Analytics",
  "location": "Mumbai (hybrid)",
  "description": "Build and maintain batch and streaming data pipelines...",
  "required_skills": ["python", "sql", "etl", "airflow", "aws"],
  "preferred_skills": ["spark", "kafka", "docker", "data modeling"],
  "min_years_experience": 3,
  "education": "Bachelor's in a technical field",
  "top_k": 20,
  "target_category": "INFORMATION-TECHNOLOGY"
}
```

| Field | Meaning |
|---|---|
| `title`, `department`, `location` | Basic details of the role |
| `description` | Free-text job description |
| `required_skills` | Skills the candidate must have |
| `preferred_skills` | Nice-to-have skills |
| `min_years_experience` | Minimum years of experience |
| `education` | Education requirement |
| `top_k` | How many top-ranked candidates to show in the report |
| `target_category` | Resume category the job belongs to |

<<Confirm that the app reads all of these fields. Delete any rows the app ignores.>>

## How the Fairness Audit Works

- **FairCVdb targets:** fairness = min/max ratio of selection rates across groups. The default pass threshold is **0.8**.
- **Kaggle 24-class resumes:** fairness = min/max ratio of accuracy across groups.
- Groups are intersectional labels such as `female_groupB`, created by swapping names into otherwise identical resumes.
- Confidence intervals use a cluster bootstrap by `original_id`, plus a paired improved-minus-baseline bootstrap.

## Datasets

Datasets are **not** included in this repo. To retrain or re-run the audit, download them and place them as shown:

- Kaggle 24-class Resume dataset → `data/raw/kaggle/` <<add link>>
- FairCVdb ([BiDAlab/FairCVtest](https://github.com/BiDAlab/FairCVtest)) → `data/raw/faircvdb/`

## Reproducing the Experiments (Google Colab notebook)

All data preparation, model training, fairness auditing and report generation were run in **`SE_project.ipynb`**, a Google Colab notebook. Use it if you want to re-run the experiments or retrain the models. The recruiter app only *loads* the artifacts the notebook produces.

1. Upload `SE_project.ipynb` to [Google Colab](https://colab.research.google.com/) (File → Upload notebook).
2. Switch to a GPU runtime: Runtime → Change runtime type → **T4 GPU**.
3. Create the project folder in your Google Drive and copy the repo's `code/` folder into it, with the datasets in `data/raw/`. The notebook expects:
   ```
   /content/drive/MyDrive/bias_detection_project/
   ```
4. Mount Drive when the notebook asks, and check that `PROJECT_ROOT` in the first cell points at the folder above.
5. Run the cells from top to bottom. The notebook builds the pipeline in this order:
   1. Preprocessing (`data/processed/preprocessed_resumes.csv`)
   2. Name-swap counterfactuals (`counterfactual_resumes.csv`)
   3. Baseline TF-IDF scorer
   4. BERTidRAAF-lite scorer
   5. Improved (IBS) scorer
   6. Fairness audit (`reports/fairness_comparison.csv`)
   7. Report dashboard (`reports/fairness_report.html`)
   8. HR agent interceptor demo
6. Download the trained `artifacts/` folder from Drive and put it in your local copy of the repo to use it with the app.

Notes:

- The `improved` model takes about 9 minutes to fit on a T4 GPU.
- Using a standard consumer GPU instead of Colab's T4 can shift results slightly from the reported numbers.
- The notebook is where the interactive `ipywidgets` fairness dashboard runs. The local app uses the HTML report instead.

## Running the Tests

```bash
pip install pytest
python -m pytest code/tests tests
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `python is not recognized` | Reinstall Python and tick **Add Python to PATH** |
| `FileNotFoundError: artifacts/...` | Model files are missing or in the wrong folder (see Quick Start step 2) |
| `improved` model is very slow | Expected on CPU. Install the CUDA build of PyTorch, or use `tf-idf` only |
| `torch.cuda.is_available()` is `False` | You have the CPU build, or your NVIDIA drivers need updating |
| `CUDA out of memory` | Close other GPU apps, or fall back to CPU |
| App window is blank | Open the localhost URL in a browser instead |
| Port already in use | Close the old app window, or run `streamlit run app/main.py --server.port 8502` |
| Rows missing from the ranking | Check the error flags in the app. The row probably has no ID or empty `resume_text` |
| Notebook can't find files in Colab | Check that Drive is mounted and `PROJECT_ROOT` points to the right folder |

## Built With

Streamlit · pywebview · pandas · NumPy · scikit-learn · PyTorch · Hugging Face Transformers (DistilBERT) · pytest · Google Colab

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
Datasets and pretrained models are covered by their own licenses and are not redistributed here.

## Authors

Sarah Gonsalves (Mukesh Patel School of Technology, Management, and Engineering, Mumbai, India)
Swara Shah (Mukesh Patel School of Technology, Management, and Engineering, Mumbai, India)
