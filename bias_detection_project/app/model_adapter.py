"""Loads the three trained models from artifacts/ and returns Kaggle-category probabilities.

Uses only existing project code (code/*.py). Models are loaded one at a time and released,
so all three can be run in one session on a 16 GB machine.
"""
import gc, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("BIAS_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
for p in (ROOT / "code",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MODELS = {
    "tfidf": {"label": "TF-IDF baseline", "path": "artifacts/baseline/kaggle", "kind": "baseline",
              "files": ["model.joblib"]},
    "bertidraaf": {"label": "BERTidRAAF-lite (IDF weights)", "path": "artifacts/bertidraaf_lite",
                   "kind": "neural", "files": ["manifest.json", "weights.pt", "term_weights.joblib"]},
    "ibs": {"label": "BERTidRAAF + IBS + embeddings (improved)", "path": "artifacts/improved",
            "kind": "neural", "files": ["manifest.json", "weights.pt", "term_weights.joblib"]},
}
CLASSES = ["ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "AVIATION",
           "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER",
           "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
           "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"]


class ModelError(Exception):
    pass


def artifact_status() -> dict:
    """{key: (ok, message)} - checks that every required file exists."""
    out = {}
    for k, m in MODELS.items():
        d = ROOT / m["path"]
        missing = [f for f in m["files"] if not (d / f).exists()]
        out[k] = (not missing, "ready" if not missing else f"missing in {d}: {', '.join(missing)}")
    code_missing = [f for f in ("model_artifacts.py", "improved_scorer.py", "embedding_engine.py",
                                "preprocessor.py", "name_swap_auditor.py", "fairness_auditor.py")
                    if not (ROOT / "code" / f).exists()]
    out["code"] = (not code_missing, "ready" if not code_missing else f"missing in {ROOT/'code'}: {code_missing}")
    return out


def cuda_info() -> str:
    try:
        import torch
        return f"CUDA available: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() \
            else "CUDA NOT available - neural models will run on CPU (slow)"
    except Exception as e:  # noqa: BLE001
        return f"torch import failed: {e}"


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def kaggle_probabilities(key: str, texts: list, max_length=None):
    """Returns (probs[n, 24], classes). `texts` must be cleaned, name-swapped text (as in training)."""
    m = MODELS[key]
    d = ROOT / m["path"]
    ok, msg = artifact_status()[key]
    if not ok:
        raise ModelError(msg)
    if m["kind"] == "baseline":
        import joblib
        pay = joblib.load(d / "model.joblib")
        x = pay["vectorizer"].transform([str(t) for t in texts])
        clf = pay["classifier"]
        return clf.predict_proba(x), [str(c) for c in clf.classes_]
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from model_artifacts import load_neural
    scorer = None
    try:
        scorer = load_neural(d)
        if max_length:  # shorter inputs = much faster on CPU (slightly different from training length)
            scorer.engine.max_length = int(max_length)
        df = pd.DataFrame({"counterfactual_text": [str(t) for t in texts]})
        sizes = (16, 4) if has_cuda() else (8,)
        for bs in sizes:
            try:
                scorer.embed_batch_size = bs
                with torch.no_grad():
                    lk, _, _ = scorer._forward(df, "counterfactual_text")
                break
            except (torch.cuda.OutOfMemoryError, MemoryError):
                _free()
                if bs == sizes[-1]:
                    raise ModelError("Out of memory. Close other programs (browser tabs), lower the max "
                                     "tokens, or upload fewer resumes.")
        probs = F.softmax(lk, dim=-1).numpy()
        return probs, list(scorer.classes_)
    finally:
        del scorer
        _free()


def suggest_category(jd_text: str):
    """Cheap CPU guess of the job category from the description, using the TF-IDF model."""
    from preprocessor import Preprocessor
    probs, classes = kaggle_probabilities("tfidf", [Preprocessor().clean(jd_text)])
    i = int(np.argmax(probs[0]))
    return classes[i], float(probs[0][i])
