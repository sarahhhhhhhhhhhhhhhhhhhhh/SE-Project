
"""Persistence and application loading for the bias detection scorers.

Only load checkpoints from your own trusted training runs. PyTorch/joblib use pickle.
"""
from __future__ import annotations
import json
from pathlib import Path


def save_baseline(scorer, destination, task):
    import joblib
    p = Path(destination); p.mkdir(parents=True, exist_ok=True)
    if scorer.vectorizer_ is None:
        raise RuntimeError('Baseline not fitted')
    clf = scorer._kaggle_clf if task == 'kaggle' else scorer._faircv_clf
    if clf is None:
        raise RuntimeError(f'Baseline {task} classifier not fitted')
    # Do not serialize training rows or the training feature matrix.
    payload = {'vectorizer': scorer.vectorizer_, 'classifier': clf,
               'task': task, 'text_col': scorer.text_col,
               'threshold': scorer.faircv_threshold_ if task != 'kaggle' else None,
               'source_class': 'BaselineScorer'}
    joblib.dump(payload, p / 'model.joblib', compress=3)
    return str(p / 'model.joblib')


def load_baseline(destination):
    import joblib
    p = Path(destination)
    return joblib.load(p / 'model.joblib')


def predict_baseline_texts(destination, texts):
    """Returns predictions for raw text strings; same preprocessing as the saved TF-IDF vectorizer."""
    model = load_baseline(destination)
    x = model['vectorizer'].transform([str(x) for x in texts])
    clf = model['classifier']
    return {'predictions': clf.predict(x).tolist(),
            'probabilities': clf.predict_proba(x).tolist(),
            'classes': clf.classes_.tolist()}


def save_neural(scorer, destination, common_kwargs, model_kind):
    import torch
    p = Path(destination); p.mkdir(parents=True, exist_ok=True)
    if scorer.model is None or scorer.engine is None or scorer.engine.model is None:
        raise RuntimeError(f'{model_kind} is not fitted; no checkpoint can be saved')
    weights = {'head': {k: v.detach().cpu() for k, v in scorer.model.state_dict().items()},
               'backbone': {k: v.detach().cpu() for k, v in scorer.engine.model.state_dict().items()}}
    torch.save(weights, p / 'weights.pt')
    import joblib
    joblib.dump(scorer.ibs_lookup_, p / 'term_weights.joblib', compress=3)
    manifest = {'format_version': 1, 'model_kind': model_kind,
                'model_name': scorer.model_name, 'weight_mode': scorer.weight_mode,
                'common_kwargs': common_kwargs, 'classes': list(scorer.classes_),
                'faircv_targets': list(scorer.faircv_targets_),
                'thresholds': {k: float(v) for k, v in scorer.thresholds_.items()},
                'decision_thresholds': {k: float(v) for k, v in scorer.decision_thr_.items()},
                'oov_weight': float(scorer.oov_weight),
                'fine_tune': bool(scorer.fine_tune),
                'head_hidden_size': int(scorer.model.ctx_norm.normalized_shape[0]),
                'gate_hidden': int(scorer.gate_hidden), 'dropout': float(scorer.dropout),
                'faircv_aux_reg': bool(scorer.faircv_aux_reg),
                'text_col': 'counterfactual_text'}
    (p / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return str(p)


def load_neural(destination, device=None):
    """Return a fully configured ImprovedScorer or BERTidRAAFLiteScorer for inference.

    Requires embedding_engine.py and its dependencies in the Python path.
    """
    import joblib
    import torch
    from embedding_engine import EmbeddingEngine
    from improved_scorer import ImprovedScorer, FusedClassifier
    from bertidraaf_lite_scorer import BERTidRAAFLiteScorer
    p = Path(destination)
    m = json.loads((p / 'manifest.json').read_text())
    if m['model_kind'] not in ('improved', 'bertidraaf_lite'):
        raise ValueError('Not a supported neural artifact')
    kwargs = dict(m['common_kwargs'])
    kwargs['faircv_thresholds'] = m['thresholds']
    cls = ImprovedScorer if m['model_kind'] == 'improved' else BERTidRAAFLiteScorer
    if cls is ImprovedScorer:
        kwargs['weight_mode'] = m['weight_mode']
    scorer = cls(**kwargs)
    scorer.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    scorer.engine = EmbeddingEngine(m['model_name'])
    scorer.engine.model.load_state_dict(
        torch.load(p / 'weights.pt', map_location='cpu', weights_only=True)['backbone'])
    scorer.engine.model.to(scorer.device)
    scorer.engine.set_train(False)
    scorer.model = FusedClassifier(m['head_hidden_size'], len(m['classes']),
                                   len(m['faircv_targets']), m['gate_hidden'],
                                   m['dropout'], aux_reg=m['faircv_aux_reg'])
    scorer.model.load_state_dict(
        torch.load(p / 'weights.pt', map_location='cpu', weights_only=True)['head'])
    scorer.model.to(scorer.device).eval()
    scorer.ibs_lookup_ = joblib.load(p / 'term_weights.joblib')
    scorer.classes_ = list(m['classes'])
    scorer.thresholds_ = m['thresholds']
    scorer.decision_thr_ = m['decision_thresholds']
    scorer.oov_weight = m['oov_weight']
    return scorer
