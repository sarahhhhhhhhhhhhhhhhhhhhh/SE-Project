"""
ibs_calculator.py  --  Intersectional Bias Score (IBS)

IBS for a term = min(rate_g) / max(rate_g), where rate_g is the fraction of
resumes in intersectional group g that contain the term, taken over ALL
intersectional groups jointly (not per attribute).

    IBS == 1.0  -> term appears equally often in every group (neutral)
    IBS -> 0.0  -> term is concentrated in some groups and (nearly) absent in others

Same divide-by-zero convention as the baseline p-ratio: if max(rates) == 0
(term appears in no group) the term is neutral and IBS = 1.0.

NOTE: this module deliberately has no torch / transformers dependency.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Dict, Iterable, List

_TERM_RE = re.compile(r"[a-z0-9]+")


def tokenize_terms(text: str) -> List[str]:
    """Lower-cased alphanumeric word tokens. Shared by IBS building and by the
    EmbeddingEngine's WordPiece alignment so both sides use the same term form."""
    return _TERM_RE.findall(str(text).lower())


class IBSCalculator:
    # ------------------------------------------------------------------ core
    def compute_ibs(self, term_presence_by_group: Dict[str, float]) -> float:
        """min(rates)/max(rates) across all groups jointly; 1.0 if max is 0."""
        rates = list(term_presence_by_group.values())
        if not rates:
            return 1.0
        lo, hi = min(rates), max(rates)
        if lo < 0:
            raise ValueError("Presence rates must be non-negative.")
        if hi <= 0:
            return 1.0  # term absent everywhere -> neutral (matches baseline p-ratio)
        return float(lo / hi)

    # ------------------------------------------------------------ vocabulary
    def build_vocab(self, texts: Iterable[str], min_df: int = 5,
                    max_vocab: int = 30000) -> List[str]:
        """Terms with document frequency >= min_df, most frequent first, capped."""
        df = Counter()
        for t in texts:
            df.update(set(tokenize_terms(t)))
        terms = [(w, c) for w, c in df.items() if c >= min_df]
        terms.sort(key=lambda x: (-x[1], x[0]))
        return [w for w, _ in terms[:max_vocab]]

    # ---------------------------------------------------------------- lookup
    def build_ibs_lookup(self, resumes_by_group: Dict[str, List[str]],
                         vocab: List[str]) -> Dict[str, float]:
        """IBS for every vocabulary term.

        Presence rate of term t in group g = (#resumes in g containing t) / (#resumes in g).
        Groups with zero resumes are skipped (with a warning) rather than being
        allowed to force every term's IBS to 0.
        """
        group_df: Dict[str, Counter] = {}
        group_n: Dict[str, int] = {}
        for g, texts in resumes_by_group.items():
            if not texts:
                warnings.warn(f"IBS: group '{g}' has no resumes; skipping it.")
                continue
            c = Counter()
            for t in texts:
                c.update(set(tokenize_terms(t)))
            group_df[g], group_n[g] = c, len(texts)

        lookup: Dict[str, float] = {}
        for term in vocab:
            rates = {g: group_df[g].get(term, 0) / group_n[g] for g in group_df}
            lookup[term] = self.compute_ibs(rates)
        return lookup
