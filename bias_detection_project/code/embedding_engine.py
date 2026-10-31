"""
embedding_engine.py  --  contextual embeddings for BERTidRAAF-lite

Backbone defaults to distilbert-base-uncased (NOT BERT-large) so everything
fits comfortably on one Colab T4.

Two entry points:
  embed(texts, batch_size)            -> pooled [CLS] embeddings, (N, H)
  embed_weighted(texts, term_weights) -> ([CLS] embeddings, rare-term vectors)

embed_weighted mirrors BERTidRAAF's "evidence injected AFTER contextualization"
idea: term-level weights are aligned to WordPiece tokens and used to pool the
*contextual* token states (weighted mean), rather than being fed in before BERT.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Tuple

import numpy as np
import torch


class EmbeddingEngine:
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        from transformers import AutoModel, AutoTokenizer  # lazy: keeps tests/imports light

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for WordPiece<->word alignment.")
        self.model = AutoModel.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model.to("cuda")
        else:
            self.device = "cpu"
            print("WARNING: CUDA not available - EmbeddingEngine is running on CPU. "
                  "This will be MUCH slower. In Colab: Runtime > Change runtime type > T4 GPU.")
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self._use_fp16 = self.device == "cuda"
        print(f"EmbeddingEngine: {model_name} on {self.device} "
              f"(hidden={self.hidden_size}, max_length={max_length}, fp16={self._use_fp16})")

    # ----------------------------------------------------------------- utils
    def _autocast(self):
        if self._use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _batches(self, texts: List[str], batch_size: int):
        # Length-sorted batching: far less padding -> faster, lower peak memory.
        order = np.argsort([len(t) for t in texts], kind="stable")
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            yield idx, [texts[i] for i in idx]

    def _encode(self, batch: List[str]):
        enc = self.tokenizer(batch, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        inputs = {k: enc[k].to(self.device) for k in ("input_ids", "attention_mask")}
        return enc, inputs

    # ------------------------------------------------------------------ API
    @torch.inference_mode()
    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Pooled [CLS] embeddings, shape (len(texts), hidden_size), float32.

        DistilBERT has no pooler layer, so "pooled [CLS]" = final-layer hidden
        state at position 0.
        """
        texts = [str(t) for t in texts]
        out = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
        for idx, batch in self._batches(texts, batch_size):
            _, inputs = self._encode(batch)
            with self._autocast():
                h = self.model(**inputs).last_hidden_state
            out[idx] = h[:, 0, :].float().cpu().numpy()
        return out

    def _split_long(self, texts: List[str], window: int = 480, max_chunks: int = 8):
        """Cut texts longer than max_length tokens into <=`window`-token pieces so nothing is
        silently truncated (Kaggle resumes are far longer than 512 tokens; FairCV texts are not)."""
        pieces, owner = [], []
        for i, t in enumerate(texts):
            if len(t) > 1800:  # cheap pre-check; ~4.5 chars/token
                offs = self.tokenizer(t, add_special_tokens=False, return_offsets_mapping=True,
                                      truncation=False, verbose=False)["offset_mapping"]
                n = len(offs)
                if n > self.max_length - 2:
                    for k in range(0, min(n, window * max_chunks), window):
                        pieces.append(t[offs[k][0]:offs[min(k + window, n) - 1][1]])
                        owner.append(i)
                    continue
            pieces.append(t)
            owner.append(i)
        return pieces, np.array(owner)

    def _word_weights(self, enc, batch, term_weights, oov_weight, L) -> torch.Tensor:
        """(B, L) CPU tensor: per-WordPiece weight of the word each token belongs to."""
        w = torch.zeros(len(batch), L, dtype=torch.float32)
        for b in range(len(batch)):
            cache: Dict[int, float] = {}
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is None:  # [CLS], [SEP], padding
                    continue
                if wid not in cache:
                    span = enc.word_to_chars(b, wid)
                    cache[wid] = term_weights.get(batch[b][span.start:span.end].lower(), oov_weight)
                w[b, pos] = cache[wid]
        return w

    def set_train(self, flag: bool):
        self.model.train(flag)

    def train_forward(self, batch: List[str], term_weights: Dict[str, float], oov_weight: float = 0.0):
        """Gradient-enabled forward for fine-tuning: (cls, rare), each (B, H) float32 on device."""
        enc, inputs = self._encode(batch)
        with self._autocast():
            h = self.model(**inputs).last_hidden_state
        h = h.float()
        w = self._word_weights(enc, batch, term_weights, oov_weight, h.shape[1]).to(h.device)
        rare = (w.unsqueeze(-1) * h).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return h[:, 0, :], rare

    @torch.inference_mode()
    def _embed_pieces(self, texts, term_weights, batch_size, oov_weight):
        n = len(texts)
        cls = np.zeros((n, self.hidden_size), dtype=np.float32)
        rare = np.zeros((n, self.hidden_size), dtype=np.float32)
        wsum = np.zeros(n, dtype=np.float32)
        for idx, batch in self._batches(texts, batch_size):
            enc, inputs = self._encode(batch)
            with self._autocast():
                h = self.model(**inputs).last_hidden_state.float()  # (B, L, H)
            w = self._word_weights(enc, batch, term_weights, oov_weight, h.shape[1])
            w = w.to(h.device)
            pooled = (w.unsqueeze(-1) * h).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            cls[idx] = h[:, 0, :].cpu().numpy()
            rare[idx] = pooled.cpu().numpy()
            wsum[idx] = w.sum(dim=1).cpu().numpy()
        return cls, rare, wsum

    def embed_weighted(self, texts: List[str], term_weights: Dict[str, float],
                       batch_size: int = 16, oov_weight: float = 0.0,
                       chunk_long: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (cls, rare), each (N, H).

        rare = sum_i w_i * h_i / sum_i w_i over real WordPiece tokens, where w_i is the weight of
        the word token i belongs to (term_weights, else oov_weight). All-zero weights -> zero vector.
        Texts longer than max_length are split into windows; cls = mean of window [CLS] vectors and
        rare = weight-sum-weighted mean of window rare vectors (i.e. one pooled average over ALL tokens).
        """
        texts = [str(t) for t in texts]
        n = len(texts)
        pieces, owner = self._split_long(texts) if chunk_long else (texts, np.arange(n))
        cls_p, rare_p, w_p = self._embed_pieces(pieces, term_weights, batch_size, oov_weight)
        if len(pieces) == n:
            return cls_p, rare_p
        counts = np.bincount(owner, minlength=n).astype(np.float32)
        cls = np.zeros((n, self.hidden_size), dtype=np.float32)
        rare = np.zeros((n, self.hidden_size), dtype=np.float32)
        wtot = np.zeros(n, dtype=np.float32)
        np.add.at(cls, owner, cls_p)
        np.add.at(rare, owner, rare_p * w_p[:, None])
        np.add.at(wtot, owner, w_p)
        return cls / counts[:, None], rare / np.maximum(wtot, 1e-8)[:, None]
