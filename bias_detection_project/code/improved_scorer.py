"""
improved_scorer.py  --  BERTidRAAF-lite fused scorer   (v2: v1 + two OPT-IN accuracy options)

v2 changes (both default OFF, so a v2 call with v1 arguments behaves exactly like v1):
  label_smoothing  : label smoothing on the 24-class Kaggle cross-entropy (Kaggle CE reaches ~0.09 on train,
                     i.e. the head memorises; smoothing regularises it).
  swa_last_epochs  : fine-tune path only. Average backbone+gate+head weights over the last K epochs
                     (checkpoint averaging). With a validation set the average is kept only if its mean
                     val accuracy is >= the best single epoch; otherwise the best epoch is restored as in v1.
No signature reordering: the two new arguments are appended after `seed`. Output columns are unchanged.

Pipeline
  1. Build IBS lookup from the TRAINING split only (no test leakage).
  2. Frozen DistilBERT -> per-text [CLS] vector (global) and IBS-weighted pooled
     token vector (rare-term). Features are extracted once and cached.
  3. Learned gate g = sigmoid(MLP([ctx ; rare])) in (0,1), per example:
         fused = g * ctx + (1 - g) * rare
  4. Heads on the fused vector: 24-class Kaggle head + one binary head PER FairCV
     target (score_blind / score_biased_gender / score_biased_ethnicity), matching
     the baseline's three separate FairCV evaluations. Gate + heads train jointly
     with a masked loss.

FairCV labels are binarized at per-target thresholds. Pass the BASELINE's
thresholds (faircv_thresholds=...) so both models predict identical labels;
otherwise the train-set median is used.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_engine import EmbeddingEngine
from ibs_calculator import IBSCalculator

# ---- schema assumptions (edit here if your columns differ) -------------------
KAGGLE_SOURCE_KEY = "kaggle"      # substring of source_dataset (case-insensitive)
FAIRCV_SOURCE_KEY = "faircv"
DEFAULT_FAIRCV_TARGETS = ["score_blind", "score_biased_gender", "score_biased_ethnicity"]
PASSTHROUGH_COLS = ["resume_id", "original_id", "intersectional_group"]  # = BaselineScorer._base_frame
# -----------------------------------------------------------------------------


class FusionGate(nn.Module):
    """Small feedforward gate: [ctx ; rare] -> fusion weight in (0, 1)."""

    def __init__(self, hidden: int, gate_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, ctx: torch.Tensor, rare: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ctx, rare], dim=-1))


class FusedClassifier(nn.Module):
    def __init__(self, hidden: int, n_classes: int, n_targets: int,
                 gate_hidden: int = 128, dropout: float = 0.1, aux_reg: bool = False):
        super().__init__()
        self.ctx_norm = nn.LayerNorm(hidden)    # put both streams on a comparable scale
        self.rare_norm = nn.LayerNorm(hidden)
        self.gate = FusionGate(hidden, gate_hidden)
        self.dropout = nn.Dropout(dropout)
        self.kaggle_head = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(512, n_classes))
        self.faircv_head = nn.Linear(hidden, max(n_targets, 1))
        self.faircv_reg = nn.Linear(hidden, max(n_targets, 1)) if aux_reg else None

    def fuse(self, ctx, rare):
        c, r = self.ctx_norm(ctx), self.rare_norm(rare)
        g = self.gate(c, r)                      # (B, 1)
        return g * c + (1.0 - g) * r, g

    def forward(self, ctx, rare):
        fused, g = self.fuse(ctx, rare)
        fused = self.dropout(fused)
        return self.kaggle_head(fused), self.faircv_head(fused), g.squeeze(-1)  # (B,C), (B,T), (B,)

    def forward_full(self, ctx, rare):
        """Like forward() plus the auxiliary regression output (B,T) on the z-scored FairCV scores."""
        fused, g = self.fuse(ctx, rare)
        fused = self.dropout(fused)
        return (self.kaggle_head(fused), self.faircv_head(fused), g.squeeze(-1),
                self.faircv_reg(fused) if self.faircv_reg is not None else None)


class ImprovedScorer:
    def __init__(self, engine: Optional[EmbeddingEngine] = None,
                 model_name: str = "distilbert-base-uncased",
                 weight_mode: str = "disparity",
                 faircv_targets: Optional[List[str]] = None,
                 faircv_thresholds: Optional[Dict[str, float]] = None,
                 epochs: int = 4, lr: float = 1e-3, train_batch_size: int = 64,
                 embed_batch_size: int = 16,
                 min_df: int = 5, max_vocab: int = 30000, oov_weight: Optional[float] = None,
                 restore_best: bool = True,
                 fine_tune: bool = False, backbone_lr: float = 3e-5,
                 ft_batch_faircv: int = 32, ft_batch_kaggle: int = 16,
                 chunk_long: Optional[bool] = None,
                 faircv_aux_reg: bool = False, aux_reg_weight: float = 1.0,
                 gate_hidden: int = 128, dropout: float = 0.1,
                 time_budget_minutes: float = 20.0, seed: int = 42,
                 label_smoothing: float = 0.0, swa_last_epochs: int = 0):
        """
        label_smoothing: smoothing for the Kaggle cross-entropy (0 = off, v1 behaviour)
        swa_last_epochs: average weights of the last K epochs (fine_tune=True only; 0 = off, v1 behaviour)
        weight_mode: "disparity" -> term weight = 1 - IBS ; "parity" -> term weight = IBS
        """
        if weight_mode not in ("disparity", "parity"):
            raise ValueError("weight_mode must be 'disparity' or 'parity'")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0 <= swa_last_epochs <= epochs:
            raise ValueError("swa_last_epochs must be between 0 and epochs")
        self.engine = engine
        self.model_name = model_name
        self.weight_mode = weight_mode
        self.faircv_targets_ = list(faircv_targets or DEFAULT_FAIRCV_TARGETS)
        self.thresholds_: Dict[str, float] = dict(faircv_thresholds or {})
        self.epochs, self.lr, self.train_batch_size = epochs, lr, train_batch_size
        self.embed_batch_size = embed_batch_size
        self.min_df, self.max_vocab = min_df, max_vocab
        # A term outside the IBS vocabulary has no evidence of disparity, i.e. IBS = 1 (neutral):
        # weight 1-IBS = 0 in 'disparity' mode, weight IBS = 1 in 'parity' mode.
        self.oov_weight = oov_weight if oov_weight is not None else (0.0 if weight_mode == 'disparity' else 1.0)
        self.restore_best = restore_best
        self.fine_tune, self.backbone_lr = fine_tune, backbone_lr
        self.ft_batch_faircv, self.ft_batch_kaggle = ft_batch_faircv, ft_batch_kaggle
        # window long texts only for the frozen-feature path; fine-tuning trains on the first 512 tokens
        self.chunk_long = (not fine_tune) if chunk_long is None else chunk_long
        self.faircv_aux_reg, self.aux_reg_weight = faircv_aux_reg, aux_reg_weight
        self.decision_thr_: Dict[str, float] = {}   # per-target probability cut-off; empty => 0.5
        self.gate_hidden, self.dropout, self.seed = gate_hidden, dropout, seed
        self.time_budget_s = time_budget_minutes * 60.0
        self.label_smoothing, self.swa_last_epochs = label_smoothing, swa_last_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ibs_calc = IBSCalculator()
        self.model: Optional[FusedClassifier] = None
        self.ibs_lookup_: Dict[str, float] = {}
        self.classes_: list = []
        self.history_: list = []
        self.fit_seconds_: float = 0.0

    # ------------------------------------------------------------- helpers
    def _get_engine(self):
        if self.engine is None:
            self.engine = EmbeddingEngine(self.model_name)
        return self.engine

    def _source_mask(self, df: pd.DataFrame, key: str) -> pd.Series:
        if "source_dataset" not in df.columns:
            return pd.Series(True, index=df.index)
        return df["source_dataset"].astype(str).str.lower().str.contains(key, regex=False)

    def kaggle_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self._source_mask(df, KAGGLE_SOURCE_KEY)]

    def faircv_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self._source_mask(df, FAIRCV_SOURCE_KEY)]

    def _term_weights(self, ibs_lookup: Dict[str, float]) -> Dict[str, float]:
        if self.weight_mode == "disparity":
            return {t: 1.0 - v for t, v in ibs_lookup.items()}
        return dict(ibs_lookup)

    def _require_fitted(self):
        if self.model is None:
            raise RuntimeError("ImprovedScorer is not fitted. Call fit() first.")

    def _set_thresholds(self, df: pd.DataFrame):
        src_f = self._source_mask(df, FAIRCV_SOURCE_KEY).values
        for t in self.faircv_targets_:
            if t in self.thresholds_:
                continue
            if t not in df.columns:
                if src_f.any():
                    raise ValueError(f"FairCV target column '{t}' not found in training data. "
                                     f"Columns: {list(df.columns)}")
                continue
            s = pd.to_numeric(df[t], errors="coerce").values
            self.thresholds_[t] = float(np.nanmedian(s[src_f])) if src_f.any() else 0.5
            print(f"NOTE: no threshold given for '{t}'; using train median {self.thresholds_[t]:.4f}. "
                  f"Pass faircv_thresholds=<baseline's> for an exact comparison.")

    def _build_targets(self, df: pd.DataFrame, learn_classes: bool = False):
        src_k = self._source_mask(df, KAGGLE_SOURCE_KEY).values
        src_f = self._source_mask(df, FAIRCV_SOURCE_KEY).values
        kmask = src_k & df["category_label"].notna().values
        if learn_classes:
            self.classes_ = sorted(df.loc[kmask, "category_label"].unique().tolist())
        idx = {c: i for i, c in enumerate(self.classes_)}
        yk = df["category_label"].map(idx).fillna(-1).astype(int).values
        kmask = kmask & (yk >= 0)
        T = len(self.faircv_targets_)
        yf = np.zeros((len(df), T), dtype=np.float32)
        mf = np.zeros((len(df), T), dtype=bool)
        for j, t in enumerate(self.faircv_targets_):
            if t in df.columns:
                s = pd.to_numeric(df[t], errors="coerce").values
                ok = src_f & ~np.isnan(s)
                mf[:, j] = ok
                yf[ok, j] = (s[ok] >= self.thresholds_[t]).astype(np.float32)
        return kmask, yk, yf, mf

    def faircv_binary(self, df: pd.DataFrame, target: str) -> np.ndarray:
        return (pd.to_numeric(df[target]).values >= self.thresholds_[target]).astype(int)

    def _extract_features(self, texts, ibs_lookup, verbose=True, desc="features"):
        """Chunked extraction with a running time projection (early slow-run warning)."""
        engine = self._get_engine()
        tw = self._term_weights(ibs_lookup)
        n, H = len(texts), engine.hidden_size
        ctx = np.zeros((n, H), dtype=np.float32)
        rare = np.zeros((n, H), dtype=np.float32)
        chunk = self.embed_batch_size * 8
        if hasattr(engine, "set_train"):
            engine.set_train(False)
        order = np.random.RandomState(0).permutation(n)   # random order => representative speed estimate
        t0, warned = time.time(), False
        for i, s in enumerate(range(0, n, chunk)):
            sel = order[s:s + chunk]
            c, r = engine.embed_weighted([texts[j] for j in sel], tw,
                                         batch_size=self.embed_batch_size,
                                         oov_weight=self.oov_weight, chunk_long=self.chunk_long)
            ctx[sel], rare[sel] = c, r
            done = min(s + chunk, n)
            elapsed = time.time() - t0
            if verbose and (i % 10 == 0 or done == n):
                print(f"  [{desc}] {done}/{n} texts  ({done / max(elapsed, 1e-9):.1f} texts/s, "
                      f"{elapsed / 60:.1f} min elapsed)")
            projected = elapsed / done * n
            if not warned and done >= min(n, 3 * chunk) and projected > self.time_budget_s:
                warned = True
                print(f"\n*** WARNING: {desc} projected to take {projected / 60:.0f} min "
                      f"(> {self.time_budget_s / 60:.0f} min budget) at {done / elapsed:.1f} texts/s. "
                      f"This suggests a device/batching problem, not just 'wait longer'. "
                      f"Check: device={getattr(engine, 'device', '?')} (should be 'cuda'), "
                      f"embed_batch_size={self.embed_batch_size}, "
                      f"max_length={getattr(engine, 'max_length', '?')}. ***\n")
        return ctx, rare

    # ----------------------------------------------------------------- fit
    def fit(self, train_df: pd.DataFrame, text_col: str = "counterfactual_text",
            val_df: Optional[pd.DataFrame] = None):
        """val_df (optional) is only REPORTED each epoch; it is not used for model selection."""
        t_start = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        df = train_df.reset_index(drop=True)
        texts = df[text_col].astype(str).tolist()

        # 1) IBS lookup from training data only
        groups = defaultdict(list)
        for g, t in zip(df["intersectional_group"], texts):
            groups[g].append(t)
        vocab = self.ibs_calc.build_vocab(texts, self.min_df, self.max_vocab)
        self.ibs_lookup_ = self.ibs_calc.build_ibs_lookup(groups, vocab)
        vals = np.array(list(self.ibs_lookup_.values()))
        print(f"IBS lookup: {len(groups)} groups, {len(vocab)} terms | "
              f"IBS<1: {(vals < 1).sum()} terms, IBS==0: {(vals == 0).sum()} terms, "
              f"mean IBS {vals.mean():.3f}")

        # 2) labels / masks
        self._set_thresholds(df)
        kmask, yk, yf, mf = self._build_targets(df, learn_classes=True)
        if not kmask.any() and not mf.any():
            seen = df["source_dataset"].unique()[:10] if "source_dataset" in df else "n/a"
            raise ValueError(f"No Kaggle/FairCV rows matched. source_dataset values: {seen}. "
                             f"Adjust KAGGLE_SOURCE_KEY / FAIRCV_SOURCE_KEY.")
        if kmask.any() and len(self.classes_) != 24:
            print(f"NOTE: expected 24 Kaggle classes, found {len(self.classes_)}.")
        print(f"Training rows: Kaggle={int(kmask.sum())}  "
              + "  ".join(f"{t}={int(mf[:, j].sum())}" for j, t in enumerate(self.faircv_targets_))
              + f"  device={self.device}")
        print(f"FairCV thresholds: { {t: round(v, 4) for t, v in self.thresholds_.items()} }")

        if self.fine_tune:
            return self._fit_finetune(df, texts, kmask, yk, yf, mf, val_df, text_col, t_start)

        # 3) features (frozen backbone, extracted once)
        print("Extracting frozen-backbone features ...")
        ctx, rare = self._extract_features(texts, self.ibs_lookup_, desc="train")
        dev = self.device
        ctx_t, rare_t = torch.tensor(ctx, device=dev), torch.tensor(rare, device=dev)
        yk_t = torch.tensor(yk, dtype=torch.long, device=dev)
        yf_t = torch.tensor(yf, device=dev)
        mk_t, mf_t = torch.tensor(kmask, device=dev), torch.tensor(mf, device=dev)

        val = None
        if val_df is not None and len(val_df):
            vdf = val_df.reset_index(drop=True)
            vk, vyk, vyf, vmf = self._build_targets(vdf)
            vc, vr = self._extract_features(vdf[text_col].astype(str).tolist(),
                                            self.ibs_lookup_, desc="val")
            val = tuple(torch.tensor(a, device=dev) for a in (vc, vr, vyk, vk, vyf, vmf))

        # 4) train gate + heads
        self.model = FusedClassifier(ctx.shape[1], max(len(self.classes_), 1),
                                     len(self.faircv_targets_), self.gate_hidden,
                                     self.dropout).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        n, bs = len(df), self.train_batch_size
        n_k, n_f = max(int(kmask.sum()), 1), max(int(mf.sum()), 1)
        self.history_ = []
        best_score, best_epoch, best_state = -1.0, 0, None
        for epoch in range(1, self.epochs + 1):
            t_ep = time.time()
            self.model.train()
            perm = torch.randperm(n, device=dev)
            lk_sum = lf_sum = g_sum = torch.zeros((), device=dev)
            for s in range(0, n, bs):
                ix = perm[s:s + bs]
                lk, lf, g = self.model(ctx_t[ix], rare_t[ix])
                mk, mfb = mk_t[ix], mf_t[ix]
                loss_k = (F.cross_entropy(lk[mk], yk_t[ix][mk], reduction="sum",
                                          label_smoothing=self.label_smoothing)
                          if mk.any() else lk.sum() * 0.0)
                loss_f = (F.binary_cross_entropy_with_logits(lf, yf_t[ix], reduction="none")
                          * mfb).sum()
                loss = (loss_k + loss_f) / len(ix)
                opt.zero_grad()
                loss.backward()
                opt.step()
                lk_sum = lk_sum + loss_k.detach()
                lf_sum = lf_sum + loss_f.detach()
                g_sum = g_sum + g.detach().sum()
            rec = {"epoch": epoch,
                   "loss": float((lk_sum + lf_sum) / n),
                   "kaggle_ce": float(lk_sum / n_k),
                   "faircv_bce": float(lf_sum / n_f),
                   "mean_gate": float(g_sum / n)}
            msg = (f"Epoch {epoch}/{self.epochs} | loss {rec['loss']:.4f} | "
                   f"kaggle CE {rec['kaggle_ce']:.4f} | faircv BCE {rec['faircv_bce']:.4f} | "
                   f"mean gate {rec['mean_gate']:.3f}")
            if val is not None:
                rec.update(self._val_metrics(*val))
                sc = float(np.nanmean([rec['val_kaggle_acc'], rec['val_faircv_acc']]))
                if self.restore_best and sc > best_score:
                    best_score, best_epoch = sc, epoch
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                msg += (f" | val kaggle acc {rec['val_kaggle_acc']:.4f} | "
                        f"val faircv acc {rec['val_faircv_acc']:.4f}")
            self.history_.append(rec)
            print(msg + f" | {time.time() - t_ep:.1f}s")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best epoch {best_epoch} (mean val acc {best_score:.4f}).")
        self._finish(t_start)
        return self

    def _finish(self, t_start):
        losses = [h["loss"] for h in self.history_]
        if not np.all(np.isfinite(losses)):
            print("*** WARNING: non-finite loss - training diverged. ***")
        elif len(losses) >= 2 and losses[-1] > 0.95 * losses[0]:
            print("*** WARNING: loss barely decreased - training may be stalled "
                  "(check labels, lr, feature scale). ***")
        gm = self.history_[-1]["mean_gate"]
        if gm < 0.02 or gm > 0.98:
            print(f"NOTE: gate has collapsed to one stream (mean gate {gm:.3f}); "
                  f"the fusion is effectively ignoring one representation.")

        self.fit_seconds_ = time.time() - t_start
        if self.fit_seconds_ > self.time_budget_s:
            print(f"*** WARNING: fit took {self.fit_seconds_ / 60:.1f} min "
                  f"(> {self.time_budget_s / 60:.0f} min). Check GPU is active, "
                  f"embed_batch_size, and max_length. ***")

    def _fit_finetune(self, df, texts, kmask, yk, yf, mf, val_df, text_col, t_start):
        """End-to-end: backbone + gate + heads. Each epoch draws ONE random name variant per profile
        (counterfactual augmentation, and ~4x cheaper than all variants). Batches are single-task so
        short FairCV texts are not padded to Kaggle length."""
        from transformers import get_linear_schedule_with_warmup
        engine, dev = self._get_engine(), self.device
        tw = self._term_weights(self.ibs_lookup_)
        oid = (df["original_id"] if "original_id" in df.columns else df["resume_id"]).values

        def groups(mask):
            g = defaultdict(list)
            for i in np.flatnonzero(mask):
                g[oid[i]].append(i)
            return [np.array(v) for v in g.values()]
        gk, gf = groups(kmask), groups(mf.any(axis=1))
        rng = np.random.RandomState(self.seed)

        def plan():
            batches = []
            for task, grp, bs in (("k", gk, self.ft_batch_kaggle), ("f", gf, self.ft_batch_faircv)):
                if not grp:
                    continue
                pick = np.array([g[rng.randint(len(g))] for g in grp])
                rng.shuffle(pick)
                batches += [(task, pick[i:i + bs]) for i in range(0, len(pick), bs)]
            rng.shuffle(batches)
            return batches

        self.model = FusedClassifier(engine.hidden_size, max(len(self.classes_), 1),
                                     len(self.faircv_targets_), self.gate_hidden, self.dropout,
                                     aux_reg=self.faircv_aux_reg).to(dev)
        yk_t = torch.tensor(yk, dtype=torch.long, device=dev)
        yf_t = torch.tensor(yf, device=dev)
        mf_t = torch.tensor(mf, device=dev)
        zs_t = None
        if self.faircv_aux_reg:   # z-scored continuous scores (TRAIN rows only) as extra supervision
            S = np.stack([pd.to_numeric(df[t], errors='coerce').values if t in df.columns
                          else np.full(len(df), np.nan) for t in self.faircv_targets_], axis=1)
            Sm = np.where(mf, S, np.nan)
            mu, sd = np.nanmean(Sm, axis=0), np.nanstd(Sm, axis=0) + 1e-8
            zs_t = torch.tensor(np.nan_to_num((S - mu) / sd), dtype=torch.float32, device=dev)

        vdf = None
        if val_df is not None and len(val_df):
            vdf = val_df.reset_index(drop=True)
            vk, vyk, vyf, vmf = self._build_targets(vdf)
            vtexts = vdf[text_col].astype(str).tolist()

        bb = list(engine.model.parameters())
        hd = list(self.model.parameters())
        opt = torch.optim.AdamW([{"params": bb, "lr": self.backbone_lr, "weight_decay": 0.01},
                                 {"params": hd, "lr": self.lr, "weight_decay": 1e-2}])
        steps_per_epoch = len(plan())
        total = steps_per_epoch * self.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
        print(f"Fine-tuning backbone (lr {self.backbone_lr}) + gate/heads (lr {self.lr}): "
              f"{steps_per_epoch} steps/epoch x {self.epochs} epochs")

        self.history_ = []
        best_score, best_epoch, best_state = -1.0, 0, None
        swa, swa_n = None, 0          # running mean of (head, backbone) weights over the last K epochs
        for epoch in range(1, self.epochs + 1):
            t_ep = time.time()
            engine.set_train(True)
            self.model.train()
            sums, cnt, g_sum, g_n = {"k": 0.0, "f": 0.0}, {"k": 0, "f": 0}, 0.0, 0
            for task, ix in plan():
                batch = [texts[i] for i in ix]
                ixt = torch.as_tensor(ix, device=dev)
                ctx, rare = engine.train_forward(batch, tw, self.oov_weight)
                if self.faircv_aux_reg and task == "f":
                    lk, lf, g, reg = self.model.forward_full(ctx, rare)
                else:
                    lk, lf, g = self.model(ctx, rare)
                    reg = None
                if task == "k":
                    loss = F.cross_entropy(lk, yk_t[ixt], label_smoothing=self.label_smoothing)
                else:
                    m = mf_t[ixt]
                    loss = (F.binary_cross_entropy_with_logits(lf, yf_t[ixt], reduction="none") * m).sum() \
                        / m.sum().clamp_min(1)
                    if reg is not None:
                        loss = loss + self.aux_reg_weight * (((reg - zs_t[ixt]) ** 2) * m).sum() \
                            / m.sum().clamp_min(1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(bb + hd, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                sums[task] += loss.item()
                cnt[task] += 1
                g_sum += g.detach().sum().item()
                g_n += len(ix)
            rec = {"epoch": epoch,
                   "kaggle_ce": sums["k"] / max(cnt["k"], 1),
                   "faircv_bce": sums["f"] / max(cnt["f"], 1),
                   "loss": (sums["k"] + sums["f"]) / max(cnt["k"] + cnt["f"], 1),
                   "mean_gate": g_sum / max(g_n, 1)}
            msg = (f"Epoch {epoch}/{self.epochs} | loss {rec['loss']:.4f} | kaggle CE {rec['kaggle_ce']:.4f} | "
                   f"faircv BCE {rec['faircv_bce']:.4f} | mean gate {rec['mean_gate']:.3f}")
            if vdf is not None:
                vc, vr = self._extract_features(vtexts, self.ibs_lookup_, verbose=False, desc="val")
                rec.update(self._val_metrics(*(torch.tensor(a_, device=dev)
                                               for a_ in (vc, vr, vyk, vk, vyf, vmf))))
                msg += (f" | val kaggle acc {rec['val_kaggle_acc']:.4f} | val faircv acc {rec['val_faircv_acc']:.4f}")
                sc = float(np.nanmean([rec["val_kaggle_acc"], rec["val_faircv_acc"]]))
                if self.restore_best and sc > best_score:
                    best_score, best_epoch = sc, epoch
                    best_state = ({k_: v_.detach().clone() for k_, v_ in self.model.state_dict().items()},
                                  {k_: v_.detach().clone() for k_, v_ in engine.model.state_dict().items()})
            if self.swa_last_epochs > 0 and epoch > self.epochs - self.swa_last_epochs:
                swa = self._swa_accumulate(swa, (self.model, engine.model), swa_n)
                swa_n += 1
            self.history_.append(rec)
            ep_s = time.time() - t_ep
            print(msg + f" | {ep_s / 60:.1f} min")
            if epoch == 1 and ep_s * self.epochs > self.time_budget_s:
                print(f"*** WARNING: at {ep_s / 60:.1f} min/epoch, {self.epochs} epochs would take "
                      f"{ep_s * self.epochs / 60:.0f} min (> {self.time_budget_s / 60:.0f} budget). "
                      f"Lower epochs or ft_batch sizes/max_length. ***")
        swa_used = False
        if swa is not None:
            self.model.load_state_dict(swa[0])
            engine.model.load_state_dict(swa[1])
            swa_used = True       # no validation / restore_best off: the average replaces the last epoch
            if vdf is not None and best_state is not None:
                swa_sc = self._ft_val_score(vtexts, vk, vyk, vyf, vmf)
                swa_used = swa_sc >= best_score
                print(f"Weight average of last {swa_n} epochs: mean val acc {swa_sc:.4f} vs best epoch "
                      f"{best_epoch} {best_score:.4f} -> {'using the average' if swa_used else 'keeping best epoch'}.")
            del swa
        if best_state is not None and not swa_used:
            self.model.load_state_dict(best_state[0])
            engine.model.load_state_dict(best_state[1])
            print(f"Restored best epoch {best_epoch} (mean val acc {best_score:.4f}).")
        engine.set_train(False)
        self.model.eval()
        self._finish(t_start)
        return self

    @staticmethod
    def _swa_accumulate(acc, modules, n):
        """Running mean over the floating-point tensors of each module's state_dict (n = #already averaged).
        Non-float buffers (e.g. position ids) are simply carried over."""
        states = [m.state_dict() for m in modules]
        if acc is None:
            return [{k: (v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
                     for k, v in sd.items()} for sd in states]
        for a, sd in zip(acc, states):
            for k, v in sd.items():
                if v.is_floating_point():
                    a[k].mul_(n).add_(v.detach().float()).div_(n + 1)
                else:
                    a[k] = v.detach().clone()
        return acc

    @torch.no_grad()
    def _ft_val_score(self, vtexts, vk, vyk, vyf, vmf) -> float:
        """Mean of val Kaggle / FairCV accuracy for the CURRENT weights (same criterion as epoch selection)."""
        vc, vr = self._extract_features(vtexts, self.ibs_lookup_, verbose=False, desc="val")
        m = self._val_metrics(*(torch.tensor(a_, device=self.device) for a_ in (vc, vr, vyk, vk, vyf, vmf)))
        return float(np.nanmean([m["val_kaggle_acc"], m["val_faircv_acc"]]))

    @torch.no_grad()
    def _val_metrics(self, vc, vr, vyk, vk, vyf, vmf):
        self.model.eval()
        lk, lf, _ = self.model(vc, vr)
        acc_k = float((lk.argmax(-1)[vk] == vyk[vk]).float().mean()) if vk.any() else float("nan")
        acc_f = float((((lf > 0).float() == vyf)[vmf]).float().mean()) if vmf.any() else float("nan")
        return {"val_kaggle_acc": acc_k, "val_faircv_acc": acc_f}

    # ------------------------------------------------------------- predict
    @torch.no_grad()
    def _forward(self, df: pd.DataFrame, text_col: str):
        self._require_fitted()
        texts = df[text_col].astype(str).tolist()
        ctx, rare = self._extract_features(texts, self.ibs_lookup_, desc="predict")
        self.model.eval()
        outs = []
        for s in range(0, len(texts), 256):
            c = torch.tensor(ctx[s:s + 256], device=self.device)
            r = torch.tensor(rare[s:s + 256], device=self.device)
            lk, lf, g = self.model(c, r)
            outs.append((lk.cpu(), lf.cpu(), g.cpu()))
        T = len(self.faircv_targets_)
        lk = torch.cat([o[0] for o in outs]) if outs else torch.zeros(0, 1)
        lf = torch.cat([o[1] for o in outs]) if outs else torch.zeros(0, T)
        g = torch.cat([o[2] for o in outs]) if outs else torch.zeros(0)
        return lk, lf, g

    def _base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in PASSTHROUGH_COLS if c in df.columns]
        return df[cols].reset_index(drop=True).copy()

    def predict_kaggle(self, df: pd.DataFrame, text_col: str = "counterfactual_text") -> pd.DataFrame:
        """Same columns as BaselineScorer.predict_kaggle:
        resume_id, original_id, intersectional_group, true_label, predicted_label, confidence
        (+ fusion_gate, extra)."""
        if not self.classes_:
            raise RuntimeError("No Kaggle classes were learned (no Kaggle rows in training data).")
        lk, _, g = self._forward(df, text_col)
        conf, pred = F.softmax(lk, dim=-1).max(dim=-1)
        out = self._base_frame(df)
        out["true_label"] = df["category_label"].to_numpy()
        out["predicted_label"] = [self.classes_[i] for i in pred.tolist()]
        out["confidence"] = conf.numpy()
        out["fusion_gate"] = g.numpy()
        return out

    def predict_faircv(self, df: pd.DataFrame, target: Optional[str] = None,
                       text_col: str = "counterfactual_text") -> pd.DataFrame:
        """Same columns as BaselineScorer.predict_faircv, with a leading 'target' column:
        target, resume_id, original_id, intersectional_group, true_label, predicted_label,
        confidence, prob_positive (+ fusion_gate, extra).

        target=None -> all targets stacked in long format (ONE forward pass), which is
        how the baseline's saved predictions are laid out once 'dataset' is inserted.
        """
        _, lf, g = self._forward(df, text_col)
        targets = self.faircv_targets_ if target is None else [target]
        frames = []
        for t in targets:
            j = self.faircv_targets_.index(t)
            p = 1.0 / (1.0 + np.exp(-lf[:, j].numpy().astype(np.float64)))
            out = self._base_frame(df)
            if t in df.columns:
                s = pd.to_numeric(df[t], errors="coerce").reset_index(drop=True)
                lab = (s >= self.thresholds_[t]).astype("Int64")
                lab[s.isna()] = pd.NA
            else:
                lab = pd.array([pd.NA] * len(df), dtype="Int64")
            out["true_label"] = lab
            out["predicted_label"] = (p >= self.decision_thr_.get(t, 0.5)).astype(int)
            out["confidence"] = np.maximum(p, 1.0 - p)
            out["prob_positive"] = p
            out["fusion_gate"] = g.numpy()
            out.insert(0, "target", t)
            frames.append(out)
        return pd.concat(frames, ignore_index=True)

    def calibrate_decision_thresholds(self, val_df: pd.DataFrame, text_col: str = "counterfactual_text",
                                      grid=None) -> Dict[str, float]:
        """Pick, per FairCV target, the probability cut-off that maximises accuracy on the VALIDATION
        split (never the test split). Ties go to the value closest to 0.5."""
        grid = np.round(np.arange(0.30, 0.701, 0.02), 2) if grid is None else grid
        self.decision_thr_ = {}
        out = self.predict_faircv(val_df, text_col=text_col)
        for t in self.faircv_targets_:
            o = out[out["target"] == t]
            ok = o["true_label"].notna().values
            if not ok.any():
                continue
            y, p = o["true_label"].astype(int).values[ok], o["prob_positive"].values[ok]
            accs = np.array([((p >= c).astype(int) == y).mean() for c in grid])
            best = accs.max()
            cands = grid[accs >= best - 1e-12]
            self.decision_thr_[t] = float(cands[np.argmin(np.abs(cands - 0.5))])
            print(f"  calibrated cut-off for {t}: {self.decision_thr_[t]:.2f} "
                  f"(val acc {accs[list(grid).index(0.5)] if 0.5 in grid else float('nan'):.4f} at 0.50 -> {best:.4f})")
        return self.decision_thr_

    @torch.no_grad()
    def fair_representation(self, text: str, ibs_lookup: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Gate-fused representation (hidden_size,) for one text."""
        self._require_fitted()
        lookup = self.ibs_lookup_ if ibs_lookup is None else ibs_lookup
        ctx, rare = self._extract_features([text], lookup, verbose=False)
        self.model.eval()
        fused, _ = self.model.fuse(torch.tensor(ctx, device=self.device),
                                   torch.tensor(rare, device=self.device))
        return fused[0].cpu().numpy()


# Suggested opt-in settings for the two v2 options (pass to the improved AND, for a fair comparison, the control).
RECOMMENDED_V2_KWARGS = dict(label_smoothing=0.1, swa_last_epochs=3)
