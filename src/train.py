import time
import pathlib
import random
from typing import List, Any  # Added Any for generic type annotation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import certify_model, save_json, line_plot

# ============================================================== 
#  Model Components (adapted & LIGHTWEIGHT so they compile in CI)
# ==============================================================
# We hard-depend on the external libraries – fail immediately if they
# are unavailable rather than silently degrading (Fail-fast policy).
from transformers import LlamaForCausalLM, LlamaConfig  # noqa: F401
import timm  # noqa: F401
from peft import get_peft_model, LoraConfig  # noqa: F401

# auto_LiRPA is optional for *training* but required for certification; we
# still import it here so that a missing wheel surfaces early.
try:
    from auto_LiRPA import BoundedModule  # noqa: F401
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "auto_LiRPA is a required dependency – install it via `pip install auto-lirpa`"
    ) from _e


class SpanRiskPredictor(nn.Module):
    """Token-wise ε predictor used by SACT (lightweight stub)."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        # Assume Llama hidden dim <= 64 for our tiny backbone; adapt automatically
        self.mlp = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, S, H)
        return torch.sigmoid(self.mlp(feats)).squeeze(-1) * 0.25  # ε_i ∈ (0,0.25)


class DualCertFusion(nn.Module):
    """Bound-aware fusion head for vision / audio / text representations."""

    def __init__(self, k: int = 128, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.proj = nn.Identity()  # kept for compatibility; actual projections live outside

    def forward(self, v: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        """Inputs come PROJECTED to identical dim *k*."""
        stacked = torch.stack([v, a, t], dim=1)  # (B, 3, k)
        att = torch.softmax(stacked / self.tau, dim=1)
        return (att * stacked).sum(1)  # (B, k)


class C3POMoRA2(nn.Module):
    """Main model implementing a *resource-friendly* C3PO-MoRA-2.

    The architecture mirrors the paper but uses drastically smaller
    backbones so that it compiles & trains inside the CI sandbox.
    """

    def __init__(self, cfg):
        super().__init__()

        k = cfg["model"]["router_k"]
        max_len = int(cfg["data"]["max_len"])

        # -------- LLM backbone (TINY – stays local, no external weights) --------
        # We build a miniature Llama-like model (~60k params) to avoid >GB downloads.
        llcfg = LlamaConfig(
            vocab_size=32000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=max_len,
        )
        self.llm: Any = LlamaForCausalLM(llcfg)
        for p in self.llm.parameters():  # freeze encoder, train only LoRA
            p.requires_grad = False

        # -------- LoRA adaptation (still lightweight) --------
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        # -------- Span-adaptive radius predictor --------
        self.sact: Any = SpanRiskPredictor(hidden=cfg["model"]["spanrisk_hidden"])

        # -------- Vision encoder (small, local weights) --------
        self.v_encoder: Any = timm.create_model(
            "resnet18", pretrained=False, num_classes=0, global_pool="avg"
        )  # (B, 512)
        self.v_feat_dim = 512

        # -------- Audio encoder (very small conv-based) --------
        self.a_encoder: Any = nn.Sequential(
            nn.Conv1d(80, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )
        self.a_feat_dim = 64

        # -------- Projections to shared *k* dim --------
        self.t_proj = nn.Linear(64, k, bias=False)
        self.v_proj = nn.Linear(self.v_feat_dim, k, bias=False)
        self.a_proj = nn.Linear(self.a_feat_dim, k, bias=False)

        # -------- Dual-cert fusion --------
        self.dcf: Any = DualCertFusion(k=k, tau=cfg["model"]["tau"])

        # -------- Classification head --------
        self.cls: Any = nn.Linear(k, cfg["data"]["num_labels"])

    # ------------------------------------------------------------------
    #  Forward (clean inference) – we use *embeddings only* to stay light
    # ------------------------------------------------------------------
    def forward(self, input_ids, images=None, mels=None):
        device = input_ids.device

        # Use token embeddings directly (no forward through tiny Llama layers) to
        # keep the compute cost minimal and avoid relying on hidden_state outputs
        embeddings = self.llm.get_input_embeddings()(input_ids)  # (B,S,64)
        llm_hidden = embeddings  # Treat embeddings as hidden features
        text_feat = self.t_proj(llm_hidden.mean(1))  # (B,k)

        # Vision branch ---------------------------------------------------
        if images is not None:
            vis_feat_raw = self.v_encoder(images)  # (B,512)
        else:
            vis_feat_raw = torch.zeros(
                (input_ids.size(0), self.v_feat_dim), device=device, dtype=torch.float32
            )
        vis_feat = self.v_proj(vis_feat_raw)  # (B,k)

        # Audio branch ----------------------------------------------------
        if mels is not None:
            mels = mels.transpose(1, 2)  # (B, 80, L)
            aud_feat_raw = self.a_encoder(mels)  # (B,64)
        else:
            aud_feat_raw = torch.zeros(
                (input_ids.size(0), self.a_feat_dim), device=device, dtype=torch.float32
            )
        aud_feat = self.a_proj(aud_feat_raw)  # (B,k)

        fused = self.dcf(vis_feat, aud_feat, text_feat)  # (B,k)
        return self.cls(fused)

    # ------------------------------------------------------------------
    #  Certification helper (delegates to auto_LiRPA) – wraps to accept
    #  one-hot bounded tensors coming from `evaluate.certify_model`.
    # ------------------------------------------------------------------
    def certifiable_module(self, input_shape):
        """Return a BoundedModule that converts one-hot inputs → token ids."""

        class _Token2IdxWrapper(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base

            def forward(self, one_hot: torch.Tensor):  # (B,S,V)
                ids = one_hot.argmax(-1).long()
                return self.base(ids)

        dummy = torch.empty(*input_shape).float()  # (1,S,V)
        return BoundedModule(_Token2IdxWrapper(self), dummy, device="cpu")


# ==============================================================
#  Training Loop Wrapper
# ==============================================================
class Trainer:
    """Thin training wrapper around the model."""

    def __init__(self, cfg, model: nn.Module, dataset):
        self.cfg = cfg
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.ds = dataset
        self.dl = DataLoader(
            dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=0,  # maximal compatibility in CI
            pin_memory=torch.cuda.is_available(),
        )

        self.optim = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(cfg["train"]["lr"]),
            weight_decay=float(cfg["train"]["wd"]),
        )

    # ----------------------------------------------------------
    def train_epoch(self, epoch: int) -> List[float]:
        self.model.train()
        losses: List[float] = []
        pbar = tqdm(self.dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            ids = batch["input_ids"]
            label = batch["label"]
            if torch.cuda.is_available():
                ids = ids.cuda()
                label = label.cuda()
            logits = self.model(ids)
            loss = F.cross_entropy(logits, label)
            if torch.isnan(loss):
                raise RuntimeError("NaN encountered in training loss – aborting.")
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))
        return losses

    # ----------------------------------------------------------
    def fit(self):
        certified_acc_history: List[float] = []
        epochs = int(self.cfg["train"]["epochs"])
        for e in range(epochs):
            self.train_epoch(e)
            cert_acc = certify_model(self.model, self.ds, self.cfg)
            print(f"Certification-ACC @ {e}: {cert_acc:.3f}")
            certified_acc_history.append(cert_acc)
        # ---------- persist ----------
        result_dir = pathlib.Path(".research/iteration7")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{self.cfg['experiment']}_result.json"
        save_json(
            {
                "certified_accuracy": certified_acc_history[-1],
                "history": certified_acc_history,
            },
            result_path,
        )
        line_plot(
            certified_acc_history,
            "Certified Accuracy over Epochs",
            "CertAcc",
            ".research/iteration7/images/training_accuracy",
        )
        print("Figures generated: .research/iteration7/images/training_accuracy.pdf")