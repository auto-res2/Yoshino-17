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
#  Model Components (adapted unchanged from original script)
# ==============================================================
try:
    from transformers import LlamaForCausalLM, LlamaConfig, Wav2Vec2Model
    import timm
    from peft import get_peft_model, LoraConfig
    from auto_LiRPA import BoundedModule
except Exception as e:
    # Delay heavy import errors until model construction
    LlamaForCausalLM = LlamaConfig = Wav2Vec2Model = timm = get_peft_model = LoraConfig = BoundedModule = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class SpanRiskPredictor(nn.Module):
    """Token-wise ε predictor used by SACT."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, S, 512)
        return torch.sigmoid(self.mlp(feats)).squeeze(-1) * 0.25  # ε_i ∈ (0,0.25)


class DualCertFusion(nn.Module):
    """Bound-aware fusion head for vision / audio / text representations."""

    def __init__(self, d_model: int = 4096, k: int = 128, tau: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, k, bias=False)
        self.tau = tau

    def forward(self, v: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Inputs come PROJECTED to same dimension k."""
        stacked = torch.stack([v, a, t], dim=1)  # (B, 3, k)
        att = torch.softmax(stacked / self.tau, dim=1)
        return (att * stacked).sum(1)  # (B, k)


class C3POMoRA2(nn.Module):
    """Main model implementing C3PO-MoRA-2 (text + vision + audio)."""

    def __init__(self, cfg):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Required libraries for the full model could not be imported: "
                f"{_IMPORT_ERROR}"
            )
        super().__init__()
        # -------- LLM backbone (frozen) --------
        llcfg = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")  # noqa: F841
        self.llm: Any = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16
        )
        for p in self.llm.parameters():
            p.requires_grad = False
        # -------- LoRA adaptation --------
        lora_cfg = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        # -------- Span-adaptive radius predictor --------
        self.sact: Any = SpanRiskPredictor(hidden=cfg["model"]["spanrisk_hidden"])
        # -------- Dual-cert fusion (vision & audio) --------
        self.dcf: Any = DualCertFusion(
            k=cfg["model"]["router_k"], tau=cfg["model"]["tau"]
        )
        # Vision & audio encoders (frozen) --------
        self.v_encoder: Any = timm.create_model("vit_base_patch16_224", pretrained=True)
        from transformers import Wav2Vec2Model as _Wav2Vec2Model

        self.a_encoder: Any = _Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        for p in list(self.v_encoder.parameters()) + list(self.a_encoder.parameters()):
            p.requires_grad = False
        # Classification head --------
        self.cls: Any = nn.Linear(cfg["model"]["router_k"], cfg["data"]["num_labels"])

    # ------------------------------------------------------------------
    #  Forward (clean inference)
    # ------------------------------------------------------------------
    def forward(self, input_ids, images=None, mels=None):
        llm_out = self.llm(input_ids).last_hidden_state  # (B,S,H)
        text_feat = llm_out.mean(1)
        if images is not None:
            vis_feat = self.v_encoder.forward_features(images)
        else:
            vis_feat = torch.zeros_like(text_feat[:, :512])
        if mels is not None:
            aud_feat = self.a_encoder(mels.transpose(1, 2)).last_hidden_state.mean(1)
        else:
            aud_feat = torch.zeros_like(text_feat[:, :512])
        fused = self.dcf(vis_feat, aud_feat, text_feat)
        return self.cls(fused)

    # ------------------------------------------------------------------
    #  Certification helper
    # ------------------------------------------------------------------
    def certifiable_module(self, input_shape):
        if BoundedModule is None:
            raise RuntimeError("auto_LiRPA is required for certification.")
        return BoundedModule(self, torch.empty(*input_shape).long(), device="cuda")


# ==============================================================
#  Training Loop Wrapper
# ==============================================================
class Trainer:
    """Thin training wrapper around the model."""

    def __init__(self, cfg, model: nn.Module, dataset):
        self.cfg = cfg
        # Avoid half precision for the minimalist fallback model – it causes
        # instability / NaNs on some GPUs.  The full model uses FP16 anyway via
        # its internal weights, so the outer cast is unnecessary.
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.ds = dataset
        self.dl = DataLoader(
            dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=0,  # maximal compatibility in CI
            pin_memory=torch.cuda.is_available(),
        )

        # Ensure learning-rate and weight-decay are numeric (yaml may load them as strings)
        lr = float(cfg["train"]["lr"])
        wd = float(cfg["train"]["wd"])
        self.optim = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=wd,
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
        result_dir = pathlib.Path(".research/iteration4")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{self.cfg['experiment']}_result.json"
        save_json(
            {"certified_accuracy": certified_acc_history[-1], "history": certified_acc_history},
            result_path,
        )
        line_plot(
            certified_acc_history,
            "Certified Accuracy over Epochs",
            "CertAcc",
            ".research/iteration4/images/training_accuracy",
        )
        print("Figures generated: .research/iteration4/images/training_accuracy.pdf")
