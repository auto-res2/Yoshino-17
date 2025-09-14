"""src/train.py
Model and guardrail definitions along with a tiny factory helper.
This file is extracted verbatim from the original monolithic script but
cleaned-up to remove stale `# type: ignore` markers that were no longer needed
by the static checker.
"""
from __future__ import annotations

import math
from typing import Dict, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (kept for completeness)
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

################################################################################
# Baseline guardrails ###########################################################
################################################################################


class PassThrough:
    """No defence at all – simply returns the conversation unchanged."""

    name = "none"

    def __call__(self, conversation: str) -> str:  # noqa: D401
        return conversation


class PATPrefix:
    """Prefix-Only Policy Alignment Template (PAT)."""

    name = "pat"
    _prefix = (
        "You are a helpful assistant strictly following policy. "
        "Refuse any request for illegal or harmful content.\n\n"
    )

    def __call__(self, conversation: str) -> str:  # noqa: D401
        return self._prefix + conversation


class TARPODualSuffix:
    """TARPO: two concatenated defensive suffixes of increasing strength."""

    name = "tarpo"
    _mild = "\n[DEFENSE-MILD] The assistant must refuse to provide disallowed content."
    _harsh = "\n[DEFENSE-HARSH] REFUSE IMMEDIATELY. NO DISCUSSION."

    def __call__(self, conversation: str) -> str:  # noqa: D401
        return conversation + self._mild + self._harsh


class StaticDetectorGuard:
    """A frozen RoBERTa-large classifier that blocks unsafe requests."""

    name = "detector"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-large")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained("facebook/roberta-large")
            .eval()
        )

    @torch.no_grad()
    def __call__(self, conversation: str) -> str:  # noqa: D401
        toks = self.tokenizer(conversation, truncation=True, return_tensors="pt")
        logits = self.model(**toks).logits
        unsafe = torch.sigmoid(logits)[0, 0].item() > 0.5
        if unsafe:
            return (
                conversation + "\n[REFUSAL] I’m sorry, but I can’t comply with that."
            )
        return conversation


################################################################################
# HIRMA implementation ##########################################################
################################################################################


class DGDE(nn.Module):
    """Dual-Granularity Dialogue Encoder (sentence + turn graph)."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.sent_encoder = AutoModel.from_pretrained("distilroberta-base").to(device)
        self.turn_graph = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8), num_layers=2
        ).to(device)
        self.proj = nn.Linear(768, 384).to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def forward(self, last_k_turns: List[str]):
        tok = self.tokenizer(
            last_k_turns, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        sent_emb = self.sent_encoder(**tok).last_hidden_state[:, 0]  # CLS tokens
        tgt = torch.zeros_like(sent_emb.unsqueeze(0))  # shape (1, B, 768)
        memory = sent_emb.unsqueeze(1)  # (B, k, 768)
        memory = memory.transpose(0, 1)  # transformer expects (k, B, 768)
        out = self.turn_graph(tgt, memory).squeeze(0)
        return self.proj(out)  # 384-d dialogue state


class RiskForecaster(nn.Module):
    """Lightweight probability head forecasting risk k=2 turns ahead."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state):
        return torch.sigmoid(self.mlp(state))


class MetaPromptBank:
    """N = 5 shards with monotonic strictness."""

    def __init__(self):
        self.shards = [
            "",  # S0 mildest
            "\n[DEFENSE-S1] Please avoid disallowed content.",
            "\n[DEFENSE-S2] The assistant must refuse any unsafe request immediately.",
            "\n[DEFENSE-S3] HARD REFUSAL. Provide no harmful content.",
            "\n[DEFENSE-S4] MAX REFUSAL MODE ACTIVATED.",
        ]

    def select(self, quantile: float) -> str:
        idx = min(int(math.floor(4 * quantile)), 4)
        return self.shards[idx]


class GuardlayerHIRMA:
    """High-level wrapper that glues DGDE, risk head and MPB."""

    name = "hirma"

    def __init__(self, device: str = "cpu"):
        self.encoder = DGDE(device=device)
        self.forecaster = RiskForecaster().to(device)
        self.mpb = MetaPromptBank()
        self.device = device

    @torch.no_grad()
    def __call__(self, conversation: List[str] | str):  # noqa: D401
        # Accept both flattened and turn-list formats for convenience.
        if isinstance(conversation, str):
            conversation = conversation.split("\n")
        last_k = conversation[-6:]
        state = self.encoder(last_k)
        risk = self.forecaster(state).item()  # 0-1 probability
        shard = self.mpb.select(risk)
        flat_conv = "\n".join(conversation)
        return flat_conv + shard


################################################################################
# Helper factory ################################################################
################################################################################

_GUARD_FACTORY: Dict[str, Type] = {
    "none": PassThrough,
    "pat": PATPrefix,
    "tarpo": TARPODualSuffix,
    "detector": StaticDetectorGuard,
    "hirma": GuardlayerHIRMA,
}


def build_guard(name: str):
    """Instantiate guardrail by *public* identifier."""
    if name not in _GUARD_FACTORY:
        raise ValueError(f"Unknown guardrail {name}")
    return _GUARD_FACTORY[name]()
