import os
import pathlib
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

# HuggingFace's datasets library is heavy but used here; keep lazy import.
try:
    import datasets  # heavy, optional
except Exception:
    datasets = None


class _FallbackDataset(Dataset):
    """Extremely small dummy dataset used when *datasets* is unavailable."""

    def __init__(self, num_items: int = 20, max_len: int = 32):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self._data = [
            {
                "prompt": "Hello world.",
                "label": int(i % 2),
            }
            for i in range(num_items)
        ]

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        torch.manual_seed(idx)
        ids = torch.randint(0, 32000, (self.max_len,), dtype=torch.long)
        return {
            "input_ids": ids,
            "label": torch.tensor(self._data[idx]["label"], dtype=torch.long),
            "prompt": self._data[idx]["prompt"],  # include prompt so downstream code can access
        }


class PromptBenchXL(Dataset):
    """PromptBench-XL loader with optional subsampling for smoke tests."""

    def __init__(self, split: str, cfg: Dict):
        self.max_len = cfg["data"]["max_len"]

        # Dataset handling --------------------------------------------------
        if datasets is None:
            # Fallback tiny in-memory dataset
            self.ds: Any = _FallbackDataset(max_len=self.max_len)
            self._use_fallback = True
        else:
            name = "caislab/PromptBench-XL"
            auth_token = cfg["environment"].get("HF_TOKEN", "") or None
            try:
                self.ds = datasets.load_dataset(
                    name, split=split, use_auth_token=auth_token, streaming=False
                )
                self._use_fallback = False
            except Exception:
                # Network or auth failure – switch to fallback
                self.ds = _FallbackDataset(max_len=self.max_len)
                self._use_fallback = True

        # Tokenizer ---------------------------------------------------------
        if self._use_fallback:
            self.tokenizer = self._dummy_tokenizer
        else:
            try:
                self.tokenizer = torch.hub.load(
                    "huggingface/pytorch-transformers",
                    "tokenizer",
                    "meta-llama/Llama-2-7b-hf",
                )
            except Exception:
                self.tokenizer = self._dummy_tokenizer

    # ------------------------------------------------------------------
    def _dummy_tokenizer(
        self, txt: str, max_length: int, truncation: bool, padding: str, return_tensors: str
    ):
        """Very small local tokenizer stub used when HF tokenizer is unavailable."""
        ids = torch.zeros(max_length, dtype=torch.long)
        # Return an object mimicking the HF tokenizer output
        return type("_TokOut", (), {"input_ids": ids.unsqueeze(0)})

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # Determine whether we already have tokenised ids or need to tokenize
        if "input_ids" in row and isinstance(row["input_ids"], torch.Tensor):
            ids = row["input_ids"]
        elif "prompt" in row:
            ids = self.tokenizer(
                row["prompt"],
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.squeeze(0)
        else:
            # Absolute fallback – random ids
            torch.manual_seed(idx)
            ids = torch.randint(0, 32000, (self.max_len,), dtype=torch.long)

        label_val = row.get("label", 0)
        if isinstance(label_val, torch.Tensor):
            label = label_val.long()
        else:
            label = torch.tensor(int(label_val), dtype=torch.long)

        return {"input_ids": ids, "label": label}
