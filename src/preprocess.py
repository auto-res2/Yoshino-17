"""src/preprocess.py
Dataset loading utilities + generic helpers (seed, timing, json).
Removed superfluous `# type: ignore` markers flagged by static validation.
"""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import datasets
import numpy as np
import yaml

###############################################################################
# Generic helpers ##############################################################
###############################################################################

def set_seed(seed: int):  # noqa: D401
    random.seed(seed)
    np.random.seed(seed)
    import torch  # local import to avoid hard dependency in non-torch envs

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timeit(description: str):
    start = time.time()
    yield
    dur = time.time() - start
    print(f"[TIMER] {description}: {dur:.3f}s")


def save_json(obj: Any, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


###############################################################################
# YAML → dataclass loader (used by main) #######################################
###############################################################################

from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentConfig:  # noqa: D101 (straightforward container)
    experiment_name: str
    random_seeds: List[int]
    run_mode: str  # "smoke" | "full"
    hardware: Dict[str, Any]
    models: Dict[str, Any]
    llm_apis: Dict[str, Any]
    guardrails: List[str]
    datasets: Dict[str, Any]
    metrics: List[str]
    output_dir: str

    @staticmethod
    def load(path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return ExperimentConfig(**cfg)


###############################################################################
# Dataset download / preprocessing ############################################
###############################################################################

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _download_hf_dataset(repo: str, split: str):
    try:
        return datasets.load_dataset(repo, split=split, token=os.getenv("HF_TOKEN"))
    except Exception as exc:  # pragma: no cover – network errors are runtime
        raise RuntimeError(f"Failed to download dataset {repo}: {exc}")


def load_dataset(cfg_entry: Dict[str, str]):
    """Return a 🤗 `datasets.Dataset` object according to the YAML spec."""

    if "hf_repo" in cfg_entry:
        return _download_hf_dataset(cfg_entry["hf_repo"], cfg_entry.get("split", "test"))

    if "url" in cfg_entry:
        url = cfg_entry["url"]
        local_path = _DATA_ROOT / os.path.basename(url)
        if not local_path.exists():
            datasets.utils.file_utils.download_url(url, local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        return datasets.Dataset.from_list(lines)

    raise ValueError("Unknown dataset entry config – expected 'hf_repo' or 'url'.")
