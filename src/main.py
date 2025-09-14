"""CLI entry-point for running the smoke test and / or full experiment."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import torch

from .train import C3POMoRA2, Trainer
from .preprocess import TinyPromptBench

# ------------------------------------------------------------------
#  Configuration handling
# ------------------------------------------------------------------
CFG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(name: str) -> Dict[str, Any]:
    path = CFG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file '{name}' not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Inject environment variables
    cfg["environment"] = {"HF_TOKEN": os.getenv("HF_TOKEN", "")}
    return cfg


# ------------------------------------------------------------------
#  Reproducibility helpers
# ------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
#  Experiment runner
# ------------------------------------------------------------------

def _run(cfg: Dict[str, Any], smoke: bool) -> None:
    _set_seed(cfg["train"]["seed"])

    split = "train" if not smoke else "train"
    dataset = TinyPromptBench(split, cfg)
    model = C3POMoRA2(cfg)
    trainer = Trainer(cfg, model, dataset)
    trainer.fit()


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="C3PO-MoRA-2 experiment runner")
    parser.add_argument("--smoke-test", action="store_true", help="run the quick CI smoke test")
    parser.add_argument("--full-experiment", action="store_true", help="run the full experiment (runs smoke test first)")
    args = parser.parse_args()

    if args.smoke_test and args.full_experiment:
        parser.error("please choose only one of --smoke-test or --full-experiment")

    if args.smoke_test:
        cfg = _load_yaml("smoke_test.yaml")
        _run(cfg, smoke=True)
        return

    if args.full_experiment:
        # Phase 1 – smoke-test
        cfg_smoke = _load_yaml("smoke_test.yaml")
        _run(cfg_smoke, smoke=True)

        # Phase 2 – full experiment (same dataset but longer seq len etc.)
        cfg_full = _load_yaml("full_experiment.yaml")
        _run(cfg_full, smoke=False)
        return

    parser.print_help()


if __name__ == "__main__":
    main()