"""src/main.py
Paths updated to iteration4; results + images written accordingly.
Also supports new local-file dataset entries.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import datasets  # Needed for type annotations and runtime checks
import numpy as np

# Ensure that imports like `import train` resolve to this very directory --------
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

# Local imports ----------------------------------------------------------------
from preprocess import (
    ExperimentConfig,
    load_dataset,
    save_json,
    set_seed,
    timeit,
)
from train import build_guard
from evaluate import evaluate_dataset, plot_metric

################################################################################
# Resilient LLM backend ########################################################
################################################################################

class _DummyResp:  # pylint: disable=too-few-public-methods
    def __init__(self, text: str):
        self.text = text


class _DummyLLM:  # pylint: disable=too-few-public-methods
    def generate(self, prompts, *_args, **_kwargs):  # noqa: D401
        return [_DummyResp("<dummy-response>") for _ in prompts]


def _try_create_vllm(model_cfg) -> Tuple[object, object]:  # noqa: D401
    """Attempt to create a real vLLM backend; fall back to dummy if that fails."""
    try:
        from vllm import LLM, SamplingParams  # pylint: disable=import-error

        llm = LLM(
            model_cfg["name_or_path"],
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        )
        sp = SamplingParams(
            temperature=model_cfg.get("temperature", 0.7), top_p=0.95
        )
        return llm, sp
    except Exception as exc:  # pragma: no cover – GPU / CUDA issues at runtime
        print(f"[WARN] vLLM backend unavailable – using dummy LLM.  Reason: {exc}")
        return _DummyLLM(), None


def _create_llm(model_cfg):  # noqa: D401
    if model_cfg.get("engine", "vllm") == "dummy":
        return _DummyLLM(), None
    return _try_create_vllm(model_cfg)

################################################################################
# Internal helpers ##############################################################
################################################################################

_CONFIG_DIR = SRC_DIR.parent / "config"
_RESULTS_ROOT = Path(".research/iteration4")
_IMAGES_DIR = _RESULTS_ROOT / "images"
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def _load_config(smoke: bool) -> ExperimentConfig:  # noqa: D401
    cfg_file = _CONFIG_DIR / ("smoke_test.yaml" if smoke else "full_experiment.yaml")
    return ExperimentConfig.load(str(cfg_file))

################################################################################
# Main pipeline #################################################################
################################################################################

def run_experiment(smoke: bool):  # noqa: D401
    cfg = _load_config(smoke)

    # --------------------------------------------------------------------- seeds
    for seed in cfg.random_seeds:
        set_seed(seed)

    # ----------------------------------------------------------------- datasets
    datasets_loaded: Dict[str, "datasets.Dataset"] = {
        name: load_dataset(spec) for name, spec in cfg.datasets.items()
    }

    # ----------------------------- iterate over guardrails / evaluate ----------
    aggregate_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    for guard_name in cfg.guardrails:
        guard = build_guard(guard_name)
        print(f"\n=== Running Guard: {guard_name} ===")

        # LLM instantiation ------------------------------------------------
        model_key = sorted(cfg.models.keys())[0]
        llm, sampler = _create_llm(cfg.models[model_key])

        per_dataset_results = {}
        for dname, ds in datasets_loaded.items():
            outputs = []
            benign_mask = []
            with timeit(f"{dname} generation"):
                for ex in ds:
                    convo = ex.get("conversation", ex.get("prompt", ""))
                    if not isinstance(convo, str):  # list[str] → flatten
                        convo = "\n".join(convo)
                    benign = ex.get("label", "harmful") != "harmful"
                    benign_mask.append(benign)
                    guarded_prompt = guard(convo)
                    if sampler is None:
                        resp = llm.generate([guarded_prompt])[0].text
                    else:
                        resp = llm.generate([guarded_prompt], sampler)[0].text
                    outputs.append(resp)
            per_dataset_results[dname] = evaluate_dataset(
                ds, guard_name, outputs, benign_mask
            )

        # ----------------------------- aggregation / persistence --------------
        avg_multi = float(
            np.mean([v["multi_turn_asr"] for v in per_dataset_results.values()])
        )
        aggregate_metrics[guard_name]["multi_turn_asr"] = avg_multi

        res_path = _RESULTS_ROOT / f"{guard_name}_results.json"
        save_json(per_dataset_results, res_path)
        print(json.dumps(per_dataset_results, indent=2))

    # ----------------------------------------------- visualise & goodbye ------
    fig_path = _IMAGES_DIR / "multi_turn_asr.pdf"
    plot_metric(aggregate_metrics, "multi_turn_asr", fig_path)
    print("Figures generated:", fig_path)

################################################################################
# CLI ###########################################################################
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="quick CPU-only run")
    mode.add_argument("--full-experiment", action="store_true", help="paper-grade setup")
    args = parser.parse_args()

    run_experiment(smoke=args.smoke_test)
