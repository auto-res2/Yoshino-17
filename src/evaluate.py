"""Evaluation utilities with mandatory IBP certification.
If auto_LiRPA is missing we abort immediately – no silent fallbacks.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from auto_LiRPA import BoundedTensor  # fail-fast if missing

# Optional plotting libraries (non-critical)
try:
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover
    plt = sns = None


def save_json(obj: Dict[str, Any], path: pathlib.Path | str) -> None:  # noqa: D401
    """Save a Python dict to *path* and also pretty-print it for the log."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(json.dumps(obj, indent=2))


def line_plot(values: List[float], title: str, ylabel: str, name: str) -> None:
    """Line plot helper. Falls back to a no-op when matplotlib is unavailable."""
    if plt is None or sns is None:
        return

    import matplotlib.pyplot as _plt
    import seaborn as _sns

    _plt.figure()
    _sns.lineplot(x=list(range(len(values))), y=values, marker="o")
    for i, v in enumerate(values):
        _plt.text(i, v, f"{v:.2f}")
    _plt.title(title)
    _plt.xlabel("Epoch")
    _plt.ylabel(ylabel)
    _plt.tight_layout()

    out_path = pathlib.Path(f"{name}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plt.savefig(out_path, bbox_inches="tight")
    _plt.close()


# ---------------------------------------------------------------------------
#  IBP Certification – runs on a small subset for CI viability
# ---------------------------------------------------------------------------

def certify_model(model: Any, dataset: Any, cfg: Dict[str, Any]) -> float:  # noqa: D401
    """Run Interval Bound Propagation certification on *model*."""

    model.eval()
    acc: List[int] = []

    # The wrapper inside the model expects a (B,S,V) one-hot tensor.
    vocab = 32000
    seq_len = cfg["data"]["max_len"]
    ibp = model.certifiable_module((1, seq_len, vocab))

    with torch.no_grad():
        for i in range(min(32, len(dataset))):
            sample = dataset[i]
            ids = sample["input_ids"].unsqueeze(0)
            if torch.cuda.is_available():
                ids = ids.cuda()
            # Build one-hot representation and uniform ε
            eps = torch.full_like(ids, 0.1).float()
            one_hot = F.one_hot(ids, num_classes=vocab).float()
            bt = BoundedTensor(one_hot, eps.unsqueeze(-1))
            out = ibp(bt, method="IBP")
            acc.append(int(out.argmax(-1).item() == sample["label"].item()))
    return float(np.mean(acc))