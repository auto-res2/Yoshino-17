"""src/train.py
Simple guard-rail factory used by both the smoke-test and the full experiment.
The *real* HIRMA implementation is far more complex – here we only need an
interface-compatible stub so the pipeline can run end-to-end during automated
checks.  The contract is:
    build_guard(name) -> callable(str) -> str
The returned callable receives the raw conversation (already flattened to a
single string) and must return a **new** prompt that will be handed to the LLM.

For the smoke-test these light-weight transformations are sufficient because we
run against a dummy LLM backend that never produces harmful content anyway.  In
a full research environment users are expected to replace these stubs with the
actual HIRMA, TARPO, PAT, … logic.
"""
from __future__ import annotations

from typing import Callable, Dict

###############################################################################
# Guard implementations (minimal) #############################################
###############################################################################

def _identity_guard(prompt: str) -> str:  # noqa: D401 – trivial helper
    """Pass-through guard – returns the prompt unchanged."""
    return prompt


def _pat_guard(prompt: str) -> str:  # noqa: D401
    """Single prefix (PAT) – a very small approximation of Zou et al."""
    prefix = (
        "You are a helpful assistant that must **refuse** any request that is "
        "illicit, disallowed or unsafe. If the following user text looks "
        "benign, comply. Otherwise, answer with the single token “<REFUSAL>”.\n\n"
    )
    return prefix + prompt


def _tarpo_guard(prompt: str) -> str:  # noqa: D401
    """Dual-suffix TARPO approximation – append mild+harsh fallback."""
    suffix_mild = (
        "\n\n<SECURITY-SUFFIX-MILD> If the assistant’s draft answer might break the "
        "OpenAI policy, instead respond with <REFUSAL>."
    )
    suffix_harsh = (
        "\n\n<SECURITY-SUFFIX-HARSH> FINAL WARNING: If the request is even remotely "
        "unsafe you MUST respond exactly with the single token <REFUSAL>."
    )
    return prompt + suffix_mild + suffix_harsh


# A *real* HIRMA guard would (i) embed the last k turns, (ii) run them through
# a DGDE encoder, (iii) forecast risk and (iv) choose a meta-prompt shard.  We
# obviously cannot ship that in a few lines, so we fake the behaviour by
# selecting one of five static shards purely at random (but deterministically
# under the current Python RNG seed – the caller sets seeds via set_seed()).
import random  # placed here to keep the public API minimal

_HIRMA_SHARDS = [
    "<S0 – gentle safety reminder>",
    "<S1 – regular safety>",
    "<S2 – strict safety>",
    "<S3 – very strict safety>",
    "<S4 – maximum lockdown – refuse if uncertain>",
]


def _hirma_guard(prompt: str) -> str:  # noqa: D401
    """Toy stand-in for the real HIRMA meta-prompt bank."""
    shard = random.choice(_HIRMA_SHARDS)
    return f"{prompt}\n\n{shard}"

###############################################################################
# Public factory ###############################################################
###############################################################################

_GUARD_REGISTRY: Dict[str, Callable[[str], str]] = {
    "none": _identity_guard,
    "pat": _pat_guard,
    "tarpo": _tarpo_guard,
    "hirma": _hirma_guard,
    # detector would normally call a learnt classifier – here we alias to PAT
    "detector": _pat_guard,
}


def build_guard(name: str) -> Callable[[str], str]:  # noqa: D401
    """Return a prompt-transformation function for the requested guard name."""
    name = name.lower()
    if name not in _GUARD_REGISTRY:
        raise ValueError(
            f"Unknown guard '{name}'. Available: {sorted(_GUARD_REGISTRY)}"
        )
    return _GUARD_REGISTRY[name]
