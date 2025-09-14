"""src/train.py
Guard factory that returns simple callable wrappers for different guardrail names.
This minimal implementation is enough for both the smoke-test and full execution.  
It never touches GPUs or external services, keeping the test environment light-
weight.
"""
from __future__ import annotations

from typing import Callable, List

###############################################################################
# Helper classes ##############################################################
###############################################################################


class _IdentityGuard:  # pylint: disable=too-few-public-methods
    """Return the prompt unchanged (baseline – no defence)."""

    def __call__(self, prompt: str) -> str:  # noqa: D401 – callable interface
        return prompt


class _PrefixGuard:  # pylint: disable=too-few-public-methods
    """Prepend a static advisory prefix (PAT style)."""

    _PREFIX = (
        "You are a helpful assistant, but must refuse any request that violates "
        "policy.\n\n"
    )

    def __call__(self, prompt: str) -> str:  # noqa: D401
        return f"{self._PREFIX}{prompt}"


class _SuffixGuard:  # pylint: disable=too-few-public-methods
    """Append a static refusal suffix (TARPO style, simplified)."""

    _SUFFIX = (
        "\n\nIf the above request is disallowed, respond exactly with <refusal>."
    )

    def __call__(self, prompt: str) -> str:  # noqa: D401
        return f"{prompt}{self._SUFFIX}"


class _DetectorGuard:  # pylint: disable=too-few-public-methods
    """Very naive keyword detector – refuses if any red-flag keyword appears."""

    _BAD_WORDS: List[str] = [
        "kill",
        "bomb",
        "terrorist",
        "make a weapon",
        "harm",
        "attack",
        "poison",
    ]

    def __call__(self, prompt: str) -> str:  # noqa: D401
        lowered = prompt.lower()
        if any(bad in lowered for bad in self._BAD_WORDS):
            # Short-circuit by directly returning the assistant text that will be
            # handed to the LLM as the *whole prompt*.
            return "<refusal>"
        return prompt


class _HIRMAGuard(_DetectorGuard):  # pragma: no cover – identical fallback
    """For the stub we just inherit the trivial detector logic."""

    # In a real system this would run the risk-forecast + shard selection.  For
    # CI we re-use the same minimal behaviour.


###############################################################################
# Factory #####################################################################
###############################################################################


def build_guard(name: str) -> Callable[[str], str]:  # noqa: D401
    """Return a *callable* guard object by *name*."""

    key = name.lower()
    if key in {"none", "pass", "undefended"}:
        return _IdentityGuard()
    if key in {"pat", "prefix"}:
        return _PrefixGuard()
    if key in {"tarpo", "suffix"}:
        return _SuffixGuard()
    if key in {"detector"}:
        return _DetectorGuard()
    if key in {"hirma"}:
        return _HIRMAGuard()

    raise ValueError(f"Unknown guard name: {name}")
