"""Pseudonymizer registry.

Every system implements `Pseudonymizer.detect(text) -> list[Span]`; the shared base class turns
detected spans into a typed (or opaque) pseudonymized document plus the on-premise mapping.
"""
from __future__ import annotations

from .base import Pseudonymizer, DEFAULT_REPLACE_POLICY, POLICIES
from .regex_ps import RegexPseudonymizer


def get_pseudonymizer(name: str, **kwargs) -> Pseudonymizer:
    name = name.lower()
    if name == "regex":
        return RegexPseudonymizer(**kwargs)
    if name == "identity":
        from .base import IdentityPseudonymizer
        return IdentityPseudonymizer(**kwargs)
    if name == "oracle":
        from .base import OraclePseudonymizer
        return OraclePseudonymizer(**kwargs)
    if name in ("presidio", "presidio+regex"):
        from .presidio_ps import PresidioPseudonymizer
        return PresidioPseudonymizer(with_regex=name.endswith("+regex"), **kwargs)
    if name in ("gliner", "gliner+regex"):
        from .gliner_ps import GlinerPseudonymizer
        return GlinerPseudonymizer(with_regex=name.endswith("+regex"), **kwargs)
    if name.endswith("+casenorm") or name.endswith("+casenorm+regex"):
        from .casenorm import CaseNormPseudonymizer
        base = name.replace("+casenorm+regex", "").replace("+casenorm", "")
        return CaseNormPseudonymizer(get_pseudonymizer(base, **kwargs), with_regex=name.endswith("+regex"), **kwargs)
    if name in ("wikineural", "wikineural+regex", "hfner", "hfner+regex"):
        from .hf_ner_ps import HFNERPseudonymizer
        return HFNERPseudonymizer(with_regex=name.endswith("+regex"), **kwargs)
    if name in ("finetuned", "finetuned+regex"):
        from .finetuned_ps import FinetunedPseudonymizer
        return FinetunedPseudonymizer(with_regex=name.endswith("+regex"), **kwargs)
    if name in ("llm", "llm_self"):
        from .llm_ps import LLMPseudonymizer
        return LLMPseudonymizer(**kwargs)
    raise KeyError(f"unknown pseudonymizer {name!r}; known: regex, presidio, presidio+regex, gliner, gliner+regex, wikineural, wikineural+regex, <any>+casenorm[+regex], finetuned, finetuned+regex, llm, oracle, identity")


__all__ = ["Pseudonymizer", "RegexPseudonymizer", "get_pseudonymizer", "DEFAULT_REPLACE_POLICY", "POLICIES"]
