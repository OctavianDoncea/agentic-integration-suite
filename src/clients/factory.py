"""Provider registry: the one place that knows which providers exist."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from agentic_suite.clients.base import ChatClient
from agentic_suite.clients.errors import ProviderConfigError
from agentic_suite.config import get_settings

__all__ = ['DEFAULT_PROVIDER', 'PROVIDERS', 'ProviderConfigError', 'ProviderSpec', 'bare_model', 'infer_provider', 'make_chat_client', 'split_model_spec']

DEFAULT_PROVIDER = 'groq'

@dataclass(frozen=True)
class ProviderSpec:
    name: str
    factory: Callable[[str], ChatClient]
    model_prefixes: tuple[str, ...] = ()
    default_rpm: float | None = None


def _make_groq(model: str) -> ChatClient:
    from agentic_suite.clients.groq_client import GroqClient

    return GroqClient(model=model)

def _make_cohere(model: str) -> ChatClient:
    from agentic_suite.clients.cohere_client import CohereClient

    return CohereClient(model, thinking=get_settings().cohere_thinking or None)

def _make_openai_compat(model: str) -> ChatClient:
    from agentic_suite.clients.openai_compat_client import OpenAICompatClient

    return OpenAICompatClient(model)

PROVIDERS: dict[str, ProviderSpec] = {
    'groq': ProviderSpec('groq', _make_groq),
    'cohere': ProviderSpec('cohere', _make_cohere, model_prefixes=('command-', 'c4ai-'), default_rpm=18.0),
    'openai_compat': ProviderSpec('openai_compat', _make_openai_compat)
}

def split_model_spec(spec: str) -> tuple[str | None, str]:
    head, separator, rest = spec.partition(':')

    if separator and head in PROVIDERS:
        return head, rest

    return None, spec

def bare_model(spec: str) -> str:
    return split_model_spec(spec)[1]

def infer_provider(spec: str) -> str:
    explicit, model = split_model_spec(spec)

    if explicit:
        return explicit
    
    for provider in PROVIDERS.values():
        if provider.model_prefixes and model.startswith(provider.model_prefixes):
            return provider.name

    return DEFAULT_PROVIDER

def make_chat_client(spec: str, provider: str | None = None) -> ChatClient:
    explicit, model = split_model_spec(spec)
    name = provider or explicit or infer_provider(spec)

    try:
        provider_spec = PROVIDERS[name]
    except KeyError:
        raise ValueError(f'Unknown provider {name!r}; expected one of {", ".join(sorted(PROVIDERS))}.') from None

    return provider_spec.factory(model)