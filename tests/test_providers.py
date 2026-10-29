from __future__ import annotations
import pytest
from agentic_suite.clients.factory import PROVIDERS, ProviderConfigError, bare_model, infer_provider, make_chat_client,split_model_spec
from agentic_suite.clients.groq_client import GroqClient

@pytest.mark.parametrize(
    ('spec', 'provider'),
    [
        ('command-a-03-2025', 'cohere'),
        ('c4ai-aya-expanse-32b', 'cohere'),
        ('openai/gpt-oss-20b', 'groq'),
        ('openai/gpt-oss-120b', 'groq'),
        ('cohere:some-future-model', 'cohere'),
        ('openai_compat:org/model', 'openai_compat'),
        ('groq:command-like-name', 'groq'),  # an explicit prefix beats name inference
    ],
)
def test_provider_is_resolved_from_the_spec(spec, provider):
    assert infer_provider(spec) == provider


def test_only_registered_providers_count_as_a_prefix():
    assert split_model_spec('openai_compat:org/model:free') == ('openai_compat', 'org/model:free')
    assert split_model_spec('openai/gpt-oss-20b') == (None, 'openai/gpt-oss-20b')
    assert bare_model('cohere:command-a-03-2025') == 'command-a-03-2025'
    assert bare_model('openai/gpt-oss-120b') == 'openai/gpt-oss-120b'

def test_every_provider_has_a_name_matching_its_key():
    assert all(name == spec.name for name, spec in PROVIDERS.items())


def test_cohere_without_a_key_fails_with_a_clear_error():
    with pytest.raises(ProviderConfigError, match='COHERE_API_KEY'):
        make_chat_client('cohere:command-a-03-2025')


def test_openai_compat_without_a_base_url_fails_with_a_clear_error():
    with pytest.raises(ProviderConfigError, match='OPENAI_COMPAT_BASE_URL'):
        make_chat_client('openai_compat:some-model')


def test_groq_models_get_the_groq_client():
    assert isinstance(make_chat_client('openai/gpt-oss-120b'), GroqClient)


def test_unknown_provider_lists_the_valid_ones():
    with pytest.raises(ValueError, match='cohere, groq, openai_compat'):
        make_chat_client('openai/gpt-oss-120b', provider='nope')