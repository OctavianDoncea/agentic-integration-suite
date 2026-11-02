from __future__ import annotations
from typing import Any
import pytest
from agentic_suite.agents.policy import ALLOW, WritePolicy, evaluate, mentioned_in
from agentic_suite.sdk.base import BaseTool

class NoteTool(BaseTool):
    """Post a note to a channel."""

    tool_name = 'note_tool'
    target_fields = ('channel',)

    channel: str
    text: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {'ok': True}


class UntargetedWriteTool(BaseTool):
    """A write with no addressable destination."""

    tool_name = 'untargeted_write_tool'

    text: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {'ok': True}


class LookupTool(BaseTool):
    """Read something."""

    tool_name = 'lookup_tool'
    read_only = True

    query: str

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {'found': True}


PROMPT = 'Tell #eng-leads the deploy finished.'
GOOD = {'channel': '#eng-leads', 'text': 'Deploy finished.'}
ENABLED = WritePolicy(allow_writes=True)
DISABLED = WritePolicy(allow_writes=False)

def decide(tool_cls, arguments, policy, prompt=PROMPT):
    return evaluate(policy, tool_cls.tool_name, tool_cls, arguments, prompt)

def test_read_only_tools_are_never_gated():
    strictest = WritePolicy(allow_writes=False, allowed_tools=frozenset())
    assert decide(LookupTool, {'query': 'x'}, strictest) == ALLOW

def test_unknown_tools_are_left_to_the_executor():
    assert evaluate(DISABLED, 'nope', None, {}, PROMPT) == ALLOW

def test_writes_are_previewed_when_disabled():
    decision = decide(NoteTool, GOOD, DISABLED)
    assert decision.action == 'preview'
    assert decision.rule == 'writes_disabled'

def test_writes_run_when_enabled_and_the_target_is_named():
    assert decide(NoteTool, GOOD, ENABLED) == ALLOW

def test_invalid_arguments_are_left_to_the_executor():
    # Missing `text`: the executor reports it and nothing runs, so no preview.
    assert decide(NoteTool, {'channel': '#eng-leads'}, DISABLED) == ALLOW

def test_the_tool_allowlist_blocks_other_write_tools():
    policy = WritePolicy(allow_writes=True, allowed_tools=frozenset({'other_tool'}))
    decision = decide(NoteTool, GOOD, policy)
    assert (decision.action, decision.rule) == ('block', 'tool_not_allowed')

    nothing_allowed = WritePolicy(allow_writes=True, allowed_tools=frozenset())
    assert decide(NoteTool, GOOD, nothing_allowed).action == 'block'

def test_an_unnamed_target_is_blocked_even_when_writes_are_enabled():
    decision = decide(NoteTool, {'channel': '#general', 'text': 'hi'}, ENABLED)
    assert (decision.action, decision.rule) == ('block', 'target_not_in_request')
    assert decision.detail == {'field': 'channel', 'value': '#general'}

def test_a_block_wins_over_a_preview():
    decision = decide(NoteTool, {'channel': '#general', 'text': 'hi'}, DISABLED)
    assert decision.action == 'block'

def test_targets_match_with_or_without_a_hash_and_ignoring_case():
    assert decide(NoteTool, GOOD, ENABLED, prompt='post to eng-leads please') == ALLOW
    assert decide(NoteTool, {'channel': '#ENG-LEADS', 'text': 'x'}, ENABLED) == ALLOW

@pytest.mark.parametrize(
    ("value", "text"),
    [
        ('#eng', 'Tell #eng-leads the deploy finished.'),
        ('#eng-leads', 'Tell #eng the deploy finished.'),
        ('#eng', 'Send the engineering update.'),
        ('', 'anything'),
        ('#', 'anything'),
    ]
)
def test_a_longer_or_shorter_name_does_not_match(value, text):
    assert mentioned_in(value, text) is False

def test_the_target_check_can_be_turned_off():
    policy = WritePolicy(allow_writes=True, require_request_target=False)
    assert decide(NoteTool, {'channel': '#general', 'text': 'hi'}, policy) == ALLOW

def test_tools_without_targets_are_gated_only_by_the_flag():
    assert decide(UntargetedWriteTool, {'text': 'x'}, ENABLED) == ALLOW
    assert decide(UntargetedWriteTool, {'text': 'x'}, DISABLED).action == 'preview'