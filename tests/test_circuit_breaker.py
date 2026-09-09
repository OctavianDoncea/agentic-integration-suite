from __future__ import annotations
import pytest
from collections.abc import Awaitable, Callable
from agentic_suite.middleware.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError
from agentic_suite.tools.mock.jira_tool import JiraIssueTool, JiraServerError

class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CallCounter:
    def __init__(self, behaviour: Callable[[], Awaitable]):
        self._behaviour = behaviour
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        return await self._behaviour()


async def _fails() -> None:
    raise JiraServerError()

async def _succeeds() -> str:
    return 'ok'

@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()

@pytest.fixture
def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(name='test', failure_threshold=3, cooldown_seconds=30.0, clock=clock)

async def _trip(breaker: CircuitBreaker, fn=_fails) -> None:
    for _ in range(breaker.failure_threshold):
        with pytest.raises(JiraServerError):
            await breaker.call(fn)

async def test_starts_closed(breaker: CircuitBreaker):
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0

async def test_three_consecutive_failures_open_the_circuit(breaker: CircuitBreaker):
    await _trip(breaker)
    assert breaker.state is CircuitState.OPEN

async def test_circuit_stays_closed_below_the_threshold(breaker: CircuitBreaker):
    for _ in range(2):
        with pytest.raises(JiraServerError):
            await breaker.call(_fails)

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 2

async def test_a_success_resets_the_failure_counter(breaker: CircuitBreaker):
    for _ in range(2):
        with pytest.raises(JiraServerError):
            await breaker.call(_fails)

    await breaker.call(_succeeds)
    assert breaker.failure_count == 0

    for _ in range(2):
        with pytest.raises(JiraServerError):
            await breaker.call(_fails)

    assert breaker.state is CircuitState.CLOSED

async def test_failures_propagate_unchanged_while_closed(breaker: CircuitBreaker):
    with pytest.raises(JiraServerError):
        await breaker.call(_fails)

async def test_open_circuit_rejects_without_invoking_the_function(breaker):
    await _trip(breaker)

    counter = CallCounter(_fails)
    with pytest.raises(CircuitOpenError):
        await breaker.call(counter)

    assert counter.calls == 0

async def test_open_circuit_reports_time_until_retry(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(10)

    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.call(_fails)

    assert exc_info.value.seconds_until_retry == pytest.approx(20.0)
    assert exc_info.value.name == 'test'

async def test_rejection_does_not_extend_the_cooldown(breaker, clock: FakeClock):
    await _trip(breaker)

    for _ in range(5):
        clock.advance(1)
        with pytest.raises(CircuitOpenError):
            await breaker.call(_fails)

    assert breaker.seconds_until_retry() == pytest.approx(25.0)

async def test_cooldown_elapsing_admits_a_trial_call(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)

    counter = CallCounter(_succeeds)
    await breaker.call(counter)

    assert counter.calls == 1

async def test_one_second_before_cooldown_is_still_rejected(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(29)

    with pytest.raises(CircuitOpenError):
        await breaker.call(_succeeds)

async def test_successful_trial_closes_the_circuit(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)

    assert await breaker.call(_succeeds) == 'ok'
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0

async def test_traffic_flows_freely_after_recovery(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)
    await breaker.call(_succeeds)

    counter = CallCounter(_succeeds)
    for _ in range(10):
        await breaker.call(counter)

    assert counter.calls == 10

async def test_failed_trial_reopens_the_circuit(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)

    with pytest.raises(JiraServerError):
        await breaker.call(_fails)

    assert breaker.state is CircuitState.OPEN

async def test_failed_trial_restarts_the_full_cooldown(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)

    with pytest.raises(JiraServerError):
        await breaker.call(_fails)
    
    assert breaker.seconds_until_retry() == pytest.approx(30.0)

    clock.advance(29)
    counter = CallCounter(_succeeds)
    with pytest.raises(CircuitOpenError):
        await breaker.call(counter)
    assert counter.calls == 0

    clock.advance(1)
    await breaker.call(counter)
    assert counter.calls == 1

async def test_failed_trial_does_not_need_the_threshold_again(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)

    with pytest.raises(JiraServerError):
        await breaker.call(_fails)

    counter = CallCounter(_succeeds)
    with pytest.raises(CircuitOpenError):
        await breaker.call(counter)
    assert counter.calls == 0

async def test_half_open_admits_only_one_trial(breaker, clock: FakeClock):
    await _trip(breaker)
    clock.advance(30)

    slow_probe_started = False

    async def hangs_then_fails():
        nonlocal slow_probe_started
        slow_probe_started = True
        raise JiraServerError()

    with pytest.raises(JiraServerError):
        await breaker.call(hangs_then_fails)
    assert slow_probe_started

    second = CallCounter(_succeeds)
    with pytest.raises(CircuitOpenError):
        await breaker.call(second)
    assert second.calls == 0

async def test_terminal_errors_do_not_trip_the_circuit(breaker: CircuitBreaker):
    async def bad_input():
        raise ValueError('invalid argument')

    for _ in range(5):
        with pytest.raises(ValueError):
            await breaker.call(bad_input)

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0

async def test_terminal_error_during_a_trial_keeps_the_slot_available(breaker, clock):
    await _trip(breaker)
    clock.advance(30)

    async def bad_input():
        raise ValueError('invalid argument')

    with pytest.raises(ValueError):
        await breaker.call(bad_input)

    counter = CallCounter(_succeeds)
    await breaker.call(counter)
    assert counter.calls == 1
    assert breaker.state is CircuitState.CLOSED

async def test_rate_limits_count_as_failures(breaker: CircuitBreaker):
    from agentic_suite.tools.mock.github_tool import GitHubRateLimitError

    for _ in range(3):
        with pytest.raises(GitHubRateLimitError):
            await breaker.call(_raise_rate_limit)

    assert breaker.state is CircuitState.OPEN

async def _raise_rate_limit():
    from agentic_suite.tools.mock.github_tool import GitHubRateLimitError

    raise GitHubRateLimitError()

async def test_circuit_open_error_is_not_retryable():
    from agentic_suite.middleware.retry import is_retryable

    assert is_retryable(CircuitOpenError('test', 30.0)) is False

async def test_retry_inside_breaker_counts_one_failure(clock: FakeClock):
    from agentic_suite.middleware.retry import RetryExhaustedError, with_retry

    async def noop_sleep(_: float) -> None:
        return None

    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0, clock=clock)
    guarded = with_retry(max_attempts=3, sleep=noop_sleep)(_fails)

    with pytest.raises(RetryExhaustedError):
        await breaker.call(guarded)

    assert breaker.failure_count == 1

async def test_decorator_form(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=30.0, clock=clock)

    @breaker
    async def flaky() -> None:
        """Docstring that must survieve wrapping."""
        raise JiraServerError()

    for _ in range(2):
        with pytest.raises(JiraServerError):
            await flaky()

    assert breaker.state is CircuitState.OPEN
    assert flaky.__name__ == 'flaky'
    assert 'must survive' in flaky.__doc__

async def test_reset_forces_the_circuit_closed(breaker: CircuitBreaker):
    await _trip(breaker)
    breaker.reset()

    assert breaker.state is CircuitState.CLOSED
    await breaker.call(_succeeds)

async def test_separate_breakers_are_independent(clock: FakeClock):
    jira = CircuitBreaker(name='jira', failure_threshold=3, clock=clock)
    github = CircuitBreaker(name='github', failure_threshold=3, clock=clock)

    await _trip(jira)

    assert jira.state is CircuitState.OPEN
    assert github.state is CircuitState.CLOSED
    assert await github.call(_succeeds) == 'ok'

@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'failure_threshold': 0}, 'at least 1'),
        ({'cooldown_seconds': -1}, 'not be negative')
    ]
)
def test_invalid_configuration_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CircuitBreaker(**kwargs)

async def test_guards_a_real_mock_tool(clock: FakeClock):
    breaker = CircuitBreaker(name='jira', failure_threshold=3, clock=clock)

    async def create_issue(fail: bool) -> dict:
        return await JiraIssueTool(title='Fix login bug', priority='High', project_key='ENG', inject_500_error=fail).execute()

    for _ in range(3):
        with pytest.raises(JiraServerError):
            await breaker.call(create_issue, fail=True)

    assert breaker.state is CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        await breaker.call(create_issue, fail=False)

    clock.advance(30)
    assert (await breaker.call(create_issue, fail=False))['status'] == 'created'
    assert breaker.state is CircuitState.CLOSED