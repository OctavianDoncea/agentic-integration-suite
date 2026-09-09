from __future__ import annotations
import asyncio
import json
import logging
import pytest
from agentic_suite.middleware.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from agentic_suite.middleware.retry import RetryExhaustedError, with_retry
from agentic_suite.sdk.registry import ToolRegistry
from agentic_suite.tools.mock.jira_tool import JiraIssueTool, JiraServerError

N_REQUESTS = 20
FAIL_FIRST = 5
THRESHOLD = 3
COOLDOWN = 30.0

TELEMETRY_LOGGER = 'telemetry.tool_invocation'

class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FlakyDependency:
    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.entered: list[int] = []
        self.succeeded: list[int] = []
        self.failed: list[int] = []

    async def __call__(self, request_id: int) -> dict:
        should_fail = len(self.entered) < self.fail_first
        self.entered.append(request_id)

        await asyncio.sleep(0)

        tool = JiraIssueTool(title=f'Request {request_id}', priority='High', project_key='ENG', inject_500_error=should_fail)
        try:
            result = await tool.execute()
        except JiraServerError:
            self.failed.append(request_id)
            raise

        self.succeeded.append(request_id)
        return result


async def _always_fails() -> None:
    raise JiraServerError()

@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()

@pytest.fixture
def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(name='load-test', failure_threshold=THRESHOLD, cooldown_seconds=COOLDOWN, clock=clock)

@pytest.fixture
def dependency() -> FlakyDependency:
    return FlakyDependency(fail_first=FAIL_FIRST)

async def _burst(breaker: CircuitBreaker, dependency, start: int = 0, n: int = N_REQUESTS):
    return await asyncio.gather(*(breaker.call(dependency, i) for i in range(start, start + n)), return_exceptions=True)

async def _trip(breaker: CircuitBreaker) -> None:
    for _ in range(THRESHOLD):
        with pytest.raises(JiraServerError):
            await breaker.call(_always_fails)

async def test_concurrent_burst_trips_the_circuit_exactly_once(breaker, dependency):
    await _burst(breaker, dependency)

    assert breaker.state is CircuitState.OPEN
    assert breaker.open_transitions == 1

async def test_failure_count_stops_at_the_threshold(breaker, dependency):
    await _burst(breaker, dependency)

    assert breaker.failure_count == THRESHOLD

async def test_every_request_is_executed_at_most_once(breaker, dependency):
    await _burst(breaker, dependency)

    assert len(dependency.entered) == len(set(dependency.entered))
    assert len(dependency.failed) + len(dependency.succeeded) == len(dependency.entered)

async def test_admitted_and_rejected_requests_account_for_all_traffic(breaker, dependency):
    outcomes = await _burst(breaker, dependency)

    rejected = [o for o in outcomes if isinstance(o, CircuitOpenError)]
    assert len(dependency.entered) + len(rejected) == N_REQUESTS

async def test_in_flight_requests_are_not_retroactively_rejected(breaker, dependency):
    outcomes = await _burst(breaker, dependency)

    assert len(dependency.entered) == N_REQUESTS
    assert not any(isinstance(o, CircuitOpenError) for o in outcomes)
    assert len(dependency.failed) == FAIL_FIRST
    assert len(dependency.succeeded) == N_REQUESTS - FAIL_FIRST

async def test_second_burst_is_shed_without_touching_the_dependency(breaker, dependency):
    await _burst(breaker, dependency)
    entered_after_first = len(dependency.entered)
    outcomes = await _burst(breaker, dependency, start=100, n=10)

    assert all(isinstance(o, CircuitOpenError) for o in outcomes)
    assert len(dependency.entered) == entered_after_first

async def test_in_flight_failure_after_the_trip_does_not_extend_the_cooldown(breaker: CircuitBreaker, clock: FakeClock):
    release = asyncio.Event()

    async def slow_failure() -> None:
        await release.wait()
        raise JiraServerError()

    straggler = asyncio.create_task(breaker.call(slow_failure))
    await asyncio.sleep(0)

    await _trip(breaker)
    assert breaker.state is CircuitState.OPEN
    assert breaker.seconds_until_retry() == pytest.approx(COOLDOWN)

    clock.advance(10)
    release.set()
    with pytest.raises(JiraServerError):
        await straggler

    assert breaker.seconds_until_retry() == pytest.approx(20.0)
    assert breaker.open_transitions == 1

async def test_in_flight_success_after_the_trip_does_not_close_the_circuit(breaker: CircuitBreaker):
    release = asyncio.Event()

    async def slow_success() -> dict:
        await release.wait()
        return {'id': 'JIRA-123', 'status': 'created'}

    straggler = asyncio.create_task(breaker.call(slow_success))
    await asyncio.sleep(0)

    await _trip(breaker)
    assert breaker.state is CircuitState.OPEN

    release.set()
    assert await straggler == {'id': 'JIRA-123', 'status': 'created'}

    assert breaker.state is CircuitState.OPEN
    assert breaker.failure_count == THRESHOLD

async def test_recovery_still_works_after_a_stale_burst(breaker, dependency, clock):
    await _burst(breaker, dependency)
    assert breaker.state is CircuitState.OPEN

    clock.advance(COOLDOWN)
    healthy = FlakyDependency(fail_first=0)
    assert (await breaker.call(healthy, 999))['status'] == 'created'

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0

async def test_half_open_admits_exactly_one_concurrent_trial(breaker, clock):
    await _trip(breaker)
    clock.advance(COOLDOWN)

    probe = FlakyDependency(fail_first=0)
    outcomes = await asyncio.gather(*(breaker.call(probe, i) for i in range(10)), return_exceptions=True)

    assert len(probe.entered) == 1
    assert sum(isinstance(o, CircuitOpenError) for o in outcomes) == 9
    assert breaker.state is CircuitState.CLOSED

async def test_concurrent_failed_trials_reopen_the_circuit_once(breaker, clock):
    await _trip(breaker)
    clock.advance(COOLDOWN)

    failing = FlakyDependency(fail_first=100)
    outcomes = await asyncio.gather(*(breaker.call(failing, i) for i in range(10)), return_exceptions=True)

    assert len(failing.entered) == 1
    assert sum(isinstance(o, CircuitOpenError) for o in outcomes) == 9
    assert breaker.state is CircuitState.OPEN
    assert breaker.open_transitions == 2
    assert breaker.seconds_until_retry() == pytest.approx(COOLDOWN)

async def _noop_sleep(_: float) -> None:
    return None

async def test_retry_and_breaker_compose_under_concurrent_load(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=THRESHOLD, cooldown_seconds=COOLDOWN, clock=clock)
    dependency = FlakyDependency(fail_first=10_000)
    guarded = with_retry(max_attempts=3, sleep=_noop_sleep)(dependency)
    outcomes = await asyncio.gather(*(breaker.call(guarded, i) for i in range(N_REQUESTS)), return_exceptions=True)

    assert all(isinstance(o, RetryExhaustedError) for o in outcomes)
    assert len(dependency.entered) == N_REQUESTS * 3
    assert breaker.state is CircuitState.OPEN
    assert breaker.open_transitions == 1

async def test_open_breaker_short_circuits_before_any_retry(clock: FakeClock):
    breaker = CircuitBreaker(failure_threshold=THRESHOLD, cooldown_seconds=COOLDOWN, clock=clock)
    dependency = FlakyDependency(fail_first=10_000)
    guarded = with_retry(max_attempts=3, sleep=_noop_sleep)(dependency)

    await asyncio.gather(*(breaker.call(guarded, i) for i in range(N_REQUESTS)), return_exceptions=True)
    calls_before = len(dependency.entered)

    outcomes = await asyncio.gather(*(breaker.call(guarded, i) for i in range(100, 110)), return_exceptions=True)

    assert all(isinstance(o, CircuitOpenError) for o in outcomes)
    assert len(dependency.entered) == calls_before

async def test_every_concurrent_invocation_emits_exactly_one_telemetry_line(caplog):
    registry = ToolRegistry()
    registry.register(JiraIssueTool)

    args = [
        {
            'title': f'Request {i}',
            'priority': 'High',
            'project_key': 'ENG',
            'inject_500_error': i < FAIL_FIRST
        } for i in range(N_REQUESTS)
    ]

    with caplog.at_level(logging.INFO, logger=TELEMETRY_LOGGER):
        await asyncio.gather(*(registry.execute('jira_issue_tool', a) for a in args), return_exceptions=True)

    records = [json.loads(r.message) for r in caplog.records if r.name == TELEMETRY_LOGGER]

    assert len(records) == N_REQUESTS
    assert sum(r['status'] == 'success' for r in records) == N_REQUESTS - FAIL_FIRST
    assert sum(r['status'] == 'tool_error' for r in records) == FAIL_FIRST
    assert all(r['latency_ms'] >= 0 for r in records)