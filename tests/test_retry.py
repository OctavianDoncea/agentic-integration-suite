from __future__ import annotations
import asyncio
import pytest
from collections.abc import Awaitable, Callable
from pydantic import ValidationError
from agentic_suite.middleware.retry import RetryExhaustedError, compute_backoff, get_retry_after, is_retryable, with_retry
from agentic_suite.tools.mock.github_tool import GitHubPRTool, GitHubRateLimitError
from agentic_suite.tools.mock.jira_tool import JiraIssueTool, JiraServerError

class SleepRecorder:
    """Replacement for 'asyncio.sleep' that records delays instead of waiting."""
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)

    @property
    def call_count(self) -> int:
        return len(self.delays)

def fixed_rng(value: float = 1.0) -> Callable[[], float]:
    return lambda: value

def failing_n_times(n: int, exc: BaseException) -> Callable[[], Awaitable[str]]:
    state = {'calls': 0}

    async def fn() -> str:
        state['calls'] += 1
        if state['calls'] <= n:
            raise exc
        
        return 'ok'

    fn.call_count = lambda: state['calls']

    return fn

def test_backoff_is_exponential_with_jitter_pinned_high():
    assert compute_backoff(0, base_delay=1.0, rng=fixed_rng(1.0)) == 1.0
    assert compute_backoff(1, base_delay=1.0, rng=fixed_rng(1.0)) == 2.0
    assert compute_backoff(2, base_delay=1.0, rng=fixed_rng(1.0)) == 4.0
    assert compute_backoff(3, base_delay=1.0, rng=fixed_rng(1.0)) == 8.0

def test_full_jitter_scales_the_whole_interval():
    assert compute_backoff(2, base_delay=1.0, rng=fixed_rng(0.5)) == 2.0
    assert compute_backoff(2, base_delay=1.0, rng=fixed_rng(0.0)) == 0.0

def test_backoff_is_capped_at_max_wait():
    assert compute_backoff(10, base_delay=1.0, max_wait=30.0, rng=fixed_rng(1.0)) == 30.0

def test_real_rng_stays_within_the_expected_interval():
    for _ in range(200):
        assert 0.0 <= compute_backoff(3, base_delay=1.0, max_wait=30.0) <= 8.0

async def test_retries_then_succeeds():
    sleeper = SleepRecorder()
    fn = failing_n_times(2, JiraServerError())
    wrapped = with_retry(max_attempts=3, sleep=sleeper, rng=fixed_rng())(fn)

    assert await wrapped() == 'ok'
    assert fn.call_count() == 3
    assert sleeper.call_count == 2

async def test_stops_after_max_attempts():
    sleeper = SleepRecorder()
    fn = failing_n_times(99, JiraServerError())
    wrapped = with_retry(max_attempts=3, sleep=sleeper, rng=fixed_rng())(fn)

    with pytest.raises(RetryExhaustedError) as exc_info:
        await wrapped()

    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, JiraServerError)
    assert isinstance(exc_info.value.__cause__, JiraServerError)
    assert fn.call_count() == 3
    assert sleeper.call_count == 2

async def test_no_sleep_after_the_final_failed_attempts():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=1, sleep=sleeper, rng=fixed_rng())(failing_n_times(99, JiraServerError()))

    with pytest.raises(RetryExhaustedError):
        await wrapped()

    assert sleeper.call_count == 0

async def test_success_on_first_call_never_sleeps():
    sleeper = SleepRecorder()

    @with_retry(max_attempts=3, sleep=sleeper)
    async def always_works() -> str:
        return 'ok'

    assert await always_works() == 'ok'
    assert sleeper.call_count == 0

async def test_delays_follow_the_exponential_sequence():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=4, base_delay=1.0, sleep=sleeper, rng=fixed_rng())(failing_n_times(99, JiraServerError()))

    with pytest.raises(RetryExhaustedError):
        await wrapped()

    assert sleeper.delays == [1.0, 2.0, 4.0]

async def test_retry_after_takes_precedence_over_jitter():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=3, sleep=sleeper, rng=fixed_rng(1.0))(failing_n_times(2, GitHubRateLimitError(retry_after_seconds='2')))

    assert await wrapped() == 'ok'
    assert sleeper.delays == [2.0, 2.0]

async def test_retry_after_is_still_capped_by_max_wait():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=2, max_wait=30.0, sleep=sleeper, rng=fixed_rng())(failing_n_times(99, GitHubRateLimitError(retry_after_seconds='86400')))

    with pytest.raises(RetryExhaustedError):
        await wrapped()

    assert sleeper.delays == [30.0]

async def test_respect_retry_after_can_be_disabled():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=2, sleep=sleeper, rng=fixed_rng(1.0), respect_retry_after=False)(failing_n_times(99, GitHubRateLimitError(retry_after_seconds='2')))

    with pytest.raises(RetryExhaustedError):
        await wrapped()

    assert sleeper.delays == [1.0]

def test_retry_after_extraction():
    assert get_retry_after(GitHubRateLimitError('7')) == 7.0
    assert get_retry_after(JiraServerError()) is None
    assert get_retry_after(ValueError('no headers')) is None

def test_unparseable_retry_after_falls_back_to_jitter():
    exc = GitHubRateLimitError()
    exc.headers = {'Retry-After': 'Wed, 21 Oct 2026 7:28:00 GMT'}
    assert get_retry_after(exc) is None

@pytest.mark.parametrize(
    ('exc', 'expected'),
    [
        (JiraServerError(500), True),
        (JiraServerError(503), True),
        (GitHubRateLimitError(), True),
        (asyncio.TimeoutError(), True),
        (ConnectionError(), True),
        (ValueError('bad input'), False),
        (KeyError('missing'), False)
    ]
)
def test_retryable_classification(exc: BaseException, expected: bool):
    assert is_retryable(exc) is expected

async def test_terminal_errors_are_raised_immediately_and_unwrapped():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=3, sleep=sleeper)(failing_n_times(99, ValueError('bad argument')))

    with pytest.raises(ValueError, match='bad argument'):
        await wrapped()

    assert sleeper.call_count == 0

async def test_pydantic_validation_errors_are_not_retried():
    sleeper = SleepRecorder()

    @with_retry(max_attempts=3, sleep=sleeper)
    async def bad_arguments():
        return JiraIssueTool.validate_arguments({'title': 'only this'})

    with pytest.raises(ValidationError):
        await bad_arguments()

    assert sleeper.call_count == 0

async def test_custom_predicate_overrides_the_default():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=3, sleep=sleeper, rng=fixed_rng(), retryable=lambda exc: isinstance(exc, ValueError))(failing_n_times(1, ValueError('transient here')))

    assert await wrapped() == 'ok'
    assert sleeper.call_count == 1

async def test_wraps_a_real_mock_tool_recovering_from_a_500():
    sleeper = SleepRecorder()
    state = {'calls': 0}

    @with_retry(max_attempts=3, sleep=sleeper, rng=fixed_rng())
    async def create_issue() -> dict:
        state['calls'] += 1
        tool = JiraIssueTool(title='Fix login bug', priority='High', project_key='ENG', inject_500_error=state['calls'] < 3)
        return await tool.execute()

    result = await create_issue()
    assert result['status'] == 'created'
    assert sleeper.delays == [1.0, 2.0]

async def test_wraps_a_real_mock_tool_hitting_a_rate_limit():
    sleeper = SleepRecorder()
    wrapped = with_retry(max_attempts=2, sleep=sleeper, rng=fixed_rng(1.0))(GitHubPRTool(repo='name/hello-world', pr_id=1, inject_429_rate_limit=True).execute)

    with pytest.raises(RetryExhaustedError):
        await wrapped()

    assert sleeper.delays == [2.0]

def test_decorator_preserves_function_metadata():
    @with_retry(max_attempts=2)
    async def documented_function() -> None:
        """A docstring that must survive wrapping."""

    assert documented_function.__name__ == 'documented_function'
    assert 'must survive' in documented_function.__doc__

def test_zero_max_attempts_is_rejected_at_decoration_time():
    with pytest.raises(ValueError, match='at least 1'):
        with_retry(max_attempts=0)