from __future__ import annotations
from evals.throttle import MinIntervalLimiter, resolve_rpm

class FakeClock:
    """A clock whose sleep advances time instead of waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_spaces_back_to_back_requests():
    clock = FakeClock()
    limiter = MinIntervalLimiter(60, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        await limiter.wait()
    assert clock.sleeps == [1.0, 1.0]

async def test_does_not_wait_when_requests_are_already_spaced():
    clock = FakeClock()
    limiter = MinIntervalLimiter(60, clock=clock, sleep=clock.sleep)
    await limiter.wait()
    clock.now += 5
    await limiter.wait()
    assert clock.sleeps == []

async def test_disabled_limiter_never_waits():
    clock = FakeClock()
    limiter = MinIntervalLimiter(None, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        await limiter.wait()
    assert clock.sleeps == []

def test_rpm_comes_from_the_registry_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv('EVAL_RPM', raising=False)
    assert resolve_rpm('cohere') == 18.0
    assert resolve_rpm('groq') is None
    assert resolve_rpm('openai_compat') is None
    assert resolve_rpm('not-a-provider') is None

    monkeypatch.setenv('EVAL_RPM', '30')
    assert resolve_rpm('groq') == 30.0

    monkeypatch.setenv('EVAL_RPM', '0')
    assert resolve_rpm('cohere') is None