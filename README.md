# Agentic Integration Suite

An SDK, Slack integration, resilience middleware, and evaluation harness for LLM tool calling.

The package name on disk is `src/`; imports use `agentic_suite` (see `pyproject.toml`). Python 3.11+.

## What it does

- **Tool SDK.** Subclass `BaseTool`, get JSON Schema the model can call, validate arguments, and dispatch through `ToolRegistry`.
- **Mock tools.** Offline Jira, GitHub, and Slack tools with injectable failures so retry and circuit-breaker tests do not need live APIs.
- **Slack app.** OAuth install, Fernet-encrypted tokens in Postgres, HMAC request verification, event dispatch, and `event_id` deduplication.
- **Resilience.** Async retries with full-jitter backoff (honors `Retry-After`), a concurrent-safe circuit breaker, and redacted structured telemetry on every registry invocation.
- **Groq client.** Thin async wrapper that parses tool-call arguments and records JSON parse errors instead of crashing.
- **Evals.** Versioned benchmark JSON, Pydantic schema, scoring for tool selection / argument validity / argument match, and dataset integrity tests. Five smoke `direct_intent` cases are in place; the live Groq eval runner is not finished yet.

## Layout

```
src/
  config.py                 Pydantic settings from env / .env
  db.py                     SQLAlchemy engine, sessions, URL normalizer
  logging_config.py         App vs telemetry log formatters
  main.py                   FastAPI: health, Slack OAuth, Slack events
  sdk/base.py               BaseTool ABC + schema generation
  sdk/registry.py           Register / list schemas / execute + telemetry
  tools/mock/               JiraIssueTool, GitHubPRTool, SlackMessageTool
  integrations/slack/       OAuth, crypto, models, signature, events
  middleware/               retry, circuit_breaker, telemetry
  clients/groq_client.py    Async Groq chat + tool-call parsing
migrations/                 Alembic (slack_installations table)
evals/
  schema.py                 BenchmarkCase / ExpectedOutcome loader
  scoring.py                score_case + RunSummary metrics
  data/benchmark.json       Bootstrap: 5 direct-intent smoke cases
  data/system_prompt.txt    Tool-calling conventions for the model
  test_benchmark_data.py    Dataset integrity (no model calls)
tests/                      Unit and integration tests
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `.env`. Required secrets (empty values fail settings validation):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API |
| `SLACK_CLIENT_ID` / `SLACK_CLIENT_SECRET` | OAuth app credentials |
| `SLACK_SIGNING_SECRET` | Incoming request HMAC |
| `SLACK_TOKEN_ENCRYPTION_KEY` | Fernet key for tokens at rest |
| `DATABASE_URL` | Postgres (or SQLite in tests) |

Optional: `GROQ_MODEL_SMOKE` (default `llama-3.1-8b-instant`), `GROQ_MODEL_FULL` (default `openai/gpt-oss-120b`), `APP_BASE_URL`, `ENVIRONMENT`, `LOG_LEVEL`, `SQL_ECHO`, `OAUTH_STATE_TTL_SECONDS` (default 600).

Generate an encryption key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Apply migrations against a real database:

```bash
alembic upgrade head
```

`postgres://` and `postgresql://` URLs are rewritten to `postgresql+psycopg2://`.

## Run the API

```bash
uvicorn agentic_suite.main:app --reload
```

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/slack/install` | Redirect to Slack OAuth (signed, single-use `state`) |
| `GET` | `/slack/oauth/callback` | Verify state, exchange code, persist installation |
| `POST` | `/slack/events` | Signature check, URL challenge, dedupe, dispatch |

Logging is configured at import time. Telemetry lines are raw JSON; everything else is timestamped.

## Tool SDK

`BaseTool` is a Pydantic model plus an async `execute`. The class docstring (or `tool_description`) is what the model reads. `tool_name` defaults to snake_case of the class name.

`ConfigField` marks fields that stay off the LLM schema (injection flags, runtime knobs). `get_schema()` emits OpenAI/Groq function-calling JSON. `validate_arguments()` is `model_validate`.

`ToolRegistry` refuses silent overwrites, lists schemas, and `execute`s by name. Outcomes are logged as `success`, `not_registered`, `validation_error`, `circuit_open`, or `tool_error`.

### Mock tools

| Tool | Name | Arguments | Failure injection |
|---|---|---|---|
| `JiraIssueTool` | `jira_issue_tool` | `title`, `priority`, `project_key` | `inject_500_error` → `JiraServerError` (status 500) |
| `GitHubPRTool` | `github_pr_tool` | `repo`, `pr_id` (int ≥ 1) | `inject_429_rate_limit` → `GitHubRateLimitError` with `Retry-After` |
| `SlackMessageTool` | `slack_message_tool` | `channel` (must start with `#`), `message` | — |

These are for unit tests and evals, not live APIs.

## Slack integration

- **OAuth.** Signed, TTL-limited, single-use state tokens. Token exchange via `oauth.v2.access`. Workspace installs stored as `SlackInstallation`.
- **Tokens at rest.** `EncryptedString` (Fernet) on `access_token` and `refresh_token`.
- **Lifecycle.** Revocation / auth errors set `needs_reauth` (`invalid_auth`, `token_revoked`, `token_expired`, `account_inactive`, `not_authed`).
- **Signatures.** Slack `v0` HMAC; 5-minute timestamp skew; `hmac.compare_digest`.
- **Events.** `url_verification` returns the challenge. `event_callback` drops duplicate `event_id`s (1h TTL). `message` events are logged; bot messages ignored.

Default bot scopes: `chat:write,channels:read`. Redirect URI is `{APP_BASE_URL}/slack/oauth/callback`.

## Middleware

**Retry** (`with_retry`): up to 3 attempts, full-jitter exponential backoff (base 1s, cap 30s). Retries timeouts, connection errors, 429, and 5xx. Does **not** retry `CircuitOpenError`. Honors `Retry-After` when present. Exhaustion raises `RetryExhaustedError`. `RetryTracker` records attempt counts for telemetry.

**Circuit breaker:** closed → open after 3 consecutive countable failures; cooldown 30s; half-open admits one trial. Concurrent in-flight trials on an open/half-open circuit are rejected. `RetryExhaustedError` is unwrapped so the underlying error is what counts. Non-retryable errors do not trip the circuit. Admit/record paths are locked.

**Telemetry:** one JSON line per invocation. Sensitive keys (`token`, `access_token`, `password`, `secret`, `api-key`) are redacted; long strings become `str(len=N)`.

## Groq client

`GroqClient.complete(messages, tools=..., tool_choice="auto")` uses temperature 0. Each tool call is parsed as JSON into `ToolCall`; malformed arguments set `parse_error` and empty `arguments` instead of raising. `ModelResponse` exposes `called_a_tool` and `first_tool_call`.

## Evaluation harness

Dataset: `evals/data/benchmark.json` (schema version 1). Categories: `direct_intent`, `ambiguous`, `edge_case_schema`. Expected behaviours: `tool_call`, `clarify`, `clarify_or_default`, `no_tool_call`.

Current bootstrap: **five `direct_intent` smoke cases** — Jira create (explicit args), GitHub PR status with integer `pr_id`, Slack post preserving `#`, Jira with reordered/lowercase priority, GitHub PR with repo in a second sentence.

`evals/scoring.py` scores:

1. **Selection** — right tool, or a clarifying question (`?`), or no tool when none applies.
2. **Argument validity** — Pydantic schema of the registered tool.
3. **Argument match** — exact fields plus non-empty `flexible_arguments`.

`RunSummary` reports pass rate, selection accuracy, argument precision, and per-category breakdowns.

`evals/data/system_prompt.txt` documents conventions the model should follow (default Jira project `ENG`, priority mapping, `#` on Slack channels, ISO-8601 timestamps, clarify instead of guessing).

Dataset tests (`evals/test_benchmark_data.py`) load the JSON, check named tools exist, expected args validate, smoke subset size, distinct prompts, and that ambiguous cases record notes. They do not call a model.

## Tests

```bash
pytest
```

Coverage includes settings, `BaseTool` schema generation, each mock tool, registry dispatch, OAuth install flow, token encryption/revocation, Slack signatures and `/slack/events`, retry backoff and error classification, circuit-breaker transitions/cooldown/concurrency, and telemetry redaction.

Tests inject a fake environment via `tests/conftest.py`; they do not need a real `.env` or Groq key.

```bash
ruff check .
```