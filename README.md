# Agentic Integration Suite

A Python toolkit for LLM tool calling: a tool SDK, a Slack app, an agent loop, an MCP server, resilience middleware, and an evaluation harness.

The package on disk is `src/`; imports use `agentic_suite` (see `pyproject.toml`). Requires Python 3.11 or 3.12.

## What it does

- **Tool SDK.** Subclass `BaseTool`, get a JSON Schema the model can call, validate arguments, and dispatch through `ToolRegistry`.
- **Agent.** `POST /agent/run` runs a prompt through a chat model and the live tools until the model answers or hits a step limit. Model calls are retried; each tool call goes through a circuit breaker.
- **MCP server.** The same live registry is exposed over stdio as `agentic-suite-mcp`, so another client can list and call the tools.
- **Slack app.** OAuth install, Fernet-encrypted tokens in Postgres, HMAC request verification, event dispatch, and `event_id` deduplication. The live Slack tool posts with the stored bot token.
- **Mock tools.** Offline Jira, GitHub, Slack, and calendar tools, with injectable failures so retry and circuit-breaker tests do not need live APIs.
- **Providers.** One `ChatClient` interface over Groq, Cohere, and any OpenAI-compatible endpoint. Tool-call arguments that are not valid JSON are recorded instead of crashing the caller.
- **Resilience.** Async retries with full-jitter backoff (honors `Retry-After`), a concurrent-safe circuit breaker, and redacted structured telemetry on every tool invocation.
- **Evals.** A 50-case benchmark, scoring for tool selection and argument quality, CI thresholds derived from committed baselines, and a cross-model comparison report.

## Layout

```
src/
  config.py                 Pydantic settings from env / .env
  db.py                     SQLAlchemy engine, sessions, URL normalizer
  logging_config.py         App vs telemetry log formatters
  main.py                   FastAPI: health, Slack OAuth, events, /agent/run
  mcp_server.py             stdio MCP server over the live registry
  agents/                   loop, executor, live registry, system prompt
  sdk/base.py               BaseTool ABC + schema generation
  sdk/registry.py           Register / list schemas / execute + telemetry
  tools/mock/               Jira, GitHub, Slack, calendar (offline)
  integrations/slack/       OAuth, crypto, models, signature, events, live post tool
  middleware/               retry, circuit_breaker, telemetry
  clients/                  ChatClient, Groq, Cohere, OpenAI-compatible, schema adapter
migrations/                 Alembic (slack_installations table)
evals/
  schema.py                 BenchmarkCase / ExpectedOutcome loader
  scoring.py                score_case + RunSummary metrics
  threshold.py              CI floors = baseline pass rate − 5 points
  baseline.py               Record a sweep (CI baseline or a comparison run)
  compare.py                Wilson intervals and a disagreement table
  throttle.py               Per-provider request spacing
  test_evals.py             Live smoke and full sweeps (pytest markers)
  data/benchmark.json       50 cases; 5 marked smoke
  baseline_results.json     Full-sweep baseline (gpt-oss-120b)
  baseline_smoke.json       Smoke baseline (gpt-oss-20b)
  comparisons/              Saved cross-model runs and REPORT.md
tests/                      Unit and integration tests
```

## Setup

```bash
python -m venv .venv
```

Activate it (`.\.venv\Scripts\Activate.ps1` on PowerShell, `source .venv/bin/activate` elsewhere), then:

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` on PowerShell, `cp .env.example .env` elsewhere) and fill it in. Empty required values fail settings validation.

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq API (also the default eval provider) |
| `SLACK_CLIENT_ID` / `SLACK_CLIENT_SECRET` | yes | OAuth app credentials |
| `SLACK_SIGNING_SECRET` | yes | Incoming request HMAC |
| `SLACK_TOKEN_ENCRYPTION_KEY` | yes | Fernet key for tokens at rest |
| `DATABASE_URL` | yes | Postgres (tests can use SQLite) |
| `AGENT_API_KEY` | for `/agent/run` | Shared secret checked as `X-Agent-Key`. Unset disables the endpoint (503) |
| `AGENT_MODEL` | no | `provider:model` or a bare model id. Falls back to `GROQ_MODEL_FULL` |
| `DEFAULT_SLACK_TEAM_ID` | no | Workspace used when a request does not name one |
| `COHERE_API_KEY` | for Cohere | Cohere chat |
| `COHERE_THINKING` | no | Passed through to the Cohere client when set |
| `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY` | for that provider | OpenAI-compatible chat endpoint |
| `OPENAI_COMPAT_ADAPT_SCHEMA` | no | Rewrite numeric/length constraints into the field description (default `false`; the example sets `true`) |

Other defaults: `GROQ_MODEL_SMOKE=openai/gpt-oss-20b`, `GROQ_MODEL_FULL=openai/gpt-oss-120b`, `APP_BASE_URL=http://localhost:8000`, `ENVIRONMENT=development`, `LOG_LEVEL=INFO`, `SQL_ECHO=false`, `OAUTH_STATE_TTL_SECONDS=600`, `SLACK_BOT_SCOPES=chat:write,channels:read`.

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
| `POST` | `/agent/run` | Run one prompt. Requires header `X-Agent-Key` |

`POST /agent/run` body:

```json
{"prompt": "Post hello to #general", "team_id": "T0123"}
```

`team_id` is optional. Resolution order is the request value, then `DEFAULT_SLACK_TEAM_ID`, then the only Slack installation that does not need reauth. A missing workspace is fine for mock tools; the Slack tool reports it as a tool error the model can explain.

The response includes `model`, `stop_reason` (`completed` or `max_steps`), `final_text`, and a `steps` list of `{tool, status, error}`. The loop allows 5 model turns. A provider failure returns 502.

Logging is configured at import time. Telemetry lines are raw JSON; everything else is timestamped.

## MCP server

```bash
agentic-suite-mcp
```

This is the `project.scripts` entry point (`agentic_suite.mcp_server:main`). It speaks MCP over stdio and serves the live registry: real Slack plus the Jira, GitHub, and calendar mocks. Simulated tools append a note to their description. Logs go to stderr so they do not corrupt the stdio stream. Each call opens a database session, resolves the Slack team, and commits or rolls back with the call.

Point an MCP client at the command. It needs the same `.env` as the API (`DATABASE_URL`, Slack credentials, encryption key).

## Tool SDK

`BaseTool` is a Pydantic model plus an async `execute`. The class docstring (or `tool_description`) is what the model reads. `tool_name` defaults to snake_case of the class name. Class flags (`read_only`, `destructive`, `idempotent`, `open_world`, `simulated`, `display_title`) become MCP annotations.

`ConfigField` marks fields that stay off the LLM schema (injection flags, runtime knobs). `get_schema()` emits OpenAI-style function-calling JSON. `validate_arguments()` is `model_validate`.

`ToolRegistry` refuses silent overwrites, lists schemas, and executes by name. `ToolExecutor` is the path the agent and the MCP server share: per-tool circuit breaker, retry, then the registry. Outcomes are logged as `success`, `not_registered`, `validation_error`, `circuit_open`, or `tool_error`.

### Tools

| Tool | Name | Role | Arguments | Failure injection |
|---|---|---|---|---|
| `JiraIssueTool` | `jira_issue_tool` | mock | `title`, `priority`, `project_key` | `inject_500_error` → `JiraServerError` (status 500) |
| `GitHubPRTool` | `github_pr_tool` | mock, read-only | `repo`, `pr_id` (int ≥ 1) | `inject_429_rate_limit` → `GitHubRateLimitError` with `Retry-After` |
| `CalendarEventTool` | `calendar_event_tool` | mock | `title`, `start_time` (ISO-8601), `duration_minutes` (5–400), optional `attendees`, `visibility`, `send_invites` | — |
| `SlackMessageTool` | `slack_message_tool` | mock, evals and unit tests | `channel` (must start with `#`), `message` | — |
| `SlackPostMessageTool` | `slack_message_tool` | live (`chat.postMessage`) | same as the mock | Slack 429 and auth errors |

The live registry registers `SlackPostMessageTool`, not the mock. Evals register the mock so a sweep never posts to a workspace. Both classes share the name `slack_message_tool`, so a model trained against the benchmark calls the same tool the agent exposes.

## Slack integration

- **OAuth.** Signed, TTL-limited, single-use state tokens. Token exchange via `oauth.v2.access`. Workspace installs stored as `SlackInstallation`.
- **Tokens at rest.** `EncryptedString` (Fernet) on `access_token` and `refresh_token`.
- **Lifecycle.** Revocation and auth errors set `needs_reauth` (`invalid_auth`, `token_revoked`, `token_expired`, `account_inactive`, `not_authed`).
- **Signatures.** Slack `v0` HMAC; 5-minute timestamp skew; `hmac.compare_digest`.
- **Events.** `url_verification` returns the challenge. `event_callback` drops duplicate `event_id`s (1h TTL). `message` events are logged; bot messages are ignored.

Default bot scopes: `chat:write,channels:read`. Redirect URI is `{APP_BASE_URL}/slack/oauth/callback`.

## Providers

`make_chat_client(spec)` builds a `ChatClient`. A spec is `provider:model` or a bare model id.

| Provider | How it is selected | Notes |
|---|---|---|
| `groq` | default | `GroqClient`. Used for CI baselines |
| `cohere` | prefix `cohere:`, or a model id starting with `command-` or `c4ai-` | Default eval throttle is 18 requests/minute |
| `openai_compat` | prefix `openai_compat:` | Needs `OPENAI_COMPAT_BASE_URL` and `OPENAI_COMPAT_API_KEY` |

`complete(messages, tools=..., tool_choice="auto")` uses temperature 0. Each tool call is parsed into `ToolCall`; malformed arguments set `parse_error` and empty `arguments` instead of raising. `ModelResponse` exposes `called_a_tool` and `first_tool_call`. Provider-specific message formatting stays on the client (`format_assistant_turn`, `format_tool_result`).

`OPENAI_COMPAT_ADAPT_SCHEMA` moves constraints some endpoints reject (`minimum`, `maximum`, `minLength`, `maxLength`, and the related bounds) into the field description so the schema still reaches the model.

## Middleware

**Retry** (`with_retry`): up to 3 attempts, full-jitter exponential backoff (base 1s, cap 30s). Retries timeouts, connection errors, 429, and 5xx. Does **not** retry `CircuitOpenError`. Honors `Retry-After` when present. Exhaustion raises `RetryExhaustedError`. `RetryTracker` records attempt counts for telemetry. Eval sweeps use a wider budget (5 attempts, 60s cap) plus `MinIntervalLimiter`.

**Circuit breaker:** closed → open after 3 consecutive countable failures; cooldown 30s; half-open admits one trial. Concurrent in-flight trials on an open or half-open circuit are rejected. `RetryExhaustedError` is unwrapped so the underlying error is what counts. Non-retryable errors do not trip the circuit. Admit and record paths are locked. The executor keeps one breaker per tool name.

**Telemetry:** one JSON line per invocation. Sensitive keys (`token`, `access_token`, `password`, `secret`, `api-key`) are redacted; long strings become `str(len=N)`.

## Evaluation harness

Dataset: `evals/data/benchmark.json` (schema version 1), 50 cases.

| Category | Cases | What it checks |
|---|---|---|
| `direct_intent` | 15 | The request names the tool and the arguments |
| `ambiguous` | 15 | The model should clarify, or apply a documented default |
| `edge_case_schema` | 20 | Types, ranges, and formats the schema enforces |

Five `direct_intent` cases are marked `smoke`. Expected behaviours: `tool_call`, `clarify`, `clarify_or_default`, `no_tool_call`.

Two metrics are scored separately, because picking the wrong tool and filling the right tool badly are different failures:

1. **Selection** — right tool, or a clarifying question (`?`), or no tool when none applies.
2. **Argument precision** — arguments validate against the tool's Pydantic model, and expected fields match. Non-empty `flexible_arguments` only have to be present. Averaged only over cases where a tool call was expected and made, so a correct abstention does not inflate the number.

`RunSummary` also reports pass rate and a per-category breakdown. The system prompt the model sees is `src/agents/system_prompt.txt` (default Jira project `ENG`, priority mapping, `#` on Slack channels, ISO-8601 timestamps, clarify instead of guessing).

Dataset tests (`evals/test_benchmark_data.py`) load the JSON, check named tools exist, expected args validate, the smoke subset size, distinct prompts, and that ambiguous cases record notes. They do not call a model.

### Baseline and CI

CI thresholds come from recorded sweeps. `openai/gpt-oss-120b` scored **90%** over all 50 cases; `openai/gpt-oss-20b` scored **100%** over the 5-case smoke subset. Both runs are committed as `evals/baseline_results.json` and `evals/baseline_smoke.json`. `evals/threshold.py` subtracts a 5-point margin from those files, so the threshold and the baseline cannot drift apart. The margin covers run-to-run variance (two consecutive `temperature=0` sweeps differed by 1 case) and provider-side model updates. Last refreshed: **2026-09-29**.

GitHub Actions:

- **Unit tests** on every push and pull request (Python 3.11 and 3.12), plus a migration job against Postgres.
- **Smoke eval** on every push and same-repo pull request, against `EVAL_MODEL_SMOKE` or `openai/gpt-oss-20b`.
- **Full sweep** nightly (`0 3 * * *`) and on manual dispatch. A model other than the baseline model is exploratory: a miss means "different", not "regressed".

Smoke and sweep tests are pytest-marked and excluded from a plain `pytest` run (`-m 'not smoke and not sweep'`).

### Refreshing a baseline

Refresh when the benchmark changes, when `src/agents/system_prompt.txt` changes (ambiguous cases depend on the defaults it documents), or when a model version changes. CI baselines are Groq runs; any other provider must be written somewhere else with `--out`.

```bash
python -m evals.baseline --model openai/gpt-oss-120b
python -m evals.baseline --model openai/gpt-oss-20b --smoke
```

Commit the regenerated baseline files in the same commit as whatever caused the refresh, so the diff shows the cause and the effect together.

Never raise a threshold to make a red build green. A failing sweep means either a real regression or a bad case.

### Comparing models

```bash
python -m evals.baseline --model cohere:command-a-03-2025 --out evals/comparisons/command-a-run1.json
python -m evals.compare evals/comparisons --out evals/comparisons/REPORT.md
```

`compare` prints pass rate, selection accuracy, argument precision, a 95% Wilson interval, per-category counts, and the cases where runs disagree. Overlapping intervals are called out; overlap is a conservative check, not a significance test. Set `EVAL_RPM` to override a provider's default spacing. Cohere defaults to 18 requests per minute. `EVAL_RPM=0` turns the limiter off.

### Known limitations

Scoring:

- Clarification detection is a substring check for a question mark, not a model-graded judgment.
- `temperature=0` reduces but does not eliminate variance on a hosted API.
- Argument precision ignores cases that correctly abstain.
- 50 cases give wide confidence intervals. The benchmark and the system prompt were developed against the Groq baseline, so a gap in that model's favour is partly home advantage.

The running app:

- **Single shared key.** `/agent/run` is gated by one API key, not per-user auth, and has no per-key rate limit.
- **Untrusted tool output.** Tool results go back to the model verbatim. A Slack message or an external API response could carry a prompt injection, and the agent loop has no isolation boundary or write-approval policy.
- **In-memory OAuth state.** Install `state` values live in the process. They are lost on restart and do not work across multiple instances.
- **One workspace.** The agent and the MCP server act in the workspace from `DEFAULT_SLACK_TEAM_ID`, or the single healthy installation, not a workspace chosen by the MCP client.
- **No conversation memory.** Each `/agent/run` starts fresh. Past runs are not stored.
- **Ambiguous post outcomes.** A Slack post that times out may or may not have landed. The tool does not retry it, which avoids duplicates and can leave the outcome unknown.
- **stdio-only MCP.** There is no remote MCP endpoint. Serving the Slack tool over HTTP would need MCP authorization so only approved clients can post as the bot.
- **Mocked integrations.** Jira, GitHub, and Calendar do not touch real services.

## Future work

Ordered by how much each one improves the project. The first two address limitations above.

1. **Write-action policy and adversarial evals.** Add an `allow_writes` request flag so non-read-only tools return a preview instead of executing, using the behaviour metadata that already drives MCP annotations. Add benchmark cases where a tool result or user message carries injected instructions, scored on whether the agent attempts the write.
2. **Multi-step trajectory evals.** Add cases where the second call depends on the first call's result, scored on the whole run as a separate category so the single-call numbers stay comparable.
3. **Persist agent runs.** Store each run's steps, errors, tokens, and latency, with a gated `GET /agent/runs/{id}`. Failures stay debuggable after the fact, and the comparison report can grow cost and latency columns.
4. **Per-key auth and rate limiting on `/agent/run`.** Store hashed, labelled keys in the database, rate-limit per key, and log which key made each call.
5. **Idempotent posting.** Record each intended post before sending, and after a timeout check the channel history before retrying. This needs an extra Slack scope, so the app must be reinstalled.
6. **GitHub as a second real OAuth provider.** Make the OAuth module provider-agnostic and add GitHub. The MCP server builds from the registry, so the new tool would show up in hosts without server changes. It also forces a real answer to how the connected account is chosen.
7. **Smaller follow-ups:**
   - A short screen recording of an MCP host posting to Slack, with the approval prompt visible.
   - A one-page write-up of the cross-model results.
   - Hypothesis property tests on schema generation and the schema adapter.
   - Dependabot, CodeQL, `pip-audit`, and PyPI publishing through trusted publishing so no token is stored.

Remote MCP over HTTP stays out of scope. Doing it properly means implementing MCP's OAuth authorization and securing a public endpoint that can post as the bot.

## Tests

```bash
pytest
```

That runs unit tests and the eval tests that do not call a model. Coverage includes settings, `BaseTool` schema generation, each mock tool, registry dispatch, the agent loop and `/agent/run`, OAuth install flow, token encryption and revocation, Slack signatures and `/slack/events`, the live Slack post tool, retry backoff and error classification, circuit-breaker transitions, cooldown, and concurrency, telemetry redaction, provider clients, the schema adapter, and the MCP server.

Tests inject a fake environment via `tests/conftest.py`. They do not need a real `.env` or a Groq key.

Live evals (these call the provider named by `EVAL_MODEL`):

```bash
pytest -m smoke
pytest -m sweep
```
