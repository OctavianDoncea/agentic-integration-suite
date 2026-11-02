# Adding a tool

A tool is a `BaseTool` subclass. The class is the schema the model sees, the validator for the arguments that come back, and the function that runs.

## 1. Subclass `BaseTool`

Put a mock under `src/tools/mock/`. Put a tool that calls a real API under `src/integrations/`.

The class docstring is the description the model reads. `tool_name` defaults to the class name in snake_case. Override it when the live tool and its offline stand-in must share a name, the way `SlackPostMessageTool` and `SlackMessageTool` both use `slack_message_tool`.

Arguments are Pydantic fields. `ConfigField` keeps a field off the model schema (failure injection, runtime knobs). `execute` is async and returns a dict. Keyword arguments that are not schema fields (`session`, `team_id`) are runtime context from the caller.

## 2. Set the behaviour flags

These class variables are not schema fields. MCP hosts read them as annotations, and the write policy reads two of them.

| Flag | Default | Effect |
|---|---|---|
| `display_title` | class name | Title shown by an MCP host |
| `read_only` | `False` | Read-only tools skip the write policy. MCP marks them read-only |
| `destructive` | `True` | MCP hint. Only sent when the tool is not read-only |
| `idempotent` | `False` | MCP hint. Only sent when the tool is not read-only |
| `open_world` | `True` | MCP hint: the tool talks to a system outside this process |
| `simulated` | `False` | Appends a note that the tool returns mock data |
| `target_fields` | `()` | Argument names that choose a destination |

If the tool writes to a destination named by one of its arguments (a channel, a project, a repository), list that field in `target_fields`. The agent's write policy then requires the value to appear in the user's own request. `SlackPostMessageTool` lists `channel`. `JiraIssueTool` lists `project_key`. `GitHubPRTool` is `read_only`, so it has no target fields.

Leave `target_fields` empty when a write has no destination argument. The policy still previews it unless the request sets `allow_writes`.

## 3. Register it

The live agent and the MCP server use `build_live_registry` in `src/agents/registry.py`. Add the class there.

The 50-case benchmark uses the registry in `evals/test_evals.py` and `evals/test_benchmark_data.py`. Register the offline tool there, not the live one, so a sweep cannot post to Slack or call a real API.

`ToolRegistry.register` refuses a second tool with the same name.

## 4. Test it

Add a unit test that builds the tool with valid arguments, checks `execute`, and checks that invalid arguments fail `model_validate`. If the tool writes, assert the `target_fields` tuple and that a missing target is blocked by `agentic_suite.agents.policy.evaluate`.

## 5. Add benchmark cases

When the model should be scored on the tool, add cases to `evals/data/benchmark.json`. Expected arguments must validate against the tool. Keep the category counts the dataset tests enforce (15 direct, 15 ambiguous, 20 edge-case), or update those tests in the same change. A prompt or system-prompt change invalidates `evals/baseline_results.json`. Refresh it in the same commit.

Adversarial cases, for a write the policy should refuse, go in `evals/adversarial/cases.json`.
