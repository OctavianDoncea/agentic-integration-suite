# Cross-model eval comparison

## Overall

| Run | Model | Passed | Pass rate | 95% CI | Tool selection | Argument precision | API errors |
|---|---|---|---|---|---|---|---|
| command-a-plus-run1 | `cohere:command-a-plus-05-2026` | 46/50 | 92.0% | 81.2%-96.8% | 92.0% | 100.0% | 0 |
| command-a-run1 | `cohere:command-a-03-2025` | 46/50 | 92.0% | 81.2%-96.8% | 92.0% | 100.0% | 0 |
| command-r7b-run1 | `cohere:command-r7b-12-2024` | 44/50 | 88.0% | 76.2%-94.4% | 92.0% | 94.6% | 0 |
| gemini-3.5-flash-lite-run2 | `openai_compat:gemini-3.5-flash-lite` | 43/50 | 86.0% | 73.8%-93.0% | 86.0% | 100.0% | 0 |
| gpt-oss-120b-run1 | `openai/gpt-oss-120b` | 45/50 | 90.0% | 78.6%-95.7% | 96.0% | 92.3% | 0 |
| gpt-oss-20b-run1 | `groq:openai/gpt-oss-20b` | 44/50 | 88.0% | 76.2%-94.4% | 94.0% | 92.1% | 1 |

The 95% intervals of **command-a-plus-run1** and **command-a-run1**, **command-r7b-run1**, **gemini-3.5-flash-lite-run2**, **gpt-oss-120b-run1**, **gpt-oss-20b-run1** overlap, so at 50 cases their difference isn't statistically meaningful.(Overlap is a conservative check, not a formal significance test.)

## By category

| Category | command-a-plus-run1 | command-a-run1 | command-r7b-run1 | gemini-3.5-flash-lite-run2 | gpt-oss-120b-run1 | gpt-oss-20b-run1 |
|---|---|---|---|---|---|---|
| ambiguous | 12/15 | 11/15 | 10/15 | 8/15 | 12/15 | 12/15 |
| direct_intent | 15/15 | 15/15 | 14/15 | 15/15 | 13/15 | 13/15 |
| edge_case_schema | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 |

## Cases where runs disagree (12)

| Case | command-a-plus-run1 | command-a-run1 | command-r7b-run1 | gemini-3.5-flash-lite-run2 | gpt-oss-120b-run1 | gpt-oss-20b-run1 |
|---|---|---|---|---|---|---|
| `ambiguous-002` | pass | fail: did not ask a clarifying question | fail: called 'github_pr_tool' instead of asking | fail: called 'github_pr_tool' instead of asking | pass | pass |
| `ambiguous-003` | pass | pass | pass | fail: called 'slack_message_tool' instead of asking | pass | pass |
| `ambiguous-004` | pass | fail: called 'calendar_event_tool' instead of asking | pass | fail: called 'calendar_event_tool' instead of asking | pass | fail: API error: BadRequestError: BadRequestError('Error code: 40... |
| `ambiguous-005` | pass | pass | pass | pass | fail: argument 'project_key': expected 'PLATFORM' (str), got 'ENG... | pass |
| `ambiguous-006` | pass | pass | fail: argument 'priority': expected 'High' (str), got 'critical' ... | pass | pass | pass |
| `ambiguous-009` | pass | pass | fail: called 'github_pr_tool' instead of asking | fail: called 'slack_message_tool' instead of asking | pass | pass |
| `ambiguous-010` | fail: called 'jira_issue_tool' instead of asking | pass | pass | fail: called 'slack_message_tool' instead of asking | fail: called 'slack_message_tool' instead of asking | fail: called 'jira_issue_tool' instead of asking |
| `ambiguous-011` | fail: called 'github_pr_tool' instead of asking | fail: called 'github_pr_tool' instead of asking | fail: called 'github_pr_tool' instead of asking | fail: called 'github_pr_tool' instead of asking | pass | pass |
| `direct-008` | pass | pass | pass | pass | fail: argument 'project_key': expected 'INFRA' (str), got 'ENG' (... | fail: argument 'project_key': expected 'INFRA' (str), got 'ENG' (... |
| `direct-012` | pass | pass | fail: argument 'project_key': expected 'DATA' (str), got 'ENG' (s... | pass | fail: argument 'project_key': expected 'DATA' (str), got 'ENG' (s... | fail: argument 'project_key': expected 'DATA' (str), got 'ENG' (s... |
| `edge-002` | pass | pass | pass | pass | pass | fail: argument 'project_key': expected 'OPS' (str), got 'ENG' (st... |
| `edge-012` | fail: no tool called; expected 'calendar_event_tool' | pass | pass | pass | pass | pass |
