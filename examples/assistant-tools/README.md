# Assistant tools

Small [MCP](../../docs/glossary.md#mcp) tools for the gmlx
[assistant](../../docs/assistant.md). Each script lists its own
dependencies, so `uv run --script` runs it with no setup.

| Script | What it does |
|--------|--------------|
| `decide-tool.py` | Asks `/v1/systemone` a question with fixed options and returns the top answer with its score. [Decisions as a tool](../../docs/assistant.md#decisions-as-a-tool) shows its setup. |

Add a tool to `assistant.mcp` with its full path:

```yaml
assistant:
  mcp:
    - name: decide
      command: [uv, run, --script, /Users/me/gmlx/examples/assistant-tools/decide-tool.py]
```
