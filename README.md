# mcp-scalpel

**Local semantic tool-filtering proxy for the [Docker MCP Gateway](https://docs.docker.com/desktop/features/gordon/mcp/). Cuts per-turn input tokens by routing `tools/list` to the relevant subset — no AWS, no cloud, no external LLM on the hot path.**

## The problem

When an MCP client connects to a gateway exposing many tools, the **entire tool catalog** (names + descriptions + JSON input schemas) is injected into **every** LLM call. On a real Docker MCP Gateway with 49 tools that's **~18,700 input tokens per turn**, re-sent on every message, forever.

```
measured on a live gateway:
  49 tools = 71,994 chars ≈ 18,663 tokens / turn
  heaviest: SearchSpecificAwsIaModules (1,888 tok), the whole Terraform/AWS family
```

## What scalpel does

It sits **between your MCP client and the gateway** as a stdio proxy, intercepts `tools/list`, and returns only the tools relevant to the current session — typically 15 instead of 49. Everything else is relayed verbatim.

```
Your Agent ──stdio──► [mcp-scalpel] ──stdio──► docker mcp gateway run ──► N MCP servers
                          │
                          └─► tools/list returns ~15 routed tools, not 49
```

Measured on the same gateway: **~52% fewer catalog tokens per turn**, with the correct tool surfaced in the top result every time.

## Why not just use an existing token-saver library?

Most gateway-side filters ([mcp-token-saver](https://github.com/victorTAKI/mcp-token-saver) et al.) assume the router sees the **user's prompt** and default to **AWS Bedrock** for embeddings + an LLM router. A stdio proxy sees neither the prompt nor needs the cloud. scalpel is built for that reality:

- **A proxy never sees the prompt** — only JSON-RPC frames. So the filter is fed by **session context**: a configurable task hint plus the names of recently-called tools.
- **Progressive disclosure** — an always-present meta tool `scalpel_search_tools(query)` lets the agent pull in any hidden tool's full schema on demand; the proxy then emits `notifications/tools/list_changed`.
- **Safety net** — low-signal queries fall back to the full catalog, and `tools/call` is always relayed, so a hidden tool is never *un*callable.
- **Zero cloud** — pure-numpy TF-IDF cosine by default (ms latency, no downloads). Swap in real MiniLM embeddings via the `[embeddings]` extra by replacing one `Vectorizer`.

## Install

```bash
pip install mcp-scalpel                 # core (numpy TF-IDF)
pip install "mcp-scalpel[embeddings]"   # optional: MiniLM semantic embeddings
```

## Use it with your MCP client

Point your client at `mcp-scalpel` instead of `docker mcp gateway run`. Example client config:

```json
{
  "mcpServers": {
    "docker-filtered": {
      "command": "mcp-scalpel",
      "env": {
        "SCALPEL_MAX_TOOLS": "15",
        "SCALPEL_TASK_HINT": "devsecops: aws, terraform, kubernetes, linkedin jobs"
      }
    }
  }
}
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SCALPEL_UPSTREAM` | `docker mcp gateway run` | Upstream MCP command to proxy |
| `SCALPEL_MAX_TOOLS` | `15` | Max tools returned per `tools/list` |
| `SCALPEL_TASK_HINT` | `""` | Free-text bias for routing (your recurring domains) |
| `SCALPEL_LOG` | *(stderr)* | Optional file to also write routing decisions to |

Every routing decision is logged:

```
[scalpel] indexed 49 tools (~18663 tok full catalog)
[scalpel] tools/list -> 16/49 tools | 8977 tok | saved 9686 (52%) | ctx='devsecops...'
```

## How routing works

1. On first `tools/list`, scalpel asks the upstream gateway for the full catalog and indexes each tool's name + description (TF-IDF, or embeddings if installed).
2. On every `tools/list`, it scores the catalog against `SCALPEL_TASK_HINT` + recently-called tool names, keeps the top-`MAX_TOOLS`, and prepends the `scalpel_search_tools` meta tool.
3. When the agent needs something hidden, it calls `scalpel_search_tools("...")`; matching tools are revealed and stay callable for the session.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python tests/test_filter.py         # unit tests on a real catalog snapshot
python tests/test_e2e_gateway.py    # end-to-end against live docker mcp gateway
```

## License

MIT
