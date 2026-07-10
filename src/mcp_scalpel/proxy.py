"""mcp-scalpel proxy: a stdio MCP middleman for the Docker MCP Gateway.

Sits between an MCP client (Hermes, Claude Code, Cursor…) and the upstream
`docker mcp gateway run`. Intercepts `tools/list` and returns a semantically
filtered subset instead of the full catalog, cutting per-turn input tokens.

Design constraints (why this is not just mcp-token-saver):
  * A stdio proxy never sees the user's prompt — only JSON-RPC frames. So the
    filter is fed by *session context*: a configurable task hint plus the
    names/args of recently issued tools/call requests.
  * Progressive disclosure: a always-present meta tool `scalpel_search_tools`
    lets the agent pull in full schemas on demand; the proxy then re-advertises
    with `notifications/tools/list_changed`.
  * Safety net: unknown / low-signal queries fall back to the full catalog —
    never silent tool loss. tools/call is always relayed verbatim, so a hidden
    tool is still callable if the agent knows its name.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
from collections import deque

from .filter import ToolFilter
from .tokens import catalog_tokens, estimate_tokens

UPSTREAM_CMD = os.environ.get(
    "SCALPEL_UPSTREAM", "docker mcp gateway run"
).split()
MAX_TOOLS = int(os.environ.get("SCALPEL_MAX_TOOLS", "15"))
TASK_HINT = os.environ.get("SCALPEL_TASK_HINT", "")
LOG_PATH = os.environ.get("SCALPEL_LOG", "")

SEARCH_TOOL = {
    "name": "scalpel_search_tools",
    "description": (
        "Search the FULL tool catalog by natural-language query and load the "
        "matching tools' full schemas into context. Use when the tool you need "
        "is not in the current visible list. Returns tool names + descriptions; "
        "the matching tools then become directly callable."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you want to do"},
        },
        "required": ["query"],
    },
}


def _log(msg: str) -> None:
    line = f"[scalpel] {msg}"
    print(line, file=sys.stderr, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


class ScalpelProxy:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            UPSTREAM_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin and self.proc.stdout  # for type-checkers
        self.full_catalog: list[dict] = []
        self.filter: ToolFilter | None = None
        self.recent: deque[str] = deque(maxlen=8)  # session context
        self.exposed: set[str] = set()             # progressively revealed
        self._lock = threading.Lock()

    # ---- upstream plumbing -------------------------------------------------
    def _to_upstream(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _to_client(self, obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def _upstream_request(self, obj: dict, want_id) -> dict | None:
        """Send a request upstream and read until the matching id comes back."""
        self._to_upstream(obj)
        for _ in range(200):
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want_id:
                return msg
            # a stray notification while we wait — forward it
            if "id" not in msg:
                self._to_client(msg)
        return None

    # ---- catalog handling --------------------------------------------------
    def _ensure_catalog(self) -> None:
        if self.filter is not None:
            return
        resp = self._upstream_request(
            {"jsonrpc": "2.0", "id": "__scalpel_catalog__",
             "method": "tools/list", "params": {}},
            "__scalpel_catalog__",
        )
        tools = (resp or {}).get("result", {}).get("tools", [])
        self.full_catalog = tools
        self.filter = ToolFilter(tools, max_tools=MAX_TOOLS)
        _log(f"indexed {len(tools)} tools "
             f"(~{catalog_tokens(tools)} tok full catalog)")

    def _visible_tools(self, query_ctx: str) -> list[dict]:
        assert self.filter is not None
        extra = " ".join(self.recent)
        routed = self.filter.route(query_ctx, extra_terms=extra)
        # always include progressively-revealed tools + the meta search tool
        chosen = {t["name"]: t for t in routed}
        for name in self.exposed:
            if name in self.filter.by_name:
                chosen[name] = self.filter.by_name[name]
        out = [SEARCH_TOOL] + list(chosen.values())
        return out

    def _handle_tools_list(self, msg: dict) -> None:
        self._ensure_catalog()
        query_ctx = TASK_HINT
        visible = self._visible_tools(query_ctx)
        full_tok = catalog_tokens(self.full_catalog)
        vis_tok = estimate_tokens({"tools": visible})
        saved = full_tok - vis_tok
        pct = round(100 * saved / full_tok) if full_tok else 0
        _log(f"tools/list -> {len(visible)}/{len(self.full_catalog)} tools "
             f"| {vis_tok} tok | saved {saved} ({pct}%) | ctx='{query_ctx[:40]}'")
        self._to_client({"jsonrpc": "2.0", "id": msg["id"],
                         "result": {"tools": visible}})

    def _handle_search(self, msg: dict) -> None:
        self._ensure_catalog()
        assert self.filter is not None
        query = (msg.get("params", {}).get("arguments", {}) or {}).get("query", "")
        hits = self.filter.search(query, limit=MAX_TOOLS)
        for name, _score in hits:
            self.exposed.add(name)
        lines = [f"{n} — {self.filter.by_name[n].get('description','')[:80]}"
                 for n, _ in hits]
        text = ("Matching tools now loaded and callable:\n" +
                "\n".join(lines)) if lines else "No matching tools."
        self._to_client({"jsonrpc": "2.0", "id": msg["id"],
                         "result": {"content": [{"type": "text", "text": text}]}})
        # tell the client the visible toolset changed
        self._to_client({"jsonrpc": "2.0",
                         "method": "notifications/tools/list_changed"})

    # ---- main loop ---------------------------------------------------------
    def run(self) -> None:
        while True:
            raw = sys.stdin.readline()
            if not raw:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            method = msg.get("method")

            if method == "tools/list" and "id" in msg:
                self._handle_tools_list(msg)
                continue
            if method == "tools/call":
                name = msg.get("params", {}).get("name", "")
                if name == "scalpel_search_tools":
                    self._handle_search(msg)
                    continue
                self.recent.append(name)  # feed session context
            # everything else (initialize, notifications, real tools/call): relay
            if "id" in msg:
                resp = self._upstream_request(msg, msg["id"])
                if resp is not None:
                    self._to_client(resp)
            else:
                self._to_upstream(msg)

        self.proc.terminate()


def main() -> None:
    ScalpelProxy().run()


if __name__ == "__main__":
    main()
