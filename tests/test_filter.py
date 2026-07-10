"""Real tests against the actual Docker MCP Gateway catalog snapshot."""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from mcp_scalpel.filter import ToolFilter  # noqa: E402
from mcp_scalpel.tokens import catalog_tokens  # noqa: E402

SNAP = pathlib.Path(__file__).parent / "fixtures" / "tools_list.json"


def load_tools():
    return json.loads(SNAP.read_text())["tools"]


def test_catalog_indexes():
    tools = load_tools()
    assert len(tools) >= 40
    f = ToolFilter(tools, max_tools=15)
    assert len(f.names) == len(tools)


def test_routing_is_relevant():
    """Each query must surface its obviously-correct tool in the top result."""
    f = ToolFilter(load_tools(), max_tools=15)
    cases = {
        "search LinkedIn for a devsecops job": "search_jobs",
        "navigate to a url and fill a form": "browser_",
        "run terraform and scan with checkov": "Checkov",
        "fetch a url as markdown": "fetch",
    }
    for query, expect in cases.items():
        routed = f.route(query)
        names = [t["name"] for t in routed]
        assert any(expect.lower() in n.lower() for n in names[:3]), \
            f"{query!r} -> {names[:3]} missing {expect}"


def test_filtering_saves_tokens():
    tools = load_tools()
    f = ToolFilter(tools, max_tools=15)
    full = catalog_tokens(tools)
    routed = f.route("search linkedin jobs")
    kept = catalog_tokens(routed)
    assert kept < full
    saved_pct = 100 * (full - kept) / full
    assert saved_pct > 30, f"only saved {saved_pct:.0f}%"


def test_never_returns_empty():
    f = ToolFilter(load_tools(), max_tools=15)
    routed = f.route("zzz totally unrelated gibberish qwxyz")
    assert len(routed) > 0  # safety net: full catalog fallback


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
