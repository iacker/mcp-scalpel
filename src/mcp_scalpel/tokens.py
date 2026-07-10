"""Token estimation for MCP tool catalogs.

JSON schemas are dense structural text; empirically ~3.7 chars/token with
cl100k_base. Good enough for routing decisions and savings reporting without
pulling a tokenizer dependency into the hot path.
"""
from __future__ import annotations
import json
from typing import Any

CHARS_PER_TOKEN = 3.7


def estimate_tokens(obj: Any) -> int:
    """Estimate token count of an arbitrary JSON-serialisable object."""
    if isinstance(obj, str):
        text = obj
    else:
        text = json.dumps(obj, separators=(",", ":"))
    return int(len(text) / CHARS_PER_TOKEN)


def catalog_tokens(tools: list[dict]) -> int:
    """Tokens for a full tools/list result payload."""
    return estimate_tokens({"tools": tools})
