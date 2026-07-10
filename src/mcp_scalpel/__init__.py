"""mcp-scalpel: local semantic tool-filtering proxy for the Docker MCP Gateway."""
from .filter import ToolFilter, TfidfVectorizer, tool_signal
from .tokens import estimate_tokens, catalog_tokens

__version__ = "0.1.0"
__all__ = ["ToolFilter", "TfidfVectorizer", "tool_signal",
           "estimate_tokens", "catalog_tokens"]
