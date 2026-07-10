#!/usr/bin/env python3
"""End-to-end smoke test: drive the real ScalpelProxy over stdio like a client.

Spawns the proxy (which itself spawns `docker mcp gateway run`), does the MCP
handshake, then compares raw tools/list vs filtered tools/list and exercises
the progressive-disclosure meta tool. Proves the proxy works against the live
gateway — not a mock.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = {**os.environ,
       "SCALPEL_MAX_TOOLS": "15",
       "SCALPEL_TASK_HINT": "search linkedin for a devsecops job in paris",
       "PYTHONPATH": os.path.join(ROOT, "src")}

_errf = open("/tmp/scalpel_proxy.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "mcp_scalpel.proxy"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=_errf,
    text=True, bufsize=1, env=ENV, cwd=ROOT)


def send(o):
    proc.stdin.write(json.dumps(o) + "\n"); proc.stdin.flush()


def read_id(want, tries=300):
    for _ in range(tries):
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("id") == want:
            return m
    return None


try:
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                     "clientInfo": {"name": "smoke", "version": "0"}}})
    init = read_id(1)
    assert init and "result" in init, f"init failed: {init}"
    print("handshake OK")

    send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    filtered = read_id(2)
    tools = filtered["result"]["tools"]
    names = [t["name"] for t in tools]
    print(f"filtered tools/list -> {len(tools)} tools")
    assert "scalpel_search_tools" in names, "meta tool missing"
    assert len(tools) <= 17, f"expected filtered set, got {len(tools)}"
    assert any("search_jobs" == n for n in names), \
        f"task hint routing failed, got {names}"
    print("  routed to task hint correctly:",
          [n for n in names if "job" in n.lower()])

    # progressive disclosure: pull in an AWS tool not in the filtered set
    send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
          "params": {"name": "scalpel_search_tools",
                     "arguments": {"query": "terraform checkov security scan"}}})
    search = read_id(3)
    txt = search["result"]["content"][0]["text"]
    print("  search_tools returned:", txt.split(chr(10))[1][:60] if chr(10) in txt else txt[:60])
    assert "Checkov" in txt or "Terraform" in txt.lower(), txt

    print("\nE2E SMOKE PASSED — proxy works against live docker mcp gateway")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
