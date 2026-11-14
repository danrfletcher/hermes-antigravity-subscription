#!/usr/bin/env python3
"""Stdio MCP server that hands Hermes' tools to agy as native tools, without running any of them.

agy loads it from ``<isolated HOME>/.gemini/config/mcp_config.json`` (server ``hermes``), so the
model sees each Hermes tool as a real function (``mcp_hermes_<name>``) instead of writing
``<tool_call>`` text, which Gemini 4 garbles on about one turn in five (Google: "Always declare
tools natively"). A ``tools/call`` is parked: the call goes to ``<dir>/calls/<seq>.json``, the
plugin ends the Hermes completion with that call, Hermes runs it with its own approvals and
sandbox, and the plugin writes the result to ``<dir>/results/<seq>.json``, which this server
returns to agy so the model carries on in the same turn.

``<dir>`` is the worker's bridge directory, named by ``HERMES_AGY_BRIDGE_DIR`` in the environment
agy passes on. Without it (a one-shot agy process, which keeps the text route) the server offers
no tools. Standard library only: it runs outside the plugin package, under agy.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

PROTOCOL_VERSION = "2024-11-05"
# agy's own per-server timeout is set to 3600 s in mcp_config.json; answer just before it.
RESULT_WAIT_SECONDS = float(os.environ.get("HERMES_AGY_BRIDGE_WAIT", "3540"))
POLL_SECONDS = 0.05


_SEND_LOCK = threading.Lock()


def _send(obj: dict) -> None:
    # A parked call answers from its own thread, so pings keep being answered while it waits.
    with _SEND_LOCK:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def _log(bridge_dir: str | None, msg: str) -> None:
    if not bridge_dir:
        return
    try:
        with open(os.path.join(bridge_dir, "bridge.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{time.time():.3f} {msg}\n")
    except OSError:
        pass


def _load_tools(bridge_dir: str | None) -> list[dict]:
    if not bridge_dir:
        return []
    try:
        with open(os.path.join(bridge_dir, "tools.json"), encoding="utf-8") as fh:
            tools = json.load(fh)
    except (OSError, ValueError):
        return []
    return [t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)] \
        if isinstance(tools, list) else []


def _write_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _park(bridge_dir: str, seq: int, name: str, arguments) -> dict:
    """Write the call, wait for Hermes' result, return MCP ``tools/call`` result content."""
    calls = os.path.join(bridge_dir, "calls")
    results = os.path.join(bridge_dir, "results")
    os.makedirs(calls, exist_ok=True)
    os.makedirs(results, exist_ok=True)
    _write_atomic(os.path.join(calls, f"{seq:06d}.json"),
                  {"seq": seq, "name": name, "arguments": arguments, "t": time.time()})
    _log(bridge_dir, f"parked {seq} {name}")
    result_path = os.path.join(results, f"{seq:06d}.json")
    deadline = time.monotonic() + RESULT_WAIT_SECONDS
    while time.monotonic() < deadline:
        if os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as fh:
                    res = json.load(fh)
            except (OSError, ValueError):
                time.sleep(POLL_SECONDS)
                continue
            _log(bridge_dir, f"answered {seq}")
            return {"content": [{"type": "text", "text": str(res.get("text", ""))}],
                    "isError": bool(res.get("is_error"))}
        time.sleep(POLL_SECONDS)
    _log(bridge_dir, f"timed out {seq}")
    return {"content": [{"type": "text", "text": "Hermes did not answer this tool call in time."}],
            "isError": True}


def main() -> int:
    bridge_dir = os.environ.get("HERMES_AGY_BRIDGE_DIR") or None
    if bridge_dir and not os.path.isdir(bridge_dir):
        bridge_dir = None
    tools = _load_tools(bridge_dir)
    # agy's name -> Hermes' name (an alias when Hermes' name is too long for agy).
    names = {t["name"]: str(t.get("hermes_name") or t["name"]) for t in tools}
    _log(bridge_dir, f"start pid={os.getpid()} ppid={os.getppid()} tools={len(tools)}")
    seq = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": (msg.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hermes", "version": "1"}}})
        elif method == "tools/list":
            _log(bridge_dir, f"listed {len(tools)}")
            if bridge_dir:  # the plugin waits for this before it sends the first prompt
                try:
                    open(os.path.join(bridge_dir, "listed"), "w").close()
                except OSError:
                    pass
            _send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": t["name"], "description": t.get("description") or "",
                 "inputSchema": t.get("inputSchema") or {"type": "object", "properties": {}}}
                for t in tools]}})
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = str(params.get("name") or "")
            arguments = params.get("arguments")
            if not bridge_dir or name not in names:
                _send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}})
            else:
                seq += 1
                args = arguments if isinstance(arguments, dict) else {}

                def answer(mid=mid, seq=seq, name=name, args=args) -> None:
                    _send({"jsonrpc": "2.0", "id": mid, "result": _park(bridge_dir, seq, names[name], args)})

                threading.Thread(target=answer, daemon=True).start()
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            _send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
        # notifications (initialized, cancelled, ...) need no answer
    return 0


if __name__ == "__main__":
    sys.exit(main())
