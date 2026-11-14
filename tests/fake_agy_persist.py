#!/usr/bin/env python3
"""Fake agy for persistent-conversation tests.

Mimics agy print mode with --input-format stream-json: one conversation per process (new, or
resumed with --conversation <id>), conversation records under $HOME/.gemini/antigravity-cli/
conversations/<id>.db, cumulative usage on result events, per-call usage on agent_response steps.
Every prompt it receives is appended to $FAKE_AGY_LOG as {"argv", "conversation_id", "prompt"}.
Replies "ACK <n>" where n counts prompts in the conversation (persisted across resumes).
FAKE_AGY_RUNAWAY="2,3" makes the 2nd and 3rd prompt of a conversation run away (call + invented result).
Native tools: like agy, it starts the MCP server named in $HOME/.gemini/config/mcp_config.json and lists
its tools. FAKE_AGY_NATIVE_CALLS=N makes each prompt call the first listed tool N times, one after another
(an ACTIVE tool step, the MCP tools/call, a DONE step), then reply "ACK <n>: <last result>".
"""
import json
import os
import sys
import uuid
from pathlib import Path

argv = sys.argv[1:]
conv = None
if "--conversation" in argv:
    conv = argv[argv.index("--conversation") + 1]
data = Path(os.environ["HOME"]) / ".gemini" / "antigravity-cli" / "conversations"
data.mkdir(parents=True, exist_ok=True)
if not conv or not (data / f"{conv}.db").exists():
    conv = str(uuid.uuid4())
record = data / f"{conv}.db"
state = json.loads(record.read_text()) if record.exists() else {"prompts": [], "tokens": 0}
record.write_text(json.dumps(state))

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

mcp = None
mcp_tools = []
mcp_id = 0


def mcp_request(method, params=None):
    global mcp_id
    mcp_id += 1
    mcp.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mcp_id, "method": method, "params": params or {}}) + "\n")
    mcp.stdin.flush()
    while True:
        msg = json.loads(mcp.stdout.readline())
        if msg.get("id") == mcp_id:
            return msg.get("result") or {}


mcp_config = Path(os.environ["HOME"]) / ".gemini" / "config" / "mcp_config.json"
if mcp_config.exists() and os.environ.get("FAKE_AGY_NO_MCP") != "1":
    import subprocess
    server = json.loads(mcp_config.read_text())["mcpServers"]["hermes"]
    mcp = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, bufsize=1)
    mcp_request("initialize", {"protocolVersion": "2024-11-05"})
    mcp_tools = [t["name"] for t in mcp_request("tools/list").get("tools", [])]

emit({"event": "init", "conversation_id": conv, "init": {"conversation_id": conv}})
cumulative_in = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    prompt = json.loads(line)["message"]["content"]
    state["prompts"].append(prompt)
    state["tokens"] += len(prompt) // 4
    record.write_text(json.dumps(state))
    with open(os.environ["FAKE_AGY_LOG"], "a") as log:
        log.write(json.dumps({"argv": argv, "conversation_id": conv, "prompt": prompt}) + "\n")
    new = len(prompt) // 4
    cached = state["tokens"] - new
    cumulative_in += new
    reply = f"ACK {len(state['prompts'])}"
    native_calls = int(os.environ.get("FAKE_AGY_NATIVE_CALLS", "0") or 0)
    if native_calls and mcp_tools:
        last = ""
        for i in range(native_calls):
            name = "mcp_hermes_" + mcp_tools[0]
            args = {"command": f"echo {i + 1}"}
            emit({"event": "step_update", "step_update": {"conversation_id": conv, "step_index": 10 + i,
                  "step_type": "tool", "state": "ACTIVE", "tool_name": name,
                  "tool_info": {"name": name, "parameters": args}}})
            res = mcp_request("tools/call", {"name": mcp_tools[0], "arguments": args})
            last = res["content"][0]["text"]
            emit({"event": "step_update", "step_update": {"conversation_id": conv, "step_index": 10 + i,
                  "step_type": "tool", "state": "DONE", "tool_name": name, "output": last}})
        reply = f"ACK {len(state['prompts'])}: {last}"
    if str(len(state["prompts"])) in os.environ.get("FAKE_AGY_RUNAWAY", "").split(","):
        # A runaway turn: a tool call, then an invented result for it.
        call = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        reply = "<tool_call>" + json.dumps(call) + "</tool_call>\nTool Result (call_1):\nmade up\n"
    if os.environ.get("FAKE_AGY_CHECKPOINT") == str(len(state["prompts"])):
        emit({"event": "step_update", "step_update": {"step_type": "checkpoint", "state": "DONE"}})
    emit({"event": "step_update", "step_update": {"conversation_id": conv, "step_type": "agent_response",
          "state": "DONE", "text_delta": reply,
          "usage": {"input_tokens": new, "output_tokens": 2, "cache_read_tokens": cached, "total_tokens": new + 2}}})
    emit({"event": "result", "result": {"conversation_id": conv, "status": "SUCCESS", "response": reply,
          "usage": {"input_tokens": cumulative_in, "output_tokens": 2, "total_tokens": cumulative_in + 2}}})
