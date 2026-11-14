#!/usr/bin/env python3
"""Fake agy for persistent-conversation tests.

Mimics agy print mode with --input-format stream-json: one conversation per process (new, or
resumed with --conversation <id>), conversation records under $HOME/.gemini/antigravity-cli/
conversations/<id>.db, cumulative usage on result events, per-call usage on agent_response steps.
Every prompt it receives is appended to $FAKE_AGY_LOG as {"argv", "conversation_id", "prompt"}.
Replies "ACK <n>" where n counts prompts in the conversation (persisted across resumes).
FAKE_AGY_RUNAWAY="2,3" makes the 2nd and 3rd prompt of a conversation run away (call + invented result).
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
