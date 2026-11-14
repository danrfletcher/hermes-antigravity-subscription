#!/usr/bin/env python3
"""agy PreToolUse hook: deny every native agy tool except Hermes' own tools and the media-attach read.

Installed by ``process.install_tool_gate`` as ``<isolated HOME>/.gemini/config/hooks.json``
(matcher ``*``), so it runs before EVERY native tool agy's model tries: run_command, write_file,
schedule, invoke_subagent, search_web, ... (agy 1.5.9 offers 60). Hermes runs tools itself, from
native ``mcp_hermes_*`` calls (the plugin's MCP server, mcp_bridge.py) or ``<tool_call>`` text blocks; a
native agy tool would run outside the repo, invisible to Hermes' loop and
session log. A denial returns to the model in about 0.1 s with the reason below, and the model then
re-emits the request as a text tool call.

The one exception is agy's media pipeline: it attaches an image/video/audio by a native
``read_file``/``view_file`` on the path the prompt references. That read is allowed only when

* the path is one the plugin granted for this conversation (a ``read_file(<path>)`` /
  ``view_file(<path>)`` allow rule in the isolated settings.json, written per referenced media path),
* it has a media file extension, and
* it does not resolve into the isolated HOME (the OAuth token link lives there).

agy runs hooks with a stripped environment and fails closed (a missing script, a crash, bad JSON or
a timeout all deny the tool). Usage: ``tool_gate.py <settings.json> <isolated HOME>``; standard
library only, no plugin imports (the hook runs outside the plugin package).
"""

from __future__ import annotations

import json
import os
import sys

MEDIA_TOOLS = ("read_file", "view_file")
MEDIA_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".heif",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".ogv",
    ".mp3", ".wav", ".ogg", ".oga", ".m4a", ".aac", ".flac", ".opus",
}
DENY_REASON = (
    "This is headless inference mode: the runtime's built-in tools are unavailable here. Use the "
    "Hermes tools you were given instead (the mcp_hermes_* functions, or a <tool_call> text block "
    "if your instructions describe that format), then wait for the result."
)
# The plugin's own MCP server (mcp_bridge.py) serves Hermes' tools under this prefix. It is the only
# MCP server in the isolated HOME and runs nothing itself: Hermes runs each call.
HERMES_SERVER = "hermes"
HERMES_PREFIX = "mcp_hermes_"


def is_hermes_tool(name: str, args) -> bool:
    if name.startswith(HERMES_PREFIX) and len(name) > len(HERMES_PREFIX):
        return True
    if name == "call_mcp_tool" and isinstance(args, dict):
        server = args.get("ServerName") or args.get("serverName") or args.get("server_name")
        return str(server or "").lower() == HERMES_SERVER
    return False


def _granted_paths(settings_path: str) -> set[str]:
    try:
        with open(settings_path, encoding="utf-8") as fh:
            allow = json.load(fh).get("permissions", {}).get("allow", [])
    except (OSError, ValueError, AttributeError):
        return set()
    granted = set()
    for rule in allow if isinstance(allow, list) else []:
        rule = str(rule)
        for tool in MEDIA_TOOLS:
            if rule.startswith(tool + "(") and rule.endswith(")"):
                granted.add(os.path.normpath(rule[len(tool) + 1:-1]))
    return granted


def _candidate_paths(args) -> list[str]:
    if not isinstance(args, dict):
        return []
    return [str(v) for k, v in args.items()
            if isinstance(v, str) and k.lower() in ("path", "absolutepath", "file_path", "filepath", "uri")]


def is_permitted_media_read(name: str, args, settings_path: str, isolated_home: str) -> bool:
    if name not in MEDIA_TOOLS:
        return False
    paths = _candidate_paths(args)
    if not paths:
        return False
    granted = _granted_paths(settings_path)
    home = os.path.realpath(isolated_home)
    for raw in paths:
        path = os.path.normpath(raw[7:] if raw.startswith("file://") else raw)
        if path not in granted:
            return False
        if os.path.splitext(path)[1].lower() not in MEDIA_EXTS:
            return False
        real = os.path.realpath(path)
        if real == home or real.startswith(home + os.sep):
            return False
    return True


def decide(payload, settings_path: str, isolated_home: str) -> dict:
    call = payload.get("toolCall") if isinstance(payload, dict) else None
    call = call if isinstance(call, dict) else {}
    name = str(call.get("name") or "")
    if is_hermes_tool(name, call.get("args")):
        return {"decision": "allow"}
    if is_permitted_media_read(name, call.get("args"), settings_path, isolated_home):
        return {"decision": "allow"}
    return {"decision": "deny", "reason": DENY_REASON}


def main(argv: list[str]) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    settings_path = argv[1] if len(argv) > 1 else ""
    isolated_home = argv[2] if len(argv) > 2 else "/nonexistent"
    try:
        out = decide(payload, settings_path, isolated_home)
    except Exception:  # fail closed
        out = {"decision": "deny", "reason": DENY_REASON}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
