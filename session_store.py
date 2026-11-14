"""Persistent agy conversations bound to Hermes sessions.

One Hermes chat maps to one agy conversation. agy owns the conversation state (including its
own compaction); the plugin only sends the messages agy has not seen, and after a Hermes or
desktop restart it resumes the same conversation with ``agy --conversation <id>``.

What is stored per Hermes session (``$HERMES_HOME/plugin-data/<plugin>/sessions.json``):
the agy conversation id, the model/effort it last ran with, and one short hash per Hermes
message agy has already received. No message text is stored here; the text lives in agy's own
conversation record inside this plugin's private agy HOME.

A per-session ``flock`` makes sure only one live client drives a conversation at a time; any
other client for the same session (for example Hermes' background review) falls back to a
throwaway one-shot run instead of writing into the shared conversation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

try:  # POSIX only; on Windows persistence degrades to no locking (single client assumed).
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PLUGIN_DIR_NAME = "antigravity-subscription-directsdk"
PERSIST_ENV = "ANTIGRAVITY_PERSIST_CONVERSATIONS"
DATA_DIR_ENV = "ANTIGRAVITY_PLUGIN_DATA_DIR"
# agy keeps a single stream-json user message only up to roughly 185k characters (measured on
# agy 1.3.3: 185k read in full, 200k cut at the end). Full-history prompts are kept below this.
MAX_PROMPT_CHARS_ENV = "ANTIGRAVITY_MAX_PROMPT_CHARS"
DEFAULT_MAX_PROMPT_CHARS = 150_000
SESSION_RETENTION_SECONDS = 30 * 24 * 3600
ORPHAN_CONVERSATION_SECONDS = 24 * 3600
_SAFE_KEY = re.compile(r"[^A-Za-z0-9_.-]")
_CONVERSATION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def persistence_enabled() -> bool:
    raw = str(os.environ.get(PERSIST_ENV, "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def max_prompt_chars() -> int:
    try:
        value = int(str(os.environ.get(MAX_PROMPT_CHARS_ENV, "")).strip() or DEFAULT_MAX_PROMPT_CHARS)
    except ValueError:
        return DEFAULT_MAX_PROMPT_CHARS
    return value if value > 10_000 else DEFAULT_MAX_PROMPT_CHARS


def data_dir() -> Path:
    explicit = os.environ.get(DATA_DIR_ENV, "").strip()
    if explicit:
        base = Path(explicit).expanduser()
    else:
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:
            home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
        base = home / "plugin-data" / PLUGIN_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(base, 0o700)
    return base


# ── Message identity ─────────────────────────────────────────────────────────────────────────

def _normalized_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return arguments
    return arguments


def message_hash(message: Any) -> str:
    """Stable identity of one Hermes message: role, content, tool calls (id/name/args), tool id.

    Volatile bookkeeping fields Hermes may add between turns (reasoning, display metadata,
    response item ids) are deliberately ignored."""
    if not isinstance(message, dict):
        return "invalid"
    # Tool calls are identified by id + name only. Hermes rewrites a past call's arguments when
    # they were invalid JSON ("corrupted tool_call arguments repaired"); that is not an edit by
    # the user and must not look like one (it used to trigger a full resend mid-turn).
    calls = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            function = call.get("function") or {}
            calls.append([call.get("id"), function.get("name")])
    identity = [
        str(message.get("role") or ""),
        message.get("content"),
        calls,
        message.get("tool_call_id"),
    ]
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def message_hashes(messages: Sequence[Any]) -> list[str]:
    return [message_hash(m) for m in messages]


def instructions_fingerprint(messages: Sequence[Any], tools: Any, tool_choice: Any) -> str:
    """Identity of the standing instructions agy must hold: system prompt + tool catalog.

    When it changes (new tools, edited system prompt) or agy compacted its conversation (which
    may summarize the instructions away), the next turn re-sends them in full."""
    system = [m.get("content") for m in messages
              if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    raw = json.dumps([system, tools or [], tool_choice], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


# ── History reconciliation ───────────────────────────────────────────────────────────────────

ANCHOR_MATCH = 3


def reconcile(known: Sequence[str], incoming: Sequence[str]) -> tuple[str, int, int]:
    """Decide how ``incoming`` (Hermes' current history) relates to ``known`` (what agy has).

    Returns ``(mode, start, keep)``:
      * ``("continue", start, _)``: agy already holds everything before ``incoming[start]``;
        send ``incoming[start:]``. Covers plain continuation and Hermes-side compaction or a
        changed system prompt (anchored on the most recent messages both sides share).
      * ``("rewind", start, keep)``: Hermes took back messages (edit, retry, undo). agy must
        disregard what it saw after ``incoming[keep - 1]`` and continue with ``incoming[start:]``.
      * ``("fresh", 0, 0)``: no usable overlap; start a new agy conversation.
    """
    known, incoming = list(known), list(incoming)
    if not known or not incoming:
        return "fresh", 0, 0
    n = len(known)
    if len(incoming) > n and incoming[:n] == known:
        return "continue", n, n
    # Anchor on the most recent messages agy saw (Hermes compaction rewrites the head and keeps
    # the tail). Widest match first; the newest occurrence wins, and the final incoming message
    # is never an anchor (it is the new turn).
    for width in range(min(ANCHOR_MATCH, n), 0, -1):
        tail = known[-width:]
        for j in range(len(incoming) - 2, width - 2, -1):
            if incoming[j - width + 1:j + 1] == tail:
                return "continue", j + 1, j + 1
    # Divergence: the longest shared prefix, ignoring a changed leading system message.
    offset = 1 if known and incoming and known[0] != incoming[0] else 0
    prefix = offset
    while prefix < min(n, len(incoming)) and known[prefix] == incoming[prefix]:
        prefix += 1
    if prefix - offset >= 1:
        if prefix < len(incoming):
            return "rewind", prefix, prefix
        # incoming is a strict prefix of what agy saw (retry / regenerate the last turn).
        return "rewind", len(incoming), len(incoming)
    return "fresh", 0, 0


# ── Session mapping ──────────────────────────────────────────────────────────────────────────

class SessionStore:
    def __init__(self, base: Path | None = None) -> None:
        self.base = Path(base) if base else data_dir()
        self.path = self.base / "sessions.json"
        self.lock_dir = self.base / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _file_lock(self):
        handle = open(self.base / "sessions.lock", "a+")
        try:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._file_lock():
            entry = self._read().get(session_id)
        if not isinstance(entry, dict) or not _CONVERSATION_ID.match(str(entry.get("conversation_id") or "")):
            return None
        return entry

    def put(self, session_id: str, *, conversation_id: str, model: str | None, effort: str | None,
            hashes: Sequence[str], tools: str | None = None) -> None:
        if not _CONVERSATION_ID.match(conversation_id or ""):
            return
        with self._file_lock():
            data = self._read()
            data[session_id] = {
                "conversation_id": conversation_id,
                "model": model,
                "effort": effort,
                "hashes": list(hashes),
                "tools": tools,
                "updated": time.time(),
            }
            self._write(data)

    def mark_bad(self, session_id: str, reason: str) -> None:
        """The conversation's latest turn went bad: it must be rewound before the next resume.

        Survives a gateway restart, so a resume never continues on top of a bad turn.
        """
        with self._file_lock():
            data = self._read()
            entry = data.get(session_id)
            if isinstance(entry, dict):
                entry["bad"] = str(reason)[:300]
                self._write(data)

    def forget(self, session_id: str) -> None:
        with self._file_lock():
            data = self._read()
            if data.pop(session_id, None) is not None:
                self._write(data)

    def referenced_conversations(self) -> set[str]:
        with self._file_lock():
            return {str(v.get("conversation_id")) for v in self._read().values() if isinstance(v, dict)}

    def prune(self, gemini_dir: Path | None) -> None:
        """Drop sessions idle for 30 days, then delete plugin-private agy conversation records
        that no session references and that are older than a day (one-shot and auxiliary runs).
        Only touches this plugin's private agy HOME, never the user's real ~/.gemini."""
        now = time.time()
        with self._file_lock():
            data = self._read()
            stale = [k for k, v in data.items()
                     if not isinstance(v, dict) or now - float(v.get("updated") or 0) > SESSION_RETENTION_SECONDS]
            for key in stale:
                data.pop(key, None)
            if stale:
                self._write(data)
            keep = {str(v.get("conversation_id")) for v in data.values() if isinstance(v, dict)}
        if not gemini_dir:
            return
        conversations = Path(gemini_dir) / "conversations"
        if not conversations.is_dir():
            return
        for path in conversations.iterdir():
            conversation_id = path.name.split(".", 1)[0]
            if not _CONVERSATION_ID.match(conversation_id) or conversation_id in keep:
                continue
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime > ORPHAN_CONVERSATION_SECONDS:
                    path.unlink()

    # Per-session ownership: held for as long as a live worker is bound to the session.
    def try_claim(self, session_id: str):
        path = self.lock_dir / (_SAFE_KEY.sub("_", session_id)[:120] + ".lock")
        handle = open(path, "a+")
        if fcntl is None:
            return handle
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        return handle

    @staticmethod
    def release(handle) -> None:
        if handle is None:
            return
        with contextlib.suppress(OSError):
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
