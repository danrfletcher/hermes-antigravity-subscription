"""Rewind a bad turn out of agy's on-disk conversation, so the next turn resumes from the last good state.

agy keeps each conversation in ``conversations/<id>.db`` (SQLite, WAL) and resumes it from that DB
alone. A rewind deletes everything after a recorded *rewind point*: the DB's shape just before the
plugin sent the turn that went bad (which is also "the conversation as of the last successful
request", what the session store's hashes describe). The delta the plugin sends next re-supplies
the bad turn's request, any calls Hermes already ran and their results.

Every step is checked; anything unexpected means "do not touch the DB" and the caller falls back
to a fresh conversation with the full history. The reasoning and the live probes are in the
design review (agy 1.5.9): backups go through SQLite's backup API (a plain copy of ``<id>.db``
misses the WAL), the cut point is recorded before each turn rather than worked out afterwards,
and agy's own presence lock is the only sign that another agy still holds the conversation.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # Windows: no presence check, so never rewind there
    fcntl = None  # type: ignore[assignment]

# sha256(repr(sqlite_master rows))[:16] of the conversation DB schema rewinds were verified on
# (agy 1.5.9). A new agy schema is pinned deliberately, after re-running the rewind probes.
KNOWN_SCHEMAS = frozenset({"a87618921be9966b"})
USER_STEP = 14  # step_type of a user message
PROMPT_PROBE_CHARS = 120


class RewindRefused(Exception):
    """The DB is not in the state the rewind point describes; nothing was changed."""


def _connect(db: Path) -> sqlite3.Connection:
    # A read-only URI open fails on a WAL DB once agy has stopped (no -shm), so open normally.
    conn = sqlite3.connect(str(db), timeout=2.0, isolation_level=None)
    return conn


def _schema_fp(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "select type, name, tbl_name, sql from sqlite_master where name not like 'sqlite_%' order by type, name"
    ).fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def _steps_hash(conn: sqlite3.Connection, below: int) -> str:
    digest = hashlib.sha256()
    for row in conn.execute(
        "select idx, step_type, status, length(step_payload) from steps where idx < ? order by idx", (below,)
    ):
        digest.update(repr(tuple(row)).encode())
    return digest.hexdigest()[:16]


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def _fields(buf: bytes) -> list[tuple[int, Any]]:
    """Top-level protobuf fields: varints as ints, length-delimited as bytes. Raises on garbage."""
    out, i = [], 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if field == 0:
            raise ValueError("field 0")
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = buf[i:i + size], i + size
            if i > len(buf):
                raise ValueError("truncated")
        else:
            raise ValueError(f"wire type {wire}")
        out.append((field, value))
    return out


def _gen_steps(data: bytes) -> list[int]:
    """gen_metadata field 2: the packed step indices that generation produced (may be absent)."""
    steps: list[int] = []
    for field, value in _fields(data or b""):
        if field == 2:
            if isinstance(value, int):
                steps.append(value)
            else:
                j = 0
                while j < len(value):
                    item, j = _varint(value, j)
                    steps.append(item)
    return steps


def _exe_last_step(data: bytes) -> int | None:
    """executor_metadata field 3: the last step index of that execution."""
    for field, value in _fields(data or b""):
        if field == 3 and isinstance(value, int):
            return value
    return None


def read_point(db: Path, *, conversation_id: str, prompt: str, agy_version: str | None) -> dict[str, Any] | None:
    """The rewind point for the turn about to be sent: call it while agy is idle (before the send)."""
    db = Path(db)
    if not db.exists():
        return None
    conn = _connect(db)
    try:
        conn.execute("pragma query_only = 1")
        steps, max_idx = conn.execute("select count(*), coalesce(max(idx) + 1, 0) from steps").fetchone()
        if steps != max_idx or steps == 0:
            return None
        gen = conn.execute("select coalesce(max(idx) + 1, 0) from gen_metadata").fetchone()[0]
        exe = conn.execute("select coalesce(max(idx) + 1, 0) from executor_metadata").fetchone()[0]
        return {
            "conversation_id": conversation_id,
            "steps": steps,
            "gen": gen,
            "exe": exe,
            "steps_hash": _steps_hash(conn, steps),
            "schema": _schema_fp(conn),
            "agy_version": agy_version,
            "prompt_probe": prompt.strip()[:PROMPT_PROBE_CHARS],
            "taken": time.time(),
        }
    finally:
        conn.close()


def presence_held(gemini_dir: Path, conversation_id: str) -> bool:
    """True when some process still holds agy's presence lock on this conversation."""
    assert fcntl is not None
    lock = Path(gemini_dir) / "presence" / f"{conversation_id}.lock"
    if not lock.exists():
        return False
    with open(lock, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def _check_before(conn: sqlite3.Connection, point: dict[str, Any]) -> None:
    if _schema_fp(conn) not in KNOWN_SCHEMAS or _schema_fp(conn) != point.get("schema"):
        raise RewindRefused("unknown conversation schema")
    if conn.execute("pragma user_version").fetchone()[0] != 1:
        raise RewindRefused("unknown user_version")
    count, max_idx = conn.execute("select count(*), coalesce(max(idx) + 1, 0) from steps").fetchone()
    cut = int(point["steps"])
    if count != max_idx:
        raise RewindRefused("step indices are not contiguous")
    if count <= cut:
        raise RewindRefused("nothing after the rewind point")
    if _steps_hash(conn, cut) != point.get("steps_hash"):
        raise RewindRefused("steps before the rewind point changed (compaction?)")
    step_type, payload = conn.execute("select step_type, step_payload from steps where idx = ?", (cut,)).fetchone()
    if step_type != USER_STEP:
        raise RewindRefused(f"step {cut} is type {step_type}, not the bad turn's user step")
    probe = str(point.get("prompt_probe") or "")
    if probe and probe.encode("utf-8") not in bytes(payload or b""):
        raise RewindRefused("step after the rewind point is not the prompt the plugin sent")
    for table in ("parent_references", "battle_mode_infos"):
        if conn.execute(f"select count(*) from {table}").fetchone()[0]:
            raise RewindRefused(f"{table} is not empty (subagents are not rewound)")


def _check_after(conn: sqlite3.Connection, point: dict[str, Any]) -> None:
    cut, gen, exe = int(point["steps"]), int(point["gen"]), int(point["exe"])
    if conn.execute("select count(*) from steps").fetchone()[0] != cut:
        raise RewindRefused("step count after the cut")
    if conn.execute("select count(*), coalesce(max(idx) + 1, 0) from gen_metadata").fetchone()[1] != gen:
        raise RewindRefused("gen_metadata after the cut")
    if conn.execute("select coalesce(max(idx) + 1, 0) from executor_metadata").fetchone()[0] != exe:
        raise RewindRefused("executor_metadata after the cut")
    for (data,) in conn.execute("select data from gen_metadata"):
        if any(step >= cut for step in _gen_steps(data)):
            raise RewindRefused("a kept generation refers to a deleted step")
    for (data,) in conn.execute("select data from executor_metadata"):
        last = _exe_last_step(data)
        if last is not None and last >= cut:
            raise RewindRefused("a kept execution refers to a deleted step")


def _restore(backup: Path, db: Path) -> None:
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{db}{suffix}")
    shutil.copyfile(backup, db)


def rewind(gemini_dir: Path, point: dict[str, Any], *, backup_dir: Path, agy_version: str | None) -> None:
    """Cut the conversation back to ``point``. agy must not be running on it.

    Raises ``RewindRefused`` (or an OSError/sqlite3 error) when it cannot; the DB is then exactly
    as it was. On success the backup is kept until the next successful turn (``drop_backups``).
    """
    if fcntl is None:
        raise RewindRefused("no presence check on this platform")
    conversation_id = str(point["conversation_id"])
    gemini_dir = Path(gemini_dir)
    db = gemini_dir / "conversations" / f"{conversation_id}.db"
    if not db.exists():
        raise RewindRefused("conversation DB is gone")
    if point.get("agy_version") and agy_version and point["agy_version"] != agy_version:
        raise RewindRefused(f"agy changed version ({point['agy_version']} -> {agy_version})")
    if presence_held(gemini_dir, conversation_id):
        raise RewindRefused("another agy process holds the conversation")

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{conversation_id}.{int(time.time() * 1000)}.db"
    conn = _connect(db)
    try:
        dest = sqlite3.connect(str(backup))
        try:
            conn.backup(dest)  # includes what is still only in the WAL
        finally:
            dest.close()
        try:
            conn.execute("begin immediate")
            _check_before(conn, point)
            cut = int(point["steps"])
            conn.execute("delete from steps where idx >= ?", (cut,))
            conn.execute("delete from gen_metadata where idx >= ?", (int(point["gen"]),))
            conn.execute("delete from executor_metadata where idx >= ?", (int(point["exe"]),))
            _check_after(conn, point)
            conn.execute("commit")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("rollback")
            raise
        conn.execute("pragma wal_checkpoint(TRUNCATE)")
        if conn.execute("pragma integrity_check").fetchone()[0] != "ok":
            conn.close()
            _restore(backup, db)
            raise RewindRefused("integrity check failed after the cut; restored")
    except (ValueError, IndexError) as exc:  # protobuf decoding: never guess
        raise RewindRefused(f"unreadable metadata: {exc}") from exc
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()

    # Outputs of deleted tool steps: indices are reused, so a stale file would sit beside a new one.
    steps_dir = gemini_dir / "brain" / conversation_id / ".system_generated" / "steps"
    if steps_dir.is_dir():
        for child in steps_dir.iterdir():
            if child.name.isdigit() and int(child.name) >= int(point["steps"]):
                shutil.rmtree(child, ignore_errors=True)


def drop_backups(backup_dir: Path, conversation_id: str | None = None) -> None:
    """Delete rewind backups (of one conversation, or all) once a turn has succeeded."""
    backup_dir = Path(backup_dir)
    if not backup_dir.is_dir():
        return
    pattern = f"{conversation_id}.*.db" if conversation_id else "*.db"
    for path in backup_dir.glob(pattern):
        with contextlib.suppress(OSError):
            path.unlink()
