"""Antigravity Subscription DirectSDK client for Hermes Agent.

Drives the official Antigravity CLI (`agy`) as an external process in
stream-json mode to provide request-scoped completions using the user's
existing Antigravity / Gemini subscription quota.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# prompt.py imports agent.acp_openai_bridge lazily during the first turn, while Hermes
# starts the title-generation thread in parallel. Two threads first-importing
# openai.types.chat race and the loser sees a partially initialized module, so the
# session title is lost. Importing the bridge here runs that import on the loading thread.
with contextlib.suppress(Exception):
    import agent.acp_openai_bridge  # noqa: E402,F401

try:
    from .models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _normalize_effort,
        resolve_model_and_effort,
    )
    from .process import (
        AGY_MARKER_BASE_URL,
        _kill_process_tree,
        _own_process_group,
        build_child_env,
        install_tool_gate,
        is_authenticated,
        read_media_rules,
        reap_orphaned_plugin_workers,
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        start_stderr_drainer,
        sync_isolated_token,
        terminate_process,
        write_media_read_permissions,
    )
    from .prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _latest_user_text,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
        _is_plugin_owned_media_path,
        media_reference_paths,
    )
    from .delivery import DeliveryWatch
    from .stream import AntigravityStream, collect_stream_completion
    from .debug import dump_prompt_debug
    from . import rewind as _rewind
    from . import session_store as _session_store
except ImportError:
    from models import (
        _FALLBACK_MODELS,
        _MODEL_ALIASES,
        _normalize_effort,
        resolve_model_and_effort,
    )
    from process import (
        AGY_MARKER_BASE_URL,
        _kill_process_tree,
        _own_process_group,
        build_child_env,
        install_tool_gate,
        is_authenticated,
        read_media_rules,
        reap_orphaned_plugin_workers,
        resolve_agy_command,
        resolve_real_token_path,
        setup_isolated_home,
        start_stderr_drainer,
        sync_isolated_token,
        terminate_process,
        write_media_read_permissions,
    )
    from prompt import (
        _PROMPT_PREAMBLE,
        _ROLE_LABELS,
        _format_delta_prompt,
        _latest_user_text,
        _format_messages_as_prompt,
        _longest_tool_call_prefix_match,
        _messages_match_prefix,
        _parse_tool_block,
        _render_message_content,
        _is_plugin_owned_media_path,
        media_reference_paths,
    )
    from delivery import DeliveryWatch
    from stream import AntigravityStream, collect_stream_completion
    from debug import dump_prompt_debug
    import rewind as _rewind
    import session_store as _session_store

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 300.0
# Bad turns of one request (same messages) that are rewound and retried before giving up.
MAX_RECOVERIES_PER_REQUEST = 3


class AntigravityRecoveryExhausted(RuntimeError):
    """The same request went bad more than MAX_RECOVERIES_PER_REQUEST times in a row."""

# agy prints these when its token source cannot be built — the signature of
# OAuth contention with ANOTHER agy session on the same Google account (each
# process refreshes the shared token; the loser logs this and produces no
# result). Observed verbatim in isolated-home cli logs, 2026-11-11.
_AUTH_CONTENTION_SIGNATURES = (
    "not logged into antigravity",
    "error getting token source",
)


def _empty_no_result(completion: Any) -> bool:
    """True when a completion carries neither content, tool calls, nor a
    finish reason — the shape ``collect_stream_completion`` returns when the
    worker ended its turn without ever emitting a result event."""
    try:
        choice = completion.choices[0]
    except (AttributeError, IndexError, TypeError):
        return False
    message = getattr(choice, "message", None)
    if getattr(message, "content", None):
        return False
    if getattr(message, "tool_calls", None):
        return False
    return getattr(choice, "finish_reason", None) is None


def _stderr_shows_auth_contention(proc: Any) -> bool:
    drainer = getattr(proc, "_stderr_drainer", None)
    if drainer is None:
        return False
    try:
        tail = (drainer.get_tail() or "").lower()
    except Exception:
        return False
    return any(signature in tail for signature in _AUTH_CONTENTION_SIGNATURES)

# The persistent worker is only ever replaced by the next request (model
# change, non-continuation history) or by close(). Hermes retires an evicted
# agent's client without calling close(), so without an idle bound a finished
# conversation leaves its agy process (~200 MB RSS) running for the lifetime
# of the gateway. Terminating an idle worker is safe: it clears the recorded
# history, so the next turn takes the full-prompt branch on a fresh worker.
# ANTIGRAVITY_WORKER_IDLE_SECONDS overrides the bound; 0 or less disables it.
_DEFAULT_WORKER_IDLE_SECONDS = 900.0

# A turn abandoned mid-stream keeps _worker_lock held: the stream's generator is
# reclaimable only by the cyclic GC, so its finally block -- the only place that
# releases the lock -- can be postponed indefinitely by a lingering reference.
# The countdown then re-arms forever, which is how three dashboard workers were
# measured still alive past the bound (November 2026) while the gateway reaped
# Such a turn is dead by then -- but silence alone is not proof: a long
# reasoning turn or a slow first token can emit nothing for minutes while the
# worker is busy, and a caller may set a read deadline well past the default.
# So the second opinion needs BOTH no output for the abandon grace AND no CPU
# burned since the last output (_worker_cpu_seconds). An abandoned worker sits
# at 0% -- measured at 2.5 s CPU across sixteen minutes of life -- whereas a
# live reasoning turn does not. Where CPU time cannot be read (non-Linux, or a
# /proc entry that has gone) the worker is left alone: refusing to kill is
# always the safe failure here.
# ANTIGRAVITY_WORKER_ABANDON_SECONDS overrides the grace verbatim; 0 or less
# disables the second opinion and restores the old re-arm-forever behaviour.
# The default grace is one idle window, floored at one read deadline plus a
# minute, so it lands at ~15 minutes -- inside the external reaper's fence,
# which is what keeps that backstop silent.
_WORKER_ABANDON_GRACE_WINDOWS = 1.0
_WORKER_ABANDON_MIN_SECONDS = _DEFAULT_TIMEOUT_SECONDS + 60.0
# CPU seconds a worker may burn between signs of life before it counts as busy
# rather than abandoned. agy's own start-up costs a few tenths of a second, so
# this clears a worker that has barely started but not one that is thinking.
_WORKER_ABANDON_CPU_SECONDS = 1.0


def _worker_cpu_seconds(proc: subprocess.Popen) -> float | None:
    """Total CPU seconds the worker has consumed, or None when unreadable.

    Linux only: /proc/<pid>/stat carries utime and stime as fields 14 and 15,
    measured from *after* the comm field, which is parenthesised and may itself
    contain spaces and parentheses -- hence the split on the last ')'.
    """
    pid = getattr(proc, "pid", None)
    if not pid or not os.path.isdir("/proc"):
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read().decode("ascii", "replace")
        fields = data[data.rindex(")") + 2:].split()
        ticks = os.sysconf("SC_CLK_TCK")
        return (int(fields[11]) + int(fields[12])) / ticks
    except (OSError, ValueError, IndexError):
        return None


def _worker_idle_seconds() -> float:
    raw = os.environ.get("ANTIGRAVITY_WORKER_IDLE_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_WORKER_IDLE_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_WORKER_IDLE_SECONDS


def _worker_abandon_seconds() -> float:
    raw = os.environ.get("ANTIGRAVITY_WORKER_ABANDON_SECONDS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    # Never shorter than one read deadline plus a margin, so a live turn that is
    # merely slow to speak cannot be mistaken for an abandoned one.
    return max(
        _WORKER_ABANDON_GRACE_WINDOWS * _worker_idle_seconds(),
        _WORKER_ABANDON_MIN_SECONDS,
    )

# Issue #4 (Windows, WinError 32): terminating agy's process tree
# (taskkill /F /T) is asynchronous, so a child -- or a grandchild -- can
# still hold conversations/*.db for a short moment after termination
# returns, and TemporaryDirectory.cleanup() then raises PermissionError.
# Three quick attempts (0.1s, then 0.25s apart) cover that handle-release
# window: the worst case adds ~0.35s to a close() that was failing anyway,
# while the normal case (POSIX, or Windows where the handle is already
# gone) succeeds on the first attempt and pays none of it. After the last
# attempt a forced removal drops whatever is still unlocked, so close()
# never raises and never leaves the workspace behind needlessly.
#
# The tuple is the single source of truth for the retry budget: the
# attempts count is DERIVED from it (one try plus one backoff per retry),
# so extending the budget can never desynchronize the two -- a longer
# budget than tuple would index past its end and the IndexError would be
# swallowed by _remove_temp_dir's outer guard, turning a retry budget
# into a silent workspace leak.
_TEMP_DIR_CLEANUP_BACKOFF_SECONDS = (0.1, 0.25)
_TEMP_DIR_CLEANUP_ATTEMPTS = len(_TEMP_DIR_CLEANUP_BACKOFF_SECONDS) + 1

__all__ = [
    "AGY_MARKER_BASE_URL",
    "AntigravityClient",
    "AntigravityStream",
    "_FALLBACK_MODELS",
    "_MODEL_ALIASES",
    "_PROMPT_PREAMBLE",
    "_ROLE_LABELS",
    "_TEMP_DIR_CLEANUP_ATTEMPTS",
    "_TEMP_DIR_CLEANUP_BACKOFF_SECONDS",
    "_force_rmtree",
    "_format_delta_prompt",
    "_latest_user_text",
    "_format_messages_as_prompt",
    "_kill_process_tree",
    "_longest_tool_call_prefix_match",
    "_messages_match_prefix",
    "_normalize_effort",
    "_own_process_group",
    "_parse_tool_block",
    "_render_message_content",
    "is_authenticated",
    "resolve_agy_command",
    "resolve_model_and_effort",
]


def _force_rmtree(path: str) -> None:
    """Best-effort forced removal of a workspace; never raises (issue #4).

    ``shutil.rmtree(path, ignore_errors=True)`` alone is WEAKER than the
    ``tempfile._rmtree`` that ``TemporaryDirectory.cleanup()`` uses:
    the latter chmod-resets read-only subtrees and retries them, the
    former silently skips them, which is exactly the residual leak this
    issue is about (a read-only ``conversations`` tree surviving the
    forced pass on the Windows shape of the bug). So: forced removal
    first; if anything survives, grant the owner write (and, for
    directories, search) permission -- the POSIX analogue of "the handle
    is gone now, try again" -- and sweep once more.

    The permission sweep deliberately SKIPS links using ``os.lstat`` so
    nothing is ever followed:

    * ``setup_isolated_home`` links the user's REAL OAuth token into the
      isolated HOME with ``os.symlink``, ``os.link``, or a ``copy2``
      fallback (process.py). A surviving token link is precisely the
      residual-leak shape of #4, and following it would chmod the user's
      real 0400 token OUTSIDE the workspace to 0600 -- out-of-scope
      permission widening on user data, on the exact path this issue is
      about. The residual link is a documented lesser evil. The
      hardlink case (the ``os.link`` fallback, ``st_nlink > 1``) is
      skipped for the same reason -- that inode is the user's token.
      On Windows the skip is deliberate for an additional reason:
      DeleteFile fails with ACCESS_DENIED on a file carrying
      FILE_ATTRIBUTE_READONLY, unlike POSIX where unlinking needs only
      parent-directory write, so a readonly hardlinked file -- and its
      now-unneeded parent chain -- survives removal THERE. Clearing
      that attribute would clear it on the shared file record, i.e. on
      the user's real token: the same out-of-scope widening the skip
      exists to prevent. That Windows remnant is accepted on purpose;
      the field shape (agy writes tokens writable, and #4's actual
      conversations/*.db is nlink=1) is swept and removed normally.
      On macOS, ``setup_isolated_home`` also links the user's real
      ``~/Library/Keychains`` directory into the isolated HOME
      (``_link_macos_keychains``, process.py), so the symlink skip
      protects that keychain data as well.
    * Skipping is also sufficient: unlinking an entry needs write
      permission on its PARENT directory -- which the sweep grants via
      the walk's own directory chmods -- never on the entry's target.
    * ``os.walk`` never follows symlinked directories either
      (``followlinks=False``), so recursion cannot escape the workspace
      through a linked directory.

    Both rmtree passes ignore errors and every chmod is individually
    suppressed, so this function cannot raise.
    """
    shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        return
    for dirpath, dirnames, filenames in os.walk(path):
        for name in (*dirnames, *filenames):
            entry = os.path.join(dirpath, name)
            with contextlib.suppress(OSError):
                entry_stat = os.lstat(entry)
                is_link = stat.S_ISLNK(entry_stat.st_mode)
                is_dir = stat.S_ISDIR(entry_stat.st_mode)
                # The hardlink check is scoped to non-directories: an
                # ordinary directory already has st_nlink >= 2 (itself
                # plus one link per subdirectory), so skipping every
                # nlink > 1 entry would skip the whole directory tree --
                # the sweep would stop working for the read-only-tree
                # case it exists for. Only a FILE's inode can be shared
                # with the outside world (the token os.link fallback).
                if is_link or (not is_dir and entry_stat.st_nlink > 1):
                    continue
                os.chmod(
                    entry,
                    entry_stat.st_mode
                    | stat.S_IWUSR
                    | (stat.S_IXUSR if is_dir else 0),
                )
    shutil.rmtree(path, ignore_errors=True)


# Spec and plan reads among the first steps of a worker's history stay whole in a fitted resend.
_PINNED_READ_WINDOW = 12
_SPEC_OR_PLAN_READ = re.compile(r"(?:briefs/[^\"'\s]*\.spec\.md|ticket-plan\.md)")


def _step_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Messages grouped into steps: an assistant message with the tool results that answer it."""
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


def _reads_spec_or_plan(group: list[dict[str, Any]]) -> bool:
    head = group[0]
    if head.get("role") != "assistant" or len(group) < 2:
        return False
    for call in head.get("tool_calls") or []:
        fn = call.get("function") if isinstance(call, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if _SPEC_OR_PLAN_READ.search(args if isinstance(args, str) else json.dumps(args or {})):
            return True
    return False


def _migrate_store_home(root: Path) -> None:
    """Move a store HOME from inside the workspace (older layout) to beside it.

    Keeps every stored conversation resumable after the cwd move. Never overwrites.
    """
    old, new = root / "workspace" / "home", root / "home"
    if old.is_dir() and not new.exists():
        try:
            os.replace(old, new)
            logger.info("Antigravity: moved isolated HOME out of agy's cwd (%s -> %s)", old, new)
        except OSError as exc:
            logger.warning("Antigravity: could not move isolated HOME out of the workspace: %s", exc)


class AntigravityClient:
    """OpenAI-compatible client facade driving Antigravity CLI.

    Async compatibility boundary (issue #8)
    --------------------------------------
    Hermes drives this client from its async auxiliary path: with
    ``HERMES_SKIP_ASYNC_WRAP`` the client is used as-is ("already async-safe"),
    and the caller then does ``await client.chat.completions.create(...)``.
    Everything the facade returns is therefore awaitable-yielding-itself —
    ``stream._AwaitableCompletion`` for the non-streaming plan (which simply
    wraps the already-assembled response) and ``stream.AntigravityStream`` for
    the streaming plan (awaitable plus ``async for`` via ``__anext__``). Sync
    callers are untouched: the objects keep the same attributes and the same
    blocking behavior, so no existing caller needs to change.

    Known limitation (documented on purpose, not fixed here): the non-streaming
    ``create()`` executes the whole subprocess round-trip synchronously before
    it returns, so awaiting it still blocks the event loop for the request's
    duration. This is identical to the pre-existing sync behavior and to
    Hermes' own CopilotACPClient shim, and there is no plugin-side fix without
    a Hermes hook that would let ``create()`` hand back a coroutine instead of
    a finished value.
    """

    # Instruct Hermes not to wrap this client in wire transports or async adapters
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "antigravity-directsdk"
        self.base_url = base_url or AGY_MARKER_BASE_URL
        self._command = command or resolve_agy_command()
        self._args = list(args or ["--output-format", "stream-json", "--disable-slash-commands"])
        self._temp_dir = None
        # Persistent conversations (dan/persistent-conversations): one agy conversation per
        # Hermes session, resumable after restarts. Needs agy's conversation records to outlive
        # this client, so the agy HOME lives in plugin data instead of a per-client temp dir.
        self._store = None
        # agy runs reads relative to its cwd, so the isolated HOME (which links the OAuth token)
        # sits BESIDE agy's cwd, never inside it: <root>/home and <root>/workspace.
        home_root = None
        if not cwd and _session_store.persistence_enabled():
            try:
                self._store = _session_store.SessionStore()
                home_root = self._store.base
                workspace = home_root / "workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                _migrate_store_home(home_root)
                cwd = str(workspace)
            except Exception as exc:  # never block the provider on plugin-data trouble
                logger.warning("Antigravity: persistent conversations disabled (%s)", exc)
                self._store = None
                home_root = None
                cwd = None
        if cwd:
            self._cwd = cwd
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="hermes_agy_")
            home_root = Path(self._temp_dir.name)
            workspace = home_root / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            self._cwd = str(workspace)

        # Isolate agy state and session index from the user's real ~/.gemini/antigravity-cli.
        # A caller-supplied cwd keeps its HOME inside it (the caller owns that layout).
        self._isolated_home, self._isolated_gemini_dir = setup_isolated_home(
            home_root if home_root is not None else self._cwd)
        # Gate agy's own tools before any agy process starts in this HOME.
        try:
            install_tool_gate(self._isolated_home, self._isolated_gemini_dir)
        except Exception:
            logger.warning("Antigravity: could not install the native-tool gate", exc_info=True)
        # Bad-turn recovery (see _mark_bad_turn / _recover_bad_turn).
        self._pending_recovery: dict[str, Any] | None = None
        self._recovery_key: tuple[int, str] | None = None
        self._recovery_count = 0
        self._recovery_note: str | None = None
        # The conversation's shape just before the turn in flight was sent (rewind.read_point).
        self._rewind_point: dict[str, Any] | None = None
        self._rewind_backups = Path(self._isolated_home).parent / "rewind-backups"

        # A Hermes process that died without close() leaves its agy worker
        # orphaned, holding a conversation open on the shared account (the
        # 2026-11-11 multi-session interference). Best-effort, Linux-only,
        # scoped to workers whose owner tempdir is already gone.
        try:
            reap_orphaned_plugin_workers()
        except Exception:
            logger.debug("Orphaned-worker sweep failed", exc_info=True)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False
        # Valid fresh-context full-send samples only. Worker deltas and
        # concurrent oneshots are not comparable: see delivery.py and issue #29.
        self._delivery_watch = DeliveryWatch()
        self._active_processes: set[subprocess.Popen] = set()
        # RLock (NOT a plain Lock) deliberately -- issue #10.
        #
        # A suspended AntigravityStream._stream_generator forms a
        # reference cycle with the stream (stream._generator <-> generator
        # frame <-> stream), so an ABANDONED stream (consumer stopped
        # iterating without close()) is reclaimable only by the cyclic
        # GC. That pass fires at an arbitrary allocation -- possibly on
        # the SAME thread while it already holds this lock (client.close(),
        # _get_or_spawn_worker, _create_chat_completion, ...). The
        # GC-finalized generator's finally block then runs stream.close()
        # -> client._terminate_worker()/_update_worker_history() ->
        # `with self._lock` on a thread that already holds it: a plain
        # Lock self-deadlocks here (deterministic hangs were observed
        # while developing #9). RLock makes that same-thread re-entrance
        # succeed; cross-thread mutual exclusion is unchanged.
        #
        # Trade-off, documented on purpose: re-entrance can MASK a future
        # lock-ordering bug that would otherwise deadlock loudly -- a
        # code path that re-acquires _lock without expecting to already
        # hold it is now silently allowed instead of hanging. Mutating
        # shared state under _lock must therefore never call back into
        # the client (a nested _lock acquisition, a state-dependent
        # branch, another lock); the region is for flat
        # acquire -> mutate -> release only. Known, accepted exceptions
        # inside the regions below are the bounded-but-blocking process
        # calls (Popen spawn, proc.wait(timeout=2), Windows taskkill
        # without timeout): they touch no client state, and moving them
        # out of the locked region would trade the deadlock for races.
        self._lock = threading.RLock()
        self._worker_proc: subprocess.Popen | None = None
        self._worker_model: str | None = None
        self._worker_effort: str | None = None
        self._worker_history: list[dict[str, Any]] = []
        self._worker_conversation_id: str | None = None
        self._conversation_history: list[dict[str, Any]] = []
        # Media rules the LIVE worker's settings.json contained at spawn:
        # agy reads permissions only at spawn, so a turn referencing new
        # media paths must respawn the worker (conversation preserved).
        self._worker_media_rules: frozenset[str] = frozenset()
        # Media files this session materialized (downloads, cache files);
        # deleted at close() so no fetched media outlives the session.
        self._materialized_media: set[str] = set()
        # Cumulative usage snapshot of the current worker session. agy 1.2.10+
        # persistent workers report cumulative session usage, so per-turn
        # deltas need this baseline. The reference is replaced on every spawn
        # and termination; the dict contents are advanced in place by the
        # owning session's stream. Each stream captures the dict of the
        # session it was created for, so a stream left over from a terminated
        # session cannot corrupt a later session's baseline.
        self._worker_usage_baseline: dict[str, int] = {}
        self._worker_lock = threading.Lock()
        self._worker_idle_timer: threading.Timer | None = None
        # Last moment each worker showed a sign of life: its spawn, or the last
        # stdout line it produced (stamped by AntigravityStream). Read by the
        # abandon check in _reap_idle_worker, which needs to tell a genuinely
        # busy worker from one whose turn died still holding _worker_lock.
        self._worker_last_activity: dict[subprocess.Popen, float] = {}
        # CPU seconds at the last sign of life: the discriminator between a
        # silent abandoned turn and a silent busy one.
        self._worker_cpu_at_activity: dict[subprocess.Popen, float] = {}
        # What the live agy conversation already holds (message hashes, see session_store),
        # which agy conversation it is, and the tool set it was told about.
        self._worker_hashes: list[str] = []
        self._worker_tools_fp: str | None = None
        self._seen_conversation_id: str | None = None
        # The Hermes session this client is bound to, and its ownership lock (session_store).
        self._session_id: str | None = None
        self._session_claim = None
        if self._store is not None:
            with contextlib.suppress(Exception):
                self._store.prune(self._isolated_gemini_dir)

    @staticmethod
    def _resolve_real_token_path() -> Path | None:
        return resolve_real_token_path()

    def _resolve_model_and_effort(
        self,
        model: str | None,
        reasoning_effort: str | None = None,
    ) -> tuple[str, str | None]:
        return resolve_model_and_effort(model, reasoning_effort)

    def _child_env(self) -> dict[str, str]:
        return build_child_env(self._isolated_home)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        terminate_process(proc)

    def __enter__(self) -> "AntigravityClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        # Phase 1 -- terminate every child process BEFORE the workspace is
        # touched (issue #4): on Windows a still-running agy keeps
        # conversations/*.db open, and taskkill /F /T is asynchronous, so
        # removing the workspace first would race the OS and fail with
        # WinError 32. Ordering is therefore load-bearing, not cosmetic.
        with self._lock:
            self.is_closed = True
            self._terminate_worker_locked()
            procs = tuple(self._active_processes)
            self._active_processes.clear()
            self._worker_last_activity.clear()
            self._worker_cpu_at_activity.clear()
        for proc in procs:
            self._terminate_process(proc)
        # Phase 1.5 -- delete the media files this session materialized.
        # After every child is dead (they may still have had the files open),
        # before the workspace removal; must never raise (issue #4 contract).
        with self._lock:
            materialized = set(self._materialized_media)
            self._materialized_media.clear()
        for path in materialized:
            try:
                # Defense in depth: even if a foreign path ever reached the
                # sink, close() deletes only files inside this plugin's own
                # media cache dir — never a user-referenced file (review
                # 5887913331, point 1).
                if not _is_plugin_owned_media_path(path):
                    logger.debug(
                        "Skipping non-plugin-owned media path on close: %s", path
                    )
                    continue
                os.remove(path)
            except OSError:
                pass
        self._release_session()
        # Phase 2 -- remove the private workspace, retrying transient
        # failures. close() must never raise from cleanup and must never
        # leave the client half-closed (issue #4).
        self._remove_temp_dir()

    def _remove_temp_dir(self) -> None:
        """Remove the private workspace; cannot raise, cannot half-finish.

        Issue #4 (reported on Windows 11, agy 1.2.3): cleanup raised
        PermissionError [WinError 32] on conversations/*.db because the
        agy child still held the file after taskkill /F /T returned (that
        kill is asynchronous). The old code swallowed the error and gave
        up, leaking the whole workspace. Now: bounded retries with short
        backoff (see _TEMP_DIR_CLEANUP_ATTEMPTS/_BACKOFF constants above
        for the sizing), then a forced removal of whatever is unlocked.

        Live Windows verification by the reporter is still pending and
        will be requested on the issue after this merges; the retry budget
        is deliberately small so the worst case stays under ~0.5s and the
        common clean path pays nothing.

        The client is already fully closed when this runs (is_closed set,
        every process terminated in close() phase 1), and _temp_dir is
        cleared unconditionally, so a second close() is a no-op and a
        lingering workspace never blocks a fresh client on the same path.
        """
        temp_dir = self._temp_dir
        if temp_dir is None:
            return
        self._temp_dir = None
        try:
            self._cleanup_temp_dir_with_retries(temp_dir)
        except Exception as exc:
            # Defense in depth: a cleanup path must never break close().
            # Only the workspace path is logged; the isolated HOME inside
            # it holds a token link but no secret content of its own, and
            # the path itself is this client's own private directory.
            logger.debug(
                "Antigravity workspace %s cleanup raised unexpectedly: %s",
                temp_dir.name,
                exc,
            )

    def _cleanup_temp_dir_with_retries(self, temp_dir: tempfile.TemporaryDirectory) -> None:
        """Try cleanup up to _TEMP_DIR_CLEANUP_ATTEMPTS times, then force.

        The loop catches EVERY exception -- not just OSError -- because the
        goal is that nothing can escape into close(): OSError
        (PermissionError/WinError 32, EBUSY, ENOTEMPTY -- the "handle
        still held" symptoms) is retried with backoff, while any other
        exception breaks out immediately and lands in the forced removal
        below. Retrying a failed TemporaryDirectory.cleanup() is safe: its
        finalizer is detached on the first call, so each attempt is a
        fresh rmtree pass.
        """
        for attempt in range(1, _TEMP_DIR_CLEANUP_ATTEMPTS + 1):
            try:
                temp_dir.cleanup()
                return
            except Exception as exc:
                retryable = isinstance(exc, OSError) and attempt < _TEMP_DIR_CLEANUP_ATTEMPTS
                if retryable:
                    logger.debug(
                        "Antigravity workspace %s cleanup attempt %d/%d failed (%s); retrying.",
                        temp_dir.name,
                        attempt,
                        _TEMP_DIR_CLEANUP_ATTEMPTS,
                        exc,
                    )
                    time.sleep(_TEMP_DIR_CLEANUP_BACKOFF_SECONDS[attempt - 1])
                    continue
                logger.debug(
                    "Antigravity workspace %s cleanup failed on attempt %d/%d (%s: %s); "
                    "forcing removal of whatever is unlocked.",
                    temp_dir.name,
                    attempt,
                    _TEMP_DIR_CLEANUP_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                )
                break
        if temp_dir.name:
            _force_rmtree(temp_dir.name)

    def _terminate_worker_locked(self, keep_conversation: bool = False) -> None:
        self._cancel_idle_timer_locked()
        if self._worker_proc is not None:
            proc = self._worker_proc
            self._worker_proc = None
            self._worker_model = None
            self._worker_effort = None
            self._worker_history = []
            self._active_processes.discard(proc)
            self._worker_last_activity.pop(proc, None)
            self._worker_cpu_at_activity.pop(proc, None)
            self._terminate_process(proc)
        # The session is gone: any later usage from it belongs to a dead
        # session, and the next spawn must start from a clean baseline.
        self._worker_usage_baseline = {}
        # The agy conversation itself survives on disk (session_store keeps the last good state).
        # keep_conversation=True (idle reap, media respawn, auth retry) keeps what the conversation
        # holds so the next turn resumes it with --conversation and a delta.
        if not keep_conversation:
            self._worker_conversation_id = None
            self._conversation_history = []
            self._worker_hashes = []
            self._worker_tools_fp = None
        self._seen_conversation_id = None

    def _terminate_worker(self, keep_conversation: bool = False) -> None:
        with self._lock:
            self._terminate_worker_locked(keep_conversation=keep_conversation)

    def _set_worker_conversation_id(self, conv_id: str) -> None:
        with self._lock:
            if conv_id:
                self._worker_conversation_id = conv_id

    def _purge_isolated_conversations(self) -> None:
        """Purge isolated conversation database files on session reset or new conversation."""
        conv_dir = self._isolated_gemini_dir / "conversations"
        if conv_dir.is_dir():
            for entry in conv_dir.glob("*.db*"):
                with contextlib.suppress(OSError):
                    if entry.is_file():
                        entry.unlink()

    def _note_worker_conversation(self, conversation_id: str | None) -> None:
        """Called by a worker stream before it records a successful turn."""
        if conversation_id:
            with self._lock:
                self._seen_conversation_id = conversation_id

    def _update_worker_history(self, messages: list[dict[str, Any]] | None) -> None:
        hashes = _session_store.message_hashes(list(messages or []))
        with self._lock:
            self._recovery_key, self._recovery_count = None, 0
            msg_list = list(messages or [])
            self._worker_history = msg_list
            self._conversation_history = msg_list
            self._worker_hashes = hashes
            requested = self._worker_conversation_id
            seen = self._seen_conversation_id
            if seen:
                self._worker_conversation_id = seen
            conversation_id = self._worker_conversation_id
            session_id = self._session_id if self._session_claim is not None else None
            model, effort, tools_fp = self._worker_model, self._worker_effort, self._worker_tools_fp
            self._arm_idle_timer_locked()
        if requested and seen and requested != seen:
            logger.warning("Antigravity: resumed conversation %s came back as %s", requested, seen)
        if self._store is not None and session_id and conversation_id:
            try:
                self._store.put(session_id, conversation_id=conversation_id, model=model, effort=effort,
                                hashes=hashes, tools=tools_fp)
            except Exception as exc:
                logger.warning("Antigravity: could not record conversation for %s: %s", session_id, exc)
        if conversation_id:
            _rewind.drop_backups(self._rewind_backups, conversation_id)

    # ── Bad-turn recovery ─────────────────────────────────────────────────────────────────

    def _mark_bad_turn(self, reason: str, messages: list[dict[str, Any]] | None,
                       conversation_id: str | None = None) -> None:
        """A worker turn went bad (runaway, native tool, silence, failure); called by the stream.

        Records what the conversation held BEFORE the turn (the in-memory state is only ever
        advanced by a successful turn), so the next request can rewind the turn out of agy's
        conversation and resume from there. The store entry is flagged too, so a resume after a
        gateway restart never continues on top of the bad turn.
        """
        hashes = _session_store.message_hashes(list(messages or []))
        key = (len(hashes), hashes[-1] if hashes else "")
        with self._lock:
            if key == self._recovery_key:
                self._recovery_count += 1
            else:
                self._recovery_key, self._recovery_count = key, 1
            if self._pending_recovery is None:
                self._pending_recovery = {
                    "conversation_id": self._worker_conversation_id or conversation_id or None,
                    "good_hashes": list(self._worker_hashes),
                    "tools_fp": self._worker_tools_fp,
                    "reason": reason,
                    "point": self._rewind_point,
                }
            session_id = self._session_id if self._session_claim is not None else None
        if self._store is not None and session_id:
            try:
                self._store.mark_bad(session_id, reason)
            except Exception as exc:
                logger.warning("Antigravity: could not flag bad turn for %s: %s", session_id, exc)

    def _rewind_bad_turn(self, pending: dict[str, Any]) -> bool:
        """Delete the bad turn from agy's conversation so it resumes at the last good state.

        False means the conversation could not be rewound safely; the caller falls back to a
        fresh conversation with a fitted resend. agy is already stopped (``_recover_bad_turn``).
        """
        point = pending.get("point")
        if not isinstance(point, dict) or point.get("conversation_id") != pending.get("conversation_id"):
            logger.info("Antigravity rewind: no rewind point for this conversation")
            return False
        try:
            _rewind.rewind(self._isolated_gemini_dir, point, backup_dir=self._rewind_backups,
                           agy_version=self._agy_cli_version())
        except _rewind.RewindRefused as exc:
            logger.warning("Antigravity rewind refused: %s", exc)
            return False
        except Exception as exc:  # sqlite/OS trouble: the transaction rolled back
            logger.warning("Antigravity rewind failed: %s", exc, exc_info=True)
            return False
        return True

    def _agy_cli_version(self) -> str | None:
        """Identity of the agy binary (resolved path, size, mtime): an upgrade changes it.

        A stat, not ``agy --version``: it runs before every turn on a known conversation.
        """
        try:
            path = shutil.which(self._command) or self._command
            real = os.path.realpath(path)
            st = os.stat(real)
            return f"{real}:{st.st_size}:{int(st.st_mtime)}"
        except (OSError, TypeError):
            return None

    def _record_rewind_point(self, session_id: str | None, persistent: bool,
                             conversation_id: str | None, prompt: str) -> None:
        """Before a turn on a known conversation is sent: remember where to cut back to."""
        point = None
        if conversation_id:
            try:
                point = _rewind.read_point(
                    Path(self._isolated_gemini_dir) / "conversations" / f"{conversation_id}.db",
                    conversation_id=conversation_id, prompt=prompt, agy_version=self._agy_cli_version())
            except Exception as exc:
                logger.debug("Antigravity: could not read the rewind point: %s", exc)
        with self._lock:
            self._rewind_point = point
        if persistent:
            try:
                self._store.set_rewind_point(session_id, point)
            except Exception as exc:
                logger.debug("Antigravity: could not store the rewind point: %s", exc)

    def _recover_bad_turn(self, session_id: str | None, persistent: bool) -> None:
        """Before any send: clear up a turn the previous request left bad (GP4 / GP6 / GP7)."""
        with self._lock:
            pending = self._pending_recovery
            count = self._recovery_count
        stored = None
        if pending is None and persistent:
            entry = self._store.get(session_id)
            if entry and entry.get("bad"):
                stored = entry
                pending = {"conversation_id": entry["conversation_id"],
                           "good_hashes": list(entry.get("hashes") or []),
                           "tools_fp": entry.get("tools"), "reason": entry["bad"],
                           "point": entry.get("rewind")}
        if pending is None:
            return
        if count > MAX_RECOVERIES_PER_REQUEST:
            raise AntigravityRecoveryExhausted(
                f"Antigravity: this request went bad {count} times in a row "
                f"(last: {pending.get('reason')}); giving up after {MAX_RECOVERIES_PER_REQUEST} recoveries.")
        # agy must not hold the conversation while it is edited.
        self._terminate_worker(keep_conversation=True)
        conversation_id = pending.get("conversation_id")
        rewound = bool(conversation_id and pending.get("good_hashes")
                       and self._conversation_exists(conversation_id)
                       and self._rewind_bad_turn(pending))
        logger.warning("Antigravity recovery (%s): %s", pending.get("reason"),
                       "rewound the bad turn" if rewound else "fresh conversation with a fitted resend")
        reason = str(pending.get("reason") or "")
        with self._lock:
            self._pending_recovery = None
            self._rewind_point = None
            if "' ran" in reason:
                # The rewind takes the attempt out of the model's memory, not out of the world.
                self._recovery_note = (
                    f"[Note] An earlier attempt at this turn used a tool outside Hermes ({reason}); "
                    "its effects may already be present.")
            if rewound:
                self._worker_conversation_id = conversation_id
                self._worker_hashes = list(pending["good_hashes"])
                self._conversation_history = []
                self._worker_tools_fp = pending.get("tools_fp")
        if not rewound:
            self._terminate_worker()  # drops the in-memory conversation
        if persistent:
            try:
                if rewound:
                    entry = stored or self._store.get(session_id) or {}
                    self._store.put(session_id, conversation_id=conversation_id,
                                    model=entry.get("model"), effort=entry.get("effort"),
                                    hashes=pending["good_hashes"], tools=pending.get("tools_fp"))
                else:
                    self._store.forget(session_id)
            except Exception as exc:
                logger.warning("Antigravity: could not update the store after recovery: %s", exc)

    # ── Persistent conversations ───────────────────────────────────────────────────────────

    def _release_session(self) -> None:
        claim, self._session_claim, self._session_id = self._session_claim, None, None
        if self._store is not None:
            self._store.release(claim)

    def _bind_session(self, session_id: str | None) -> bool:
        """True when this client owns ``session_id``'s agy conversation (resume + record)."""
        if self._store is None or not session_id:
            return False
        if self._session_id == session_id and self._session_claim is not None:
            return True
        if self._session_id is not None and self._session_id != session_id:
            # Same Hermes agent, new chat (/new, /reset): the old conversation stays on disk.
            self._terminate_worker()
            self._release_session()
        claim = self._store.try_claim(session_id)
        self._session_id = session_id
        if claim is None:
            # Another live client drives this conversation (e.g. Hermes' background review):
            # work in a private conversation instead of writing into the shared one.
            return False
        self._session_claim = claim
        return True

    @staticmethod
    def _instruction_header(messages, tools, tool_choice) -> str:
        """Tool protocol, tool catalog and Hermes system prompt, as in the first full prompt."""
        system = [m for m in messages if isinstance(m, dict) and m.get("role") in ("system", "developer")]
        header = _format_messages_as_prompt(system, tools=tools, tool_choice=tool_choice)
        tail = "Continue the conversation from the latest message."
        if header.endswith(tail):
            header = header[: -len(tail)].rstrip()
        return "### STANDING INSTRUCTIONS (restated; they replace any earlier version):\n\n" + header

    def _note_checkpoint(self) -> None:
        """agy compacted the conversation: restate the standing instructions on the next turn."""
        with self._lock:
            self._worker_tools_fp = None

    def _conversation_exists(self, conversation_id: str | None) -> bool:
        if not conversation_id:
            return False
        return (Path(self._isolated_gemini_dir) / "conversations" / f"{conversation_id}.db").exists()

    def _fit_full_prompt(self, messages, *, model, tools, tool_choice) -> tuple[str, bool]:
        """Full-history prompt that stays under agy's single-message limit (UTF-8 bytes).

        agy silently keeps only the beginning of an oversized message. In order, until it fits:
        older tool results kept long (skills, workflow files), then cut short; then whole older
        steps (an assistant message with its tool results, never split) are left out, oldest first,
        with a note. Always kept: the system prompt, the first user message (a worker's brief, or
        Hermes' compaction summary), early reads of the ticket's spec and plan, and the newest steps.
        Returns ``(prompt, trimmed)``.
        """
        limit = _session_store.max_prompt_chars()
        size = _session_store.prompt_size
        fmt = functools.partial(_format_messages_as_prompt, model=model, tools=tools, tool_choice=tool_choice,
                                media_sink=self._materialized_media)
        prompt = fmt(messages, historical_tool_chars=20000)
        if size(prompt) <= limit:
            return prompt, False
        prompt = fmt(messages)
        if size(prompt) <= limit:
            return prompt, False

        valid = [m for m in messages if isinstance(m, dict)]
        system = [m for m in valid if m.get("role") in ("system", "developer")]
        body = [m for m in valid if m.get("role") not in ("system", "developer")]
        first = body[:1] if body and body[0].get("role") == "user" else []
        groups = _step_groups(body[len(first):])
        pinned = {gi for gi, group in enumerate(groups[:_PINNED_READ_WINDOW]) if _reads_spec_or_plan(group)}
        keep_whole = {id(m) for gi in pinned for m in groups[gi] if m.get("role") == "tool"}

        def render(keep_from: int, *, pin: bool = True) -> str:
            kept_early = [m for gi in sorted(pinned) if pin and gi < keep_from for m in groups[gi]]
            left_out = sum(len(groups[gi]) for gi in range(keep_from) if not (pin and gi in pinned))
            note = [{"role": "system", "content": (
                f"Note: {left_out} of the oldest messages of this conversation are not shown (earlier "
                "tool calls and their results), because the full history is larger than one Antigravity "
                "prompt allows. Files on disk already reflect that work: re-read them when you need their "
                "current state instead of redoing it.")}] if left_out else []
            tail = [m for group in groups[keep_from:] for m in group]
            return fmt(system + note + first + kept_early + tail, keep_whole=keep_whole if pin else None)

        best = None
        for pin in (True, False):
            lo, hi = 1, len(groups) - 1  # always keep the newest step
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = render(mid, pin=pin)
                if size(candidate) <= limit:
                    best, hi = candidate, mid - 1
                else:
                    lo = mid + 1
            if best is not None:
                return best, True
        logger.warning("Antigravity: even the newest step alone is larger than one prompt allows")
        return render(max(len(groups) - 1, 0), pin=False), True

    def _prepare_worker_turn(self, *, session_id, model, effort, messages, tools, tool_choice):
        """``(proc, prompt, branch)`` for the next worker turn: send agy only what it has not seen."""
        hashes = _session_store.message_hashes(messages)
        tools_fp = _session_store.instructions_fingerprint(messages, tools, tool_choice)
        persistent = self._bind_session(session_id)
        self._recover_bad_turn(session_id, persistent)
        with self._lock:
            live = self._worker_proc is not None and self._worker_proc.poll() is None
            # A worker stopped with keep_conversation (idle reap, media respawn) leaves what its
            # conversation holds in memory, so it resumes even without a session store.
            kept = (not live and bool(self._worker_conversation_id) and bool(self._worker_hashes)
                    and self._conversation_exists(self._worker_conversation_id))
            known = list(self._worker_hashes) if live or kept else []
            conversation_id = self._worker_conversation_id if live or kept else None
            known_tools = self._worker_tools_fp if live or kept else None
            same_runtime = live and self._worker_model == model and self._worker_effort == effort
        if not known and persistent:
            entry = self._store.get(session_id)
            if entry and self._conversation_exists(entry["conversation_id"]):
                known = list(entry.get("hashes") or [])
                conversation_id = entry["conversation_id"]
                known_tools = entry.get("tools")
                if live:  # a live worker with nothing recorded belongs to no conversation we know
                    self._terminate_worker()
                    live = same_runtime = False

        mode, start, keep = _session_store.reconcile(known, hashes)
        full = None
        if mode == "rewind":
            full, trimmed = self._fit_full_prompt(messages, model=model, tools=tools, tool_choice=tool_choice)
            if not trimmed or not conversation_id and not live:
                mode = "fresh"  # exact replay fits: cleaner than asking agy to disregard turns
        if mode == "continue" and not live and not conversation_id:
            mode = "fresh"

        if mode == "fresh":
            if live or conversation_id:
                self._terminate_worker()
                if self._store is None:
                    # Per-client temp HOME: nothing else resumes these (never with a shared store HOME).
                    self._purge_isolated_conversations()
            proc = self._get_or_spawn_worker(model, effort)
            prompt = full if full is not None else self._fit_full_prompt(
                messages, model=model, tools=tools, tool_choice=tool_choice)[0]
            branch = "full"
        else:
            if live and not same_runtime:
                # Model/effort switch: same conversation, new runtime (resume is model-agnostic).
                self._terminate_worker()
                live = False
            if live:
                with self._lock:
                    proc = self._worker_proc
            else:
                proc = self._get_or_spawn_worker(model, effort, conversation_id=conversation_id)
            if mode == "continue":
                delta = list(messages[start:])
                # agy produced the assistant turns that follow what it was last sent.
                while delta and isinstance(delta[0], dict) and delta[0].get("role") == "assistant":
                    delta.pop(0)
                prompt = _format_delta_prompt(delta, pending_user_text=_latest_user_text(messages),
                                              media_sink=self._materialized_media)
                branch = "delta" if live else "resume"
            else:
                resend = list(messages[keep:]) or list(messages[-1:])
                anchor_index = keep - 1 if messages[keep:] else len(messages) - 2
                anchor = (_render_message_content(messages[anchor_index].get("content"))
                          if 0 <= anchor_index < len(messages) and isinstance(messages[anchor_index], dict) else "")
                if messages[keep:]:
                    note = ("[Conversation edited] The user withdrew the latest part of this conversation. "
                            "Disregard every message after the one that begins: \"" + anchor[:200].replace("\n", " ")
                            + "\" -- your replies and any tool results after it no longer apply. "
                            "Continue from the messages below.")
                else:
                    note = ("[Retry] Disregard your previous answer to the latest message and answer it again.")
                prompt = note + "\n\n" + _format_delta_prompt(resend, pending_user_text=_latest_user_text(messages),
                                                                media_sink=self._materialized_media)
                branch = "rewind"
            if tools_fp != known_tools:
                # New tools / edited system prompt, or agy compacted its history: restate them.
                prompt = self._instruction_header(messages, tools, tool_choice) + "\n\n" + prompt
        with self._lock:
            self._worker_tools_fp = tools_fp
            note, self._recovery_note = self._recovery_note, None
            point_conversation = self._worker_conversation_id if branch != "full" else None
        if note:
            prompt = note + "\n\n" + prompt
        self._record_rewind_point(session_id, persistent, point_conversation or conversation_id
                                  if branch != "full" else None, prompt)
        return proc, prompt, branch

    def _cancel_idle_timer_locked(self) -> None:
        timer = self._worker_idle_timer
        self._worker_idle_timer = None
        if timer is not None:
            timer.cancel()

    def _arm_idle_timer_locked(self) -> None:
        """(Re)start the idle countdown; called when a worker turn finishes."""
        self._cancel_idle_timer_locked()
        proc = self._worker_proc
        idle_seconds = _worker_idle_seconds()
        if proc is None or self.is_closed or idle_seconds <= 0:
            return
        timer = threading.Timer(idle_seconds, self._reap_idle_worker, args=(proc,))
        timer.daemon = True
        self._worker_idle_timer = timer
        timer.start()

    def _reap_idle_worker(self, proc: subprocess.Popen) -> None:
        # A held worker lock means a turn is running on the worker.
        if not self._worker_lock.acquire(blocking=False):
            # Re-arm rather than give up. The countdown used to be armed only
            # by a turn that FINISHED, so a wedged or abandoned turn left the
            # worker with no timer at all and it survived the idle bound
            # entirely. Re-arming repeats the check without ever reaching the
            # terminate below, which is still what a genuinely busy worker
            # wants -- but a worker that has shown no sign of life for the
            # abandon grace is not busy, its turn is gone.
            with self._lock:
                if self._worker_proc is not proc:
                    self._worker_last_activity.pop(proc, None)
                    return
                if self._worker_is_abandoned_locked(proc):
                    last_seen = self._worker_last_activity.get(proc)
                    logger.warning(
                        "Terminating abandoned agy worker pid=%s: no output for %.0fs "
                        "while its turn still held the worker lock",
                        getattr(proc, "pid", None),
                        time.monotonic() - (last_seen if last_seen is not None else time.monotonic()),
                    )
                    self._terminate_worker_locked(keep_conversation=True)
                    return
                self._arm_idle_timer_locked()
            return
        try:
            with self._lock:
                if self._worker_proc is not proc:
                    return
                logger.info("Terminating idle agy worker pid=%s", getattr(proc, "pid", None))
                self._terminate_worker_locked(keep_conversation=True)
        finally:
            self._worker_lock.release()

    def _note_worker_activity(self, proc: subprocess.Popen) -> None:
        """Stamp a worker as alive; AntigravityStream calls this per stdout line.

        The CPU reading is taken outside the lock: it must never extend the
        critical section the reader thread shares with the timer thread.
        """
        cpu = _worker_cpu_seconds(proc)
        with self._lock:
            if self._worker_proc is proc:
                self._worker_last_activity[proc] = time.monotonic()
                if cpu is not None:
                    self._worker_cpu_at_activity[proc] = cpu

    def _worker_is_abandoned_locked(self, proc: subprocess.Popen) -> bool:
        """True when a lock-holding worker has shown no sign of life for too long.

        A held _worker_lock normally means a turn is running, and killing it
        would be a regression. But an abandoned turn never releases the lock:
        the stream generator's finally block -- the only place that releases it
        -- runs when the client/stream reference cycle is collected, and a
        lingering reference (a traceback, a cached response object) can postpone
        that indefinitely. The grace runs from the worker's last output, not
        from when the lock was taken, which nobody records -- and it is only
        acted upon when the worker has also stopped burning CPU, so a live but
        quiet turn (long reasoning, slow first token, generous read deadline) is
        never mistaken for a dead one.
        """
        abandon_after = _worker_abandon_seconds()
        if abandon_after <= 0:
            return False
        last_seen = self._worker_last_activity.get(proc)
        if last_seen is None:
            # Unknown baseline (a worker installed by an older code path): stay
            # conservative and keep re-arming rather than guess.
            return False
        if (time.monotonic() - last_seen) < abandon_after:
            return False
        # Silence is necessary but not sufficient. A live turn can be silent for
        # minutes -- a long reasoning block, a slow first token, a caller whose
        # read deadline is longer than the grace -- and it burns CPU while it
        # runs. An abandoned worker burns none. Unmeasurable CPU means unproven
        # abandonment: keep re-arming rather than risk a live turn.
        cpu_now = _worker_cpu_seconds(proc)
        cpu_then = self._worker_cpu_at_activity.get(proc)
        if cpu_now is None or cpu_then is None:
            return False
        return (cpu_now - cpu_then) < _WORKER_ABANDON_CPU_SECONDS

    def _ensure_worker_media_permissions(self, paths: set[str]) -> None:
        """Grant agy's headless read permission for this turn's media paths.

        agy auto-denies read_file/view_file without an explicit per-target
        ``read_file(<path>)`` rule in settings.json, and reads that file only
        at spawn. So: merge the rules into the isolated settings.json, and if
        a live worker was spawned without them, terminate it KEEPING the agy
        conversation — the next ``_get_or_spawn_worker`` respawns with
        ``--conversation`` and the fresh rules, so delta resume still works.
        """
        if not paths:
            return
        rules = write_media_read_permissions(self._isolated_gemini_dir, paths)
        with self._lock:
            live = (
                self._worker_proc is not None
                and self._worker_proc.poll() is None
            )
            if live and not paths <= self._worker_media_rules:
                self._terminate_worker_locked(keep_conversation=True)
            self._worker_media_rules = frozenset(rules)

    def _get_or_spawn_worker(
        self,
        model: str,
        effort: str | None,
        conversation_id: str | None = None,
    ) -> subprocess.Popen:
        with self._lock:
            if (
                self._worker_proc is not None
                and self._worker_proc.poll() is None
                and self._worker_model == model
                and self._worker_effort == effort
                and (conversation_id is None or conversation_id == self._worker_conversation_id)
            ):
                # A continuation turn reuses the worker without respawning it.
                # Without a fresh baseline the abandon grace would be measured
                # from the PREVIOUS turn's last output, so a worker idle for
                # almost a full window and then handed a new turn could be
                # reaped mid-turn. The grace also has to restart, not merely
                # re-measure: a silent first token is normal at turn start.
                proc = self._worker_proc
                self._worker_last_activity[proc] = time.monotonic()
                cpu = _worker_cpu_seconds(proc)
                if cpu is not None:
                    self._worker_cpu_at_activity[proc] = cpu
                return proc

            self._terminate_worker_locked(keep_conversation=bool(conversation_id))

            # Multiple agy processes share one Google account; whoever
            # refreshed last wins. Re-point the isolated token at the real
            # file right before every spawn so a worker never starts from a
            # stale copy ("You are not logged into Antigravity").
            sync_isolated_token(self._isolated_home)

            cmd_args = [self._command, "--input-format", "stream-json", *self._args]
            if conversation_id:
                cmd_args.extend(["--conversation", conversation_id])
            if model:
                cmd_args.extend(["--model", model])
            if effort:
                cmd_args.extend(["--effort", effort])

            proc = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._cwd,
                env=self._child_env(),
                **_own_process_group(),
            )
            start_stderr_drainer(proc)
            self._worker_proc = proc
            self._worker_model = model
            self._worker_effort = effort
            self._worker_history = list(self._conversation_history) if conversation_id else []
            if conversation_id:
                self._worker_conversation_id = conversation_id
            # Stamp what the fresh worker's settings.json granted: agy reads
            # permissions only at spawn (verified: rewriting the file under a
            # live worker changes nothing for its next turn).
            self._worker_media_rules = read_media_rules(self._isolated_gemini_dir)
            self._worker_usage_baseline = {}
            self._active_processes.add(proc)
            self._worker_last_activity[proc] = time.monotonic()
            # Give every worker a death date at spawn, not only when a turn
            # finishes. The countdown was armed solely by
            # _update_worker_history, so a worker whose first turn never
            # completed -- an abandoned stream (consumer stopped iterating
            # without close()), a cancelled request, or a capacity retry that
            # dropped the generator -- had no timer and no close() caller, and
            # lived for the lifetime of the gateway. Hermes does not call
            # close() on an evicted client (see _DEFAULT_WORKER_IDLE_SECONDS),
            # so that was unbounded: 86 such workers were measured holding
            # 15.3 GB on a 16 GB host. A live turn still holds _worker_lock,
            # which _reap_idle_worker respects, and the turn's own completion
            # re-arms the countdown, so arming here cannot cut a turn short.
            self._arm_idle_timer_locked()
            return proc

    def _run_oneshot_completion(
        self,
        *,
        model: str,
        effort: str | None,
        messages: list[dict[str, Any]],
        timeout: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        stream: bool,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            media_sink=self._materialized_media,
        )
        dump_prompt_debug(
            prompt=prompt_text,
            branch="oneshot",
            model=model,
            messages=messages,
        )
        cmd_args = [self._command, *self._args]
        if model:
            cmd_args.extend(["--model", model])
        if effort:
            cmd_args.extend(["--effort", effort])

        # Same account-contention guard as the worker spawn path.
        sync_isolated_token(self._isolated_home)

        proc = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self._cwd,
            env=self._child_env(),
            **_own_process_group(),
        )
        start_stderr_drainer(proc)

        try:
            if proc.stdin:
                proc.stdin.write(prompt_text + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except OSError:
            pass

        with self._lock:
            self._active_processes.add(proc)

        stream_iter = AntigravityStream(
            proc=proc,
            client=self,
            model=model,
            timeout=timeout,
            tools=tools,
            is_worker=False,
            prompt_chars=len(prompt_text),
            # The turn's messages feed the native media-attach allowlist
            # (read_file/view_file on referenced media paths only).
            messages=messages,
            # Oneshots run concurrently when the worker is busy. They have no
            # comparable prior sample and must not mutate the worker watch.
            delivery_watch=None,
        )
        if stream:
            return stream_iter
        return collect_stream_completion(stream_iter)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        **extra_kwargs: Any,
    ) -> Any:
        if self.is_closed:
            raise RuntimeError("AntigravityClient is closed.")

        if not is_authenticated():
            raise RuntimeError(
                "Antigravity CLI is not authenticated. Please run 'agy' in your terminal "
                "to log in with your Google account."
            )

        effort_param = reasoning_effort or extra_kwargs.get("reasoning_effort")
        resolved_model, effort = self._resolve_model_and_effort(model, effort_param)
        messages_list = list(messages or [])
        # Hermes' streaming caller passes an httpx.Timeout, not a scalar; use
        # its read budget instead of silently falling back to the default.
        if hasattr(timeout, "read"):
            read_timeout = getattr(timeout, "read")
            timeout = float("inf") if read_timeout is None else read_timeout
        effective_timeout = (
            float(timeout)
            if isinstance(timeout, (int, float)) and timeout > 0
            else _DEFAULT_TIMEOUT_SECONDS
        )

        # Multimodal translation: grant agy's headless read permission for
        # every media path this turn's messages reference, BEFORE any spawn
        # decision — both the worker and the oneshot process read settings.json
        # at spawn. A live worker lacking the new paths is respawned with its
        # agy conversation preserved.
        turn_media_paths = media_reference_paths(
            messages_list, self._materialized_media
        )
        if turn_media_paths:
            self._ensure_worker_media_permissions(turn_media_paths)
        session_id = extra_kwargs.get("hermes_session_id")
        session_id = str(session_id).strip() if session_id else None

        worker_acquired = self._worker_lock.acquire(blocking=False)
        if worker_acquired:
            try:
                proc, prompt_payload, branch = self._prepare_worker_turn(
                    session_id=session_id, model=resolved_model, effort=effort,
                    messages=messages_list, tools=tools, tool_choice=tool_choice,
                )
                comparable_full_send = branch == "full"
                logger.info("Antigravity turn: branch=%s session=%s chars=%d", branch,
                            session_id or "-", len(prompt_payload))
                dump_prompt_debug(
                    prompt=prompt_payload,
                    branch=branch,
                    model=resolved_model,
                    messages=messages_list,
                )

                event_msg = {"event": "user", "message": {"content": prompt_payload}}
                try:
                    proc.stdin.write(json.dumps(event_msg) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    comparable_full_send = True
                    self._terminate_worker()
                    if self._store is None:
                        self._purge_isolated_conversations()
                    proc = self._get_or_spawn_worker(resolved_model, effort)
                    prompt_payload = self._fit_full_prompt(
                        messages_list, model=resolved_model, tools=tools, tool_choice=tool_choice
                    )[0]
                    dump_prompt_debug(
                        prompt=prompt_payload,
                        branch="retry",
                        model=resolved_model,
                        messages=messages_list,
                    )
                    event_msg = {"event": "user", "message": {"content": prompt_payload}}
                    proc.stdin.write(json.dumps(event_msg) + "\n")
                    proc.stdin.flush()

                with self._lock:
                    worker_usage_baseline = self._worker_usage_baseline
                stream_iter = AntigravityStream(
                    proc=proc,
                    client=self,
                    model=resolved_model,
                    timeout=effective_timeout,
                    tools=tools,
                    is_worker=True,
                    worker_lock=self._worker_lock,
                    worker_lock_held=worker_acquired,
                    messages=messages_list,
                    usage_baseline=worker_usage_baseline,
                    prompt_chars=len(prompt_payload),
                    delivery_watch=self._delivery_watch if comparable_full_send else None,
                    delivery_sample_key=f"{resolved_model}:{effort}",
                )
                if stream:
                    return stream_iter
                completion: Any = None
                auth_retried = False
                while True:
                    completion = collect_stream_completion(stream_iter)
                    if auth_retried or not _empty_no_result(completion):
                        break
                    if not _stderr_shows_auth_contention(proc):
                        break
                    # Token contention with another agy session on this
                    # account: the worker produced no result and its stderr
                    # carries the token-source failure. Re-sync the token and
                    # retry the turn ONCE with a fresh worker, keeping the
                    # agy conversation so delta resume still works.
                    logger.warning(
                        "Antigravity worker ended with no result amid token-source auth errors; "
                        "re-syncing token and retrying once."
                    )
                    auth_retried = True
                    sync_isolated_token(self._isolated_home)
                    with self._lock:
                        self._terminate_worker_locked(keep_conversation=True)
                    proc = self._get_or_spawn_worker(
                        resolved_model, effort, conversation_id=self._worker_conversation_id
                    )
                    event_msg = {"event": "user", "message": {"content": prompt_payload}}
                    try:
                        if proc.stdin:
                            proc.stdin.write(json.dumps(event_msg) + "\n")
                            proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        break
                    with self._lock:
                        worker_usage_baseline = self._worker_usage_baseline
                    stream_iter = AntigravityStream(
                        proc=proc,
                        client=self,
                        model=resolved_model,
                        timeout=effective_timeout,
                        tools=tools,
                        is_worker=True,
                        worker_lock=self._worker_lock,
                        worker_lock_held=True,
                        messages=messages_list,
                        usage_baseline=worker_usage_baseline,
                        prompt_chars=len(prompt_payload),
                        delivery_watch=self._delivery_watch if comparable_full_send else None,
                        delivery_sample_key=f"{resolved_model}:{effort}",
                    )
                try:
                    return completion
                finally:
                    if self._worker_lock.locked():
                        self._worker_lock.release()
            except Exception:
                self._terminate_worker()
                if self._worker_lock.locked():
                    self._worker_lock.release()
                raise
        else:
            return self._run_oneshot_completion(
                model=resolved_model,
                effort=effort,
                messages=messages_list,
                timeout=effective_timeout,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
            )
