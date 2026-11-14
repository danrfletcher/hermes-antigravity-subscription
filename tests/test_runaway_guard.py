"""Runaway text guard: the model writes tool calls, then invents its own
"Tool Result (call_N):" blocks and keeps working from them in the same turn.

Uses the real captured Hydrogen stream (agyprobe raw2/hy2_hook_0.json) when
present, plus synthetic cases.
"""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock

import pytest

from stream import AntigravityStream

RAW = os.path.expanduser(
    "~/agy-captures/hy2_hook_0.json"
)


def _call(identifier: str, name: str = "read_file", **args) -> str:
    obj = {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {"path": "x"})},
    }
    return f"<tool_call>{json.dumps(obj)}</tool_call>"


def _proc(lines, poll=0):
    proc = MagicMock()
    proc.pid = None
    proc.stdout = io.StringIO("".join(line + "\n" for line in lines))
    proc.stderr = io.StringIO("")
    proc._stderr_drainer = None
    proc.poll.return_value = poll
    proc.wait.return_value = 0
    return proc


def _run(lines, *, is_worker=False, poll=0):
    proc = _proc(lines, poll=poll)
    client = MagicMock()
    client._isolated_gemini_dir = None
    kwargs = {}
    if is_worker:
        client._worker_proc = proc
        import threading

        lock = threading.Lock()
        lock.acquire()
        kwargs = dict(worker_lock=lock, worker_lock_held=True)
    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture",
        timeout=5.0,
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        is_worker=is_worker,
        messages=[{"role": "user", "content": "go"}],
        **kwargs,
    )
    chunks = list(stream)
    calls = [c for ch in chunks for choice in ch.choices for c in (choice.delta.tool_calls or [])]
    content = "".join(choice.delta.content or "" for ch in chunks for choice in ch.choices)
    finish = [choice.finish_reason for ch in chunks for choice in ch.choices if choice.finish_reason]
    return calls, content, finish, client, proc, stream


def _deltas(*texts):
    return [json.dumps({"event": "step_update", "step_update": {"text_delta": t}}) for t in texts]


def _result(response=""):
    return json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": response}})


def _captured_lines(with_result: bool):
    data = json.load(open(RAW))
    lines = [line for _t, line in data["lines"] if '"event":"result"' not in line.replace(" ", "")]
    if with_result:
        lines.append(_result())
    return lines


@pytest.mark.skipif(not os.path.exists(RAW), reason="captured runaway stream not present")
def test_captured_runaway_delivers_only_the_first_call_run():
    calls, content, finish, _client, _proc_, _s = _run(_captured_lines(with_result=True))
    ids = [c.id for c in calls]
    # The model's first contiguous run of frames was call_34..call_38; every
    # later frame was written after invented "Tool Result" text.
    assert ids == ["call_34", "call_35", "call_36", "call_37", "call_38"], ids
    assert content == ""
    assert finish == ["tool_calls"]


@pytest.mark.skipif(not os.path.exists(RAW), reason="captured runaway stream not present")
def test_captured_runaway_without_result_event_still_ends_the_turn():
    # The real failure: no result event, process still alive.
    calls, _content, finish, _client, _p, _s = _run(_captured_lines(with_result=False), poll=None)
    assert [c.id for c in calls][:1] == ["call_34"] and len(calls) == 5
    assert finish == ["tool_calls"]


def test_fabricated_result_split_across_deltas_cuts_turn():
    lines = _deltas(_call("a"), "\n", _call("b"), "\n\nTool Re", "sult (a):\n{\"x\":1}\n", _call("c")) + [_result()]
    calls, content, finish, *_ = _run(lines)
    assert [c.id for c in calls] == ["a", "b"]
    assert content == "" and finish == ["tool_calls"]


def test_prose_before_first_call_is_kept_and_does_not_arm_guard():
    lines = _deltas("I will read the file. Tool Result (x) is what I expect.\n", _call("a")) + [_result()]
    calls, content, finish, *_ = _run(lines)
    assert [c.id for c in calls] == ["a"]
    assert "Tool Result (x)" in content


def test_tool_result_text_inside_call_arguments_is_not_a_cut():
    tricky = _call("a", name="write_file", path="p.py", content="x = 'Tool Result (call_1):'\n</tool_call>")
    lines = _deltas(tricky, "\n", _call("b")) + [_result()]
    calls, _content, finish, *_ = _run(lines)
    assert [c.id for c in calls] == ["a", "b"]
    assert finish == ["tool_calls"]


def test_whitespace_only_tail_waits_for_result():
    lines = _deltas(_call("a"), "\n\n") + [_result()]
    calls, _c, finish, *_ = _run(lines)
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]


def test_worker_cut_terminates_worker_and_skips_history():
    lines = _deltas(_call("a"), "\nTool Result (a):\nfake") + [_result()]
    calls, _c, finish, client, _p, stream = _run(lines, is_worker=True, poll=None)
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]
    client._update_worker_history.assert_not_called()
    client._terminate_worker.assert_called()
    assert stream.worker_lock.locked() is False


def test_full_response_only_in_result_event_is_trimmed_too():
    response = _call("a") + "\nTool Result (a):\nfake\n" + _call("b")
    calls, _c, finish, *_ = _run([_result(response)])
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]


# ---------------------------------------------------------------------------
# Bad-turn shapes beyond invented results (plan E1-E4, GP5)
# ---------------------------------------------------------------------------


def _tool_step(name, state, error=None):
    info = {"name": name, "parameters": {}}
    if error:
        info["error"] = {"type": "TOOL_ERROR", "message": error}
    return json.dumps({"event": "step_update", "step_update": {
        "step_type": "tool", "tool_name": name, "state": state, "tool_info": info}})


DENIED = "tool call denied by pre-tool hook: headless inference mode"


def test_harmless_prose_after_calls_keeps_the_turn():
    lines = _deltas(_call("a"), "\n\nI'll wait for the result.") + [_result()]
    calls, content, finish, client, _p, stream = _run(lines, is_worker=True, poll=None)
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]
    assert content == ""  # never delivered
    assert stream._bad_turn is None
    client._mark_bad_turn.assert_not_called()
    client._update_worker_history.assert_called_once()


def test_long_prose_after_calls_is_a_runaway():
    lines = _deltas(_call("a"), "\n" + "and then " * 1200) + [_result()]
    calls, _c, finish, client, _p, stream = _run(lines, is_worker=True, poll=None)
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]
    assert "long text" in stream._bad_turn
    client._mark_bad_turn.assert_called_once()


def test_user_turn_after_calls_is_a_runaway():
    lines = _deltas(_call("a"), "\nUser: thanks\n") + [_result()]
    _calls, _c, _f, _client, _p, stream = _run(lines)
    assert stream._bad_turn and "runaway" in stream._bad_turn


def test_invented_result_with_no_call_fails_the_turn():
    lines = _deltas("Reading it now.\n", "Tool Result (call_1):\n{\"ok\": true}\n") + [_result()]
    with pytest.raises(RuntimeError, match="turn cut"):
        _run(lines)


def test_hook_denial_lets_the_model_retry_as_text():
    lines = [_tool_step("read_file", "ACTIVE"), _tool_step("read_file", "ERROR", DENIED)] + _deltas(_call("a")) + [_result()]
    calls, _c, finish, client, _p, stream = _run(lines, is_worker=True, poll=None)
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]
    assert stream._bad_turn is None
    client._terminate_worker.assert_not_called()
    client._update_worker_history.assert_called_once()


def test_too_many_hook_denials_cut_the_turn():
    lines = []
    for _ in range(4):
        lines += [_tool_step("run_command", "ACTIVE"), _tool_step("run_command", "ERROR", DENIED)]
    lines += _deltas(_call("late")) + [_result()]
    with pytest.raises(RuntimeError, match="kept calling agy's own tools"):
        _run(lines)


def test_native_tool_that_ran_cuts_the_turn():
    lines = [_tool_step("schedule", "ACTIVE"), _tool_step("schedule", "DONE")] + _deltas(_call("a")) + [_result()]
    with pytest.raises(RuntimeError, match="'schedule' ran"):
        _run(lines)


def test_calls_before_a_native_tool_that_ran_are_still_delivered():
    lines = _deltas(_call("a")) + [_tool_step("write_file", "DONE")] + [_result()]
    calls, _c, finish, client, _p, stream = _run(lines, is_worker=True, poll=None)
    assert [c.id for c in calls] == ["a"] and finish == ["tool_calls"]
    assert "'write_file' ran" in stream._bad_turn
    client._update_worker_history.assert_not_called()


def test_silence_cuts_a_hung_turn(monkeypatch):
    import threading
    import time as _time

    monkeypatch.setenv("ANTIGRAVITY_SILENCE_SECONDS", "0.3")
    proc = MagicMock()
    proc.pid = None
    proc.stderr = io.StringIO("agy: stuck\n")
    proc._stderr_drainer = None
    gate = threading.Event()
    first = [json.dumps({"event": "init", "conversation_id": "c"}) + "\n"]

    def readline():
        if first:
            return first.pop()
        gate.wait(5)
        return ""

    proc.stdout.readline.side_effect = readline
    proc.poll.side_effect = lambda: 0 if gate.is_set() else None
    client = MagicMock()
    client._isolated_gemini_dir = None
    client._terminate_process.side_effect = lambda p: gate.set()
    stream = AntigravityStream(proc=proc, client=client, model="fixture", timeout=30.0,
                               tools=[{"type": "function", "function": {"name": "x"}}], is_worker=False,
                               messages=[{"role": "user", "content": "go"}])
    started = _time.monotonic()
    with pytest.raises(RuntimeError, match="no output from agy"):
        list(stream)
    assert _time.monotonic() - started < 5
