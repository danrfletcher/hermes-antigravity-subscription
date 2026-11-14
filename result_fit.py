"""Keep each Hermes tool result inside what agy shows the model.

agy 1.5.9 shows the model at most 10,240 bytes of one MCP tool result, counting its own
68-byte "Created At / Completed At" header, in UTF-8 bytes (multi-byte text is cut sooner in
characters; JSON escaping doesn't count). A longer result reaches the model as a start+end
preview with ``<truncated N of M bytes from the middle of the output>`` and a pointer to a file
the model can't read here. Measured with a probe server (10,068 bytes in: whole; 10,268: cut;
splitting into several content blocks doesn't help). In a real coder run 67 of 175 results were
cut and the model kept paging into the hidden middles.

So the plugin cuts a long result itself, where the model can carry on: a file read at a line
boundary with the offset to continue from, command output keeping its end (errors and test
summaries are there), anything else keeping its start. ``fit_tool_result`` leaves a result that
already fits exactly as it is.
"""

from __future__ import annotations

import json
import re
from typing import Any

# agy's limit is 10,240 bytes including its 68-byte header; leave room for header drift.
AGY_TOOL_RESULT_MAX_BYTES = 9_800
# Lines a native read_file returns at most per call, so most reads fit without a cut.
READ_FILE_MAX_LINES = 150

_LINE_NO_RE = re.compile(r"^\s*(\d+)\|")


def _nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _head_lines(lines: list[str], budget: int) -> list[str]:
    """The leading lines that fit in ``budget`` bytes (joined with newlines)."""
    out: list[str] = []
    used = 0
    for line in lines:
        cost = _nbytes(line) + (1 if out else 0)
        if used + cost > budget:
            break
        out.append(line)
        used += cost
    return out


def _tail_lines(lines: list[str], budget: int) -> list[str]:
    """The trailing lines that fit in ``budget`` bytes."""
    out: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = _nbytes(line) + (1 if out else 0)
        if used + cost > budget:
            break
        out.append(line)
        used += cost
    out.reverse()
    return out


def _cut_bytes(text: str, budget: int, *, keep_end: bool = False) -> str:
    """At most ``budget`` bytes of ``text`` without splitting a character."""
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text
    piece = raw[-budget:] if keep_end else raw[:budget]
    return piece.decode("utf-8", errors="ignore")


def _fit_read_file(data: dict[str, Any], limit: int) -> str | None:
    content = data.get("content")
    if not isinstance(content, str) or not content:
        return None
    lines = content.split("\n")
    total = data.get("total_lines")
    numbered = [m for m in (_LINE_NO_RE.match(l) for l in lines) if m]
    if not numbered:
        return None

    def note(first: int, last: int, shown: int) -> str:
        of = f" of {total}" if isinstance(total, int) else ""
        more = isinstance(total, int) and last < total
        tail = (f" Continue with read_file offset={last + 1} (about {max(shown, 20)} lines a call fit)."
                if more or shown < len(lines) else "")
        return (f"\n[read_file: lines {first}-{last}{of} shown; the rest of this read was cut to fit the "
                f"~10 KB limit on one tool result.{tail}]")

    first = int(numbered[0].group(1))
    # The longest the note can be: both line numbers at their widest, with the continuation hint.
    widest = max(first, int(numbered[-1].group(1)), total if isinstance(total, int) else 0) + 1
    reserve = _nbytes(note(widest, widest, 999)) + 120  # + the continuation hint
    kept = _head_lines(lines, limit - reserve)
    if not kept:
        kept = [_cut_bytes(lines[0], limit - reserve)]
    last_m = None
    for line in reversed(kept):
        last_m = _LINE_NO_RE.match(line)
        if last_m:
            break
    last = int(last_m.group(1)) if last_m else first
    return "\n".join(kept) + note(first, last, len(kept))


def _fit_command(data: dict[str, Any], limit: int) -> str | None:
    output = data.get("output")
    if not isinstance(output, str):
        return None
    rest = {k: v for k, v in data.items() if k != "output"}
    head = (f"[Command output cut to fit the ~10 KB limit on one tool result: the start was left out, "
            f"the end is below. {json.dumps(rest, ensure_ascii=False)[:300]} To see another part, rerun "
            f"with a filter (grep, head, tail, sed -n) or send the output to a file and read it in "
            f"ranges.]\n")
    budget = limit - _nbytes(head)
    lines = output.split("\n")
    kept = _tail_lines(lines, budget)
    body = "\n".join(kept) if kept else _cut_bytes(lines[-1], budget, keep_end=True)
    return head + body


def fit_tool_result(name: str, text: str, limit: int = AGY_TOOL_RESULT_MAX_BYTES) -> str:
    """``text`` unchanged when it fits in ``limit`` bytes, else a cut that says how to read on."""
    if _nbytes(text) <= limit:
        return text
    data: Any = None
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    if isinstance(data, dict):
        fitted = None
        if name == "read_file":
            fitted = _fit_read_file(data, limit)
        elif "output" in data and ("exit_code" in data or name in ("terminal", "process")):
            fitted = _fit_command(data, limit)
        if fitted is not None and _nbytes(fitted) <= limit:
            return fitted
    size = _nbytes(text)
    note = (f"\n[Result cut to fit the ~10 KB limit on one tool result: the first part of {size} bytes "
            f"is shown. Ask for less at a time (a narrower search, a smaller range, a filter) to see "
            f"the rest.]")
    budget = limit - _nbytes(note)
    lines = text.split("\n")
    kept = _head_lines(lines, budget)
    body = "\n".join(kept) if kept else _cut_bytes(text, budget)
    return body + note


def clamp_read_file_args(name: str, args: Any) -> Any:
    """A native read_file call asks for at most READ_FILE_MAX_LINES lines (Hermes' default is 2,000)."""
    if name != "read_file" or not isinstance(args, dict):
        return args
    limit = args.get("limit")
    try:
        n = int(limit) if limit is not None else None
    except (TypeError, ValueError):
        n = None
    if n is None or n <= 0 or n > READ_FILE_MAX_LINES:
        args = dict(args)
        args["limit"] = READ_FILE_MAX_LINES
    return args
