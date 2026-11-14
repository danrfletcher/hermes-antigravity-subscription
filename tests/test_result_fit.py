"""Tool results kept inside agy's ~10 KB per-result limit (result_fit.py), and the post-compaction quote."""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import result_fit as RF  # noqa: E402
import prompt as P  # noqa: E402
import stream as S  # noqa: E402


def read_result(first, n, total, width=60):
    lines = [f"{i}|" + ("x" * width) for i in range(first, first + n)]
    return json.dumps({"content": "\n".join(lines), "total_lines": total, "file_size": 999999,
                       "truncated": False, "is_binary": False, "is_image": False, "not_found": False})


class FitToolResultTests(unittest.TestCase):
    def test_a_result_that_fits_is_returned_unchanged(self):
        text = read_result(1, 50, 2000)
        self.assertIs(RF.fit_tool_result("read_file", text), text)

    def test_a_long_read_is_cut_at_a_line_with_the_offset_to_continue(self):
        text = read_result(101, 400, 2600)
        out = RF.fit_tool_result("read_file", text)
        self.assertLessEqual(len(out.encode()), RF.AGY_TOOL_RESULT_MAX_BYTES)
        body, note = out.rsplit("\n[read_file:", 1)
        last = int(body.split("\n")[-1].split("|")[0])
        self.assertTrue(body.startswith("101|"))
        self.assertIn(f"lines 101-{last} of 2600 shown", note)
        self.assertIn(f"offset={last + 1}", note)
        # Every line shown is whole: no line was split.
        for line in body.split("\n"):
            self.assertTrue(line.endswith("x" * 60))

    def test_the_limit_counts_utf8_bytes(self):
        text = json.dumps({"content": "\n".join(f"{i}|" + "é" * 60 for i in range(1, 200)),
                           "total_lines": 199}, ensure_ascii=False)
        self.assertLess(len(text), 2 * RF.AGY_TOOL_RESULT_MAX_BYTES)
        out = RF.fit_tool_result("read_file", text)
        self.assertLessEqual(len(out.encode("utf-8")), RF.AGY_TOOL_RESULT_MAX_BYTES)
        self.assertIn("offset=", out)

    def test_command_output_keeps_its_end(self):
        output = "\n".join(f"line {i}" for i in range(5000)) + "\nFAILED: test_x"
        text = json.dumps({"output": output, "exit_code": 1, "error": None})
        out = RF.fit_tool_result("terminal", text)
        self.assertLessEqual(len(out.encode()), RF.AGY_TOOL_RESULT_MAX_BYTES)
        self.assertTrue(out.endswith("FAILED: test_x"))
        self.assertIn('"exit_code": 1', out)
        self.assertIn("the start was left out", out)

    def test_anything_else_keeps_its_start_and_says_it_was_cut(self):
        text = "\n".join(f"match {i}: some/path.py" for i in range(3000))
        out = RF.fit_tool_result("search_files", text)
        self.assertLessEqual(len(out.encode()), RF.AGY_TOOL_RESULT_MAX_BYTES)
        self.assertTrue(out.startswith("match 0: "))
        self.assertIn("Result cut to fit", out)

    def test_one_huge_line_is_still_cut_to_fit(self):
        out = RF.fit_tool_result("search_files", "y" * 50000)
        self.assertLessEqual(len(out.encode()), RF.AGY_TOOL_RESULT_MAX_BYTES)
        out = RF.fit_tool_result("read_file", json.dumps({"content": "1|" + "z" * 50000, "total_lines": 1}))
        self.assertLessEqual(len(out.encode()), RF.AGY_TOOL_RESULT_MAX_BYTES)

    def test_read_file_calls_ask_for_at_most_the_line_cap(self):
        self.assertEqual(RF.clamp_read_file_args("read_file", {"path": "a"})["limit"], RF.READ_FILE_MAX_LINES)
        self.assertEqual(RF.clamp_read_file_args("read_file", {"path": "a", "limit": 2000})["limit"],
                         RF.READ_FILE_MAX_LINES)
        self.assertEqual(RF.clamp_read_file_args("read_file", {"path": "a", "offset": 9, "limit": 40}),
                         {"path": "a", "offset": 9, "limit": 40})
        self.assertEqual(RF.clamp_read_file_args("terminal", {"command": "ls"}), {"command": "ls"})


class AgyCutMarkerTests(unittest.TestCase):
    def test_marker_is_found_in_a_done_tool_step(self):
        step = {"tool_name": "mcp_hermes_read_file", "state": "DONE", "tool_info": {
            "output": "The tool's output was truncated ... <truncated 452 of 10268 bytes from the middle of the output> ..."}}
        self.assertEqual(S.agy_cut_marker(step), "<truncated 452 of 10268 bytes from the middle of the output>")
        self.assertEqual(S.agy_cut_marker({"tool_info": {"output": "fine"}}), "")
        self.assertEqual(S.agy_cut_marker({}), "")


class ObjectiveQuoteAfterCompactionTests(unittest.TestCase):
    def test_quotes_the_restated_request_not_the_summary_header(self):
        brief = "# Brief: coder for PR-2\n" + "Do the work. " * 400
        carrier = ("[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted ... do nothing ...\n"
                   + "summary " * 300
                   + "\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---\n"
                   + "[STILL IN PROGRESS — this is the active request, restated after the compaction boundary "
                   + "because it was not finished yet. Continue it; do not start over.]\n\n" + brief)
        quote = P._objective_quote(carrier, "above")
        self.assertTrue(quote.startswith("[STILL IN PROGRESS"))
        self.assertNotIn("REFERENCE ONLY", quote)
        self.assertIn("# Brief: coder for PR-2", quote)

    def test_a_summary_with_nothing_after_it_is_quoted_as_before(self):
        carrier = "[CONTEXT COMPACTION] x\n--- END OF CONTEXT SUMMARY — respond below ---\n"
        self.assertEqual(P._objective_quote(carrier, "above"), carrier)

    def test_ordinary_requests_are_unchanged(self):
        self.assertEqual(P._objective_quote("fix the bug", "above"), "fix the bug")


if __name__ == "__main__":
    unittest.main()
