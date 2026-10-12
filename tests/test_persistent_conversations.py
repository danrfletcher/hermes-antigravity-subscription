"""Persistent agy conversations: one per Hermes session, delta-only turns, resume after restart."""
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN))

import session_store as ss  # noqa: E402

FAKE = Path(__file__).with_name("fake_agy_persist.py")


class ReconcileTests(unittest.TestCase):
    def test_plain_continuation(self):
        self.assertEqual(ss.reconcile(list("abc"), list("abcde")), ("continue", 3, 3))

    def test_nothing_known_is_fresh(self):
        self.assertEqual(ss.reconcile([], list("ab")), ("fresh", 0, 0))

    # ``known`` is the request agy last answered, so it always ends with a user/tool message.
    def test_hermes_compaction_anchors_on_shared_tail(self):
        known = ["s", "u1", "a1", "u2", "a2", "u3"]
        incoming = ["s", "SUMMARY", "u3", "a3", "u4"]
        mode, start, _ = ss.reconcile(known, incoming)
        self.assertEqual((mode, start), ("continue", 3))

    def test_anchor_uses_newest_occurrence_of_repeated_message(self):
        known = ["s", "ok", "a1", "u2", "a2", "ok"]
        incoming = ["s", "SUMMARY", "a2", "ok", "a3", "ok"]
        mode, start, _ = ss.reconcile(known, incoming)
        self.assertEqual((mode, start), ("continue", 4))

    def test_edit_of_last_user_message_is_rewind(self):
        known = ["s", "u1", "a1", "u2", "a2"]
        incoming = ["s", "u1", "a1", "u2-edited"]
        self.assertEqual(ss.reconcile(known, incoming), ("rewind", 3, 3))

    def test_retry_is_rewind_to_strict_prefix(self):
        known = ["s", "u1", "a1"]
        incoming = ["s", "u1"]
        self.assertEqual(ss.reconcile(known, incoming), ("rewind", 2, 2))

    def test_changed_system_prompt_still_matches_history(self):
        known = ["s1", "u1", "a1", "u2"]
        incoming = ["s2", "u1", "a1", "u2", "a2", "u3"]
        mode, start, _ = ss.reconcile(known, incoming)
        self.assertEqual((mode, start), ("continue", 4))

    def test_unrelated_history_is_fresh(self):
        self.assertEqual(ss.reconcile(["s", "x", "y"], ["t", "p", "q"]), ("fresh", 0, 0))

    def test_hash_ignores_volatile_fields(self):
        a = {"role": "assistant", "content": "hi", "reasoning": "x", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}
        b = {"role": "assistant", "content": "hi", "tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}}], "display_metadata": {}}
        self.assertEqual(ss.message_hash(a), ss.message_hash(b))


@unittest.skipIf(os.name == "nt", "POSIX fake agy")
class PersistentClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "agy.log"
        agy = self.tmp / "agy"
        agy.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
        agy.chmod(agy.stat().st_mode | stat.S_IEXEC)
        self.env = patch.dict(os.environ, {
            "ANTIGRAVITY_PERSIST_CONVERSATIONS": "1",
            "ANTIGRAVITY_PLUGIN_DATA_DIR": str(self.tmp / "data"),
            "FAKE_AGY_LOG": str(self.log),
        })
        self.env.start()
        self.auth = patch("client.is_authenticated", return_value=True)
        self.auth.start()
        import client as C
        self.C = C
        self.agy = str(agy)

    def tearDown(self):
        self.auth.stop()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def client(self):
        return self.C.AntigravityClient(command=self.agy)

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def ask(self, cl, messages, session="S1", **kw):
        res = cl.chat.completions.create(model="gemini-3.8-flash", messages=messages,
                                         hermes_session_id=session, **kw)
        return res.choices[0].message.content, res

    SYSTEM = {"role": "system", "content": "You are Vector."}

    def test_second_turn_sends_only_the_new_message(self):
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "first question"}]
            a1, _ = self.ask(cl, m)
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "second question"}]
            self.ask(cl, m)
        finally:
            cl.close()
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["conversation_id"], calls[1]["conversation_id"])
        self.assertIn("first question", calls[0]["prompt"])
        self.assertIn("You are Vector.", calls[0]["prompt"])
        self.assertIn("second question", calls[1]["prompt"])
        self.assertNotIn("first question", calls[1]["prompt"])
        self.assertNotIn("You are Vector.", calls[1]["prompt"])

    def test_restart_resumes_same_conversation_with_only_the_delta(self):
        m = [self.SYSTEM, {"role": "user", "content": "remember PELICAN"}]
        cl = self.client()
        try:
            a1, _ = self.ask(cl, m)
        finally:
            cl.close()  # Hermes / desktop restart
        cl2 = self.client()
        try:
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "what was it?"}]
            a2, _ = self.ask(cl2, m)
        finally:
            cl2.close()
        first, second = self.calls()
        self.assertIn("--conversation", second["argv"])
        self.assertEqual(first["conversation_id"], second["conversation_id"])
        self.assertNotIn("remember PELICAN", second["prompt"])
        self.assertIn("what was it?", second["prompt"])
        self.assertEqual(a2, "ACK 2")  # agy's own record carried both prompts

    def test_tool_round_trip_is_delta_only(self):
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "list files"}]
            self.ask(cl, m)
            m += [{"role": "assistant", "content": None, "tool_calls": [
                      {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "MARMALADE.txt"}]
            self.ask(cl, m)
        finally:
            cl.close()
        second = self.calls()[1]["prompt"]
        self.assertIn("MARMALADE.txt", second)
        # Only the tool result is new; the request it serves is restated once (upstream's anchor),
        # never the earlier transcript.
        self.assertEqual(second.count("list files"), 1)
        self.assertIn("USER REQUEST THIS TOOL LOOP IS SERVING", second)

    def test_edit_rewinds_in_place_without_resending_history(self):
        cl = self.client()
        try:
            base = [self.SYSTEM, {"role": "user", "content": "Q1 original"}]
            a1, _ = self.ask(cl, base)
            m = base + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Q2 original"}]
            self.ask(cl, m)
            edited = base + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Q2 edited"}]
            self.ask(cl, edited)
        finally:
            cl.close()
        third = self.calls()[2]["prompt"]
        # Small history fits in one prompt: an exact fresh replay is preferred over "disregard".
        self.assertIn("Q2 edited", third)
        self.assertNotIn("Q2 original", third)

    def test_rewind_of_oversized_history_keeps_conversation(self):
        with patch.dict(os.environ, {"ANTIGRAVITY_MAX_PROMPT_CHARS": "20000"}):
            cl = self.client()
            try:
                big = "x" * 12000
                base = [self.SYSTEM, {"role": "user", "content": "Q1 " + big}]
                a1, _ = self.ask(cl, base)
                m = base + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Q2 " + big}]
                a2, _ = self.ask(cl, m)
                edited = base + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Q2 edited " + big}]
                self.ask(cl, edited)
            finally:
                cl.close()
        calls = self.calls()
        self.assertEqual(calls[2]["conversation_id"], calls[0]["conversation_id"])
        self.assertIn("[Conversation edited]", calls[2]["prompt"])
        self.assertIn("Q2 edited", calls[2]["prompt"])
        self.assertLess(len(calls[2]["prompt"]), 20000)

    def test_fresh_full_prompt_never_exceeds_limit_and_keeps_newest(self):
        with patch.dict(os.environ, {"ANTIGRAVITY_MAX_PROMPT_CHARS": "20000"}):
            cl = self.client()
            try:
                m = [self.SYSTEM]
                for i in range(10):
                    m += [{"role": "user", "content": f"old {i} " + "y" * 3000},
                          {"role": "assistant", "content": f"answer {i}"}]
                m.append({"role": "user", "content": "NEWEST QUESTION"})
                self.ask(cl, m, session="S-big")
            finally:
                cl.close()
        prompt = self.calls()[0]["prompt"]
        self.assertLessEqual(len(prompt), 20000)
        self.assertIn("NEWEST QUESTION", prompt)
        self.assertIn("You are Vector.", prompt)
        self.assertNotIn("old 0 ", prompt)
        self.assertIn("oldest messages of this conversation are not shown", prompt)

    def test_second_client_for_same_session_does_not_touch_conversation(self):
        owner = self.client()
        other = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "main chat"}]
            a1, _ = self.ask(owner, m)
            review = m + [{"role": "assistant", "content": a1}, {"role": "user", "content": "background review"}]
            self.ask(other, review)
            m2 = m + [{"role": "assistant", "content": a1}, {"role": "user", "content": "next"}]
            self.ask(owner, m2)
        finally:
            other.close()
            owner.close()
        main1, rev, main2 = self.calls()
        self.assertNotEqual(rev["conversation_id"], main1["conversation_id"])
        self.assertEqual(main2["conversation_id"], main1["conversation_id"])
        self.assertNotIn("main chat", main2["prompt"])

    def test_usage_reports_full_context_not_the_delta(self):
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "z" * 40000}]
            a1, r1 = self.ask(cl, m)
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "short"}]
            _, r2 = self.ask(cl, m)
        finally:
            cl.close()
        self.assertGreater(r2.usage.prompt_tokens, 10000)
        self.assertGreaterEqual(r2.usage.prompt_tokens, r1.usage.prompt_tokens)

    def test_agy_checkpoint_restates_instructions_next_turn(self):
        with patch.dict(os.environ, {"FAKE_AGY_CHECKPOINT": "2"}):
            cl = self.client()
            try:
                m = [self.SYSTEM, {"role": "user", "content": "one"}]
                a1, _ = self.ask(cl, m)
                m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "two"}]
                a2, _ = self.ask(cl, m)
                m += [{"role": "assistant", "content": a2}, {"role": "user", "content": "three"}]
                self.ask(cl, m)
            finally:
                cl.close()
        calls = self.calls()
        self.assertNotIn("You are Vector.", calls[1]["prompt"])
        self.assertIn("STANDING INSTRUCTIONS", calls[2]["prompt"])
        self.assertIn("You are Vector.", calls[2]["prompt"])

    def test_hermes_repairing_past_tool_arguments_is_not_an_edit(self):
        # Seen live (session 20261012_135106_bbbf0e): Hermes rewrote an earlier call's invalid JSON
        # arguments, the plugin read it as an edit and resent the history with skill text cut to
        # 300 chars, so the model lost the question format the skill asks for.
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "go"}]
            self.ask(cl, m)
            bad = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_x", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"ls\n"}}]}
            m += [bad, {"role": "tool", "tool_call_id": "call_x", "content": "Error: Invalid JSON arguments"}]
            a2, _ = self.ask(cl, m)
            repaired = dict(bad, tool_calls=[{"id": "call_x", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
            m2 = m[:2] + [repaired, m[3], {"role": "assistant", "content": a2}, {"role": "user", "content": "next"}]
            self.ask(cl, m2)
        finally:
            cl.close()
        calls = self.calls()
        self.assertEqual(len({c["conversation_id"] for c in calls}), 1)
        self.assertNotIn("You are Vector.", calls[2]["prompt"])  # delta, not a full resend
        self.assertIn("next", calls[2]["prompt"])

    def test_full_resend_keeps_older_tool_results_intact_when_they_fit(self):
        cl = self.client()
        try:
            skill = "SKILL FORMAT: use the question emoji. " + "rule " * 400 + "END-OF-SKILL"
            m = [self.SYSTEM, {"role": "user", "content": "go"}]
            for i in range(10):
                m += [{"role": "assistant", "content": None, "tool_calls": [
                          {"id": f"c{i}", "type": "function", "function": {"name": "skill_view", "arguments": "{}"}}]},
                      {"role": "tool", "tool_call_id": f"c{i}", "content": skill if i == 0 else f"result {i}"}]
            m.append({"role": "user", "content": "now"})
            self.ask(cl, m, session="S-skill")
        finally:
            cl.close()
        self.assertIn("END-OF-SKILL", self.calls()[0]["prompt"])

    def test_new_tools_are_restated(self):
        tool = lambda n: {"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object", "properties": {}}}}
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "one"}]
            a1, _ = self.ask(cl, m, tools=[tool("alpha_tool")])
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "two"}]
            self.ask(cl, m, tools=[tool("alpha_tool"), tool("beta_tool")])
        finally:
            cl.close()
        self.assertIn("beta_tool", self.calls()[1]["prompt"])

    def test_without_session_id_behaves_like_upstream(self):
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "first"}]
            a1, _ = cl.chat.completions.create(model="gemini-3.8-flash", messages=m).choices[0].message.content, None
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "second"}]
            cl.chat.completions.create(model="gemini-3.8-flash", messages=m)
        finally:
            cl.close()
        calls = self.calls()
        self.assertEqual(calls[0]["conversation_id"], calls[1]["conversation_id"])
        self.assertNotIn("first", calls[1]["prompt"])
        self.assertEqual(json.loads((self.tmp / "data" / "sessions.json").read_text()) if (self.tmp / "data" / "sessions.json").exists() else {}, {})


if __name__ == "__main__":
    unittest.main()
