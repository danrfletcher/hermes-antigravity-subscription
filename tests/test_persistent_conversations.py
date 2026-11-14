"""Persistent agy conversations: one per Hermes session, delta-only turns, resume after restart."""
import json
import os
import shutil
import stat
import sys
import tempfile
import time
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

    TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]

    def test_bad_turn_is_never_resumed(self):
        # Turn 2 runs away. Its call is delivered, but agy's conversation holds the invented
        # result, so the next request must not continue that conversation as if it were clean.
        os.environ["FAKE_AGY_RUNAWAY"] = "2"
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "first"}]
            a1, _ = self.ask(cl, m, tools=self.TOOLS)
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "second"}]
            _, res = self.ask(cl, m, tools=self.TOOLS)
            msg = res.choices[0].message
            self.assertEqual([c.id for c in msg.tool_calls], ["call_1"])
            self.assertIsNotNone(cl._pending_recovery)
            self.assertTrue(cl._store.get("S1")["bad"].startswith("runaway"))
            m += [{"role": "assistant", "content": None, "tool_calls": [
                      {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "REAL RESULT"}]
            self.ask(cl, m, tools=self.TOOLS)
        finally:
            cl.close()
            os.environ.pop("FAKE_AGY_RUNAWAY", None)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        # The fake agy keeps no real conversation DB, so the rewind is refused -> fallback: a fresh
        # conversation with the whole history.
        self.assertNotEqual(calls[2]["conversation_id"], calls[1]["conversation_id"])
        self.assertIn("first", calls[2]["prompt"])
        self.assertIn("REAL RESULT", calls[2]["prompt"])
        self.assertIsNone(cl._pending_recovery)

    def test_rewound_bad_turn_resumes_the_same_conversation_with_a_delta(self):
        os.environ["FAKE_AGY_RUNAWAY"] = "2"
        cuts = []

        def fake_point(db, *, conversation_id, prompt, agy_version):
            return {"conversation_id": conversation_id, "steps": 2}

        def fake_rewind(gemini_dir, point, *, backup_dir, agy_version):
            cuts.append(point)

        cl = self.client()
        try:
            with patch.object(self.C._rewind, "read_point", fake_point), \
                    patch.object(self.C._rewind, "rewind", fake_rewind):
                m = [self.SYSTEM, {"role": "user", "content": "first"}]
                a1, _ = self.ask(cl, m, tools=self.TOOLS)
                m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "second"}]
                self.ask(cl, m, tools=self.TOOLS)
                m += [{"role": "assistant", "content": None, "tool_calls": [
                          {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                      {"role": "tool", "tool_call_id": "call_1", "content": "REAL RESULT"}]
                self.ask(cl, m, tools=self.TOOLS)
        finally:
            cl.close()
            os.environ.pop("FAKE_AGY_RUNAWAY", None)
        calls = self.calls()
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["conversation_id"], calls[1]["conversation_id"])
        self.assertEqual(calls[2]["conversation_id"], calls[1]["conversation_id"])
        third = calls[2]["prompt"]
        self.assertNotIn("first", third)  # agy still has turn 1
        self.assertIn("second", third)  # the bad turn's request, re-sent
        self.assertIn("REAL RESULT", third)
        self.assertNotIn("made up", third)

    def test_bad_flag_survives_a_restart(self):
        os.environ["FAKE_AGY_RUNAWAY"] = "2"
        try:
            cl = self.client()
            m = [self.SYSTEM, {"role": "user", "content": "first"}]
            a1, _ = self.ask(cl, m, tools=self.TOOLS)
            m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "second"}]
            self.ask(cl, m, tools=self.TOOLS)
            cl.close()
            cl = self.client()  # gateway restart: the in-memory pending recovery is gone
            m += [{"role": "assistant", "content": None, "tool_calls": [
                      {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "REAL RESULT"}]
            self.ask(cl, m, tools=self.TOOLS)
            cl.close()
        finally:
            os.environ.pop("FAKE_AGY_RUNAWAY", None)
        calls = self.calls()
        self.assertNotEqual(calls[2]["conversation_id"], calls[1]["conversation_id"])
        self.assertIn("first", calls[2]["prompt"])

    def test_recoveries_are_capped_per_request(self):
        cl = self.client()
        try:
            m = [self.SYSTEM, {"role": "user", "content": "q"}]
            for _ in range(self.C.MAX_RECOVERIES_PER_REQUEST + 1):
                cl._mark_bad_turn("runaway: test", m)
            with self.assertRaises(self.C.AntigravityRecoveryExhausted):
                self.ask(cl, m)
            self.assertFalse(cl._worker_lock.locked())
        finally:
            cl.close()

    def test_isolated_home_sits_beside_agys_cwd(self):
        cl = self.client()
        try:
            home, cwd = Path(cl._isolated_home), Path(cl._cwd)
            self.assertNotIn(cwd, home.parents)
            self.assertEqual(home.parent, cwd.parent)
            hooks = json.loads((home / ".gemini" / "config" / "hooks.json").read_text())
            self.assertIn("tool_gate.py", json.dumps(hooks))
            settings = json.loads((Path(cl._isolated_gemini_dir) / "settings.json").read_text())
            self.assertIn("command(*)", settings["permissions"]["deny"])
            self.assertNotIn("read_file(*)", settings["permissions"]["deny"])
        finally:
            cl.close()

    def test_old_layout_home_moves_out_of_the_workspace(self):
        old = self.tmp / "data" / "workspace" / "home" / ".gemini" / "antigravity-cli" / "conversations"
        old.mkdir(parents=True)
        (old / "keep.db").write_text("x")
        cl = self.client()
        try:
            self.assertTrue((Path(cl._isolated_gemini_dir) / "conversations" / "keep.db").exists())
            self.assertFalse((self.tmp / "data" / "workspace" / "home").exists())
        finally:
            cl.close()

    def test_fresh_start_never_purges_other_sessions_conversations(self):
        # The store's agy HOME is shared by every Hermes session: upstream's purge of
        # conversations/*.db on a fresh start must not run there.
        cl = self.client()
        try:
            self.ask(cl, [self.SYSTEM, {"role": "user", "content": "session one"}], session="S1")
            other = self.ask(cl, [self.SYSTEM, {"role": "user", "content": "unrelated"}], session="S1")
            self.assertTrue(other[0])
        finally:
            cl.close()
        first, second = self.calls()[:2]
        self.assertNotEqual(first["conversation_id"], second["conversation_id"])
        conv_dir = self.tmp / "data"
        dbs = {p.stem for p in conv_dir.rglob("conversations/*.db")}
        self.assertIn(first["conversation_id"], dbs)

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
        self.assertIn("old 0 ", prompt)  # the first user message (a worker's brief) is pinned
        self.assertNotIn("old 1 ", prompt)
        self.assertIn("oldest messages of this conversation are not shown", prompt)

    def test_fitted_resend_keeps_brief_spec_reads_and_newest_steps_whole(self):
        def step(i, path, size):
            call = {"id": f"c{i}", "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}
            return [{"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": f"RESULT {i} " + "z" * size}]

        m = [self.SYSTEM, {"role": "user", "content": "BRIEF " + "b" * 3000}]
        m += step(0, "briefs/PR-3.spec.md", 6000)
        m += step(1, "ticket-plan.md", 2000)
        for i in range(2, 30):
            m += step(i, f"src/file{i}.py", 1500)
        with patch.dict(os.environ, {"ANTIGRAVITY_MAX_PROMPT_CHARS": "30000"}):
            cl = self.client()
            try:
                prompt, trimmed = cl._fit_full_prompt(m, model="fixture", tools=None, tool_choice=None)
            finally:
                cl.close()
        self.assertTrue(trimmed)
        self.assertLessEqual(len(prompt.encode()), 30000)
        self.assertIn("BRIEF " + "b" * 3000, prompt)
        self.assertIn("RESULT 0 " + "z" * 6000, prompt)  # spec read whole, though old
        self.assertIn("RESULT 1 " + "z" * 2000, prompt)
        self.assertIn("RESULT 29 ", prompt)
        self.assertNotIn("RESULT 2 ", prompt)
        self.assertIn("Files on disk already reflect that work", prompt)
        # Steps are never split: a kept call always comes with its result.
        for i in range(2, 30):
            self.assertEqual(f'"id": "c{i}"' in prompt, f"RESULT {i} " in prompt, i)

    def test_long_request_is_not_restated_whole(self):
        brief = "BRIEF-START " + "q" * 20000
        m = [self.SYSTEM, {"role": "user", "content": brief}]
        m += [{"role": "assistant", "content": None, "tool_calls": [
                  {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "R1"}]
        cl = self.client()
        try:
            prompt, _ = cl._fit_full_prompt(m, model="fixture", tools=None, tool_choice=None)
        finally:
            cl.close()
        self.assertEqual(prompt.count("q" * 20000), 1)
        self.assertGreaterEqual(prompt.count("BRIEF-START"), 2)  # restated by its opening

    def test_tool_delta_restates_a_long_request_by_its_opening(self):
        import prompt as P
        brief = "BRIEF-START " + "q" * 20000
        delta = P._format_delta_prompt([{"role": "tool", "tool_call_id": "c1", "content": "R1"}],
                                       pending_user_text=brief)
        self.assertIn("BRIEF-START", delta)
        self.assertNotIn("q" * 2000, delta)
        self.assertIn("earlier in this conversation", delta)
        short = P._format_delta_prompt([{"role": "tool", "tool_call_id": "c1", "content": "R1"}],
                                       pending_user_text="list files")
        self.assertIn("list files", short)

    def test_full_prompt_limit_counts_utf8_bytes(self):
        # agy cuts one message by bytes: 3-byte characters fill the limit three times faster.
        with patch.dict(os.environ, {"ANTIGRAVITY_MAX_PROMPT_CHARS": "20000"}):
            cl = self.client()
            try:
                m = [self.SYSTEM]
                for i in range(6):
                    m += [{"role": "user", "content": f"old {i} " + "\u6f22" * 2000},
                          {"role": "assistant", "content": f"answer {i}"}]
                m.append({"role": "user", "content": "NEWEST QUESTION"})
                self.ask(cl, m, session="S-cjk")
            finally:
                cl.close()
        prompt = self.calls()[0]["prompt"]
        self.assertLessEqual(len(prompt.encode("utf-8")), 20000)
        self.assertIn("NEWEST QUESTION", prompt)
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


TERMINAL_TOOL = {"type": "function", "function": {
    "name": "terminal", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}


@unittest.skipIf(os.name == "nt", "POSIX fake agy")
class NativeToolRouteTests(unittest.TestCase):
    """Hermes' tools as native agy tools through mcp_bridge.py (the fake agy talks real MCP to it)."""

    SYSTEM = {"role": "system", "content": "You are a coder."}

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
            "AGY_NATIVE_TOOLS": "1",
            "FAKE_AGY_NATIVE_CALLS": "2",
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

    def prompts(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def ask(self, cl, messages, session="S1"):
        res = cl.chat.completions.create(model="gemini-3.8-flash", messages=messages, tools=[TERMINAL_TOOL],
                                         hermes_session_id=session)
        return res.choices[0]

    @staticmethod
    def answer(messages, choice, output):
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                 for tc in choice.message.tool_calls]
        return messages + [{"role": "assistant", "content": None, "tool_calls": calls}] + [
            {"role": "tool", "tool_call_id": c["id"], "content": output} for c in calls]

    def test_call_is_parked_answered_and_the_turn_carries_on(self):
        cl = self.C.AntigravityClient(command=self.agy)
        try:
            m = [self.SYSTEM, {"role": "user", "content": "run two commands"}]
            c1 = self.ask(cl, m)
            self.assertEqual(c1.finish_reason, "tool_calls")
            [tc] = c1.message.tool_calls
            self.assertEqual(tc.function.name, "terminal")
            self.assertEqual(json.loads(tc.function.arguments), {"command": "echo 1"})
            m = self.answer(m, c1, "one")
            c2 = self.ask(cl, m)
            self.assertEqual(c2.finish_reason, "tool_calls")
            self.assertEqual(json.loads(c2.message.tool_calls[0].function.arguments), {"command": "echo 2"})
            m = self.answer(m, c2, "two")
            c3 = self.ask(cl, m)
            self.assertEqual(c3.finish_reason, "stop")
            self.assertEqual(c3.message.content, "ACK 1: two")
            # One agy prompt for the whole turn: the results went through the bridge.
            prompts = self.prompts()
            self.assertEqual(len(prompts), 1)
            self.assertNotIn("<tool_call>", prompts[0]["prompt"])
            self.assertIn("mcp_hermes_", prompts[0]["prompt"])
            # The next user message continues the same conversation with only the delta.
            m = m + [{"role": "assistant", "content": c3.message.content},
                     {"role": "user", "content": "and now?"}]
            c4 = self.ask(cl, m)
            self.assertEqual(c4.finish_reason, "tool_calls")
            prompts = self.prompts()
            self.assertEqual(len(prompts), 2)
            self.assertEqual(prompts[0]["conversation_id"], prompts[1]["conversation_id"])
            self.assertIn("and now?", prompts[1]["prompt"])
            self.assertNotIn("run two commands", prompts[1]["prompt"])
        finally:
            cl.close()

    def test_parked_completions_report_the_context_size_never_zero(self):
        # Hermes reads prompt_tokens=0 as "no usage" and compresses on a rough estimate that runs high.
        cl = self.C.AntigravityClient(command=self.agy)
        try:
            m = [self.SYSTEM, {"role": "user", "content": "run two commands"}]
            r1 = cl.chat.completions.create(model="gemini-3.8-flash", messages=m, tools=[TERMINAL_TOOL],
                                            hermes_session_id="S1")
            self.assertGreater(r1.usage.prompt_tokens, 0)
            m = self.answer(m, r1.choices[0], "x" * 8000)
            r2 = cl.chat.completions.create(model="gemini-3.8-flash", messages=m, tools=[TERMINAL_TOOL],
                                            hermes_session_id="S1")
            self.assertEqual(r2.choices[0].finish_reason, "tool_calls")
            self.assertGreaterEqual(r2.usage.prompt_tokens, r1.usage.prompt_tokens + 2000)
        finally:
            cl.close()

    def test_a_long_result_reaches_agy_cut_to_fit_with_its_end_kept(self):
        # agy shows the model at most ~10 KB of one tool result (start+end, middle hidden): the plugin
        # cuts it first, where the model can carry on.
        import result_fit as RF
        cl = self.C.AntigravityClient(command=self.agy)
        try:
            m = [self.SYSTEM, {"role": "user", "content": "run two commands"}]
            c1 = self.ask(cl, m)
            big = json.dumps({"output": "\n".join(f"line {i}" for i in range(6000)) + "\nTAIL-MARK",
                              "exit_code": 0})
            m = self.answer(m, c1, big)
            c2 = self.ask(cl, m)
            self.assertEqual(c2.finish_reason, "tool_calls")
            results = sorted((self.tmp / "data").rglob("results/*.json"))
            self.assertTrue(results)
            sent = json.loads(results[0].read_text())["text"]
            self.assertLessEqual(len(sent.encode()), RF.AGY_TOOL_RESULT_MAX_BYTES)
            self.assertTrue(sent.endswith("TAIL-MARK"))
        finally:
            cl.close()

    def test_a_request_that_does_not_answer_the_parked_call_recovers_the_turn(self):
        cl = self.C.AntigravityClient(command=self.agy)
        try:
            m = [self.SYSTEM, {"role": "user", "content": "run two commands"}]
            c1 = self.ask(cl, m)
            self.assertEqual(c1.finish_reason, "tool_calls")
            # The user interrupts: Hermes sends a new message instead of the tool result.
            m2 = m + [{"role": "user", "content": "stop that, say hi"}]
            c2 = self.ask(cl, m2)
            self.assertEqual(c2.finish_reason, "tool_calls")
            prompts = self.prompts()
            self.assertEqual(len(prompts), 2)
            self.assertIn("stop that, say hi", prompts[1]["prompt"])
        finally:
            cl.close()

    def test_kill_switch_keeps_the_text_route(self):
        with patch.dict(os.environ, {"AGY_NATIVE_TOOLS": "0"}):
            cl = self.C.AntigravityClient(command=self.agy)
            try:
                c1 = self.ask(cl, [self.SYSTEM, {"role": "user", "content": "hello"}])
                self.assertEqual(c1.finish_reason, "stop")
            finally:
                cl.close()
        [p] = self.prompts()
        self.assertIn("<tool_call>", p["prompt"])

    def test_bridge_never_loaded_falls_back_to_text_calls(self):
        with patch.dict(os.environ, {"FAKE_AGY_NO_MCP": "1"}), patch.object(self.C, "_BRIDGE_LIST_WAIT_SECONDS", 1.0):
            cl = self.C.AntigravityClient(command=self.agy)
            try:
                c1 = self.ask(cl, [self.SYSTEM, {"role": "user", "content": "hello"}])
                self.assertEqual(c1.finish_reason, "stop")
                self.assertTrue(cl._native_failed)
            finally:
                cl.close()
        [p] = self.prompts()
        self.assertIn("<tool_call>", p["prompt"])

    def test_gateway_restart_mid_call_rewinds_instead_of_resuming_a_dangling_call(self):
        cl = self.C.AntigravityClient(command=self.agy)
        m = [self.SYSTEM, {"role": "user", "content": "run two commands"}]
        try:
            c1 = self.ask(cl, m)
        finally:
            cl.close()
        cl2 = self.C.AntigravityClient(command=self.agy)
        try:
            c2 = self.ask(cl2, self.answer(m, c1, "one"))
            self.assertEqual(c2.finish_reason, "tool_calls")
        finally:
            cl2.close()
        first, second = self.prompts()
        # Not a delta on top of the unanswered call: the result is re-sent with its call.
        self.assertIn("echo 1", second["prompt"])
        self.assertIn("one", second["prompt"])


class NativeGateAndPromptTests(unittest.TestCase):
    def test_gate_allows_hermes_tools_only(self):
        import tool_gate as G
        allow = lambda name, args=None: G.decide({"toolCall": {"name": name, "args": args or {}}}, "", "/nonexistent")
        self.assertEqual(allow("mcp_hermes_terminal")["decision"], "auto_approve")
        self.assertEqual(allow("call_mcp_tool", {"ServerName": "hermes", "ToolName": "terminal"})["decision"],
                         "auto_approve")
        self.assertEqual(allow("call_mcp_tool", {"ServerName": "other"})["decision"], "deny")
        self.assertEqual(allow("run_command", {"CommandLine": "ls"})["decision"], "deny")
        self.assertEqual(allow("mcp_hermes_")["decision"], "deny")

    def test_native_prompt_has_no_text_tool_format(self):
        import prompt as P
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}]
        native = P._format_messages_as_prompt(msgs, tools=[TERMINAL_TOOL], native=True)
        self.assertNotIn("<tool_call>", native)
        self.assertIn("[Called tool terminal (c1)", native)
        text = P._format_messages_as_prompt(msgs, tools=[TERMINAL_TOOL])
        self.assertIn("<tool_call>", text)
        self.assertNotIn("<tool_call>", P._format_delta_prompt(msgs[1:], native=True))

    def test_mcp_config_lists_every_tool_eager_and_keeps_earlier_names(self):
        import process as PR
        home = Path(tempfile.mkdtemp())
        try:
            PR.write_mcp_config(home, ["terminal"])
            PR.write_mcp_config(home, ["read_file"])
            cfg = json.loads((home / ".gemini" / "config" / "mcp_config.json").read_text())["mcpServers"]["hermes"]
            self.assertEqual(cfg["tools"], {"read_file": {"eager": True}, "terminal": {"eager": True}})
            self.assertEqual(cfg["timeoutSeconds"], 3600)
            self.assertTrue(cfg["args"][0].endswith("mcp_bridge.py"))
        finally:
            shutil.rmtree(home, ignore_errors=True)


class BridgeNameTests(unittest.TestCase):
    def test_long_and_unsafe_names_get_short_unique_aliases(self):
        import process as PR
        long = "mcp_on_demand_build_pipeline_steps_review_remediation_approved"
        tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
                 for n in ("terminal", long, long + "_x", "odd.name")]
        specs = PR.bridge_tool_specs(tools)
        self.assertEqual(specs[0]["name"], "terminal")
        aliases = [s["name"] for s in specs]
        self.assertEqual(len(set(aliases)), 4)
        for spec in specs:
            self.assertLessEqual(len("mcp_hermes_" + spec["name"]), 64)
            self.assertRegex(spec["name"], r"^[A-Za-z0-9_-]+$")
        self.assertEqual([s["hermes_name"] for s in specs], ["terminal", long, long + "_x", "odd.name"])
        self.assertIn(long, specs[1]["description"])

    def test_bridge_parks_the_hermes_name_for_an_alias(self):
        import subprocess
        import process as PR
        long = "mcp_on_demand_build_pipeline_steps_review_remediation_approved"
        d = Path(tempfile.mkdtemp())
        try:
            spec = PR.bridge_tool_specs([{"type": "function", "function": {"name": long, "parameters": {"type": "object"}}}])
            (d / "tools.json").write_text(json.dumps(spec))
            p = subprocess.Popen([sys.executable, str(PR.MCP_BRIDGE_SCRIPT)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True, env={**os.environ, "HERMES_AGY_BRIDGE_DIR": str(d)})
            p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                      "params": {"name": spec[0]["name"], "arguments": {"a": 1}}}) + "\n")
            p.stdin.flush()
            for _ in range(100):
                if (d / "calls" / "000001.json").exists():
                    break
                time.sleep(0.05)
            call = json.loads((d / "calls" / "000001.json").read_text())
            self.assertEqual(call["name"], long)
            (d / "results").mkdir(exist_ok=True)
            (d / "results" / "000001.json").write_text(json.dumps({"text": "done", "is_error": False}))
            reply = json.loads(p.stdout.readline())
            self.assertEqual(reply["result"]["content"][0]["text"], "done")
            p.stdin.close()
            p.wait(5)
        finally:
            shutil.rmtree(d, ignore_errors=True)
