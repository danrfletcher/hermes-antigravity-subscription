"""The PreToolUse hook that gates agy's own tools (tool_gate.py) and its installer."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN))

import process  # noqa: E402
import tool_gate  # noqa: E402


class ToolGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        self.gemini = self.home / ".gemini" / "antigravity-cli"
        self.gemini.mkdir(parents=True)
        self.settings = self.gemini / "settings.json"
        self.media = self.root / "uploads" / "cat.png"
        self.media.parent.mkdir()
        self.media.write_bytes(b"png")
        process.write_media_read_permissions(self.gemini, {str(self.media)})

    def tearDown(self):
        self._tmp.cleanup()

    def decide(self, name, **args):
        payload = {"hookEventName": "PreToolUse", "toolCall": {"name": name, "args": args}}
        return tool_gate.decide(payload, str(self.settings), str(self.home))["decision"]

    def test_every_native_tool_is_denied(self):
        for name in ("run_command", "write_file", "edit_file", "schedule", "invoke_subagent",
                     "search_web", "manage_task", "some_future_tool"):
            self.assertEqual(self.decide(name, Path=str(self.media)), "deny", name)

    def test_granted_media_read_is_allowed(self):
        self.assertEqual(self.decide("read_file", Path=str(self.media)), "allow")
        self.assertEqual(self.decide("view_file", AbsolutePath=str(self.media)), "allow")

    def test_read_of_an_ungranted_path_is_denied(self):
        other = self.media.with_name("dog.png")
        other.write_bytes(b"png")
        self.assertEqual(self.decide("read_file", Path=str(other)), "deny")

    def test_granted_non_media_path_is_denied(self):
        source = self.root / "notes.txt"
        source.write_text("x")
        process.write_media_read_permissions(self.gemini, {str(source)})
        self.assertEqual(self.decide("read_file", Path=str(source)), "deny")

    def test_token_inside_isolated_home_is_never_readable(self):
        sneaky = self.gemini / "token.png"  # a media name pointing into the isolated HOME
        sneaky.write_bytes(b"x")
        process.write_media_read_permissions(self.gemini, {str(sneaky)})
        self.assertEqual(self.decide("read_file", Path=str(sneaky)), "deny")
        link = self.root / "uploads" / "link.png"
        os.symlink(sneaky, link)
        process.write_media_read_permissions(self.gemini, {str(link)})
        self.assertEqual(self.decide("read_file", Path=str(link)), "deny")

    def test_read_without_a_path_is_denied(self):
        self.assertEqual(self.decide("read_file"), "deny")

    def test_installed_hook_command_runs_with_a_stripped_environment(self):
        hooks_path = process.install_tool_gate(self.home, self.gemini)
        command = json.loads(hooks_path.read_text())["hermes-native-tool-gate"]["PreToolUse"][0]["hooks"][0]["command"]

        def run(name, **args):
            payload = json.dumps({"toolCall": {"name": name, "args": args}})
            out = subprocess.run(shlex.split(command), input=payload, capture_output=True, text=True,
                                 env={}, timeout=30)
            return json.loads(out.stdout)

        denied = run("run_command", CommandLine="echo hi")
        self.assertEqual(denied["decision"], "deny")
        self.assertIn("<tool_call>", denied["reason"])
        self.assertEqual(run("read_file", Path=str(self.media))["decision"], "allow")

    def test_install_keeps_media_rules_and_is_idempotent(self):
        process.install_tool_gate(self.home, self.gemini)
        process.install_tool_gate(self.home, self.gemini)
        settings = json.loads(self.settings.read_text())
        self.assertEqual(settings["permissions"]["deny"], list(process.NATIVE_TOOL_DENY_RULES))
        self.assertIn(f"read_file({self.media})", settings["permissions"]["allow"])
        process.write_media_read_permissions(self.gemini, {"/x/y.png"})
        self.assertEqual(json.loads(self.settings.read_text())["permissions"]["deny"],
                         list(process.NATIVE_TOOL_DENY_RULES))

    def test_malformed_payload_fails_closed(self):
        out = subprocess.run([sys.executable, str(PLUGIN / "tool_gate.py"), str(self.settings), str(self.home)],
                             input="not json", capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(out.stdout)["decision"], "deny")


if __name__ == "__main__":
    unittest.main()
