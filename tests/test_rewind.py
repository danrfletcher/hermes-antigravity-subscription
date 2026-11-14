"""rewind.py: cutting a bad turn out of an agy conversation DB (schema of agy 1.5.9)."""
import fcntl
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rewind  # noqa: E402

DDL = [
    "CREATE TABLE `battle_mode_infos` (`idx` integer,`data` blob,PRIMARY KEY (`idx`))",
    "CREATE TABLE `executor_metadata` (`idx` integer,`data` blob,PRIMARY KEY (`idx`))",
    "CREATE TABLE `gen_metadata` (`idx` integer,`data` blob,`size` integer NOT NULL DEFAULT 0,PRIMARY KEY (`idx`))",
    "CREATE TABLE `parent_references` (`idx` integer,`data` blob,PRIMARY KEY (`idx`))",
    "CREATE TABLE `steps` (`idx` integer,`step_type` integer NOT NULL DEFAULT 0,`status` integer NOT NULL DEFAULT 0,"
    "`has_subtrajectory` numeric NOT NULL DEFAULT false,`metadata` blob,`error_details` blob,`permissions` blob,"
    "`task_details` blob,`render_info` blob,`step_payload` blob,`step_format` integer NOT NULL DEFAULT 0,PRIMARY KEY (`idx`))",
    "CREATE TABLE `trajectory_meta` (`trajectory_id` text,`cascade_id` text,`trajectory_type` integer,`source` integer,"
    "PRIMARY KEY (`trajectory_id`))",
    "CREATE TABLE `trajectory_metadata_blob` (`id` text DEFAULT \"main\",`data` blob,PRIMARY KEY (`id`))",
    "CREATE INDEX `idx_steps_status` ON `steps`(`status`)",
    "CREATE INDEX `idx_steps_step_type` ON `steps`(`step_type`)",
]
CONV = "0f3c2a1e-1111-2222-3333-444455556666"


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def gen_blob(steps):
    packed = b"".join(_varint(s) for s in steps)
    return b"\x12" + _varint(len(packed)) + packed + b"\x22\x04uuid"


def exe_blob(kind, last):
    return b"\x08" + _varint(kind) + b"\x10\x01\x18" + _varint(last)


class Conversation:
    """A fake agy conversation: turns of (user step, response step), plus optional tool steps."""

    def __init__(self, gemini: Path):
        self.gemini = gemini
        (gemini / "conversations").mkdir(parents=True)
        self.db = gemini / "conversations" / f"{CONV}.db"
        conn = sqlite3.connect(self.db)
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma user_version=1")
        for ddl in DDL:
            conn.execute(ddl)
        conn.commit()
        conn.close()
        self.steps = self.gen = self.exe = 0

    def turn(self, prompt, *, tools=0, steps_dir=False):
        conn = sqlite3.connect(self.db)
        first = self.steps
        conn.execute("insert into steps(idx, step_type, status, step_payload) values (?, 14, 3, ?)",
                     (self.steps, b"\x0a" + prompt.encode()))
        self.steps += 1
        made = []
        for _ in range(tools):
            conn.execute("insert into steps(idx, step_type, status, step_payload) values (?, 15, 3, ?)",
                         (self.steps, b"call"))
            conn.execute("insert into steps(idx, step_type, status, step_payload) values (?, 132, 7, ?)",
                         (self.steps + 1, b"denied"))
            made += [self.steps, self.steps + 1]
            if steps_dir:
                d = self.gemini / "brain" / CONV / ".system_generated" / "steps" / str(self.steps + 1)
                d.mkdir(parents=True)
                (d / "output.txt").write_text("x")
            self.steps += 2
        conn.execute("insert into steps(idx, step_type, status, step_payload) values (?, 15, 3, ?)",
                     (self.steps, b"answer"))
        made.append(self.steps)
        self.steps += 1
        conn.execute("insert into gen_metadata(idx, data) values (?, ?)", (self.gen, gen_blob(made)))
        self.gen += 1
        conn.execute("insert into executor_metadata(idx, data) values (?, ?)", (self.exe, exe_blob(4, self.steps - 1)))
        self.exe += 1
        conn.commit()
        conn.close()
        return first

    def sync(self):
        conn = sqlite3.connect(self.db)
        try:
            self.steps, self.gen, self.exe = (conn.execute(f"select coalesce(max(idx) + 1, 0) from {t}").fetchone()[0]
                                              for t in ("steps", "gen_metadata", "executor_metadata"))
        finally:
            conn.close()

    def types(self):
        conn = sqlite3.connect(self.db)
        try:
            return [r[0] for r in conn.execute("select step_type from steps order by idx")]
        finally:
            conn.close()


class RewindTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.gemini = self.root / "home" / ".gemini" / "antigravity-cli"
        self.conv = Conversation(self.gemini)
        self.backups = self.root / "rewind-backups"

    def tearDown(self):
        self._tmp.cleanup()

    def point(self, prompt, version="agy:1"):
        return rewind.read_point(self.conv.db, conversation_id=CONV, prompt=prompt, agy_version=version)

    def test_schema_fingerprint_matches_agy(self):
        self.conv.turn("first")
        self.assertEqual(self.point("x")["schema"], "a87618921be9966b")

    def test_cut_restores_the_state_before_the_bad_turn(self):
        self.conv.turn("first question")
        self.conv.turn("second question", tools=1)
        before = self.conv.types()
        point = self.point("BAD TURN prompt")
        self.conv.turn("BAD TURN prompt", tools=2, steps_dir=True)
        rewind.rewind(self.gemini, point, backup_dir=self.backups, agy_version="agy:1")
        self.assertEqual(self.conv.types(), before)
        self.assertEqual(self.point("next")["steps_hash"], point["steps_hash"])
        self.assertEqual(list((self.gemini / "brain" / CONV / ".system_generated" / "steps").iterdir()), [])
        self.assertEqual(len(list(self.backups.glob(f"{CONV}.*.db"))), 1)
        rewind.drop_backups(self.backups, CONV)
        self.assertEqual(list(self.backups.glob("*.db")), [])

    def test_same_point_rewinds_twice(self):
        self.conv.turn("first")
        point = self.point("again")
        self.conv.turn("again")
        rewind.rewind(self.gemini, point, backup_dir=self.backups, agy_version="agy:1")
        self.conv.sync()
        self.conv.turn("again", tools=1)
        rewind.rewind(self.gemini, point, backup_dir=self.backups, agy_version="agy:1")
        self.assertEqual(self.conv.types(), [14, 15])

    def _refused(self, point, match, **kwargs):
        before = self.conv.types()
        with self.assertRaisesRegex(rewind.RewindRefused, match):
            rewind.rewind(self.gemini, point, backup_dir=self.backups, agy_version=kwargs.get("version", "agy:1"))
        self.assertEqual(self.conv.types(), before)

    def test_refuses_when_the_next_step_is_not_the_sent_prompt(self):
        self.conv.turn("first")
        point = self.point("what the plugin sent")
        self.conv.turn("something else entirely")
        self._refused(point, "not the prompt the plugin sent")

    def test_refuses_when_history_before_the_point_changed(self):
        self.conv.turn("first")
        point = self.point("bad")
        self.conv.turn("bad")
        conn = sqlite3.connect(self.conv.db)
        conn.execute("update steps set step_payload = ? where idx = 1", (b"compacted summary",))
        conn.commit()
        conn.close()
        self._refused(point, "changed")

    def test_refuses_when_nothing_follows_the_point(self):
        self.conv.turn("first")
        self._refused(self.point("x"), "nothing after")

    def test_refuses_after_an_agy_upgrade(self):
        self.conv.turn("first")
        point = self.point("bad")
        self.conv.turn("bad")
        self._refused(point, "agy changed version", version="agy:2")

    def test_refuses_while_another_agy_holds_the_conversation(self):
        self.conv.turn("first")
        point = self.point("bad")
        self.conv.turn("bad")
        lock = self.gemini / "presence" / f"{CONV}.lock"
        lock.parent.mkdir()
        handle = open(lock, "a+")
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            self._refused(point, "another agy")
        finally:
            handle.close()

    def test_refuses_an_unknown_schema(self):
        self.conv.turn("first")
        point = self.point("bad")
        self.conv.turn("bad")
        conn = sqlite3.connect(self.conv.db)
        conn.execute("create table new_thing (x)")
        conn.commit()
        conn.close()
        self._refused(point, "schema")

    def test_refuses_when_a_kept_generation_points_past_the_cut(self):
        self.conv.turn("first")
        point = self.point("bad")
        self.conv.turn("bad")
        conn = sqlite3.connect(self.conv.db)
        conn.execute("update gen_metadata set data = ? where idx = 0", (gen_blob([1, 3]),))
        conn.commit()
        conn.close()
        self._refused(point, "deleted step")

    def test_large_step_indices_decode(self):
        self.assertEqual(rewind._gen_steps(gen_blob([5, 200, 70000])), [5, 200, 70000])
        self.assertEqual(rewind._exe_last_step(exe_blob(9, 300)), 300)

    def test_no_point_for_an_empty_or_missing_db(self):
        self.assertIsNone(self.point("x"))
        self.assertIsNone(rewind.read_point(self.root / "nope.db", conversation_id=CONV, prompt="x", agy_version=None))


if __name__ == "__main__":
    unittest.main()
