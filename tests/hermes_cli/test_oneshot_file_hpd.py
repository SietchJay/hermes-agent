"""Offline security checks for positional-argv-free one-shot prompt transport."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import os
import sys
import tempfile

from hermes_cli import main as cli_main


class SecureOneShotFileTests(TestCase):
    def setUp(self):
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "getuid"):
            self.skipTest("secure file prompt requires POSIX O_NOFOLLOW/UID")
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)

    def make_file(self, data=b"One shot public test!", mode=0o600) -> Path:
        path = self.root / "query.txt"
        path.write_bytes(data)
        path.chmod(mode)
        return path

    def test_private_utf8_file_roundtrip(self):
        content = "One shot secure \u2713 and $() literal"
        file = self.make_file(content.encode("utf-8"))
        self.assertEqual(cli_main._private_oneshot_file(str(file)), content)

    def test_empty_and_invalid_utf8_rejected(self):
        for data in (b"", b"\x80\x80"):
            with self.subTest(data=data):
                file = self.make_file(data)
                with self.assertRaises(SystemExit) as e:
                    cli_main._private_oneshot_file(str(file))
                self.assertEqual(e.exception.code, 2)

    def test_permissions_0644_and_0640_rejected(self):
        for mode in (0o644, 0o640, 0o666):
            with self.subTest(mode=mode):
                file = self.make_file(mode=mode)
                with self.assertRaises(SystemExit):
                    cli_main._private_oneshot_file(str(file))

    def test_symlink_and_hardlink_rejected(self):
        file = self.make_file()
        alias = self.root / "alias"
        alias.symlink_to(file)
        with self.assertRaises(SystemExit):
            cli_main._private_oneshot_file(str(alias))
        alias.unlink()
        os.link(file, alias)
        with self.assertRaises(SystemExit):
            cli_main._private_oneshot_file(str(file))

    def test_fifo_rejected_without_blocking(self):
        fifo = self.root / "pipe"
        os.mkfifo(fifo, 0o600)
        with self.assertRaises(SystemExit):
            cli_main._private_oneshot_file(str(fifo))

    def test_oversized_file_rejected(self):
        file = self.make_file(b"X" * (128 * 1024 + 1))
        with self.assertRaises(SystemExit):
            cli_main._private_oneshot_file(str(file))

    def test_parser_accepts_flag_with_usage_file(self):
        from hermes_cli._parser import build_top_level_parser
        parser = build_top_level_parser()[0]
        args = parser.parse_args(["--oneshot-file", "/tmp/private-query",
                                  "--usage-file", "/tmp/private-receipt"])
        self.assertEqual(args.oneshot_file, "/tmp/private-query")
        self.assertEqual(args.usage_file, "/tmp/private-receipt")

    def test_dispatch_keeps_source_out_of_argv(self):
        private = self.make_file(b"Private text cannot be a process argument")
        args = SimpleNamespace(oneshot=None, oneshot_file=str(private),
                               model="test-model", provider="nous",
                               usage_file=str(self.root / "receipt.json"),
                               toolsets=None, skills=None, resume=None,
                               reasoning=None)
        with patch.object(cli_main, "_confirm_startup_expensive_model_override"), \
             patch.object(cli_main, "_resolve_chat_session_args"), \
             patch.object(cli_main, "_run_and_exit_oneshot") as run:
            cli_main._run_oneshot_from_args(args)
        self.assertEqual(run.call_args.args[0],
                         "Private text cannot be a process argument")
        self.assertEqual(run.call_args.kwargs["usage_file"], args.usage_file)

    def test_mutual_exclusion_fails_before_model(self):
        path = self.make_file()
        args = SimpleNamespace(oneshot="argv prompt", oneshot_file=str(path))
        with patch.object(cli_main, "_run_and_exit_oneshot") as run:
            with self.assertRaises(SystemExit) as e:
                cli_main._run_oneshot_from_args(args)
            self.assertEqual(e.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    import unittest
    unittest.main()
