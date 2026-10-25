"""Offline fail-closed tests for opt-in HPD isolated reviewers (no model calls)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import os

from hermes_cli import config, main as cli_main, oneshot


def flags(**kw):
    defaults = dict(review_isolated=True, oneshot_file="/tmp/query.txt",
                    usage_file="/tmp/usage.json", no_fallback=True,
                    safe_mode=True, provider="custom:groq-free-review",
                    model="qwen/qwen3.8-27b", oneshot=None, resume=None,
                    continue_last=None, toolsets=None, skills=None, yolo=False,
                    accept_hooks=False, ignore_user_config=False, in_dir=None,
                    command=None, query=None)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def raw():
    return {"providers": {
        "groq-free-review": {"enabled": True, "api": "https://api.groq.com/openai/v1",
                             "key_env": "GROQ_API_KEY", "transport": "chat_completions",
                             "discover_models": False, "models": ["qwen/qwen3.8-27b"]},
        "paid-provider": {"enabled": True, "api": "https://untrusted.invalid/v1",
                          "key_env": "PAID_KEY", "transport": "chat_completions",
                          "discover_models": False, "models": ["paid"]},
    }, "model": {"provider": "openrouter", "default": "some-paid-model"},
            "fallback_providers": [{"provider": "paid-provider", "model": "paid"}],
            "mcp_servers": {"private-server": {"command": "bad"}},
            "hooks": {"run": "bad"}, "plugins": {"auto": True},
            "skills": {"everything": True}, "auxiliary": {"enabled": True}}


class IsolatedReviewerTest(TestCase):
    def test_only_pinned_named_provider_survives_config_projection(self):
        with patch.dict(os.environ, {
            "HERMES_REVIEW_PROVIDER_PIN": "custom:groq-free-review",
            "HERMES_REVIEW_MODEL_PIN": "qwen/qwen3.8-27b"}, clear=False):
            result = config._isolated_review_config(raw())
        self.assertEqual(list(result["providers"]), ["groq-free-review"])
        self.assertEqual(result["model"]["provider"], "custom:groq-free-review")
        self.assertEqual(result["fallback_providers"], [])
        self.assertEqual(result["mcp_servers"], {})
        self.assertEqual(result["plugins"], {})
        self.assertEqual(result["hooks"], {})
        self.assertNotIn("paid-provider", str(result))
        self.assertNotIn("untrusted.invalid", str(result))

    def test_nous_projection_excludes_every_custom_provider(self):
        with patch.dict(os.environ, {"HERMES_REVIEW_PROVIDER_PIN": "nous",
                                     "HERMES_REVIEW_MODEL_PIN": "stepfun/step-3.7-flash:free"}):
            result = config._isolated_review_config(raw())
        self.assertEqual(result["providers"], {})
        self.assertEqual(result["model"]["provider"], "nous")

    def test_unsafe_custom_provider_config_is_blocked(self):
        for key, value in (("key_cmd", "echo paid"), ("api_key", "inline"),
                           ("extra_headers", {"Authorization": "Bearer bad"}),
                           ("extra_body", {"model": "paid"}),
                           ("models_api", "https://other.invalid/v1/models"),
                           ("discover_models", True), ("api", "http://untrusted.invalid/v1"),
                           ("enabled", False), ("models", ["other-model"])):
            with self.subTest(key=key):
                c = raw()
                c["providers"]["groq-free-review"][key] = value
                with patch.dict(os.environ, {"HERMES_REVIEW_PROVIDER_PIN": "custom:groq-free-review",
                                             "HERMES_REVIEW_MODEL_PIN": "qwen/qwen3.8-27b"}):
                    with self.assertRaises(ValueError):
                        config._isolated_review_config(c)

    def test_review_mode_requires_secure_file_and_no_fallback(self):
        for field, val in (
            ("oneshot_file", None), ("usage_file", None),
            ("safe_mode", False), ("no_fallback", False),
            ("provider", None), ("model", None), ("oneshot", "unsafe"),
            ("resume", "latest"), ("continue_last", True),
            ("toolsets", "all"), ("skills", ["private"]),
            ("yolo", True), ("accept_hooks", True),
            ("ignore_user_config", True), ("in_dir", "/tmp"),
        ):
            with self.subTest(field=field):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaises(SystemExit) as exc:
                        cli_main._apply_review_isolated(flags(**{field: val}))
                self.assertEqual(exc.exception.code, 2)

    def test_approved_flags_set_expected_env_without_ignoring_config(self):
        with patch.dict(os.environ, {}, clear=True):
            cli_main._apply_review_isolated(flags())
            self.assertEqual(os.environ["HERMES_NO_FALLBACK"], "1")
            self.assertEqual(os.environ["HERMES_SAFE_MODE"], "1")
            self.assertEqual(os.environ["HERMES_IGNORE_RULES"], "1")
            self.assertEqual(os.environ["HERMES_REVIEW_PROVIDER_PIN"],
                             "custom:groq-free-review")
            self.assertNotIn("HERMES_IGNORE_USER_CONFIG", os.environ)
            cli_main._apply_safe_mode(flags())
            self.assertEqual(os.environ["HERMES_SAFE_MODE"], "1")
            self.assertNotIn("HERMES_IGNORE_USER_CONFIG", os.environ)

    def test_normal_safe_mode_still_ignores_user_config(self):
        with patch.dict(os.environ, {}, clear=True):
            cli_main._apply_safe_mode(flags(review_isolated=False))
            self.assertEqual(os.environ["HERMES_IGNORE_USER_CONFIG"], "1")

    def test_main_routes_review_before_boot_or_update_side_effects(self):
        fake = flags()
        with patch.object(cli_main.sys, "argv", [
                "hermes", "--review-isolated", "--oneshot-file", "/tmp/query",
                "--usage-file", "/tmp/usage", "--safe-mode", "--no-fallback",
                "--provider", "custom:groq-free-review", "--model", "qwen/qwen3.8-27b",
             ]), \
             patch.object(cli_main, "_set_process_title"), \
             patch.object(cli_main, "_warn_if_unsupervised_pid1"), \
             patch.object(cli_main, "_advertise_agent_env"), \
             patch.object(cli_main, "_light_chat_parser") as parser, \
             patch.object(cli_main, "_prepare_agent_startup") as prepare, \
             patch.object(cli_main, "_run_oneshot_from_args",
                          side_effect=SystemExit(0)) as invoke, \
             patch.object(cli_main, "_recover_update_debts_on_startup",
                          side_effect=AssertionError("unexpected updater")):
            parser.return_value.parse_args.return_value = fake
            with self.assertRaises(SystemExit) as e:
                cli_main.main()
        self.assertEqual(e.exception.code, 0)
        prepare.assert_called_once_with(fake)
        invoke.assert_called_once_with(fake)

    def test_pre_start_isolation_excludes_discovery_registration(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(cli_main, "_guard_noninteractive_user_config") as guard:
                cli_main._prepare_agent_startup(flags())
                guard.assert_called_once()
            self.assertEqual(os.environ["HERMES_SAFE_MODE"], "1")
            self.assertEqual(os.environ["HERMES_NO_FALLBACK"], "1")
            self.assertNotIn("HERMES_IGNORE_USER_CONFIG", os.environ)

    def test_isolated_tool_loader_does_not_import_plugins_or_tools(self):
        import sys
        from agent.agent_init import _load_tools
        agent = SimpleNamespace(quiet_mode=True)
        with patch.dict(os.environ, {"HERMES_REVIEW_ISOLATED": "1"}), \
             patch.dict(sys.modules, {
                 "hermes_cli.left_core_migration": None,
                 "hermes_cli.plugins": None,
                 "model_tools": None,
             }):
            _load_tools(agent, ["all"], [])
        self.assertEqual(agent.enabled_toolsets, [])
        self.assertEqual(agent.tools, [])
        self.assertEqual(agent.valid_tool_names, set())

    def test_config_projection_also_filters_cached_source(self):
        with patch.dict(os.environ, {
            "HERMES_REVIEW_ISOLATED": "1",
            "HERMES_REVIEW_PROVIDER_PIN": "custom:groq-free-review",
            "HERMES_REVIEW_MODEL_PIN": "qwen/qwen3.8-27b",
        }):
            with patch("hermes_cli.config._load_config_impl", return_value=raw()):
                a = config.load_config()
                b = config.load_config_readonly()
        self.assertEqual(a, b)
        self.assertEqual(list(a["providers"]), ["groq-free-review"])
        self.assertEqual(a["fallback_providers"], [])

    def test_model_choice_preserves_custom_name(self):
        with patch.dict(os.environ, {
            "HERMES_REVIEW_PROVIDER_PIN": "custom:groq-free-review",
            "HERMES_REVIEW_MODEL_PIN": "qwen/qwen3.8-27b",
        }):
            cfg = config._isolated_review_config(raw())
            choice = oneshot._resolve_model_and_provider(
                cfg, "qwen/qwen3.8-27b", "custom:groq-free-review"
            )
        self.assertEqual(choice.provider, "custom:groq-free-review")
        self.assertEqual(choice.model, "qwen/qwen3.8-27b")

    def test_isolated_oneshot_never_enables_yolo_or_hooks(self):
        import tempfile
        from pathlib import Path
        from hermes_cli import oneshot as one
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "usage.json")
            def observe(*args, **kwargs):
                self.assertNotIn("HERMES_YOLO_MODE", os.environ)
                self.assertNotIn("HERMES_ACCEPT_HOOKS", os.environ)
                self.assertEqual(kwargs["ledger"], True)
                return "REVIEW OK", {
                    "completed": True, "failed": False, "api_calls": 1,
                    "provider": "nous", "model": "test-free",
                    "estimated_cost_usd": 0,
                    "auxiliary_usage": {},
                }
            with patch.dict(os.environ, {"HERMES_REVIEW_ISOLATED": "1",
                                         "HERMES_YOLO_MODE": "1",
                                         "HERMES_ACCEPT_HOOKS": "1"}):
                with patch.object(one, "declare_stateless_channel"), \
                     patch.object(one, "_run_agent", side_effect=observe):
                    result = one.run_oneshot("synthetic public reviewer fixture",
                                             model="test-free", provider="nous",
                                             usage_file=path)
            self.assertEqual(result, 0)
            import json
            receipt = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(receipt["auxiliary"]["api_calls"], 0)
            self.assertEqual(receipt["api_calls"], 1)

    def test_isolated_auxiliary_sync_async_and_client_resolution_are_blocked(self):
        import asyncio
        from agent import auxiliary_client as aux
        with patch.dict(os.environ, {"HERMES_REVIEW_ISOLATED": "1"}):
            with self.assertRaisesRegex(RuntimeError, "auxiliary_model_disabled"):
                aux.resolve_provider_client("nous")
            with self.assertRaisesRegex(RuntimeError, "auxiliary_model_disabled"):
                aux.call_llm(messages=[], task="title_generation")
            with self.assertRaisesRegex(RuntimeError, "auxiliary_model_disabled"):
                asyncio.run(aux.async_call_llm(messages=[], task="title_generation"))

    def test_resolver_bypasses_fallback_chain_when_isolated(self):
        choice = SimpleNamespace(provider="custom:groq-free-review",
                                 model="qwen/qwen3.8-27b",
                                 base_url=None, api_key=None)
        with patch.dict(os.environ, {"HERMES_REVIEW_ISOLATED": "1",
                                     "HERMES_NO_FALLBACK": "1"}):
            with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       return_value={"provider": "custom"}) as direct:
                with patch("hermes_cli.runtime_provider.resolve_runtime_with_fallback",
                           side_effect=AssertionError("fallback forbidden")):
                    result, other = oneshot._resolve_one_shot_runtime({}, choice)
        self.assertEqual(result["provider"], "custom")
        self.assertIsNone(other)
        direct.assert_called_once()


if __name__ == "__main__":
    import unittest
    unittest.main()
