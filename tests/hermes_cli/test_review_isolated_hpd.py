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

    def test_pre_start_isolation_excludes_discovery_registration(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(cli_main, "_guard_noninteractive_user_config") as guard:
                cli_main._prepare_agent_startup(flags())
                guard.assert_called_once()
            self.assertEqual(os.environ["HERMES_SAFE_MODE"], "1")
            self.assertEqual(os.environ["HERMES_NO_FALLBACK"], "1")
            self.assertNotIn("HERMES_IGNORE_USER_CONFIG", os.environ)

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
