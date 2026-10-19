"""Regression guard: safe-mode pinned one-shots cannot inherit paid fallbacks."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import os

from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from agent.title_generator import _auto_title_enabled, _model_title_upgrade_enabled
from agent.auxiliary_client import _aux_openrouter_settings, _OPENROUTER_MODEL


class SafeModeProviderPinningTests(TestCase):
    def test_safe_mode_prevents_fallback_reimport(self):
        for flag in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG", "HERMES_NO_FALLBACK"):
            with self.subTest(flag=flag), patch.dict(os.environ, {flag: "1"}):
                with patch("hermes_cli.config_effective.load_user_config_effective",
                           side_effect=AssertionError("must not read live config")):
                    shell = CLIChatTurnMixin()
                    old = {"provider": "openrouter", "model": "paid-model"}
                    agent = SimpleNamespace(
                        _fallback_chain=[old], _fallback_model=old,
                        _fallback_index=0, _fallback_activated=True,
                        _rate_limited_until=float("inf"),
                        _unavailable_fallback_keys=set(),
                    )
                    shell._sync_fallback_chain_with_config(agent)
                    self.assertEqual(shell._fallback_model, [])
                    self.assertEqual(agent._fallback_chain, [])
                    self.assertIsNone(agent._fallback_model)
                    self.assertEqual(agent._fallback_index, 0)
                    self.assertFalse(agent._fallback_activated)

    def test_safe_mode_disables_background_llm_titles(self):
        with patch.dict(os.environ, {"HERMES_SAFE_MODE": "1"}):
            self.assertFalse(_auto_title_enabled())
            self.assertFalse(_model_title_upgrade_enabled())

    def test_safe_mode_enforces_zero_cost_auxiliary_fallback(self):
        with patch.dict(os.environ, {"HERMES_SAFE_MODE": "1"}):
            free_only, model = _aux_openrouter_settings()
            self.assertTrue(free_only)
            self.assertEqual(model, _OPENROUTER_MODEL)



    def test_no_fallback_oneshot_resolves_only_requested_provider(self):
        from hermes_cli.oneshot import _ModelChoice, _resolve_one_shot_runtime
        choice = _ModelChoice(model="stepfun/step-3.7-flash:free", provider="nous")
        expected = {"provider": "nous", "api_key": "opaque", "base_url": "https://example.test"}
        with patch.dict(os.environ, {"HERMES_NO_FALLBACK": "1"}),              patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=expected) as direct,              patch("hermes_cli.runtime_provider.resolve_runtime_with_fallback",
                   side_effect=AssertionError("fallback must never run")):
            runtime, entry = _resolve_one_shot_runtime({"fallback_providers": [{"provider": "openrouter", "model": "paid"}]}, choice)
        self.assertEqual(runtime, expected)
        self.assertIsNone(entry)
        direct.assert_called_once()

    def test_normal_oneshot_keeps_existing_fallback_policy(self):
        from hermes_cli.oneshot import _ModelChoice, _resolve_one_shot_runtime
        choice = _ModelChoice(model="model-x", provider="nous")
        with patch.dict(os.environ, {}, clear=True),              patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   side_effect=AssertionError("normal path must use fallback resolver")),              patch("hermes_cli.runtime_provider.resolve_runtime_with_fallback",
                   return_value=({"provider": "openrouter"}, {"provider": "openrouter", "model": "paid"})) as fallback:
            runtime, entry = _resolve_one_shot_runtime({"fallback_providers": []}, choice)
        self.assertEqual(runtime["provider"], "openrouter")
        self.assertIsNotNone(entry)
        fallback.assert_called_once()

    def test_no_fallback_auxiliary_auto_does_not_discover_alternates(self):
        from agent.auxiliary_client import _resolve_auto_route
        with patch.dict(os.environ, {"HERMES_NO_FALLBACK": "1"}),              patch("agent.auxiliary_client._normalize_main_runtime", return_value={}),              patch("agent.auxiliary_client._warn_stale_openai_base_url"),              patch("agent.auxiliary_client._main_route_target",
                   return_value=("nous", "model", "", "opaque", "chat_completions")),              patch("agent.auxiliary_client._try_main_provider_route", return_value=None),              patch("agent.auxiliary_client._try_configured_fallback_chain",
                   side_effect=AssertionError("configured fallback forbidden")),              patch("agent.auxiliary_client._try_main_fallback_chain",
                   side_effect=AssertionError("main fallback forbidden")),              patch("agent.auxiliary_client._try_discovery_chain",
                   side_effect=AssertionError("discovery forbidden")):
            self.assertEqual(_resolve_auto_route(task="title_generation"), (None, None, ""))

if __name__ == "__main__":
    import unittest
    unittest.main()
