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


if __name__ == "__main__":
    import unittest
    unittest.main()
