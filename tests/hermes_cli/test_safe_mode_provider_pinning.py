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

    def test_no_fallback_rejects_auto_provider_before_auth_resolution(self):
        from hermes_cli.oneshot import _ModelChoice, _resolve_one_shot_runtime
        for candidate in (None, "", "auto", "AUTO"):
            with self.subTest(provider=candidate), \
                 patch.dict(os.environ, {"HERMES_NO_FALLBACK":"1"}), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       side_effect=AssertionError("must not reach auth ladder")), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_with_fallback",
                       side_effect=AssertionError("must not fall back")):
                with self.assertRaisesRegex(ValueError,"requires an explicit --provider"):
                    _resolve_one_shot_runtime(
                        {"fallback_providers":[{"provider":"openrouter"}]},
                        _ModelChoice(provider=candidate,model="some-model"))

    def test_classic_chat_requires_explicit_pinned_provider(self):
        from hermes_cli.cli_agent_setup_mixin import (
            CLIAgentSetupMixin, _no_fallback_provider_missing,
        )
        with patch.dict(os.environ, {"HERMES_NO_FALLBACK":"1"}):
            for selected in (None, "", "auto", "AUTO"):
                with self.subTest(selected=selected):
                    self.assertTrue(_no_fallback_provider_missing(selected))
                    cli = CLIAgentSetupMixin()
                    cli.requested_provider = selected
                    cli._explicit_api_key = None
                    cli._explicit_base_url = None
                    ready, err = cli._probe_runtime_credentials()
                    self.assertFalse(ready)
                    self.assertIsInstance(err, ValueError)
            self.assertFalse(_no_fallback_provider_missing("nous"))

    def test_classic_chat_prevents_fallback_on_auth_error(self):
        from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
        from hermes_cli.auth import AuthError
        with patch.dict(os.environ, {"HERMES_NO_FALLBACK":"1"}):
            cli = CLIAgentSetupMixin()
            self.assertIsNone(cli._resolve_fallback_runtime(AuthError("blocked")))

    def test_safe_mode_forbids_auxiliary_dispatch(self):
        from agent.auxiliary_client import _plan_aux_call
        with patch.dict(os.environ, {"HERMES_SAFE_MODE":"1"}, clear=True), \
             patch("agent.auxiliary_client._normalize_main_runtime",
                   side_effect=AssertionError("no auxiliary provider resolution")):
            with self.assertRaisesRegex(RuntimeError,"disabled"):
                _plan_aux_call(
                    "vision", async_mode=False, provider="auto", model=None,
                    base_url=None, api_key=None, main_runtime=None,
                    messages=[{"role":"user","content":"test"}],
                    temperature=None,max_tokens=8,tools=None,timeout=None,
                    extra_body=None,reasoning_config=None,extra_headers=None,
                    api_mode=None,route_info=None)

    def test_safe_mode_auto_provider_fails_closed_in_oneshot(self):
        from hermes_cli.oneshot import _ModelChoice, _resolve_one_shot_runtime
        for flag in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
            with self.subTest(flag=flag), patch.dict(os.environ, {flag:"1"}, clear=True),                  patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       side_effect=AssertionError("must not select account")):
                with self.assertRaisesRegex(ValueError,"explicit --provider"):
                    _resolve_one_shot_runtime({}, _ModelChoice(provider="auto", model="synthetic"))

    def test_safe_mode_auto_provider_fails_closed_in_classic_chat(self):
        from hermes_cli.cli_agent_setup_mixin import (
            _no_fallback_provider_missing, CLIAgentSetupMixin,
        )
        for flag in ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"):
            with self.subTest(flag=flag), patch.dict(os.environ, {flag:"1"}, clear=True),                  patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       side_effect=AssertionError("must not select account")):
                self.assertTrue(_no_fallback_provider_missing(None))
                cli=CLIAgentSetupMixin()
                cli.requested_provider=None
                cli._explicit_api_key=None
                cli._explicit_base_url=None
                ready,error=cli._probe_runtime_credentials()
                self.assertFalse(ready)
                self.assertIsInstance(error,ValueError)

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

    def test_no_fallback_denies_auxiliary_model_call_before_provider_resolution(self):
        from agent.auxiliary_client import _plan_aux_call
        with patch.dict(os.environ, {"HERMES_NO_FALLBACK": "1"}), \
             patch("agent.auxiliary_client._normalize_main_runtime",
                   side_effect=AssertionError("must not even resolve an auxiliary provider")):
            with self.assertRaisesRegex(RuntimeError, "Auxiliary model calls disabled"):
                _plan_aux_call(
                    "vision", async_mode=False, provider="auto", model=None,
                    base_url=None, api_key=None, main_runtime=None,
                    messages=[{"role":"user","content":"test"}],
                    temperature=None,max_tokens=8,tools=None,timeout=None,
                    extra_body=None,reasoning_config=None,extra_headers=None,
                    api_mode=None,route_info=None)

    def test_ignore_user_config_denies_auxiliary_dispatch_and_titles(self):
        from agent.auxiliary_client import _plan_aux_call
        for flag in ("HERMES_IGNORE_USER_CONFIG", "HERMES_SAFE_MODE", "HERMES_NO_FALLBACK"):
            with self.subTest(flag=flag), patch.dict(os.environ, {flag:"1"}, clear=True), \
                 patch("agent.auxiliary_client._normalize_main_runtime",
                       side_effect=AssertionError("never resolve auxiliary account")):
                self.assertFalse(_auto_title_enabled())
                self.assertFalse(_model_title_upgrade_enabled())
                with self.assertRaisesRegex(RuntimeError,"Auxiliary model calls disabled"):
                    _plan_aux_call(
                        "vision", async_mode=False, provider="auto", model=None,
                        base_url=None, api_key=None, main_runtime=None,
                        messages=[{"role":"user","content":"synthetic"}],
                        temperature=None,max_tokens=8,tools=None,timeout=None,
                        extra_body=None,reasoning_config=None,extra_headers=None,
                        api_mode=None,route_info=None)

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
