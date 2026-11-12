"""Regression: an OpenRouter main router alias must not shadow free-only aux fallback.

All provider clients are synthetic; no network or paid model calls are made.
"""

from unittest.mock import patch, sentinel

import pytest


@pytest.fixture
def openrouter_aux_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: openrouter\n"
        "  default: openrouter/free\n"
        "auxiliary:\n"
        "  free_only: true\n"
        "  openrouter_model: nvidia/nemotron-3.5-lightning:free\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import auxiliary_client as ac

    with (
        patch.object(ac, "_select_pool_entry", return_value=(False, None)),
        patch.object(ac, "_scoped_key_env", return_value="synthetic-test-key"),
        patch.object(ac, "_create_openai_client", return_value=sentinel.client) as make_client,
    ):
        yield ac, make_client


def test_missing_aux_task_model_uses_configured_free_openrouter_fallback(openrouter_aux_home):
    ac, make_client = openrouter_aux_home

    client, model = ac.resolve_provider_client("openrouter")

    assert client is sentinel.client
    assert model == "nvidia/nemotron-3.5-lightning:free"
    make_client.assert_called_once()
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly()
    assert cfg["model"]["default"] == "openrouter/free"
    assert cfg["auxiliary"]["free_only"] is True


def test_explicit_main_router_alias_remains_blocked_by_aux_free_only(openrouter_aux_home):
    ac, make_client = openrouter_aux_home

    client, model = ac.resolve_provider_client("openrouter", model="openrouter/free")

    assert (client, model) == (None, None)
    make_client.assert_not_called()


def test_non_free_explicit_aux_model_stays_blocked(openrouter_aux_home):
    ac, make_client = openrouter_aux_home

    client, model = ac.resolve_provider_client("openrouter", model="vendor/paid")

    assert (client, model) == (None, None)
    make_client.assert_not_called()
