#!/usr/bin/env python3

from astra_shared.custom_antenna_schema import (
    default_custom_antenna,
    normalize_custom_antenna,
)
from astra_shared.param_parsing import _parse_custom_antenna_payload, parse_rf_params


def test_parse_rf_params_accepts_custom_antenna_json_string_payload():
    params = {
        "antenna_model": "custom",
        "frequency_ghz": "12.0",
        "custom_antenna": (
            '{"enabled": true, "source_format": "csv", "filename": "p.csv", '
            '"frequency_hz": 12000000000, "psi_deg": [0, 30], '
            '"phi_deg": [0, 90], "gain_dbi": [[10, 9], [8, 7]]}'
        ),
    }

    payload = parse_rf_params(params)["custom_antenna"]
    assert payload["enabled"] is True
    assert payload["source_format"] == "csv"
    assert payload["filename"] == "p.csv"
    assert payload["validation"]["is_valid"] is True


def test_parse_rf_params_invalid_custom_antenna_json_falls_back_to_default_payload():
    rf = parse_rf_params(
        {
            "antenna_model": "custom",
            "frequency_ghz": "12.0",
            "custom_antenna": "{not_json}",
        }
    )
    assert rf["custom_antenna"] == default_custom_antenna()


def test_parse_custom_antenna_payload_accepts_json_string_and_normalizes():
    raw_json = (
        '{"enabled": true, "source_format": "csv", "frequency_hz": 12000000000, '
        '"psi_deg": [0, 1], "phi_deg": [0, 180], "gain_dbi": [[1, 0], [0, -1]]}'
    )
    assert _parse_custom_antenna_payload(
        {"custom_antenna": raw_json}
    ) == normalize_custom_antenna(
        {
            "enabled": True,
            "source_format": "csv",
            "frequency_hz": 12_000_000_000,
            "psi_deg": [0, 1],
            "phi_deg": [0, 180],
            "gain_dbi": [[1, 0], [0, -1]],
        }
    )


def test_parse_custom_antenna_payload_bad_or_missing_values_return_default():
    assert _parse_custom_antenna_payload({}) == default_custom_antenna()
    assert (
        _parse_custom_antenna_payload({"custom_antenna": "{not-json}"})
        == default_custom_antenna()
    )
    assert (
        _parse_custom_antenna_payload({"custom_antenna": [1, 2, 3]})
        == default_custom_antenna()
    )
