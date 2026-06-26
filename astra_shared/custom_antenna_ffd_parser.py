"""Step 8.2: FFD parser with minimal field extraction + canonical mapping.

Scope (intentional constraints):
- single-frequency selection only,
- scalar gain extraction from E_theta / E_phi magnitudes,
- canonical payload normalization via normalize_custom_antenna(...).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from .custom_antenna_errors import CustomAntennaErrorCode
from .custom_antenna_schema import normalize_custom_antenna


@dataclass
class CustomAntennaFfdError(Exception):
    """Deterministic FFD parsing error for UI/API handling."""

    code: str
    detail: str
    line_no: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "detail": self.detail}
        if self.line_no is not None:
            payload["line_no"] = self.line_no
        return payload


_FREQ_RE = re.compile(r"frequency(?:_hz)?\s*[:=]\s*([0-9.eE+-]+)")
_MULTISPACE_RE = re.compile(r"\s+")


def _parse_frequency_header(text: str) -> float | None:
    m = _FREQ_RE.search(text.lower())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if math.isfinite(val) and val > 0 else None


def parse_custom_antenna_ffd_text(
    ffd_text: str,
    *,
    filename: str | None = None,
    frequency_hz: float | None = None,
) -> dict[str, Any]:
    if not isinstance(ffd_text, str):
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_INVALID_INPUT_TYPE,
            "FFD parser expects text input (str).",
        )

    text = ffd_text.strip()
    if not text:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_EMPTY_FILE, "FFD file is empty."
        )

    lines = [ln.strip() for ln in ffd_text.splitlines()]
    nonempty = [
        (i + 1, ln) for i, ln in enumerate(lines) if ln and not ln.startswith("#")
    ]

    parsed_freq = _parse_frequency_header(ffd_text)
    if frequency_hz is not None:
        try:
            selected = float(frequency_hz)
        except ValueError as exc:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_INVALID_FREQUENCY_SELECTION,
                "frequency_hz must be numeric.",
            ) from exc
        if not math.isfinite(selected) or selected <= 0:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_INVALID_FREQUENCY_SELECTION,
                "frequency_hz must be positive finite.",
            )
        if parsed_freq is not None and not math.isclose(
            parsed_freq, selected, rel_tol=1e-6, abs_tol=1.0
        ):
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_FREQUENCY_MISMATCH,
                f"Requested frequency_hz={selected} does not match FFD frequency {parsed_freq}.",
            )
        out_freq = selected
    else:
        if parsed_freq is None:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_MISSING_FREQUENCY_SELECTION,
                "No frequency header found; provide frequency_hz explicitly for Step 8.2 single-frequency mode.",
            )
        out_freq = parsed_freq

    # detect table header
    header_idx = None
    for line_no, ln in nonempty:
        low = _MULTISPACE_RE.sub(" ", ln.lower())
        if "theta" in low and "phi" in low and "e_theta" in low and "e_phi" in low:
            header_idx = (line_no, ln)
            break
    if header_idx is None:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_FORMAT_NOT_RECOGNIZED,
            "Missing required header columns: theta phi e_theta e_phi",
            1,
        )

    start_line_no = header_idx[0] + 1
    rows: list[tuple[float, float, float]] = []
    cell_power: dict[tuple[float, float], float] = {}
    for idx in range(start_line_no - 1, len(lines)):
        raw = lines[idx].strip()
        line_no = idx + 1
        if not raw or raw.startswith("#"):
            continue
        toks = raw.replace(",", " ").split()
        if len(toks) < 4:
            continue
        if len(toks) > 4:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_UNEXPECTED_COLUMN_COUNT,
                "FFD data row must contain exactly 4 columns: theta phi e_theta e_phi.",
                line_no,
            )
        try:
            theta = float(toks[0])
            phi = float(toks[1])
            e_theta = float(toks[2])
            e_phi = float(toks[3])
        except ValueError:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_NON_NUMERIC_FIELD,
                "Could not parse theta/phi/e-field numeric values.",
                line_no,
            )
        if not all(math.isfinite(v) for v in (theta, phi, e_theta, e_phi)):
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_NON_FINITE_FIELD,
                "Encountered non-finite numeric value.",
                line_no,
            )
        if theta < 0 or theta > 180:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_THETA_OUT_OF_RANGE,
                "theta must be in [0, 180] degrees.",
                line_no,
            )
        if phi < 0 or phi > 360:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_PHI_OUT_OF_RANGE,
                "phi must be in [0, 360] degrees.",
                line_no,
            )
        is_periodic_boundary = math.isclose(phi, 360.0, rel_tol=0, abs_tol=1e-9)
        if is_periodic_boundary:
            phi = 0.0
        key = (theta, phi)
        p = (e_theta * e_theta) + (e_phi * e_phi)
        if p <= 0:
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_NON_POSITIVE_POWER,
                "Computed field power must be positive.",
                line_no,
            )
        if key in cell_power:
            # 0 and 360 degrees are the same periodic sample.  Accept the
            # duplicate only when its field power agrees with the original;
            # otherwise silently keeping either row corrupts the pattern.
            if math.isclose(cell_power[key], p, rel_tol=1e-9, abs_tol=1e-12):
                continue
            raise CustomAntennaFfdError(
                CustomAntennaErrorCode.FFD_DUPLICATE_GRID_POINT,
                "Duplicate theta/phi sample encountered in FFD table.",
                line_no,
            )
        cell_power[key] = p
        rows.append((theta, phi, p))

    if not rows:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_NO_DATA_ROWS,
            "No numeric data rows found after header.",
        )

    psi = sorted({r[0] for r in rows})
    ph = sorted({r[1] for r in rows})
    lookup = {(t, p): pw for t, p, pw in rows}
    missing = [(t, p) for t in psi for p in ph if (t, p) not in lookup]
    if missing:
        raise CustomAntennaFfdError(
            CustomAntennaErrorCode.FFD_INCOMPLETE_GRID,
            "FFD table does not form a complete rectangular theta/phi grid.",
        )

    peak = max(lookup.values())
    gain = []
    for t in psi:
        row = []
        for p in ph:
            rel = lookup[(t, p)] / peak
            row.append(10.0 * math.log10(rel))
        gain.append(row)

    raw_payload = {
        "enabled": True,
        "source_format": "ffd",
        "filename": filename,
        "frequency_hz": out_freq,
        "psi_deg": psi,
        "phi_deg": ph,
        "gain_dbi": gain,
    }
    return normalize_custom_antenna(raw_payload)
