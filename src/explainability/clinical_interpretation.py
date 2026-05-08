"""Simple clinical-style narratives for prediction outputs."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _risk_band(probability: float) -> str:
    if probability >= 0.8:
        return "very high"
    if probability >= 0.6:
        return "high"
    if probability >= 0.4:
        return "intermediate"
    if probability >= 0.2:
        return "low"
    return "very low"


def extract_clinical_flags(patient_row: pd.Series) -> list[str]:
    """Produce a compact set of clinically intuitive flags."""
    flags = []
    if float(patient_row.get("PSA_level", 0)) >= 10:
        flags.append("elevated PSA")
    if float(patient_row.get("age", 0)) >= 70:
        flags.append("older age")
    if str(patient_row.get("family_history", "0")) == "1":
        flags.append("positive family history")
    if str(patient_row.get("smoking_status", "")).lower() == "current":
        flags.append("current smoking")
    if float(patient_row.get("alkaline_phosphatase", 0)) >= 120:
        flags.append("elevated alkaline phosphatase")
    return flags


def interpret_prediction(patient_row: pd.Series, malignancy_probability: float) -> Dict[str, Any]:
    """Return a structured narrative for one patient's predicted risk."""
    probability = float(malignancy_probability)
    flags = extract_clinical_flags(patient_row)
    return {
        "risk_band": _risk_band(probability),
        "malignancy_probability": probability,
        "clinical_flags": flags,
        "summary": (
            f"Predicted malignancy risk is {_risk_band(probability)} at probability {probability:.3f}. "
            f"Key observable flags: {', '.join(flags) if flags else 'none highlighted'}."
        ),
    }
