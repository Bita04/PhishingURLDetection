"""Return the three most influential, human-readable SHAP reasons."""

from __future__ import annotations

import numpy as np
import shap

from features import FEATURE_NAMES, features_frame
from train import phishing_shap_values

REASON_TEXT = {
    "hostname_length": lambda value: f"The hostname is {int(value)} characters long.",
    "dot_count": lambda value: f"The hostname contains {int(value)} dots.",
    "hyphen_count": lambda value: f"The hostname contains {int(value)} hyphens.",
    "underscore_count": lambda value: f"The hostname contains {int(value)} underscores.",
    "digit_count": lambda value: f"The hostname contains {int(value)} digits.",
    "digit_ratio": lambda value: f"Digits account for {value:.0%} of the hostname.",
    "subdomain_count": lambda value: f"The hostname has {int(value)} subdomains.",
    "path_depth": lambda value: f"The URL has {int(value)} nested path segments.",
    "query_parameter_count": lambda value: f"The URL has {int(value)} query parameters.",
    "uses_https": lambda value: "The URL uses HTTPS." if value else "The URL does not use HTTPS.",
    "hostname_is_ip": lambda value: "The hostname is an IP address." if value else "The hostname is a domain name, not an IP address.",
    "has_punycode": lambda value: "The hostname contains a punycode label." if value else "The hostname has no punycode label.",
}


def explain_url(model, explainer: shap.LinearExplainer, url: str) -> dict[str, object]:
    """Predict a URL and return phishing probability plus three directional reasons."""
    row = features_frame([url])
    classes = list(model.classes_)
    phishing_probability = float(model.predict_proba(row)[0][classes.index(0)])
    label = "suspicious" if phishing_probability >= 0.5 else "safe"

    values = phishing_shap_values(explainer, model, row)[0]
    if label == "safe":
        values = -values
    ordered = np.argsort(values)[::-1]

    reasons: list[str] = []
    for index in ordered:
        name = FEATURE_NAMES[index]
        reason = REASON_TEXT[name](row.iloc[0, index])
        direction = f"contributes toward the {label} classification"
        reasons.append(f"{reason} ({direction}).")
        if len(reasons) == 3:
            break

    return {"label": label, "probability": phishing_probability, "reasons": reasons}
