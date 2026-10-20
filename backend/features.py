"""Extract numeric signals from URL text without contacting the destination."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import pandas as pd

FEATURE_NAMES = [
    "hostname_length",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "digit_count",
    "digit_ratio",
    "subdomain_count",
    "uses_https",
    "hostname_is_ip",
    "has_punycode",
]


def extract_url_features(url: str) -> dict[str, float | int]:
    """Extract the features used by every model, without DNS or page requests."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("URL is malformed.") from error

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or any(character.isspace() for character in hostname)
    ):
        raise ValueError("URL must include an http:// or https:// scheme and hostname.")

    hostname = hostname.lower().rstrip(".")
    canonical_hostname = hostname[4:] if hostname.startswith("www.") else hostname
    labels = canonical_hostname.split(".")
    try:
        ipaddress.ip_address(canonical_hostname)
        is_ip = True
    except ValueError:
        is_ip = False

    digits = sum(character.isdigit() for character in canonical_hostname)

    return {
        "hostname_length": len(canonical_hostname),
        "dot_count": canonical_hostname.count("."),
        "hyphen_count": canonical_hostname.count("-"),
        "underscore_count": canonical_hostname.count("_"),
        "digit_count": digits,
        "digit_ratio": digits / max(len(canonical_hostname), 1),
        "subdomain_count": max(len(labels) - 2, 0) if not is_ip else 0,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "hostname_is_ip": int(is_ip),
        "has_punycode": int(any(label.startswith("xn--") for label in labels)),
    }


def features_frame(urls: list[str]) -> pd.DataFrame:
    """Return normalized hostname and scheme features in the model's fixed order.

    URL paths and query strings are intentionally excluded because the source
    labels associate them with collection artifacts rather than reliable risk.
    """
    return pd.DataFrame(
        [extract_url_features(url) for url in urls],
        columns=FEATURE_NAMES,
        dtype=float,
    )
