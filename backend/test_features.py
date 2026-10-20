"""Tests for URL validation, feature stability, and known benign-domain regressions."""

import pytest
from fastapi.testclient import TestClient

from explain import explain_url
from features import FEATURE_NAMES, extract_url_features, features_frame
from main import app, load_predictor

client = TestClient(app)


def test_extracts_expected_url_features():
    url = "https://login2.example-2.co.uk/account/reset?user=alice&code=123"
    features = extract_url_features(url)

    assert features["hostname_length"] == len("login2.example-2.co.uk")
    assert features["uses_https"] == 1
    assert features["subdomain_count"] == 2
    assert features["digit_count"] == 2
    assert features["hyphen_count"] == 1
    assert list(features) == FEATURE_NAMES


def test_detects_ip_and_punycode_hosts():
    assert extract_url_features("http://192.0.2.1/a")["hostname_is_ip"] == 1
    assert extract_url_features("https://xn--bcher-kva.example/")["has_punycode"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "",
        "example.com",
        "ftp://example.com",
        "https:///missing-host",
        "https://bad host",
        "http://example.com:not-a-port",
    ],
)
def test_rejects_invalid_urls(url):
    with pytest.raises(ValueError):
        extract_url_features(url)


def test_root_slash_is_canonicalized():
    assert features_frame(
        ["https://example.com", "https://www.example.com/account/0?user=alice"]
    ).iloc[0].equals(
        features_frame(["https://www.example.com/?different=1"]).iloc[0]
    )


def test_feature_frame_has_stable_numeric_columns():
    frame = features_frame(["https://example.com", "http://192.0.2.1/a"])

    assert frame.shape == (2, len(FEATURE_NAMES))
    assert frame.columns.tolist() == FEATURE_NAMES
    assert frame.loc[0, "dot_count"] == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://gmail.com",
        "https://github.com",
        "https://google.com",
        "https://mail.google.com/",
        "https://mail.google.com/mail/u/0/#inbox",
        "https://github.com/login",
        "https://www.github.com/Bita04/Real-Time-Visual-Perception-System",
        "https://www.google.com/search?q=phishing",
        "https://outlook.live.com/mail/0/inbox",
        "https://www.microsoft.com/",
    ],
)
def test_common_benign_domains_do_not_trigger_the_shortcut(url):
    bundle, explainer = load_predictor()
    model = bundle["models"][bundle["default_model"]]

    prediction = explain_url(model, explainer, url)

    assert prediction["label"] == "safe"
    assert prediction["probability"] < 0.5
    assert len(prediction["reasons"]) == 3


def test_prediction_exposes_phishing_examples():
    bundle, explainer = load_predictor()
    model = bundle["models"][bundle["default_model"]]

    prediction = explain_url(model, explainer, "http://www.f0519141.xsph.ru")

    assert prediction["label"] == "suspicious"
    assert prediction["probability"] >= 0.5
    assert len(prediction["reasons"]) == 3


@pytest.mark.parametrize("url", ["javascript:alert(1)", "ftp://example.com", "https:///missing-host"])
def test_api_rejects_non_http_urls(url):
    response = client.post("/predict", json={"url": url})

    assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
