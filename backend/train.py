"""Train, evaluate, and save the URL classifiers plus their SHAP plot."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract_url_features

SEED = 42
TEST_SIZE = 0.2
SHAP_BACKGROUND_SIZE = 100
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BACKEND_DIR / "data" / "phiusiil.zip"
MODEL_PATH = BACKEND_DIR / "phishing_model.joblib"
RESULTS_DIR = BACKEND_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.csv"
PLOT_PATH = RESULTS_DIR / "shap_summary.png"
METRIC_COLUMNS = [
    "model",
    "split",
    "n_test",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
]


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read the UCI ZIP/CSV; keep only its raw URL, label, and exact hostname."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the UCI archive as "
            "backend/data/phiusiil.zip; instructions are in README.md."
        )

    compression: str | None = "zip" if zipfile.is_zipfile(path) else None
    raw = pd.read_csv(path, compression=compression, usecols=["URL", "label"])
    raw = raw.dropna(subset=["URL", "label"]).copy()
    raw["label"] = pd.to_numeric(raw["label"], errors="coerce")
    raw = raw.dropna(subset=["label"])
    raw = raw[raw["label"].isin([0, 1])]

    feature_rows: list[dict[str, float | int]] = []
    labels: list[int] = []
    hostnames: list[str] = []
    for url, label in zip(raw["URL"], raw["label"], strict=True):
        try:
            feature_rows.append(extract_url_features(str(url)))
        except ValueError:
            continue
        hostname = urlsplit(str(url)).hostname
        if hostname is None:
            continue
        labels.append(int(label))
        canonical_hostname = hostname.lower().rstrip(".")
        if canonical_hostname.startswith("www."):
            canonical_hostname = canonical_hostname[4:]
        hostnames.append(canonical_hostname)

    if not feature_rows:
        raise ValueError(f"No valid HTTP(S) URL rows were found in {path}.")

    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES, dtype=float)
    y = pd.Series(labels, name="label", dtype=int)
    groups = pd.Series(hostnames, name="hostname", dtype=str)
    return X, y, groups


def make_models() -> dict[str, object]:
    """Create new deterministic models for one evaluation split."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def score_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict[str, float | int]:
    """Calculate metrics with phishing (dataset label 0) as the positive class."""
    phishing_probability = model.predict_proba(X)[:, list(model.classes_).index(0)]
    actual = (y.to_numpy() == 0).astype(int)
    predicted = (phishing_probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, phishing_probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def phishing_shap_values(explainer: shap.LinearExplainer, model, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for phishing; LinearExplainer reports class-1 logits."""
    scaled = model.named_steps["standardscaler"].transform(X)
    values = explainer.shap_values(scaled)
    if isinstance(values, list):
        values = values[0]
    if values.ndim == 3:
        values = values[:, :, 1]
    return -np.asarray(values)


def write_shap_plot(model, X_background: pd.DataFrame, X_test: pd.DataFrame) -> shap.LinearExplainer:
    """Write a global summary plot for the user-facing Logistic Regression."""
    import matplotlib.pyplot as plt

    scaler = model.named_steps["standardscaler"]
    background_sample = X_background.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_background)),
        random_state=SEED,
    )
    background = scaler.transform(background_sample)
    sample = X_test.sample(n=min(1500, len(X_test)), random_state=SEED)
    sample_scaled = scaler.transform(sample)
    explainer = shap.LinearExplainer(model.named_steps["logisticregression"], background)
    values = -np.asarray(explainer.shap_values(sample_scaled))
    if values.ndim == 3:
        values = values[:, :, 1]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        values,
        sample_scaled,
        feature_names=FEATURE_NAMES,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    return explainer


def train(data_path: Path = DEFAULT_DATA) -> list[dict[str, str | float | int]]:
    """Fit all models on both splits; save the random-split app bundle and reports."""
    X, y, hostnames = load_dataset(data_path)
    random_indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        random_indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    grouped_train, grouped_test = next(
        GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(
            X, y, hostnames
        )
    )
    if set(hostnames.iloc[grouped_train]).intersection(hostnames.iloc[grouped_test]):
        raise RuntimeError("Hostname overlap found between grouped train and test.")

    random_models = make_models()
    rows: list[dict[str, str | float | int]] = []
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    for name, model in random_models.items():
        model.fit(X_train, y_train)
        rows.append(
            {
                "model": name,
                "split": "Stratified random",
                "n_test": len(test_indices),
                **score_model(model, X_test, y_test),
            }
        )

    background = random_models["Logistic Regression"].named_steps["standardscaler"].transform(
        X_train.sample(n=min(SHAP_BACKGROUND_SIZE, len(X_train)), random_state=SEED)
    )
    bundle = {
        "models": random_models,
        "default_model": "Logistic Regression",
        "feature_names": FEATURE_NAMES,
        "shap_background": background,
        "seed": SEED,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    X_domain_train, X_domain_test = X.iloc[grouped_train], X.iloc[grouped_test]
    y_domain_train, y_domain_test = y.iloc[grouped_train], y.iloc[grouped_test]
    for name, model in make_models().items():
        model.fit(X_domain_train, y_domain_train)
        rows.append(
            {
                "model": name,
                "split": "Hostname-held-out",
                "n_test": len(grouped_test),
                **score_model(model, X_domain_test, y_domain_test),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_shap_plot(random_models["Logistic Regression"], X_train, X_test)
    metadata = {
        "seed": SEED,
        "rows": len(X),
        "phishing_rows": int((y == 0).sum()),
        "legitimate_rows": int((y == 1).sum()),
        "random_train_rows": len(train_indices),
        "random_test_rows": len(test_indices),
        "hostname_train_rows": len(grouped_train),
        "hostname_test_rows": len(grouped_test),
        "hostname_overlap": 0,
        "hostname_test_phishing_rate": float((y_domain_test == 0).mean()),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved SHAP plot: {PLOT_PATH}")
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        key: "{:.4f}".format for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }))
    print(f"Hostname overlap: {metadata['hostname_overlap']}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train(parser.parse_args().data)
