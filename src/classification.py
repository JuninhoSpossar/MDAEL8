"""Benchmark de modelos de classificação."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config.settings import (
    BENCHMARK_REPORT,
    CV_FOLDS,
    DATA_ENCODED,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    ensure_directories,
)
from src.utils import plot_confusion_matrix, show_dataframe_info

MODELS = {
    "DecisionTree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, max_depth=3, min_samples_leaf=20, min_samples_split=20,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear", random_state=RANDOM_STATE, probability=True),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=RANDOM_STATE),
}


def _evaluate_model(name: str, model, x_scaled: np.ndarray, y: np.ndarray) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        "modelo": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "cv_accuracy_mean": float(cross_val_score(model, x_scaled, y, cv=CV_FOLDS, n_jobs=-1).mean()),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": cm.tolist(),
        "roc_auc": float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
        if hasattr(model, "predict_proba") else None,
    }

    print(f"\n{name}: accuracy={results['accuracy']:.4f}, f1={results['f1']:.4f}")
    plot_confusion_matrix(cm, ["<=50K", ">50K"], f"Matriz de Confusão — {name}", save_as=f"confusion_matrix_{name.lower()}.png")
    return results


def run_benchmark(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        if not DATA_ENCODED.exists():
            raise FileNotFoundError(f"Execute feature engineering primeiro: {DATA_ENCODED}")
        df = pd.read_csv(DATA_ENCODED)

    show_dataframe_info(df, "DataFrame para classificação")
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    x_scaled = StandardScaler().fit_transform(df[features].values)
    y = df[TARGET].values

    benchmark = {
        "config": {"features": features, "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "cv_folds": CV_FOLDS},
        "modelos": [_evaluate_model(name, model, x_scaled, y) for name, model in MODELS.items()],
    }

    ensure_directories()
    with open(BENCHMARK_REPORT, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    print(f"Benchmark salvo em: {BENCHMARK_REPORT}")
    return benchmark


def run(df: pd.DataFrame | None = None) -> dict:
    return run_benchmark(df)
