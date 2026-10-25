"""
Training pipeline for the YouTube Sentiment Analyzer.

Run from the project root:

    python ml/train.py

Steps
-----
1. Load the labelled dataset         (data/training_data.csv)
2. Clean the text                    (ml/preprocess.py)
3. Train / test split (80% / 20%, stratified)
4. TF-IDF vectorization              (unigrams + bigrams)
5. Train a Logistic Regression model
6. Evaluate on the held-out test set (accuracy, precision, recall, F1, confusion matrix)
7. Save the artifacts with joblib    (models/*.pkl and models/metrics.json)

Every number written to metrics.json is calculated from the real test data.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.preprocess import clean_text  # noqa: E402  (needs the sys.path line above)

DATA_PATH = BASE_DIR / "data" / "training_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

LABELS = ["Positive", "Negative", "Neutral"]   # display order
RANDOM_STATE = 42
TEST_SIZE = 0.2

VECTORIZER_PARAMS = dict(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
MODEL_PARAMS = dict(C=4.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV, normalise labels and remove empty / duplicate rows."""
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("training_data.csv must contain 'text' and 'label' columns")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df = df[df["label"].isin(LABELS)]
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean", "label"]).reset_index(drop=True)
    if df["label"].nunique() < 3:
        raise ValueError("Training data must contain Positive, Negative and Neutral examples")
    return df


def train(save: bool = True, verbose: bool = True) -> dict:
    """Train the model, evaluate it and (optionally) save everything. Returns the metrics."""
    log = print if verbose else (lambda *a, **k: None)

    log("1/6  Loading dataset ...")
    df = load_dataset()
    log(f"     {len(df)} comments  ->  {df['label'].value_counts().to_dict()}")

    log("2/6  Splitting into train / test sets (80/20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )
    log(f"     train = {len(X_train)}, test = {len(X_test)}")

    log("3/6  TF-IDF vectorization (unigrams + bigrams) ...")
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)     # fit ONLY on training data
    X_test_vec = vectorizer.transform(X_test)
    log(f"     vocabulary size = {len(vectorizer.vocabulary_)}")

    log("4/6  Training Logistic Regression ...")
    model = LogisticRegression(**MODEL_PARAMS)
    model.fit(X_train_vec, y_train)

    log("5/6  Evaluating on the unseen test set ...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average="weighted", zero_division=0)
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    # 5-fold cross validation on the whole dataset gives a more stable estimate
    # than a single small test split.
    cv_pipeline = make_pipeline(TfidfVectorizer(**VECTORIZER_PARAMS), LogisticRegression(**MODEL_PARAMS))
    cv_scores = cross_val_score(
        cv_pipeline, df["clean"], df["label"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model_name": "Logistic Regression",
        "vectorizer_name": "TF-IDF (unigrams + bigrams)",
        "classes": LABELS,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "class_distribution": {k: int(v) for k, v in df["label"].value_counts().items()},
        "accuracy": float(accuracy),
        "precision": float(p_macro),
        "recall": float(r_macro),
        "f1_score": float(f_macro),
        "weighted": {"precision": float(p_w), "recall": float(r_w), "f1_score": float(f_w)},
        "per_class": {
            label: {
                "precision": float(p_cls[i]), "recall": float(r_cls[i]),
                "f1_score": float(f_cls[i]), "support": int(s_cls[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "cross_validation": {
            "folds": 5,
            "accuracy_mean": float(np.mean(cv_scores)),
            "accuracy_std": float(np.std(cv_scores)),
        },
        "hyperparameters": {
            "C": MODEL_PARAMS["C"],
            "class_weight": MODEL_PARAMS["class_weight"],
            "ngram_range": list(VECTORIZER_PARAMS["ngram_range"]),
        },
    }

    log(f"     Accuracy  : {accuracy:.4f}")
    log(f"     Precision : {p_macro:.4f}  (macro)")
    log(f"     Recall    : {r_macro:.4f}  (macro)")
    log(f"     F1 score  : {f_macro:.4f}  (macro)")
    log(f"     5-fold CV : {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    log("     Confusion matrix (rows = actual, columns = predicted):")
    log("                 " + "  ".join(f"{l[:8]:>8}" for l in LABELS))
    for label, row in zip(LABELS, cm):
        log(f"     {label:<12}" + "  ".join(f"{v:>8}" for v in row))

    if save:
        log("6/6  Saving artifacts ...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log(f"     saved {VECTORIZER_PATH.relative_to(BASE_DIR)}")
        log(f"     saved {MODEL_PATH.relative_to(BASE_DIR)}")
        log(f"     saved {METRICS_PATH.relative_to(BASE_DIR)}")
        log("Done.")
    return metrics


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train the YouTube sentiment model.").parse_args()
    try:
        train()
    except Exception as exc:          # friendly message instead of a long traceback
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
