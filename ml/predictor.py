"""
Loads the trained artifacts (or trains them if they are missing / incompatible)
and exposes a small, safe prediction API for the web application.
"""

import json
import logging
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
import sklearn

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.preprocess import clean_text  # noqa: E402
from ml import train as trainer  # noqa: E402

logger = logging.getLogger("yt-sentiment.predictor")

MAX_CHARS_PER_COMMENT = 2000   # longer comments are truncated before prediction

# Friendly names for the special tokens produced by preprocessing
_FRIENDLY_TERMS = {
    "emojipositive": "positive emoji",
    "emojinegative": "negative emoji",
    "emojineutral": "neutral emoji",
    "questionmark": "question mark",
    "exclamationmark": "exclamation mark",
    "timestamp": "timestamp",
    "yearnum": "year",
}
_UNINFORMATIVE_TOKENS = {"timestamp", "yearnum", "questionmark", "exclamationmark"}


class ModelNotReadyError(RuntimeError):
    """Raised when the model could not be loaded or trained."""


class SentimentPredictor:
    """Wraps the TF-IDF vectorizer and the Logistic Regression model."""

    def __init__(self, vectorizer, model, metrics: dict):
        self.vectorizer = vectorizer
        self.model = model
        self.metrics = metrics
        self.classes = list(model.classes_)
        self._feature_names = np.array(vectorizer.get_feature_names_out())

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load_or_train(cls) -> "SentimentPredictor":
        """Load saved artifacts; if that fails for any reason, retrain from the CSV."""
        try:
            return cls._load()
        except Exception as exc:      # missing files, corrupt pickle, version mismatch ...
            logger.warning("Could not load saved model (%s). Training a new one ...", exc)
        try:
            trainer.train(save=True, verbose=False)
            return cls._load()
        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            raise ModelNotReadyError(
                "The sentiment model is not available. Run 'python ml/train.py' and restart the server."
            ) from exc

    @classmethod
    # def _load(cls) -> "SentimentPredictor":
    #     for path in (trainer.VECTORIZER_PATH, trainer.MODEL_PATH, trainer.METRICS_PATH):
    #         if not path.exists():
    #             raise FileNotFoundError(path.name)
    #     metrics = json.loads(trainer.METRICS_PATH.read_text(encoding="utf-8"))
    #     # Pickles are only guaranteed to load with the scikit-learn version that wrote them.
    #     if metrics.get("sklearn_version") != sklearn.__version__:
    #         raise RuntimeError(
    #             f"saved with scikit-learn {metrics.get('sklearn_version')}, "
    #             f"installed is {sklearn.__version__}"
    #         )
    #     vectorizer = joblib.load(trainer.VECTORIZER_PATH)
    #     model = joblib.load(trainer.MODEL_PATH)
    #     return cls(vectorizer, model, metrics)

    @classmethod
    def _load(cls):
        if not trainer.VECTORIZER_PATH.exists():
            raise FileNotFoundError("vectorizer.pkl not found")

        if not trainer.MODEL_PATH.exists():
            raise FileNotFoundError("sentiment_model.pkl not found")

        # Metrics are optional for prediction
        metrics = {}

        if trainer.METRICS_PATH.exists():
            metrics = json.loads(
                trainer.METRICS_PATH.read_text(encoding="utf-8")
            )

        vectorizer = joblib.load(trainer.VECTORIZER_PATH)
        model = joblib.load(trainer.MODEL_PATH)

        return cls(vectorizer, model, metrics)


    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, texts: list) -> list:
        """Predict sentiment + class probabilities for a list of raw comments."""
        if not texts:
            return []
        cleaned = [clean_text(str(t)[:MAX_CHARS_PER_COMMENT]) for t in texts]
        matrix = self.vectorizer.transform(cleaned)
        probabilities = self.model.predict_proba(matrix)       # real probabilities
        results = []
        for row, nnz in zip(probabilities, np.diff(matrix.indptr)):
            best = int(np.argmax(row))
            results.append({
                "sentiment": self.classes[best],
                "confidence": round(float(row[best]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, row)},
                "known_terms": int(nnz),
            })
        return results

    def analyze_single(self, text: str) -> dict:
        """Predict one comment and add a simple, honest keyword-based hint."""
        text = str(text)[:MAX_CHARS_PER_COMMENT]
        result = self.predict([text])[0]
        key_terms = self._key_terms(text, result["sentiment"])
        result["key_terms"] = key_terms
        result["explanation"] = self._explain(result, key_terms)
        result["text"] = text
        return result

    # ------------------------------------------------------------------ #
    # Simple model-based insight
    # ------------------------------------------------------------------ #
    def _key_terms(self, text: str, sentiment: str, top_n: int = 4) -> list:
        """Words in the comment that pushed the model most strongly towards its prediction."""
        matrix = self.vectorizer.transform([clean_text(text)])
        if matrix.nnz == 0:
            return []
        class_index = self.classes.index(sentiment)
        coefficients = self.model.coef_[class_index]
        # Logistic regression is linear: contribution = coefficient x tf-idf weight
        indices = matrix.indices
        contributions = coefficients[indices] * matrix.data
        order = np.argsort(contributions)[::-1]
        terms = []
        for i in order:
            if contributions[i] <= 0:
                break
            words = str(self._feature_names[indices[i]]).split()
            if all(w in _UNINFORMATIVE_TOKENS for w in words):
                continue                      # a bare "timestamp" / "year" tells the reader nothing
            friendly = " ".join(_FRIENDLY_TERMS.get(w, w) for w in words)
            if friendly not in terms:
                terms.append(friendly)
            if len(terms) >= top_n:
                break
        return terms

    @staticmethod
    def _explain(result: dict, key_terms: list) -> str:
        sentiment = result["sentiment"].lower()
        confidence = result["confidence"]

        if result["known_terms"] == 0:
            return ("None of the words in this comment were seen during training, so the "
                    "prediction is not reliable.")

        if sentiment == "neutral":
            if confidence >= 0.6:
                base = ("The comment looks factual, conversational or like a question, "
                        "without strong opinion words.")
            else:
                base = "The comment has no strong positive or negative wording."
        elif confidence >= 0.8:
            base = f"The comment contains strongly {sentiment} language."
        elif confidence >= 0.55:
            base = f"The comment leans {sentiment}."
        else:
            base = (f"The model only weakly leans {sentiment}; the wording is mixed or "
                    "uses words it has rarely seen.")

        if key_terms:
            base += " Terms that pushed the model towards this label: " + ", ".join(key_terms) + "."
        return base + " (This is a simple model-based hint, not a full explanation.)"

    # ------------------------------------------------------------------ #
    # Model information for the dashboard
    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        m = self.metrics
        return {
            "model": m.get("model_name", "Logistic Regression"),
            "vectorizer": m.get("vectorizer_name", "TF-IDF"),
            "classes": m.get("classes", self.classes),
            "train_samples": m.get("train_samples"),
            "test_samples": m.get("test_samples"),
            "total_samples": m.get("total_samples"),
            "vocabulary_size": m.get("vocabulary_size"),
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1_score": m.get("f1_score"),
            "per_class": m.get("per_class"),
            "confusion_matrix": m.get("confusion_matrix"),
            "cross_validation": m.get("cross_validation"),
            "class_distribution": m.get("class_distribution"),
            "hyperparameters": m.get("hyperparameters"),
            "trained_at": m.get("trained_at"),
        }


# A lock prevents two threads from training at the same moment on first start-up.
_load_lock = threading.Lock()
_predictor = None


def get_predictor() -> SentimentPredictor:
    global _predictor
    with _load_lock:
        if _predictor is None:
            _predictor = SentimentPredictor.load_or_train()
        return _predictor
