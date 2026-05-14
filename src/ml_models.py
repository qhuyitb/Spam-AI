"""
ML models: TF-IDF + Naive Bayes, SVM (LinearSVC), Logistic Regression.
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from typing import Tuple

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TFIDF = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    min_df=2,
)

_MODELS = {
    "NaiveBayes": {
        "pipeline": Pipeline([("tfidf", TFIDF), ("clf", MultinomialNB())]),
        "params": {"clf__alpha": [0.1, 0.5, 1.0]},
    },
    "SVM": {
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000))),
        ]),
        "params": {"clf__estimator__C": [0.1, 1.0, 5.0]},
    },
    "LogisticRegression": {
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")),
        ]),
        "params": {"clf__C": [0.1, 1.0, 10.0]},
    },
}


def train_all(X_train, y_train, X_test, y_test) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for name, config in _MODELS.items():
        print(f"\nTraining {name}...")
        gs = GridSearchCV(
            config["pipeline"], config["params"],
            cv=5, scoring="f1", n_jobs=-1, verbose=0
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"  Best params: {gs.best_params_}")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
        print(f"  ROC-AUC: {roc_auc:.4f}")

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(best, model_path)
        print(f"  Saved to {model_path}")

        results[name] = {
            "model": best,
            "report": report,
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": report["accuracy"],
            "precision": report["spam"]["precision"],
            "recall": report["spam"]["recall"],
            "f1": report["spam"]["f1-score"],
        }

    return results


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


def predict(model, text: str) -> Tuple[str, float]:
    proba = model.predict_proba([text])[0][1]
    label = "spam" if proba >= 0.5 else "ham"
    return label, float(proba)
