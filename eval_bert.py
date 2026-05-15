"""
Load BERT from HuggingFace, evaluate on test set, merge into metrics.json, regenerate charts.
Run: python eval_bert.py
"""

import sys
import os
import json
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bert_model import eval_bert_from_hf
from evaluate import print_summary_table, plot_roc_curves, plot_confusion_matrices, plot_metrics_bar, save_metrics

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "data", "processed.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.json")


def main():
    # Load preprocessed test set
    print("Loading processed data...")
    data = joblib.load(PROCESSED_PATH)
    X_test_raw = data["X_test_raw"]
    y_test = data["y_test"]

    # Evaluate BERT from HF
    print("\nEvaluating BERT from HuggingFace...")
    bert_result = eval_bert_from_hf(X_test_raw, y_test)

    # Load existing ML metrics and rebuild full results dict for charts
    with open(METRICS_PATH) as f:
        saved = json.load(f)

    # Merge BERT into saved metrics and re-save
    saved["BERT"] = {
        "accuracy": round(bert_result["accuracy"], 4),
        "precision": round(bert_result["precision"], 4),
        "recall": round(bert_result["recall"], 4),
        "f1": round(bert_result["f1"], 4),
        "roc_auc": round(bert_result["roc_auc"], 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nUpdated {METRICS_PATH}")

    # Build all_results for chart functions (need y_proba and confusion_matrix)
    # Load ML model predictions from pkl for ROC + confusion matrix plots
    import joblib as jl
    from sklearn.metrics import confusion_matrix as cm_fn, roc_auc_score
    import numpy as np

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    ml_names = ["NaiveBayes", "SVM", "LogisticRegression"]

    all_results = {}
    for name in ml_names:
        model = jl.load(os.path.join(models_dir, f"{name}.pkl"))
        # Use clean text for ML models
        X_test_clean = data["X_test"]
        y_pred = model.predict(X_test_clean)
        y_proba = model.predict_proba(X_test_clean)[:, 1]
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
        all_results[name] = {
            "accuracy": report["accuracy"],
            "precision": report["spam"]["precision"],
            "recall": report["spam"]["recall"],
            "f1": report["spam"]["f1-score"],
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_proba": y_proba,
            "confusion_matrix": cm_fn(y_test, y_pred),
        }

    all_results["BERT"] = bert_result

    # Regenerate all charts
    print("\nRegenerating charts...")
    print_summary_table(all_results)
    plot_roc_curves(all_results, y_test)
    plot_confusion_matrices(all_results)
    plot_metrics_bar(all_results)

    print("\nDone. metrics.json and charts updated with BERT.")


if __name__ == "__main__":
    main()
