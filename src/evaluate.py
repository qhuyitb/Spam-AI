"""
Benchmark comparison: all models side by side — metrics table, ROC curves, confusion matrices.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def plot_roc_curves(all_results: dict, y_test):
    plt.figure(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "green", "crimson"]

    for (name, result), color in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"ROC curves saved to {path}")


def plot_confusion_matrices(all_results: dict):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, all_results.items()):
        cm = result["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"]
        )
        ax.set_title(f"{name}\nAcc={result['accuracy']:.3f} F1={result['f1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Confusion matrices saved to {path}")


def plot_metrics_bar(all_results: dict):
    models = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = [all_results[m][metric] for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric.upper())
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Metrics bar chart saved to {path}")


def print_summary_table(all_results: dict):
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        print(
            f"{name:<22} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>9.4f}"
        )
    print("=" * len(header))


def save_metrics(all_results: dict):
    serializable = {}
    for name, r in all_results.items():
        serializable[name] = {
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1"], 4),
            "roc_auc": round(r["roc_auc"], 4),
        }
    with open(METRICS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")


def plot_bert_history(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title("BERT Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_f1"], "g-o", label="Val F1 (spam)")
    axes[1].set_title("BERT Validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "bert_history.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"BERT history saved to {path}")


def evaluate_all(ml_results: dict, bert_result: dict, y_test):
    all_results = {**ml_results}
    if bert_result and "y_proba" in bert_result:
        all_results["BERT"] = bert_result
    print_summary_table(all_results)
    plot_roc_curves(all_results, y_test)
    plot_confusion_matrices(all_results)
    plot_metrics_bar(all_results)
    save_metrics(all_results)

    if "history" in bert_result:
        plot_bert_history(bert_result["history"])

    return all_results
