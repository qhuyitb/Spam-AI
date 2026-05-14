"""
Full training pipeline: preprocess -> ML models -> BERT -> evaluate.
Run: python train.py [--skip-bert]
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import prepare
from ml_models import train_all
from bert_model import train_bert
from evaluate import evaluate_all

SKIP_BERT = "--skip-bert" in sys.argv


def main():
    print("=" * 60)
    print("Spam SMS Detection — Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess
    print("\n[1/4] Preprocessing data...")
    data = prepare(save=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train_raw = data["X_train_raw"]
    X_test_raw = data["X_test_raw"]

    # Step 2: ML models
    print("\n[2/4] Training ML models (NaiveBayes, SVM, LogisticRegression)...")
    ml_results = train_all(X_train, y_train, X_test, y_test)

    # Step 3: BERT
    if SKIP_BERT:
        print("\n[3/4] Skipping BERT (--skip-bert flag set)")
        bert_result = None
    else:
        print("\n[3/4] Fine-tuning BERT...")
        bert_result = train_bert(X_train_raw, y_train, X_test_raw, y_test)

    # Step 4: Evaluate
    print("\n[4/4] Generating benchmark comparison...")
    all_results = ml_results.copy()
    if bert_result:
        all_results["BERT"] = bert_result
    evaluate_all(ml_results, bert_result or {}, y_test)

    print("\nDone. Check models/ directory for saved models and plots.")


if __name__ == "__main__":
    main()
