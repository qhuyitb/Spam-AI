"""
BERT fine-tuning for spam detection using HuggingFace Transformers.
Uses bert-base-uncased with raw (uncleaned) text.
"""

import os
import json
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BERT_DIR = os.path.join(MODELS_DIR, "bert")
MODEL_NAME = "bert-base-uncased"
HF_MODEL_ID = "qhuyisthebest/bert-spam"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_bert(X_train, y_train, X_test, y_test) -> dict:
    os.makedirs(BERT_DIR, exist_ok=True)
    print(f"\nTraining BERT on {DEVICE}...")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = SMSDataset(X_train, y_train, tokenizer)
    test_dataset = SMSDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Class weights to handle imbalance
    n_ham = (np.array(list(y_train)) == 0).sum()
    n_spam = (np.array(list(y_train)) == 1).sum()
    weight = torch.tensor([1.0, n_ham / n_spam], dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Eval ---
        val_results = _evaluate(model, test_loader, loss_fn)
        report = classification_report(
            val_results["labels"], val_results["preds"],
            target_names=["ham", "spam"], output_dict=True
        )
        f1 = report["spam"]["f1-score"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_results["loss"])
        history["val_f1"].append(f1)

        print(f"  Epoch {epoch+1} | train_loss={avg_train_loss:.4f} | val_loss={val_results['loss']:.4f} | spam_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(BERT_DIR)
            tokenizer.save_pretrained(BERT_DIR)
            print(f"  Saved best model (F1={best_f1:.4f})")

    # Final evaluation with best model
    model = BertForSequenceClassification.from_pretrained(BERT_DIR)
    model.to(DEVICE)
    final = _evaluate(model, test_loader, loss_fn)
    final_report = classification_report(
        final["labels"], final["preds"],
        target_names=["ham", "spam"], output_dict=True
    )
    roc_auc = roc_auc_score(final["labels"], final["probas"])
    print("\nBERT Final Evaluation:")
    print(classification_report(final["labels"], final["preds"], target_names=["ham", "spam"]))
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Save training history
    with open(os.path.join(BERT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    from sklearn.metrics import confusion_matrix
    return {
        "report": final_report,
        "confusion_matrix": confusion_matrix(final["labels"], final["preds"]),
        "roc_auc": roc_auc,
        "y_pred": np.array(final["preds"]),
        "y_proba": np.array(final["probas"]),
        "accuracy": final_report["accuracy"],
        "precision": final_report["spam"]["precision"],
        "recall": final_report["spam"]["recall"],
        "f1": final_report["spam"]["f1-score"],
        "history": history,
    }


def _evaluate(model, loader, loss_fn) -> dict:
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probas = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probs)

    return {
        "loss": total_loss / len(loader),
        "preds": all_preds,
        "labels": all_labels,
        "probas": all_probas,
    }


def load_bert():
    tokenizer = BertTokenizerFast.from_pretrained(HF_MODEL_ID)
    model = BertForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def eval_bert_from_hf(X_test, y_test) -> dict:
    """Load BERT from HuggingFace and evaluate on test set."""
    print(f"Loading BERT from HuggingFace ({HF_MODEL_ID})...")
    model, tokenizer = load_bert()

    dataset = SMSDataset(X_test, y_test, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    loss_fn = torch.nn.CrossEntropyLoss()

    result = _evaluate(model, loader, loss_fn)

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    report = classification_report(
        result["labels"], result["preds"],
        target_names=["ham", "spam"], output_dict=True
    )
    roc_auc = roc_auc_score(result["labels"], result["probas"])

    print(classification_report(result["labels"], result["preds"], target_names=["ham", "spam"]))
    print(f"  ROC-AUC: {roc_auc:.4f}")

    return {
        "report": report,
        "confusion_matrix": confusion_matrix(result["labels"], result["preds"]),
        "roc_auc": roc_auc,
        "y_pred": np.array(result["preds"]),
        "y_proba": np.array(result["probas"]),
        "accuracy": report["accuracy"],
        "precision": report["spam"]["precision"],
        "recall": report["spam"]["recall"],
        "f1": report["spam"]["f1-score"],
    }


def predict_bert(model, tokenizer, text: str) -> Tuple[str, float]:
    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    proba = torch.softmax(logits, dim=1)[0][1].item()
    return ("spam" if proba >= 0.5 else "ham"), proba
