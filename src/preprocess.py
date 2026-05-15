"""
Data loading, cleaning, feature engineering, and train/test split.
"""

import os
import re
import string
import pandas as pd
import numpy as np
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed.pkl")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def download_nltk_resources():
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        nltk.download(resource, quiet=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df.drop_duplicates(subset="text", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} messages | spam={df['label'].sum()} ham={(df['label']==0).sum()}")
    return df


def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                       # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["char_count"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["digit_count"] = df["text"].apply(lambda x: sum(c.isdigit() for c in x))
    df["upper_count"] = df["text"].apply(lambda x: sum(c.isupper() for c in x))
    df["exclamation_count"] = df["text"].apply(lambda x: x.count("!"))
    df["currency_count"] = df["text"].apply(lambda x: sum(c in "$£€" for c in x))
    return df


def plot_eda(df: pd.DataFrame, save_dir: str = PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # Class distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df["label"].value_counts().rename({0: "ham", 1: "spam"}).plot(
        kind="bar", ax=axes[0], color=["steelblue", "tomato"], rot=0
    )
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")

    # Message length distribution
    for label, color, name in [(0, "steelblue", "ham"), (1, "tomato", "spam")]:
        axes[1].hist(
            df[df["label"] == label]["char_count"],
            bins=50, alpha=0.6, color=color, label=name
        )
    axes[1].set_title("Message Length Distribution")
    axes[1].set_xlabel("Character count")
    axes[1].legend()

    # Word count distribution
    for label, color, name in [(0, "steelblue", "ham"), (1, "tomato", "spam")]:
        axes[2].hist(
            df[df["label"] == label]["word_count"],
            bins=40, alpha=0.6, color=color, label=name
        )
    axes[2].set_title("Word Count Distribution")
    axes[2].set_xlabel("Word count")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda.png"), dpi=120)
    plt.close()
    print(f"EDA plot saved to {save_dir}/eda.png")

    # Feature correlation heatmap
    feature_cols = ["char_count", "word_count", "digit_count", "upper_count",
                    "exclamation_count", "currency_count", "label"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[feature_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation.png"), dpi=120)
    plt.close()


def prepare(save: bool = True):
    download_nltk_resources()
    df = load_data()
    df = extract_features(df)
    plot_eda(df)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(lambda x: clean_text(x, lemmatizer, stop_words))

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    # Also split raw text for BERT (no cleaning)
    X_train_raw, X_test_raw, _, _ = train_test_split(
        df["text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    data = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "df": df,
    }

    if save:
        joblib.dump(data, PROCESSED_PATH)
        print(f"Processed data saved to {PROCESSED_PATH}")

    return data


if __name__ == "__main__":
    prepare()
