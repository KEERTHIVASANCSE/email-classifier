from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR/ "Data" / "emails.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"




def main():
    df = pd.read_csv(DATA_PATH)
    required_columns = {"text", "spam"}

    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'spam' columns.")

    df = df.dropna(subset=["text", "spam"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates().reset_index(drop=True)

    if not set(df["spam"].unique()).issubset({0, 1}):
        raise ValueError("Spam labels must be 0 or 1.")

    X = df["text"]
    y = df["spam"].astype(int)

    if y.nunique() != 2:
        raise ValueError("Dataset must contain both classes: 0 and 1.")

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(y.value_counts())

    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Create the complete ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                max_features=10000,
            ),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ])

    # 5. Train
    model.fit(X_train, y_train)

    # 6. Evaluate on unseen test data
    predictions = model.predict(X_test)

    print("\nAccuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Spam", "Spam"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # 7. Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()


    # Im excited to share one of my recent machine learning projects: A spam email classification system built using Python and Scikit-learn.

