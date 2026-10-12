
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "reviews.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "fake_review_model.joblib"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Create data/reviews.csv with review and label columns."
        )

    df = pd.read_csv(DATA_FILE)

    if not {"review", "label"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named review and label."
        )

    df = df[["review", "label"]].dropna()
    df["review"] = df["review"].astype(str).str.strip()

    label_map = {
        "genuine": 0,
        "real": 0,
        "fake": 1,
        "deceptive": 1,
    }

    df["label"] = df["label"].apply(
        lambda value: label_map.get(
            str(value).strip().lower(), value
        )
    )

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin([0, 1])]
    df = df[df["review"] != ""]

    if df["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both genuine and fake labels."
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError(
            "Add at least two examples for each class."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["label"].astype(int),
        test_size=0.25,
        random_state=42,
        stratify=df["label"].astype(int),
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nModel evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Genuine", "Fake"],
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "demo_model": True,
            "training_rows": len(X_train),
        },
        MODEL_FILE,
    )

    print(f"Model saved successfully: {MODEL_FILE}")
    print(
        "Note: The supplied sample dataset is illustrative. "
        "Its evaluation scores do not demonstrate real-world accuracy."
    )


if __name__ == "__main__":
    main()