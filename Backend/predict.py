
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run train.py first.")
        return

    model = joblib.load(MODEL_PATH)

    print("=== Spam Email Classifier ===")
    print("Enter an email below. Type 'exit' to quit.\n")

    while True:
        email = input("Email text: ").strip()

        if email.lower() == "exit":
            print("Goodbye!")
            break

        if not email:
            print("Please enter some email text.\n")
            continue

        prediction = int(model.predict([email])[0])
        probabilities = model.predict_proba([email])[0]
        spam_probability = probabilities[
            list(model.classes_).index(1)
        ]

        if prediction == 1:
            print("Prediction: SPAM")
        else:
            print("Prediction: NOT SPAM")

        print(
            f"Estimated spam probability: "
            f"{spam_probability:.2%}\n"
        )


if __name__ == "__main__":
    main()
