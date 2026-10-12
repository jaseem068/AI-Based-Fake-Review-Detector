
from pathlib import Path

import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "fake_review_model.joblib"

app = Flask(__name__)

if not MODEL_FILE.exists():
    raise RuntimeError(
        "Trained model not found. Run: python train_model.py"
    )

saved_model = joblib.load(MODEL_FILE)
model = saved_model["model"]


@app.route("/", methods=["GET", "POST"])
def index():
    review = ""
    result = None
    error = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if not review:
            error = "Please enter a review."
        elif len(review) > 5000:
            error = "Review must be 5,000 characters or fewer."
        else:
            prediction = int(model.predict([review])[0])
            probabilities = model.predict_proba([review])[0]

            classes = list(model.named_steps["classifier"].classes_)
            fake_index = classes.index(1)
            fake_probability = float(probabilities[fake_index])

            result = {
                "is_fake": prediction == 1,
                "label": (
                    "Likely Fake" if prediction == 1
                    else "Likely Genuine"
                ),
                "fake_probability": round(fake_probability * 100, 1),
                "genuine_probability": round(
                    (1 - fake_probability) * 100, 1
                ),
            }

    return render_template(
        "index.html",
        review=review,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)