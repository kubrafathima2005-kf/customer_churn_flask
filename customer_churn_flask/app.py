"""Flask web app for customer churn prediction."""
from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_pipeline.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

FORM_OPTIONS = {
    "gender": ["Female", "Male"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["Yes", "No", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "online_backup": ["Yes", "No", "No internet service"],
    "device_protection": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def form_to_dataframe(form):
    """Convert browser form data into the exact columns used during training."""
    data = {
        "gender": form.get("gender", "Female"),
        "SeniorCitizen": int(form.get("senior_citizen", 0)),
        "Partner": form.get("partner", "No"),
        "Dependents": form.get("dependents", "No"),
        "tenure": float(form.get("tenure", 12)),
        "PhoneService": form.get("phone_service", "Yes"),
        "MultipleLines": form.get("multiple_lines", "No"),
        "InternetService": form.get("internet_service", "DSL"),
        "OnlineSecurity": form.get("online_security", "No"),
        "OnlineBackup": form.get("online_backup", "No"),
        "DeviceProtection": form.get("device_protection", "No"),
        "TechSupport": form.get("tech_support", "No"),
        "StreamingTV": form.get("streaming_tv", "No"),
        "StreamingMovies": form.get("streaming_movies", "No"),
        "Contract": form.get("contract", "Month-to-month"),
        "PaperlessBilling": form.get("paperless_billing", "Yes"),
        "PaymentMethod": form.get("payment_method", "Electronic check"),
        "MonthlyCharges": float(form.get("monthly_charges", 70)),
        "TotalCharges": float(form.get("total_charges", 840)),
    }
    return pd.DataFrame([data])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", options=FORM_OPTIONS, metrics=metrics, values={})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        customer = form_to_dataframe(request.form)
        probability = float(model.predict_proba(customer)[0][1])
        prediction = int(probability >= 0.5)
        result = {
            "label": "Likely to churn" if prediction else "Likely to stay",
            "churn": bool(prediction),
            "probability": round(probability * 100, 1),
            "advice": (
                "Consider a retention offer, proactive support, or a longer-term contract."
                if prediction
                else "Maintain engagement and highlight the value of the current service."
            ),
        }
        return render_template("result.html", result=result, customer=customer.iloc[0].to_dict())
    except (TypeError, ValueError) as error:
        return render_template(
            "index.html",
            options=FORM_OPTIONS,
            metrics=metrics,
            values=request.form,
            error=f"Please check the numeric fields. Details: {error}",
        ), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
