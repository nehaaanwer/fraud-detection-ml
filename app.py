from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="Fraud Detection API")

# ----------------------------
# Load Saved Artifacts
# ----------------------------
try:
    model = joblib.load("best_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")
    best_model_name = joblib.load("best_model_name.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")


# ----------------------------
# Input Schema (IMPORTANT FIX)
# ----------------------------
class Transaction(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running successfully"}


# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict(data: Transaction):

    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])

        # ----------------------------
        # Feature Engineering
        # ----------------------------
        df['amount_to_oldbalanceOrg'] = (
            df['amount'] / (df['oldbalanceOrg'] + 1)
        )

        high_risk_types = ['TRANSFER', 'CASH_OUT']
        df['isHighRiskType'] = df['type'].apply(
            lambda x: 1 if x in high_risk_types else 0
        )

        df['balanceOrig_error'] = (
            df['oldbalanceOrg']
            - df['newbalanceOrig']
            - df['amount']
        )

        df['balanceDest_error'] = (
            df['newbalanceDest']
            - df['oldbalanceDest']
            - df['amount']
        )

        # Drop leakage columns
        df.drop(
            ['newbalanceOrig', 'newbalanceDest'],
            axis=1,
            inplace=True,
            errors='ignore'
        )

        # One-hot encoding
        df = pd.get_dummies(df, drop_first=True)

        # Align with training columns
        df = df.reindex(columns=columns, fill_value=0)

        # ----------------------------
        # Scaling (Only if required)
        # ----------------------------
        if best_model_name in ["Logistic Regression", "SVM"]:
            df_final = scaler.transform(df)
        else:
            df_final = df.values

        # ----------------------------
        # Prediction
        # ----------------------------
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df_final)[0][1]
        else:
            # Fallback for models without predict_proba
            decision = model.decision_function(df_final)[0]
            prob = 1 / (1 + np.exp(-decision))  # convert to probability

        prediction = int(prob >= 0.5)

        return {
            "Fraud_Probability": float(round(prob, 4)),
            "Prediction": "Fraud" if prediction == 1 else "Not Fraud",
            "Model_Used": best_model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
