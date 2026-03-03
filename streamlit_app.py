import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)
st.title("💳 Fraud Detection System Dashboard")
st.header("📊 Model Performance Comparison")
comparison_df = pd.read_csv("model_comparison.csv")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance Table")
    st.dataframe(comparison_df, use_container_width=True)

with col2:
    st.subheader("PR-AUC Comparison")

    fig = px.bar(
        comparison_df,
        x="Model",
        y="Val_PR_AUC",
        text="Val_PR_AUC",
        title="Precision-Recall AUC by Model"
    )

    st.plotly_chart(fig, use_container_width=True)

best_model = comparison_df.iloc[0]["Model"]
st.success(f"🏆 Best Model Selected: {best_model}")

st.divider()
st.header("🧪 Try Sample Transactions")

col1, col2 = st.columns(2)

with col1:
    if st.button("Normal Transaction"):
        sample_data = {
            "type": "PAYMENT",
            "amount": 500,
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 4500,
            "oldbalanceDest": 2000,
            "newbalanceDest": 2500
        }
        st.session_state["sample"] = sample_data

with col2:
    if st.button("Fraud Transaction 🚨"):
        sample_data = {
            "type": "TRANSFER",
            "amount": 1000000,
            "oldbalanceOrg": 1000000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 0
        }
        st.session_state["sample"] = sample_data

        st.divider()
st.header("🔎 Transaction Prediction")

sample = st.session_state.get("sample", {})

type_ = st.selectbox(
    "Transaction Type",
    ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"],
    index=["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"].index(
        sample.get("type","PAYMENT")
    )
)

amount = st.number_input(
    "Amount",
    value=float(sample.get("amount", 0))
)

oldbalanceOrg = st.number_input(
    "Old Balance Origin",
    value=float(sample.get("oldbalanceOrg", 0))
)

newbalanceOrig = st.number_input(
    "New Balance Origin",
    value=float(sample.get("newbalanceOrig", 0))
)

oldbalanceDest = st.number_input(
    "Old Balance Destination",
    value=float(sample.get("oldbalanceDest", 0))
)

newbalanceDest = st.number_input(
    "New Balance Destination",
    value=float(sample.get("newbalanceDest", 0))
)

if st.button("Predict Fraud"):

    input_data = {
        "type": type_,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    result = response.json()

    prob = result["Fraud_Probability"]
    pred = result["Prediction"]

    st.subheader("Prediction Result")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Fraud Probability"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "red"},
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    if pred == "Fraud":
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("Legitimate Transaction")