import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from services.inference import load_model, predict

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Intrusion Detection System",
    layout="wide"
)

st.title("AI-Driven Intrusion Detection System")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

ATTACK_LABELS = [
    "BENIGN", "Bot", "BruteForce", "DDoS", "DoS_Hulk",
    "DoS_Slowloris", "DoS_Slowhttptest", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan",
    "SSH-Patator", "Web_Attack_BruteForce",
    "Web_Attack_XSS", "Web_Attack_SQLInjection"
]

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload CICIDS2017 CSV",
    type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] != 78:
        st.error(f"Expected 78 features, found {numeric_df.shape[1]}")
        st.stop()

    predictions = []
    confidences = []

    with st.spinner("Running intrusion detection..."):
        for _, row in numeric_df.iterrows():
            result = predict(model, row.values)
            predictions.append(ATTACK_LABELS[result["prediction"]])
            confidences.append(result["confidence"])

    results_df = pd.DataFrame({
        "Prediction": predictions,
        "Confidence": confidences
    })

    st.subheader("Predictions")
    st.dataframe(results_df)

    summary = results_df["Prediction"].value_counts().reset_index()
    summary.columns = ["Attack Type", "Count"]

    st.subheader("Attack Distribution")
    st.bar_chart(summary.set_index("Attack Type")["Count"])

    fig, ax = plt.subplots()
    ax.pie(summary["Count"], labels=summary["Attack Type"], autopct="%1.1f%%")
    st.pyplot(fig)

    os.makedirs("logs", exist_ok=True)
    with open("logs/alerts.log", "a") as f:
        for p, c in zip(predictions, confidences):
            f.write(f"{datetime.datetime.now()},{p},{c:.4f}\n")



