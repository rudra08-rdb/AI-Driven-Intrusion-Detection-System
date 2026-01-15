import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

from services.inference import load_model, predict

st.set_page_config("AI Intrusion Detection System", layout="wide")

st.title("AI-Driven Intrusion Detection System (CNN + LSTM)")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

ATTACK_LABELS = [
    "BENIGN","Bot","BruteForce","DDoS","DoS_Hulk",
    "DoS_Slowloris","DoS_Slowhttptest","FTP-Patator",
    "Heartbleed","Infiltration","PortScan",
    "SSH-Patator","Web_Attack_BruteForce",
    "Web_Attack_XSS","Web_Attack_SQLInjection"
]

uploaded = st.file_uploader("Upload CICIDS2017 CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] != 78:
        st.error("Invalid feature count")
    else:
        preds, confs = [], []

        with st.spinner("Detecting intrusions..."):
            for _, row in numeric.iterrows():
                r = predict(model, row.values)
                preds.append(ATTACK_LABELS[r["prediction"]])
                confs.append(r["confidence"])

        results = pd.DataFrame({
            "Prediction": preds,
            "Confidence": confs
        })

        st.dataframe(results)

        summary = results["Prediction"].value_counts()
        st.bar_chart(summary)

        os.makedirs("logs", exist_ok=True)
        with open("logs/alerts.log", "a") as f:
            for p, c in zip(preds, confs):
                f.write(f"{datetime.datetime.now()},{p},{c:.4f}\n")


