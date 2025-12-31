import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os

from services.inference import load_model, predict

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Intrusion Detection System",
    layout="wide"
)

st.title("AI-Driven Intrusion Detection Dashboard (CNN + LSTM)")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload CICIDS2017 CSV File",
    type="csv"
)

ATTACK_LABELS = [
    "BENIGN", "Bot", "BruteForce", "DDoS", "DoS_Hulk",
    "DoS_Slowloris", "DoS_Slowhttptest", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan",
    "SSH-Patator", "Web_Attack_BruteForce",
    "Web_Attack_XSS", "Web_Attack_SQLInjection"
]

# ------------------ PROCESS FILE ------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] != 78:
        st.warning(
            f"Expected 78 numeric features, found {numeric_df.shape[1]}"
        )
    else:
        st.success("Correct feature count detected (78).")

        predictions = []
        confidences = []

        with st.spinner("Running intrusion detection..."):
            for _, row in numeric_df.iterrows():
                result = predict(model, row.values)

                predictions.append(
                    ATTACK_LABELS[result["prediction"]]
                )
                confidences.append(result["confidence"])

        # ------------------ RESULTS ------------------
        results_df = pd.DataFrame({
            "Prediction": predictions,
            "Confidence": confidences
        })

        st.subheader("Per-Flow Predictions")
        st.dataframe(results_df)

        # ------------------ SUMMARY ------------------
        summary = (
            results_df["Prediction"]
            .value_counts()
            .reset_index()
        )
        summary.columns = ["Attack Type", "Count"]
        summary["Percentage"] = (
            summary["Count"] / summary["Count"].sum() * 100
        ).round(2)

        st.subheader("Prediction Summary")
        st.dataframe(summary)

        # ------------------ BAR CHART ------------------
        st.subheader("Attack Distribution (Bar Chart)")
        st.bar_chart(
            summary.set_index("Attack Type")["Count"]
        )

        # ------------------ PIE CHART ------------------
        st.subheader("Attack Distribution (Pie Chart)")
        fig, ax = plt.subplots()
        ax.pie(
            summary["Count"],
            labels=summary["Attack Type"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

        # ------------------ LOGGING ------------------
        os.makedirs("logs", exist_ok=True)
        with open("logs/alerts.log", "a") as f:
            for pred, conf in zip(predictions, confidences):
                f.write(
                    f"{datetime.datetime.now()},"
                    f"{pred},"
                    f"{conf:.4f}\n"
                )
