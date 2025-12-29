
import streamlit as st
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

st.title("AI Intrusion Detection Dashboard & CNN+LSTM IDS Tester")

uploaded_file = st.file_uploader("Upload CICIDS2017 CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] != 78:
        st.warning(f"Expected 78 numeric features, found {numeric_df.shape[1]}")
    else:
        scaler = StandardScaler()
        x = scaler.fit_transform(numeric_df.values)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with st.spinner("Predicting..."):
            preds = model(x_tensor)
            pred_labels = torch.argmax(preds, dim=1).numpy()
            attack_results = ['"BENIGN"', '"Bot"', '"BruteForce"', '"DDoS"', '"DoS_Hulk"', '"DoS_Slowloris"', '"DoS_Slowhttptest"', '"FTP-Patator"', '"Heartbleed"', '"Infiltration"', '"PortScan"', '"SSH-Patator"', '"Web_Attack_BruteForce"', '"Web_Attack_XSS"', '"Web_Attack_SQLInjection"']
            attack_results = [attack_results[i] for i in pred_labels]

        df_results = pd.DataFrame({'Prediction': attack_results})
        st.write("Predictions per row:")
        st.dataframe(df_results)

        summary = df_results['Prediction'].value_counts().reset_index()
        summary.columns = ['Attack Type', 'Count']
        summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
        st.write("Prediction Summary:")
        st.dataframe(summary)

        st.markdown("### Attack Distribution Bar Chart")
        st.bar_chart(summary.set_index('Attack Type')['Count'])

        st.markdown("### Attack Distribution Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(summary['Count'], labels=summary['Attack Type'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
