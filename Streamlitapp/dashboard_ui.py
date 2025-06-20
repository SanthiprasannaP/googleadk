#Streamlitapp/dashboard_ui.py

import streamlit as st # type: ignore
import json
import os

st.set_page_config(page_title="Insurance Risk Dashboard", layout="wide")

output_path = os.path.join(os.getcwd(), "dashboard_output", "summary_output.json")

# Load summary data
try:
    with open(output_path) as f:
        dashboard = json.load(f)

    st.title("🛡️ Insurance Risk Dashboard")
    
    st.metric("Risk Score", dashboard["risk_score"])
    st.metric("Confidence", dashboard["confidence"])
    
    st.subheader("Top Risk Factors")
    st.write(dashboard["top_features"])
    
    st.subheader("Explanation")
    st.write(dashboard["explanation"])
    
    st.subheader("Visuals")
    st.image("shap_outputs/"+dashboard["force_plot_path"], caption="SHAP Force Plot")
    st.image(dashboard["decision_plot_path"], caption="SHAP Decision Plot")
    
    st.caption(f'Model Version: {dashboard["model_version"]}, Audit ID: {dashboard["audit_id"]}')
except Exception as e:
    st.error(f"Failed to load dashboard data: {e}")
