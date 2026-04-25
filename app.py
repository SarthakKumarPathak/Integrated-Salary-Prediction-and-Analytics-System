import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("salary_predictor.pkl")
columns = joblib.load("model_columns.pkl")
dropdowns = joblib.load("dropdown_options.pkl")

st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")
st.markdown("Provide your job details to predict your salary in USD and INR.")

# Allow job_title to be prefilled via session state (set by Resume Analyzer page)
prefill_job = st.session_state.get("job_title", None)
job_options = list(dropdowns.get("job_title", []))
if prefill_job:
    # If the predicted job isn't in the saved dropdowns, temporarily show it at the top
    if prefill_job not in job_options:
        job_options = [prefill_job] + job_options

    default_index = job_options.index(prefill_job) if prefill_job in job_options else 0
    job_title = st.selectbox("Job Title", job_options, index=default_index)
else:
    job_title = st.selectbox("Job Title", job_options)
company_size = st.selectbox("Company Size", dropdowns["company_size"])
employee_residence = st.selectbox("Location", dropdowns["employee_residence"])
experience_level = st.selectbox("Experience Level", dropdowns["experience_level"])
employment_type = st.selectbox("Employment Type", dropdowns["employment_type"])

remote_ratio = st.slider("Remote Work Ratio (0 = Onsite, 100 = Fully Remote)", 0, 100, 50)

if st.button("Predict Salary"):
    input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    for col in columns:
        if col.endswith(f"_{job_title}") and "job_title_" in col:
            input_data[col] = 1
        if col.endswith(f"_{company_size}") and "company_size_" in col:
            input_data[col] = 1
        if col.endswith(f"_{employee_residence}") and "employee_residence_" in col:
            input_data[col] = 1
        if col.endswith(f"_{experience_level}") and "experience_level_" in col:
            input_data[col] = 1
        if col.endswith(f"_{employment_type}") and "employment_type_" in col:
            input_data[col] = 1

    if "remote_ratio" in input_data.columns:
        input_data["remote_ratio"] = remote_ratio

    predicted_usd = model.predict(input_data)[0]
    predicted_inr = predicted_usd * 83  

    # Save predictions and current inputs to session_state so downstream pages (Salary Breakdown) can use them
    st.session_state['predicted_usd'] = float(predicted_usd)
    st.session_state['predicted_inr'] = float(predicted_inr)
    st.session_state['job_title'] = job_title
    st.session_state['company_size'] = company_size
    st.session_state['employee_residence'] = employee_residence
    st.session_state['experience_level'] = experience_level
    st.session_state['employment_type'] = employment_type
    st.session_state['remote_ratio'] = remote_ratio

    st.success(f"💰 Predicted Salary (USD): ${predicted_usd:,.2f}")
    st.success(f"🇮🇳 Predicted Salary (INR): ₹{predicted_inr:,.2f}")
