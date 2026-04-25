<<<<<<< HEAD
# 💼 Salary Predictor App

A **Streamlit web app** that predicts salaries for data professionals based on key job-related features using an **ensemble machine learning model**. This tool estimates salaries in both **USD** and **INR** and is built for quick, easy, and accurate predictions.

🔗 **Live App:**  
👉 [https://salaryprediction51.streamlit.app/](https://salaryprediction51.streamlit.app/)

---

## 🚀 Features

- 🔮 Predict salaries in both **USD** and **INR**
- 🧠 Uses an **ensemble model** (`Voting Regressor`) for high accuracy
- 📊 Clean and responsive UI with dropdown-based inputs
- ⚡ Fast predictions using only **6 key features**
- ✅ **R² Score:** **0.8473** (84.73% accuracy)

---

# Integrated Salary Prediction and Analytics System

A Streamlit app that predicts salaries (USD and INR) for tech/data roles, analyzes uploaded resumes for skills, suggests job roles, and provides an explainable salary decomposition (Skill, Experience, Market Demand, Company Premium, Location/Education). It combines a saved ML model for prediction with lightweight rule-based explainability and small utilities for mapping and persistence.

Features
- Resume Analyzer: upload PDF/DOCX, extract skills (keyword-based), and suggest role.
- Salary Prediction: loads `salary_predictor.pkl` and `model_columns.pkl` to perform inference.
- Salary Breakdown: explainable decomposition of a predicted salary into contribution factors with charts.
- Role mapping UI to reconcile analyzer role names with app dropdowns.

Quick start (Windows PowerShell)
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
streamlit run .\app.py
```

Key files
- `app.py` - main Streamlit app for salary prediction
- `pages/Resume Analyzer.py` - resume upload, extraction and role suggestion
- `pages/Salary Breakdown.py` - explainability layer and charts
- `salary_predictor.pkl`, `model_columns.pkl`, `dropdown_options.pkl` - saved model and metadata
- `requirements.txt` - Python dependencies

License
- Add a LICENSE file if you want to open-source this repository.

Contributing
- PRs welcome. For model changes, include retraining artifacts and updated metrics.
