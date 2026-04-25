# Integrated Salary Prediction and Analytics System

A **Streamlit web app** that predicts salaries for tech/data professionals based on key job-related features using an **ensemble machine learning model**. This tool estimates salaries in both **USD** and **INR**, analyzes resumes, suggests job roles, and provides salary breakdown insights.

🔗 **Live App:**  
👉 https://integrated-salary-prediction-and-analytics-system.streamlit.app/

---

## 🚀 Features

- 🔮 Predict salaries in both **USD** and **INR**
- 🧠 Uses an **ensemble model** (`Voting Regressor`) for high accuracy
- 📄 Resume Analyzer: upload PDF/DOCX and extract skills
- 💼 Suggest suitable job roles based on resume skills
- 📊 Explainable Salary Breakdown (Skill, Experience, Market Demand, Company Premium)
- 📈 Interactive charts and analytics
- ⚡ Fast predictions using optimized input features
- ✅ **R² Score:** **0.8473** (84.73% accuracy)

---

## 🛠️ Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
streamlit run .\app.py
