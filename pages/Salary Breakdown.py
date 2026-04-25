import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Small helper plotting function for Streamlit
def plot_pie(labels, values):
    if not HAS_MPL:
        return None
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig


def normalize_components(components: Dict[str, float], total: float) -> Dict[str, float]:
    """Normalize and scale components so they sum exactly to total.
    components: name -> raw score (non-negative)
    Returns: name -> scaled amount (same units as total)
    """
    # guard
    comps = {k: max(0.0, float(v)) for k, v in components.items()}
    s = sum(comps.values())
    if s == 0:
        # distribute equally
        n = len(comps)
        if n == 0:
            return {}
        each = total / n
        return {k: each for k in comps}
    factor = total / s
    scaled = {k: round(v * factor, 2) for k, v in comps.items()}
    # fix rounding so sum exact
    diff = round(total - sum(scaled.values()), 2)
    # adjust the largest component by diff
    if abs(diff) >= 0.01:
        largest = max(scaled.items(), key=lambda x: x[1])[0]
        scaled[largest] = round(scaled[largest] + diff, 2)
    return scaled


def generate_salary_breakdown(predicted_salary_inr: float, inputs: Dict[str, any]) -> Dict[str, float]:
    """Rule-based decomposition using user inputs. Returns components in INR (same units as predicted_salary_inr).
    inputs: expects keys such as 'experience_level', 'company_size', 'employee_residence', 'job_title', 'remote_ratio', 'resume_skills'
    """
    # Base component weights (raw scores) - tuned heuristics
    # We'll compute raw scores for: skill, experience, market, company, location, education
    raw = {}

    # Skill value: count of desirable keywords in resume_skills or role match strength
    skills = inputs.get('resume_skills', []) or []
    skill_score = len(skills) * 1.0
    # boost if the role suggests senior/advanced (not available here)
    raw['Skill Value'] = max(0.0, skill_score)

    # Experience value: map experience_level to score
    exp_map = {
        'EN': 1.0,  # Entry
        'MI': 2.0,  # Mid
        'SE': 3.0,  # Senior
        'EX': 4.0,  # Executive
    }
    exp_code = inputs.get('experience_level', '')
    raw['Experience Value'] = exp_map.get(exp_code, 1.5)

    # Market demand: heuristic from job title keywords and location
    job_title = (inputs.get('job_title') or '').lower()
    market = 1.0
    if any(x in job_title for x in ['data scientist', 'machine learning', 'ml', 'ai']):
        market += 1.0
    if any(x in job_title for x in ['devops', 'site reliability', 'sre']):
        market += 0.8
    if any(x in job_title for x in ['frontend', 'react', 'ui']):
        market += 0.3
    raw['Market Demand'] = market

    # Company premium: larger companies get premium
    company = (inputs.get('company_size') or '').lower()
    company_score = 1.0
    if 'large' in company or 'big' in company or 'enterprise' in company:
        company_score = 1.5
    elif 'small' in company or 'startup' in company:
        company_score = 0.8
    raw['Company Premium'] = company_score

    # Location bonus: based on residence / remote ratio
    loc = (inputs.get('employee_residence') or '').lower()
    loc_score = 1.0
    if any(x in loc for x in ['san francisco', 'new york', 'london', 'bengaluru', 'bangalore']):
        loc_score = 1.3
    raw['Location Bonus'] = loc_score if inputs.get('remote_ratio', 50) < 50 else 1.0

    # Education bonus (optional) - look for keywords in resume_skills
    edu = 0.5 if any(x in skills for x in ['masters', 'phd', 'ms', 'mtech', 'btech']) else 0.0
    raw['Education Bonus'] = edu

    # Provide at least these main keys and then normalize to predicted_salary_inr
    scaled = normalize_components(raw, predicted_salary_inr)
    return scaled


def plot_breakdown_chart(components: Dict[str, float]):
    labels = list(components.keys())
    values = list(components.values())
    fig_pie = plot_pie(labels, values)
    if HAS_MPL:
        fig_bar, ax = plt.subplots()
        ax.bar(labels, values, color='tab:blue')
        ax.set_ylabel('INR')
        ax.set_xticklabels(labels, rotation=30, ha='right')
        plt.tight_layout()
        return fig_pie, fig_bar
    else:
        return fig_pie, None


def show_breakdown_ui():
    st.title("💡 Salary Breakdown")
    # Read inputs silently from session_state (no interactive inputs on this page)
    inputs = {}
    inputs['job_title'] = st.session_state.get('job_title', None)
    inputs['company_size'] = st.session_state.get('company_size', None)
    inputs['employee_residence'] = st.session_state.get('employee_residence', None)
    inputs['experience_level'] = st.session_state.get('experience_level', None)
    inputs['employment_type'] = st.session_state.get('employment_type', None)
    inputs['remote_ratio'] = st.session_state.get('remote_ratio', 50)
    inputs['resume_skills'] = st.session_state.get('resume_skills', [])

    st.subheader('Resume skills detected (session state)')
    st.write(inputs['resume_skills'])

    # Prefer predicted salary stored in session_state by the main app when available
    predicted_inr = st.session_state.get('predicted_inr', None)
    predicted_usd = st.session_state.get('predicted_usd', None)
    stored_job = st.session_state.get('job_title', None)

    # Banner: if a stored prediction exists, show it and offer to clear
    if predicted_inr is not None:
        st.success(f"Using stored prediction from main page — Job: {stored_job or 'N/A'} | Predicted Salary: ₹{predicted_inr:,.2f}")
        if st.button("Clear stored prediction"):
            for k in ['predicted_inr','predicted_usd','job_title','company_size','employee_residence','experience_level','employment_type','remote_ratio','resume_skills']:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()

    model_available = False
    try:
        model = joblib.load('salary_predictor.pkl')
        columns = joblib.load('model_columns.pkl')
        model_available = True
    except Exception:
        model_available = False

    if predicted_inr is None:
        # allow user to compute locally with model or input manually
        if model_available and st.button('Predict salary using model (use current inputs)'):
            # build input row similar to main app
            input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
            jt = inputs.get('job_title')
            cs = inputs.get('company_size')
            er = inputs.get('employee_residence')
            ex = inputs.get('experience_level')
            et = inputs.get('employment_type')
            for col in columns:
                if jt and col.endswith(f"_{jt}") and 'job_title_' in col:
                    input_data[col] = 1
                if cs and col.endswith(f"_{cs}") and 'company_size_' in col:
                    input_data[col] = 1
                if er and col.endswith(f"_{er}") and 'employee_residence_' in col:
                    input_data[col] = 1
                if ex and col.endswith(f"_{ex}") and 'experience_level_' in col:
                    input_data[col] = 1
                if et and col.endswith(f"_{et}") and 'employment_type_' in col:
                    input_data[col] = 1
            if 'remote_ratio' in input_data.columns:
                input_data['remote_ratio'] = inputs.get('remote_ratio', 50)
            try:
                predicted_usd = model.predict(input_data)[0]
                predicted_inr = predicted_usd * 83
                st.session_state['predicted_usd'] = float(predicted_usd)
                st.session_state['predicted_inr'] = float(predicted_inr)
                st.success(f"Predicted and stored salary: ₹{predicted_inr:,.2f}")
            except Exception as e:
                st.error(f"Model predict failed: {e}")

        if predicted_inr is None:
            val = st.number_input('Enter predicted salary in INR (for breakdown)', min_value=0.0, value=500000.0, step=1000.0)
            predicted_inr = float(val)
    else:
        # make sure inputs job_title matches the stored job_title when present
        if stored_job:
            inputs['job_title'] = stored_job

    st.subheader(f'Predicted Salary: ₹{predicted_inr:,.2f}')

    components = generate_salary_breakdown(predicted_inr, inputs)

    st.subheader('Breakdown')
    for k, v in components.items():
        st.write(f"- {k}: ₹{v:,.2f}")

    # Visuals
    fig_pie, fig_bar = plot_breakdown_chart(components)
    if fig_pie is not None:
        st.pyplot(fig_pie)
    else:
        st.info("Install 'matplotlib' for pie chart visuals (added to requirements). Using fallback charts.")

    if fig_bar is not None:
        st.pyplot(fig_bar)
    else:
        # fallback to streamlit bar_chart
        try:
            df = pd.DataFrame({'component': list(components.keys()), 'amount': list(components.values())}).set_index('component')
            st.bar_chart(df)
        except Exception:
            pass

    # Progress bars
    st.subheader('Contribution Progress')
    for k, v in components.items():
        pct = 0 if predicted_inr == 0 else (v / predicted_inr)
        st.write(k)
        st.progress(int(pct * 100))

    # Explanation text
    st.markdown('---')
    primary = max(components.items(), key=lambda x: x[1])[0] if components else 'N/A'
    st.write(f"**Explanation:** Your salary is mainly driven by **{primary}** based on the inputs provided.")


if __name__ == '__main__':
    show_breakdown_ui()
