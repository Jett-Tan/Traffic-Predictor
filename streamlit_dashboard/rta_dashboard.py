import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Paths to your saved insight files
INSIGHT_DIR = "/app/data/insights"

st.set_page_config(page_title="RTA Insights Dashboard", layout="wide")

st.title("🚦 Road Traffic Accident Insights Dashboard")

# Section 1: Top Risk Conditions
st.header("📊 Top Risk Conditions")
top_risk_df = pd.read_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", index_col=0)
st.dataframe(top_risk_df)

st.image(f"{INSIGHT_DIR}/top_risk_conditions.png", caption="Top Risk Conditions (Feature Importances)")

# Section 2: Severity by Hour
st.header("🕐 Severity by Time of Day")
severity_df = pd.read_csv(f"{INSIGHT_DIR}/severity_by_hour.csv", index_col=0)
st.bar_chart(severity_df)

st.image(f"{INSIGHT_DIR}/severity_by_hour.png", caption="Accident Severity by Hour")

# Section 3: Risky Combinations
st.header("⚠️ Risky Condition Combinations")
combo_df = pd.read_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv")
st.dataframe(combo_df)

st.image(f"{INSIGHT_DIR}/risky_condition_combos.png", caption="Top Risky Condition Combos")

# Section 4: Suggested Interventions
st.header("🛠 Suggested Interventions")
intervention_df = pd.read_csv(f"{INSIGHT_DIR}/suggested_interventions.csv")
st.dataframe(intervention_df)
