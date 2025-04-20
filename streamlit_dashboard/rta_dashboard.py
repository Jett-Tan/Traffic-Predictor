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
st.divider()  # Add between each section


# Section 2: Severity by Hour
st.header("🕐 Severity by Time of Day")
severity_df = pd.read_csv(f"{INSIGHT_DIR}/severity_by_hour.csv", index_col=0)
st.bar_chart(severity_df)

st.image(f"{INSIGHT_DIR}/severity_by_hour.png", caption="Accident Severity by Hour")
st.divider()  # Add between each section

# Section 3: Risky Combinations
st.header("⚠️ Risky Condition Combinations")
combo_df = pd.read_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv")
st.dataframe(combo_df)

st.image(f"{INSIGHT_DIR}/risky_condition_combos.png", caption="Top Risky Condition Combos")
st.divider()  # Add between each section

# Section 4: Suggested Interventions
st.header("🛠 Suggested Interventions")
intervention_df = pd.read_csv(f"{INSIGHT_DIR}/suggested_interventions.csv")
st.dataframe(intervention_df)
st.divider()  # Add between each section


# Section 5: Accidents by Cause
st.header("📉 Accidents by Cause")
cause_df = pd.read_csv(f"{INSIGHT_DIR}/accidents_by_cause.csv")
st.dataframe(cause_df)

st.image(f"{INSIGHT_DIR}/accidents_by_cause.png", caption="Accidents by Cause")
st.divider()  # Add between each section



# Section 6: Age Band vs Severity
st.header("👶 Age Band vs Severity")
age_df = pd.read_csv(f"{INSIGHT_DIR}/age_band_vs_severity.csv", index_col=0)
st.dataframe(age_df)

st.image(f"{INSIGHT_DIR}/age_band_vs_severity.png", caption="Accident Severity by Age Band")
st.divider()  # Add between each section

# # Section 7: Accidents by Lane Type
# st.header("🛣 Accidents by Lane Type")
# lane_df = pd.read_csv(f"{INSIGHT_DIR}/accidents_by_lane.csv")
# st.dataframe(lane_df)

# st.image(f"{INSIGHT_DIR}/accidents_by_lane.png", caption="Accidents by Lane Type")
# st.divider()  # Add between each section



# Section 7: Accidents by Lane Type
st.header("🛣 Accidents by Lane Type")

st.write("✅ [Lane] Before reading CSV")
try:
    lane_df = pd.read_csv(f"{INSIGHT_DIR}/accidents_by_lane.csv")
    st.write("✅ [Lane] Successfully read CSV")
    st.dataframe(lane_df)
except Exception as e:
    st.error(f"❌ [Lane] Error reading CSV: {e}")

st.write("✅ [Lane] Before showing image")
try:
    st.image(f"{INSIGHT_DIR}/accidents_by_lane.png", caption="Accidents by Lane Type")
    st.write("✅ [Lane] Successfully showed image")
except Exception as e:
    st.error(f"❌ [Lane] Error showing image: {e}")

st.write("✅ [Lane] Reached end of section")
st.divider()



# Section 9 and 10 Debugging
st.write("🟩 Reached BEFORE Service Year section")
try:
    st.header("📅 Service Year of Vehicle vs Accident Frequency")
    year_df = pd.read_csv(f"{INSIGHT_DIR}/service_year_vs_accidents.csv", index_col=0)
    st.dataframe(year_df)
    st.image(f"{INSIGHT_DIR}/service_year_vs_accidents.png", caption="Accidents by Vehicle Service Year")
except Exception as e:
    st.error(f"❌ Service Year section failed: {e}")
st.write("🟩 Reached AFTER Service Year section")
st.divider()

st.write("🟩 Reached BEFORE Driver Experience section")
try:
    st.header("🧑‍✈️ Driver Experience vs Severity")
    experience_df = pd.read_csv(f"{INSIGHT_DIR}/driver_experience_vs_severity.csv")
    st.dataframe(experience_df)
    st.image(f"{INSIGHT_DIR}/driver_experience_vs_severity.png", caption="Severity by Driver Experience")
except Exception as e:
    st.error(f"❌ Driver Experience section failed: {e}")
st.write("🟩 Reached AFTER Driver Experience section")
st.divider()



# # Section 9: Service Year of Vehicle vs Accident Frequency
# st.header("📅 Service Year of Vehicle vs Accident Frequency")
# year_df = pd.read_csv(f"{INSIGHT_DIR}/service_year_vs_accidents.csv", index_col=0)
# st.dataframe(year_df)
# st.image(f"{INSIGHT_DIR}/service_year_vs_accidents.png", caption="Accidents by Vehicle Service Year")
# st.divider()  # Add between each section



# # Section 10: Driver Experience vs Severity
# st.header("🧑‍✈️ Driver Experience vs Severity")
# experience_df = pd.read_csv(f"{INSIGHT_DIR}/driver_experience_vs_severity.csv")
# st.dataframe(experience_df)
# st.image(f"{INSIGHT_DIR}/driver_experience_vs_severity.png", caption="Severity by Driver Experience")
# st.divider()  # Add between each section
