import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Paths to your saved insight files
INSIGHT_DIR = "/app/data/insights"

st.set_page_config(page_title="RTA Insights Dashboard", layout="wide")

st.title("🚦 Road Traffic Accident Insights Dashboard")

# Define Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 Risky Patterns", "🔍 Deep Dive"])

with tab1: 
    # Section: Suggested Interventions 

    # Define rule-based recommendations per group
    recommendation_rules = {
        "day": "Monitor daily patterns and target high-accident days with alerts and police patrols.",
        "age": "Customize road safety training and licensing programs for different age groups.",
        "sex": "Use data to inform inclusive policy and outreach campaigns for drivers by gender.",
        "educational": "Create visual safety materials and campaigns for drivers with lower education levels.",
        "vehicle": "Enforce maintenance checks and create awareness around high-risk vehicle types.",
        "owner": "Strengthen policies for fleet owners and institutional drivers.",
        "area": "Design interventions specific to high-risk zones such as schools, markets, or residential areas.",
        "lanes": "Improve lane markings and signage, especially for undivided or poorly marked roads.",
        "road": "Install warning signs and make structural improvements on hazardous alignments.",
        "types": "Redesign common junction types and ensure proper visibility at intersections.",
        "light": "Improve nighttime lighting and ensure drivers use lights properly. Ensure efficient route planning to accommodate day-time crowds.",
        "weather": "Implement dynamic warnings and adjust speed limits during bad weather.",
        "type": "Introduce road features to reduce frequent collision types (e.g., bumpers, barriers).",
        "vehicle": "Monitor vehicle behavior and enforce penalties for reckless movements.",
        "casualty": "Improve post-crash emergency response based on severity levels.",
        "cause": "Run awareness campaigns and strict monitoring of frequent accident causes."
    }

    # Load the top features 
    top_features_df = pd.read_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", index_col=0)
    top_factors = top_features_df.head(5).index.tolist()

    # Match the rules to the top 5 risk factors based on their group
    recommendations = {}
    for factor in top_factors:
        # Extract the root category of the feature (before the first underscore)
        group = factor.split("_")[0]
        
        # Retrieve the appropriate recommendation rule for that group
        rule = recommendation_rules.get(group, "Investigate and tailor specific interventions for this group.")
        
        # Store the recommendation in the dictionary
        recommendations[factor] = rule

    # Prepare the interventions data for the dashboard
    interventions = [(factor, recommendation) for factor, recommendation in recommendations.items()]

    # Convert the list to DataFrame
    interventions_df = pd.DataFrame(interventions, columns=["Risk Factor", "Recommended Intervention"])

    # Display the recommendations on Streamlit dashboard
    st.header("🛠 Suggested Interventions")
    st.dataframe(interventions_df)

with tab2: 
    # Section 1: Top Risk Conditions
    st.header("📊 Top Risk Conditions")
    top_risk_df = pd.read_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", index_col=0)

    # Use Plotly for an interactive bar chart
    fig = px.bar(top_risk_df, x="importance", y=top_risk_df.index, 
                labels={"importance": "Importance", "index": "Feature"})
    st.plotly_chart(fig)

    # Section 2: Severity by Hour
    st.header("🕐 Severity by Time of Day")
    severity_df = pd.read_csv(f"{INSIGHT_DIR}/severity_by_hour.csv", index_col=0)

    # Streamlit slider to select a time range
    min_hour = int(severity_df.index.min())
    max_hour = int(severity_df.index.max())
    selected_range = st.slider(
        "Select Hour Range",
        min_value=min_hour,
        max_value=max_hour,
        value=(min_hour, max_hour)
    )

    # Filter dataframe based on the selected hour range
    filtered_df = severity_df.loc[selected_range[0]:selected_range[1]]

    # Show as stacked bar chart (using matplotlib for stacked bar look)
    st.subheader("Stacked Bar Chart of Accident Severity")
    st.bar_chart(filtered_df)

    # Section 3: Risky Combinations
    st.header("⚠️ Risky Condition Combinations")
    # Load data for risky conditions
    combo_df = pd.read_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv")

    # Display the data
    st.dataframe(combo_df)

