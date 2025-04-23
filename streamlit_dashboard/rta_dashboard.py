import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Paths to your saved insight files
INSIGHT_DIR = "/app/data/insights"

st.set_page_config(page_title="RTA Insights Dashboard", layout="wide")

st.title("🚦 Road Traffic Accident Insights Dashboard")

# Define Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 Risky Patterns", "🔍 Detailed Feature Analysis"])

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
    severity_df = pd.read_csv(f"{INSIGHT_DIR}/severity_by_hour.csv")

    # Ensure correct data types
    severity_df["hour"] = severity_df["hour"].astype(int)

    # Slider: Select time range
    min_hour = int(severity_df["hour"].min())
    max_hour = int(severity_df["hour"].max())
    selected_range = st.slider(
        "Select Hour Range",
        min_value=min_hour,
        max_value=max_hour,
        value=(min_hour, max_hour)
    )

    # Filter by slider selection
    filtered_df = severity_df[
        (severity_df["hour"] >= selected_range[0]) &
        (severity_df["hour"] <= selected_range[1])
    ]

    # Plotly animation chart
    fig = px.bar(
        filtered_df,
        x="accident_severity",
        y="count",
        color="accident_severity",
        animation_frame="hour",
        animation_group="accident_severity",
        title="Accident Severity Over Time (Animated by Hour)",
        labels={"count": "Number of Accidents", "accident_severity": "Severity"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        xaxis_title="Severity Level",
        yaxis_title="Number of Accidents",
        legend_title="Severity",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # Display the animated plot
    st.plotly_chart(fig, use_container_width=True)

    # Static stacked bar plot
    # Pivot the data to get accident counts by hour and severity
    pivot_df = severity_df.pivot_table(
        index="hour", columns="accident_severity", values="count", aggfunc="sum"
    ).fillna(0)

    # Show as stacked bar chart 
    st.subheader("📊 Accident Severity by Hour (over 24 hours)")
    st.bar_chart(pivot_df)

    # Section 3: Risky Combinations
    st.header("⚠️ Risky Condition Combinations")
    # Load data for risky conditions
    combo_df = pd.read_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv")

    # Display the data
    st.dataframe(combo_df)

with tab3:
    st.header("🔍 Detailed Feature Analysis")
    st.markdown("Explore specific factors that influence accident severity and frequency.")

    # Section 1: Accident Severity by Driver Age Band
    st.subheader("👶 Accident Severity by Driver Age Band")
    age_df = pd.read_csv(f"{INSIGHT_DIR}/age_band_vs_severity.csv")
    age_df = age_df.set_index("age_band_of_driver")
    age_df = age_df.reset_index().melt(id_vars="age_band_of_driver", var_name="Severity", value_name="Count")
    
    fig1 = px.bar(
        age_df, x="age_band_of_driver", y="Count", color="Severity",
        title="Accident Severity by Driver Age Band",
        barmode="stack", height=450, color_discrete_sequence=px.colors.sequential.Reds
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Section 2: Accidents by Lane Type
    st.subheader("🛣 Accidents by Lane Type")
    lane_df = pd.read_csv(f"{INSIGHT_DIR}/accidents_by_lane.csv")
    fig2 = px.bar(
        lane_df, x="Lane Type", y="Number of Accidents",
        title="Accidents by Lane Type", color="Number of Accidents",
        color_continuous_scale="Blues", height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Section 3: Number of Accidents by Cause
    st.subheader("⚠️ Accidents by Cause")
    cause_df = pd.read_csv(f"{INSIGHT_DIR}/accidents_by_cause.csv")
    cause_df = cause_df.reset_index().rename(columns={"index": "Cause"})
    fig3 = px.bar(
        cause_df.sort_values("count"), x="count", y="Cause",
        orientation="h", title="Number of Accidents by Cause",
        color="count", color_continuous_scale="Sunset"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Section 4: Driver Experience vs Severity
    st.subheader("🧓 Accident Severity by Driver Experience")
    exp_df = pd.read_csv(f"{INSIGHT_DIR}/driver_experience_vs_severity.csv")
    exp_df = exp_df.set_index("driving_experience")
    exp_df = exp_df.reset_index().melt(id_vars="driving_experience", var_name="Severity", value_name="Count")
    
    fig5 = px.bar(
        exp_df, x="driving_experience", y="Count", color="Severity",
        title="Accident Severity by Driver Experience",
        barmode="stack", color_discrete_sequence=px.colors.sequential.Aggrnyl
    )
    st.plotly_chart(fig5, use_container_width=True)