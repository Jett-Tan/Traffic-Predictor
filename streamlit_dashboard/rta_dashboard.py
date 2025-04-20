import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Paths to your saved insight files
INSIGHT_DIR = "/app/data/insights"

st.set_page_config(page_title="RTA Insights Dashboard", layout="wide")

st.title("🚦 Road Traffic Accident Insights Dashboard")

# Section 1: Top Risk Conditions
st.header("📊 Top Risk Conditions")
top_risk_df = pd.read_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", index_col=0)

# Show data with interactive features
st.subheader("Top Risk Conditions")

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
st.subheader("Risky Condition Combinations Data")
st.dataframe(combo_df)

# Section 4: Suggested Interventions using Hugging Face
# Load pre-trained model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_recommendation(risk_factor):
    # Prepare the prompt for the model
    prompt = f"Suggest a road safety intervention for the risk factor: {risk_factor}"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Generate the recommendation using the model
    outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=3, early_stopping=True)

    # Decode and return the generated recommendation
    recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return recommendation

# Load the top features 
top_features_df = pd.read_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", index_col=0)
top_factors = top_features_df.head(5).index.tolist()

# Generate recommendations for each risk factor and store them in a dictionary
recommendations = {}
for factor in top_factors:
    recommendations[factor] = generate_recommendation(factor)

# Prepare the interventions data for the dashboard
interventions = [(factor, recommendation) for factor, recommendation in recommendations.items()]

# Convert the list to DataFrame
interventions_df = pd.DataFrame(interventions, columns=["Risk Factor", "Recommended Intervention"])

# Display the recommendations on Streamlit dashboard
st.header("🛠 Suggested Interventions")
st.dataframe(interventions_df)