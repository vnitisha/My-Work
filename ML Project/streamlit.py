import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# ---- Load trained model and dataset for feature alignment ----
model = joblib.load('estimated_revenue_model.pkl')
df = pd.read_csv('dataset.csv')  # Path to your data

# Prepare feature columns from training data
df_model = df.drop(columns=['campaign_id', 'start_date', 'end_date', 'estimated_revenue'])
feature_columns = pd.get_dummies(df_model, drop_first=True).columns.tolist()

# ---- Dropdown options ----
campaign_types     = df['campaign_type'].unique()
target_segments    = df['target_segment'].unique()
regions            = df['region'].unique()
product_categories = df['product_category'].unique()
seasons            = df['season'].unique()

st.title("Estimated Revenue & ROI Prediction")

# ---- User inputs ----
campaign_type    = st.selectbox("Campaign Type", campaign_types)
target_segment   = st.selectbox("Target Segment", target_segments)
region           = st.selectbox("Region", regions)
product_category = st.selectbox("Product Category", product_categories)
season           = st.selectbox("Season", seasons)

start_date      = st.date_input("Start Date", datetime.today())
end_date        = st.date_input("End Date", datetime.today())
budget          = st.number_input("Budget (₹)", min_value=0.0, format="%.2f")

impressions     = st.number_input("Impressions", min_value=0.0, format="%.0f")
clicks          = st.number_input("Clicks",      min_value=0.0, format="%.0f")
conversions     = st.number_input("Conversions", min_value=0.0, format="%.0f")

# ---- Derived metrics ----
duration_days   = (end_date - start_date).days or 1
ctr             = clicks / impressions   if impressions > 0 else 0
conversion_rate = conversions / clicks   if clicks > 0      else 0

# ---- Build input record ----
record = {
    'campaign_type':    campaign_type,
    'target_segment':   target_segment,
    'region':           region,
    'product_category': product_category,
    'season':           season,
    'budget':           budget,
    'duration_days':    duration_days,
    'impressions':      impressions,
    'clicks':           clicks,
    'ctr':              ctr,
    'conversions':      conversions,
    'conversion_rate':  conversion_rate
}

input_df = pd.DataFrame([record])

# One-hot encode and align with training features
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

st.write("Features for model:", input_aligned)

if st.button("Predict"):
    # Predict estimated revenue
    predicted_revenue = model.predict(input_aligned)[0]
    # Calculate ROI
    roi = (predicted_revenue - budget) / budget * 100 if budget != 0 else 0

    st.success(f"Predicted Estimated Revenue: ₹{predicted_revenue:,.2f}")
    st.success(f"Predicted ROI: {roi:.2f}%")