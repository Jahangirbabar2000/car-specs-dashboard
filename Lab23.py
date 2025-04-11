# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------- Load & Preprocess Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cars_specs.csv")

    # Clean up column names and standardize
    df.columns = df.columns.str.strip().str.lower()

    df.rename(columns={
        "car_name": "name",
        "model": "model_year"
    }, inplace=True)

    df = df.dropna(subset=["mpg", "horsepower", "weight", "model_year", "origin", "cylinders"])
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna()
    df['model_year'] = df['model_year'].astype(int)
    return df

df = load_data()

# -------------------- Sidebar Controls --------------------
st.sidebar.title("Filters")
years = st.sidebar.multiselect("Select Model Year", sorted(df['model_year'].unique()), default=sorted(df['model_year'].unique()))
origins = st.sidebar.multiselect("Select Origin", df['origin'].unique(), default=list(df['origin'].unique()))
cyls = st.sidebar.multiselect("Select Cylinders", sorted(df['cylinders'].unique()), default=sorted(df['cylinders'].unique()))
hp_range = st.sidebar.slider("Horsepower Range", int(df['horsepower'].min()), int(df['horsepower'].max()), (60, 150))
normalize = st.sidebar.checkbox("Normalize Weight & Horsepower")

# -------------------- Apply Filters --------------------
filtered_df = df[
    (df['model_year'].isin(years)) &
    (df['origin'].isin(origins)) &
    (df['cylinders'].isin(cyls)) &
    (df['horsepower'].between(hp_range[0], hp_range[1]))
]

# Optional Normalization
if normalize:
    scaler = StandardScaler()
    filtered_df[['horsepower', 'weight']] = scaler.fit_transform(filtered_df[['horsepower', 'weight']])

# -------------------- KPI Section --------------------
st.title("🚗 Car Performance Dashboard")
st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)
col1.metric("Average MPG", round(filtered_df['mpg'].mean(), 2))
col2.metric("Average Horsepower", round(filtered_df['horsepower'].mean(), 2))
col3.metric("Average Weight", round(filtered_df['weight'].mean(), 2))

st.markdown("### Most Fuel-Efficient Cars")
st.write(filtered_df.sort_values(by='mpg', ascending=False).head(5)[['name', 'mpg']] if 'name' in filtered_df else filtered_df.sort_values(by='mpg', ascending=False).head(5))

st.markdown("### Least Fuel-Efficient Cars")
st.write(filtered_df.sort_values(by='mpg', ascending=True).head(5)[['name', 'mpg']] if 'name' in filtered_df else filtered_df.sort_values(by='mpg', ascending=True).head(5))

# -------------------- Visualizations --------------------
st.header("📊 Visualizations")

# Scatter plot MPG vs Weight
st.subheader("MPG vs. Weight (Color = Cylinders)")
fig1, ax1 = plt.subplots()
sns.scatterplot(data=filtered_df, x='weight', y='mpg', hue='cylinders', palette='viridis', ax=ax1)
st.pyplot(fig1)

# Histogram of Horsepower
st.subheader("Horsepower Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['horsepower'], bins=20, kde=True, ax=ax2)
st.pyplot(fig2)

# Line chart of MPG over Years
st.subheader("MPG Trend by Model Year")
fig3, ax3 = plt.subplots()
sns.lineplot(data=filtered_df.groupby('model_year')['mpg'].mean().reset_index(), x='model_year', y='mpg', ax=ax3)
st.pyplot(fig3)

# Bar chart: Average horsepower by Origin
st.subheader("Average Horsepower by Origin")
fig4, ax4 = plt.subplots()
sns.barplot(data=filtered_df, x='origin', y='horsepower', estimator=np.mean, ax=ax4)
st.pyplot(fig4)

# -------------------- Regression --------------------
st.header("📈 MPG Prediction")

X = filtered_df[['weight', 'horsepower']]
y = filtered_df['mpg']
reg_model = LinearRegression()
reg_model.fit(X, y)

st.subheader("Regression Line: MPG vs Weight")
fig5, ax5 = plt.subplots()
sns.scatterplot(x=X['weight'], y=y, color='blue', label='Actual', ax=ax5)
ax5.plot(X['weight'], reg_model.predict(X), color='red', label='Regression Line')
ax5.legend()
st.pyplot(fig5)

st.markdown("### Predict MPG with Custom Inputs")
input_weight = st.number_input("Weight", float(df['weight'].min()), float(df['weight'].max()), float(df['weight'].mean()))
input_hp = st.number_input("Horsepower", float(df['horsepower'].min()), float(df['horsepower'].max()), float(df['horsepower'].mean()))
predicted_mpg = reg_model.predict(np.array([[input_weight, input_hp]]))[0]
st.success(f"Predicted MPG: {predicted_mpg:.2f}")
