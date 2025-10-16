# ======================================
# INTELLIGENT BUSINESS INSIGHTS DASHBOARD
# ======================================
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Business Insights Dashboard", layout="wide")
st.title("📊 Intelligent Business Insights Dashboard")
st.markdown("Analyze sales performance and forecast future revenue using Machine Learning.")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)
    return df
df = load_data()

st.sidebar.header("🔍 Filters")
region = st.sidebar.multiselect("Select Region:", options=df["Region"].unique(), default=df["Region"].unique())
category = st.sidebar.multiselect("Select Category:", options=df["Category"].unique(), default=df["Category"].unique())

filtered_df = df[(df["Region"].isin(region)) & (df["Category"].isin(category))]
# -----------------------------
# KPIs
# -----------------------------
total_sales = round(filtered_df["Sales"].sum(), 2)
total_profit = round(filtered_df["Profit"].sum(), 2)
avg_discount = round(filtered_df["Discount"].mean() * 100, 2)

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
col3.metric("🏷️ Avg Discount", f"{avg_discount}%")

st.divider()

# -----------------------------
# VISUAL 1: SALES BY CATEGORY
# -----------------------------
st.subheader("🛍️ Sales by Category and Sub-Category")
cat_sales = (
    filtered_df.groupby(["Category", "Sub-Category"], as_index=False)["Sales"].sum()
)
fig1 = px.bar(
    cat_sales,
    x="Sub-Category",
    y="Sales",
    color="Category",
    title="Sales Distribution by Sub-Category",
    template="plotly_white"
)
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# VISUAL 2: SALES TREND OVER TIME
# -----------------------------
st.subheader("📅 Monthly Sales Trend")
monthly_sales = (
    filtered_df.groupby("Order Month", as_index=False)["Sales"].sum().sort_values("Order Month")
)
fig2 = px.line(monthly_sales, x="Order Month", y="Sales", title="Sales Trend Over Time", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# VISUAL 3: REGION-WISE PROFIT
# -----------------------------
st.subheader("🌍 Profit by Region")
region_profit = (
    filtered_df.groupby("Region", as_index=False)["Profit"].sum()
)
fig3 = px.bar(region_profit, x="Region", y="Profit", color="Region", title="Regional Profit Overview", text_auto=True)
st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# FORECASTING SECTION
# -----------------------------
st.subheader("📊 Sales Forecasting (Prophet Model)")

# Prepare data for Prophet
forecast_data = (
    df.groupby("Order Date", as_index=False)["Sales"].sum()
    .rename(columns={"Order Date": "ds", "Sales": "y"})
)

model = Prophet()
model.fit(forecast_data)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

fig_forecast = px.line(forecast, x="ds", y="yhat", title="Next 3 Months Sales Forecast", labels={"ds": "Date", "yhat": "Predicted Sales"})
fig_forecast.add_scatter(x=forecast_data["ds"], y=forecast_data["y"], mode='lines', name='Actual Sales')
st.plotly_chart(fig_forecast, use_container_width=True)

st.success("✅ Dashboard successfully loaded! You can now interact with filters and explore insights.")

