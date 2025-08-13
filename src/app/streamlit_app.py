import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import joblib
import sys

# Path setup
BASE = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE))

from src.nlp.gemini_client import generate_insights

PROC = BASE / "data" / "processed"
MODEL_DIR = BASE / "model_store"

st.set_page_config(page_title='Sales Forecast & Insights', layout='wide')
st.title('📈 AI-Driven Sales Forecasting & Insight Generator')

# Load processed monthly data
proc_file = PROC / 'monthly_by_category.csv'
if proc_file.exists():
    df = pd.read_csv(proc_file, parse_dates=['ds'])
else:
    st.warning('⚠️ Processed data not found. Run `python src/data_pipeline.py` to create processed data.')
    st.stop()

# Sidebar category selection
categories = df['category'].unique().tolist()
cat = st.sidebar.selectbox('Select Category', options=categories)

# Filter data
cat_df = df[df['category'] == cat]

# KPIs
st.subheader(f'📊 Overview — {cat}')
col1, col2, col3 = st.columns(3)
col1.metric('Total Revenue', f"${cat_df['y_revenue'].sum():,.0f}")
col2.metric('Total Profit', f"${cat_df['profit'].sum():,.0f}")
col3.metric('Avg Monthly Revenue', f"${cat_df['y_revenue'].mean():,.0f}")

# Historical Revenue Chart
st.write('**Historical Monthly Revenue**')
fig = go.Figure()
fig.add_trace(go.Scatter(x=cat_df['ds'], y=cat_df['y_revenue'], mode='lines+markers', name='Revenue'))
fig.update_layout(xaxis_title='Date', yaxis_title='Revenue')
st.plotly_chart(fig, use_container_width=True)

# Load Prophet model & Forecast
prophet_model_path = MODEL_DIR / f"prophet_{cat}.pkl"
if prophet_model_path.exists():
    m = joblib.load(prophet_model_path)
    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)

    # Forecast Plot
    st.write("**📅 Forecast (Next 6 Months)**")
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig_forecast.add_trace(go.Scatter(x=cat_df['ds'], y=cat_df['y_revenue'], mode='lines+markers', name='Actual'))
    fig_forecast.update_layout(xaxis_title='Date', yaxis_title='Revenue')
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Prophet Components
    with st.expander("🔍 View Forecast Components"):
        from prophet.plot import plot_components_plotly
        fig_components = plot_components_plotly(m, forecast)
        st.plotly_chart(fig_components, use_container_width=True)

    # ✅ Scenario Simulator
    st.sidebar.header("Scenario Simulator")
    price_change = st.sidebar.slider("Price Change (%)", -20, 20, 0, step=1)
    discount_change = st.sidebar.slider("Discount Change (%)", -20, 20, 0, step=1)
    marketing_change = st.sidebar.slider("Marketing Spend Change (%)", -50, 50, 0, step=5)

    scenario_forecast = forecast.copy()
    scenario_forecast['yhat'] *= (1 + price_change / 100)
    scenario_forecast['yhat'] *= (1 - discount_change / 100)
    scenario_forecast['yhat'] *= (1 + 0.005 * marketing_change)

    # Compare baseline vs scenario revenue
    st.subheader("📊 Scenario Impact")
    rev_change = scenario_forecast['yhat'].sum() - forecast['yhat'].sum()
    col1, col2 = st.columns(2)
    col1.metric("Δ Revenue", f"${rev_change:,.0f}", delta=f"{(rev_change/forecast['yhat'].sum())*100:.1f}%")

    # Profit regression scenario prediction
    profit_model_path = MODEL_DIR / "profit_regressor.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if profit_model_path.exists() and scaler_path.exists():
        profit_model = joblib.load(profit_model_path)
        scaler = joblib.load(scaler_path)

        # Baseline Profit
        last_rev = forecast['yhat'].iloc[-1]
        prev_rev = forecast['yhat'].iloc[-2]
        X_baseline = pd.DataFrame([[last_rev, prev_rev]], columns=['y_revenue', 'prev_revenue'])
        X_baseline_scaled = scaler.transform(X_baseline)
        predicted_profit_baseline = profit_model.predict(X_baseline_scaled)[0]

        # Scenario Profit
        s_last_rev = scenario_forecast['yhat'].iloc[-1]
        s_prev_rev = scenario_forecast['yhat'].iloc[-2]
        X_scenario = pd.DataFrame([[s_last_rev, s_prev_rev]], columns=['y_revenue', 'prev_revenue'])
        X_scenario_scaled = scaler.transform(X_scenario)
        predicted_profit_scenario = profit_model.predict(X_scenario_scaled)[0]

        # Show profit change
        profit_change = predicted_profit_scenario - predicted_profit_baseline
        col2.metric("Δ Profit", f"${profit_change:,.0f}", delta=f"{(profit_change/predicted_profit_baseline)*100:.1f}%")

    # Baseline vs scenario chart
    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Baseline Forecast'))
    fig_scenario.add_trace(go.Scatter(x=scenario_forecast['ds'], y=scenario_forecast['yhat'], mode='lines', name='Scenario Forecast', line=dict(dash='dash')))
    st.plotly_chart(fig_scenario, use_container_width=True)

    # ✅ Gemini AI Insights
    kpis = {
        "total_revenue": int(cat_df['y_revenue'].sum()),
        "total_profit": int(cat_df['profit'].sum()),
        "avg_monthly_revenue": int(cat_df['y_revenue'].mean())
    }
    forecast_points = [
        {"month": str(row['ds'].date()), "forecast_revenue": float(row['yhat']),
         "forecast_profit": None}
        for _, row in scenario_forecast.head(3).iterrows()
    ]

    st.subheader("💡 AI Insights")
    insight_text = generate_insights(cat, kpis, forecast_points)
    st.write(insight_text)

else:
    st.warning(f"No Prophet model found for {cat}. Train it via `python src/models/train_forecast.py`.")

# Predict Profit for Next Month (Baseline Only)
profit_model_path = MODEL_DIR / "profit_regressor.pkl"
scaler_path = MODEL_DIR / "scaler.pkl"
if profit_model_path.exists() and scaler_path.exists():
    profit_model = joblib.load(profit_model_path)
    scaler = joblib.load(scaler_path)

    last_rev = cat_df['y_revenue'].iloc[-1]
    prev_rev = cat_df['y_revenue'].iloc[-2] if len(cat_df) > 1 else last_rev
    X_next = pd.DataFrame([[last_rev, prev_rev]], columns=['y_revenue', 'prev_revenue'])
    X_next_scaled = scaler.transform(X_next)

    predicted_profit = profit_model.predict(X_next_scaled)[0]
    st.metric("💰 Predicted Profit (Next Month — Baseline)", f"${predicted_profit:,.0f}")
else:
    st.warning("Profit regression model not found. Train it via `python src/models/train_regression.py`.")
