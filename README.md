---
title: Sales Forecasting Tool
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0 # Specify a recent Streamlit version (check latest if needed)
python_version: 3.11 # Specify Python version (adjust if you used a different one locally, 3.11 often good)
app_file: app.py
pinned: false
license: apache-2.0 # Or choose another license like mit, cc-by-4.0, etc.
---

# Sales Forecasting Tool (Streamlit MVP)

This is a Streamlit application designed to provide basic sales forecasting based on uploaded historical data.

**Features:**
*   Upload Excel data (`.xlsx`)
*   Select Date, Quantity, Account Type, and Product ID columns
*   Select specific product to analyze
*   Filter data by date range and account type ('Sales-*')
*   Display processed data summary and decomposition plots
*   Generate a 30-day forecast using ETS with 95% confidence intervals
*   Visualize results with an interactive Plotly chart
*   Download combined actuals and forecast data

Built using Streamlit, Pandas, Statsmodels, Plotly.