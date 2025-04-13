---
title: Sales Forecasting Tool
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0 # Or check latest stable version if preferred
python_version: 3.11 # Or 3.10, 3.12 - should match your local env ideally
app_file: app.py
pinned: false
license: apache-2.0 # Feel free to change if needed (e.g., mit)
---

# Sales Forecasting Tool (Streamlit MVP+)

Streamlit application for basic sales forecasting based on uploaded historical data.

**Features:**
*   Upload Excel data (`.xlsx`)
*   Select Date, Quantity, Account Type, and Product ID columns
*   Select specific product to analyze
*   Filter data by date range and account type ('Sales-*')
*   Display processed data summary and decomposition plots
*   Generate a 30-day forecast using ETS with 95% confidence intervals
*   Visualize results with an interactive Plotly chart
*   Download combined actuals and forecast data

Built using Streamlit, Pandas, Statsmodels, Plotly, Matplotlib, Openpyxl.
