import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import openpyxl
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Import Plotly
import logging
from datetime import date

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Forecasting Tool",
    page_icon="📈",
    layout="wide"
)

# --- Helper Functions ---
# (Keep load_data, generate_ets_forecast_with_intervals, to_excel functions exactly as they were in V1d)
@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_file):
    """Loads data from the uploaded Excel file."""
    try:
        logging.info(f"Attempting to load file: {uploaded_file.name}")
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        if df.empty:
            st.error("Uploaded file is empty.")
            logging.warning(f"Uploaded file {uploaded_file.name} is empty.")
            return None
        logging.info(f"Successfully loaded {len(df)} rows from {uploaded_file.name}")
        df.columns = df.columns.astype(str)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        logging.error(f"Error reading Excel file {uploaded_file.name}: {e}", exc_info=True)
        return None

@st.cache_data(show_spinner="Generating ETS forecast with intervals...")
def generate_ets_forecast_with_intervals(series, steps=30, confidence_level=0.95):
    """Generates forecast using Exponential Smoothing (Holt-Winters) and prediction intervals."""
    if series is None or series.empty:
        logging.warning("Input series for forecasting is empty.")
        return pd.Series(dtype=float), None, None

    min_length_for_ets = 2
    if len(series) < min_length_for_ets:
        st.warning(f"Insufficient data points (< {min_length_for_ets}) for ETS forecasting. Using naive forecast.")
        logging.warning(f"Insufficient data points ({len(series)}) for ETS. Using naive.")
        last_value = series.iloc[-1]
        last_date = series.index[-1]
        if not isinstance(last_date, pd.Timestamp): last_date = pd.to_datetime(last_date)
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        naive_forecast = pd.Series([last_value] * steps, index=forecast_index)
        return naive_forecast, naive_forecast, naive_forecast

    try:
        logging.info(f"Generating ETS forecast for series with {len(series)} points, steps={steps}.")
        model = ExponentialSmoothing(
            series, trend='add', damped_trend=True, seasonal=None, initialization_method='estimated'
        ).fit()
        forecast_point = model.forecast(steps=steps)
        predictions = model.get_prediction(start=forecast_point.index[0], end=forecast_point.index[-1])
        pred_summary = predictions.summary_frame(alpha=1.0-confidence_level)
        forecast_lower = pred_summary[f'pi_lower']
        forecast_upper = pred_summary[f'pi_upper']
        logging.info("ETS forecast and intervals generated successfully.")
        forecast_point[forecast_point < 0] = 0
        forecast_lower[forecast_lower < 0] = 0
        forecast_upper[forecast_upper < 0] = 0
        return forecast_point, forecast_lower, forecast_upper
    except Exception as e:
        st.error(f"Error during ETS forecasting: {e}")
        logging.error(f"Error during ETS forecasting: {e}", exc_info=True)
        return pd.Series(dtype=float), None, None

@st.cache_data
def to_excel(df):
    """Exports DataFrame to Excel format in memory."""
    output = BytesIO()
    df_processed = df.copy()
    try:
        if isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed.index = df_processed.index.strftime('%Y-%m-%d')
    except Exception as e:
        logging.warning(f"Could not format index for Excel export: {e}")
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_processed.to_excel(writer, index=True, sheet_name='ForecastData')
    processed_data = output.getvalue()
    return processed_data

# --- Session State Initialization ---
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'processed_product_data' not in st.session_state: st.session_state.processed_product_data = None
if 'forecast_df' not in st.session_state: st.session_state.forecast_df = None
if 'forecast_lower' not in st.session_state: st.session_state.forecast_lower = None
if 'forecast_upper' not in st.session_state: st.session_state.forecast_upper = None
if 'decomposition_result' not in st.session_state: st.session_state.decomposition_result = None
if 'selected_date_col' not in st.session_state: st.session_state.selected_date_col = None
if 'selected_qty_col' not in st.session_state: st.session_state.selected_qty_col = None
if 'selected_acct_col' not in st.session_state: st.session_state.selected_acct_col = None
if 'selected_product_id_col' not in st.session_state: st.session_state.selected_product_id_col = None
if 'selected_product_instance' not in st.session_state: st.session_state.selected_product_instance = None
if 'unique_products' not in st.session_state: st.session_state.unique_products = []
if 'current_file_name' not in st.session_state: st.session_state.current_file_name = None
if 'processing_error' not in st.session_state: st.session_state.processing_error = False
if 'run_forecast' not in st.session_state: st.session_state.run_forecast = False
if 'date_range_start' not in st.session_state: st.session_state.date_range_start = None
if 'date_range_end' not in st.session_state: st.session_state.date_range_end = None

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader(
        "Upload Sales Data (.xlsx)", type=['xlsx'], accept_multiple_files=False, key='file_uploader'
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.current_file_name:
        logging.info(f"New file uploaded: {uploaded_file.name}")
        st.session_state.current_file_name = uploaded_file.name
        # Reset state
        st.session_state.data_loaded = False; st.session_state.raw_df = None; st.session_state.processed_product_data = None
        st.session_state.forecast_df = None; st.session_state.forecast_lower = None; st.session_state.forecast_upper = None
        st.session_state.decomposition_result = None; st.session_state.selected_date_col = None; st.session_state.selected_qty_col = None
        st.session_state.selected_acct_col = None; st.session_state.selected_product_id_col = None; st.session_state.selected_product_instance = None
        st.session_state.unique_products = []; st.session_state.processing_error = False; st.session_state.run_forecast = False
        st.session_state.date_range_start = None; st.session_state.date_range_end = None
        # Load new data
        st.session_state.raw_df = load_data(uploaded_file)
        if st.session_state.raw_df is not None: st.session_state.data_loaded = True; st.success(f"Loaded: {uploaded_file.name}")
        else: st.session_state.data_loaded = False
        st.rerun()

    st.divider()

    if st.session_state.data_loaded and st.session_state.raw_df is not None:
        st.header("2. Data Preparation")
        df_input = st.session_state.raw_df; available_columns = [""] + df_input.columns.astype(str).tolist()
        st.subheader("Select Columns")
        date_val = st.session_state.selected_date_col if st.session_state.selected_date_col in available_columns else ""
        qty_val = st.session_state.selected_qty_col if st.session_state.selected_qty_col in available_columns else ""
        acct_val = st.session_state.selected_acct_col if st.session_state.selected_acct_col in available_columns else ""
        prod_id_val = st.session_state.selected_product_id_col if st.session_state.selected_product_id_col in available_columns else ""
        date_idx = available_columns.index(date_val) if date_val else (available_columns.index(next((c for c in available_columns if 'date' in c.lower()), '')))
        qty_idx = available_columns.index(qty_val) if qty_val else (available_columns.index(next((c for c in available_columns if 'qty' in c.lower()), '')))
        acct_idx = available_columns.index(acct_val) if acct_val else (available_columns.index(next((c for c in available_columns if 'acct' in c.lower()), '')))
        prod_id_idx = available_columns.index(prod_id_val) if prod_id_val else (available_columns.index(next((c for c in available_columns if 'fg' in c.lower()), '')))

        st.session_state.selected_date_col = st.selectbox("Date Column:", available_columns, index=date_idx, key='date_sel')
        st.session_state.selected_qty_col = st.selectbox("Quantity Column:", available_columns, index=qty_idx, key='qty_sel')
        st.session_state.selected_acct_col = st.selectbox("Account/Type Column:", available_columns, index=acct_idx, key='acct_sel')
        st.session_state.selected_product_id_col = st.selectbox("Product ID Column:", available_columns, index=prod_id_idx, key='prod_id_sel')

        st.subheader("Select Date Range for Analysis")
        min_date = pd.Timestamp.min; max_date = date.today()
        if st.session_state.selected_date_col and st.session_state.selected_date_col in df_input.columns:
            try:
                date_series = pd.to_datetime(df_input[st.session_state.selected_date_col], errors='coerce').dropna()
                if not date_series.empty: min_date = date_series.min().date(); max_date = date_series.max().date()
            except Exception as e: st.warning(f"Could not get date range: {e}")
        if st.session_state.date_range_start is None: st.session_state.date_range_start = min_date
        if st.session_state.date_range_end is None: st.session_state.date_range_end = max_date
        start_date_val = st.session_state.date_range_start; end_date_val = st.session_state.date_range_end
        st.session_state.date_range_start = st.date_input("Start Date", value=start_date_val, min_value=min_date, max_value=max_date, key='start_date')
        st.session_state.date_range_end = st.date_input("End Date", value=end_date_val, min_value=min_date, max_value=max_date, key='end_date')

        st.subheader("Select Product")
        st.session_state.unique_products = []; prod_instance_idx = 0
        if st.session_state.selected_product_id_col and st.session_state.selected_product_id_col in df_input.columns:
             try:
                st.session_state.unique_products = [""] + sorted(df_input[st.session_state.selected_product_id_col].astype(str).unique())
                prod_instance_val = st.session_state.selected_product_instance if st.session_state.selected_product_instance in st.session_state.unique_products else ""
                prod_instance_idx = st.session_state.unique_products.index(prod_instance_val) if prod_instance_val else 0
             except Exception as e: st.warning(f"Could not get unique products: {e}")
        st.session_state.selected_product_instance = st.selectbox(f"Select specific '{st.session_state.selected_product_id_col}':", st.session_state.unique_products, index=prod_instance_idx, key='prod_inst_sel')

        st.divider()
        st.header("3. Generate Forecast")
        all_selections_made = all([st.session_state.selected_date_col, st.session_state.selected_qty_col, st.session_state.selected_acct_col, st.session_state.selected_product_id_col, st.session_state.selected_product_instance, st.session_state.date_range_start, st.session_state.date_range_end])
        if st.button("Process & Forecast", disabled=not all_selections_made):
             if st.session_state.date_range_end < st.session_state.date_range_start: st.error("End Date cannot be before Start Date.")
             else: st.session_state.run_forecast = True; st.rerun()
        if not all_selections_made: st.caption("Select columns, product, and dates.")

    if st.session_state.processed_product_data is not None:
        st.divider(); st.subheader("Processed Data Summary")
        try:
            st.metric(label="Analysis Start", value=st.session_state.processed_product_data.index.min().strftime('%Y-%m-%d'))
            st.metric(label="Analysis End", value=st.session_state.processed_product_data.index.max().strftime('%Y-%m-%d'))
            st.metric(label="Total Qty Processed", value=f"{st.session_state.processed_product_data.sum():,.0f}")
            st.metric(label="Number of Days", value=len(st.session_state.processed_product_data))
        except Exception as e: st.warning(f"Metrics error: {e}")


# --- Main Page Area ---
st.title("Sales Forecasting Tool Results")
if not st.session_state.data_loaded: st.info("⬅️ Upload data and make selections in the sidebar.")

# Processing Logic
if st.session_state.run_forecast:
    st.session_state.processing_error = False; st.session_state.processed_product_data = None
    st.session_state.forecast_df = None; st.session_state.forecast_lower = None; st.session_state.forecast_upper = None
    st.session_state.decomposition_result = None
    date_col = st.session_state.selected_date_col; qty_col = st.session_state.selected_qty_col; acct_col = st.session_state.selected_acct_col
    prod_id_col = st.session_state.selected_product_id_col; prod_instance = st.session_state.selected_product_instance
    start_date = pd.to_datetime(st.session_state.date_range_start); end_date = pd.to_datetime(st.session_state.date_range_end)
    df_input = st.session_state.raw_df.copy()
    with st.spinner("Processing data and generating forecast..."):
        try:
            # (Keep the main processing block: column checks, date conversion, date range filter, sales/product filter, qty conversion/filter, aggregation, decomposition, forecasting)
            # ... [Identical data processing logic as previous version] ...
            logging.info(f"Processing: {prod_instance} from {start_date.date()} to {end_date.date()}")
            required_cols = [date_col, qty_col, acct_col, prod_id_col]
            for col in required_cols:
                if col not in df_input.columns: raise KeyError(f"Required column '{col}' not found.")
            df_input[date_col] = pd.to_datetime(df_input[date_col], errors='coerce')
            df_input.dropna(subset=[date_col], inplace=True)
            date_range_filter = (df_input[date_col] >= start_date) & (df_input[date_col] <= end_date)
            df_filtered_date = df_input[date_range_filter].copy()
            if df_filtered_date.empty: st.warning(f"No data found within selected dates."); st.session_state.processing_error = True
            else:
                sales_filter = df_filtered_date[acct_col].astype(str).str.lower().str.startswith('sales-')
                product_filter = df_filtered_date[prod_id_col].astype(str) == prod_instance
                df_filtered = df_filtered_date[sales_filter & product_filter].copy()
                if df_filtered.empty: st.warning(f"No 'Sales-' transactions for '{prod_instance}' in dates."); st.session_state.processing_error = True
                else:
                    logging.info(f"Found {len(df_filtered)} sales rows for {prod_instance} in date range.")
                    df_filtered[qty_col] = pd.to_numeric(df_filtered[qty_col], errors='coerce')
                    df_filtered = df_filtered[df_filtered[qty_col] > 0]
                    df_filtered.dropna(subset=[date_col, qty_col], inplace=True)
                    if df_filtered.empty: st.warning(f"No valid data for '{prod_instance}' after cleaning."); st.session_state.processing_error = True
                    else:
                        df_processed = df_filtered[[date_col, qty_col]].set_index(date_col).sort_index()
                        df_processed.rename(columns={qty_col: 'ActualQuantity'}, inplace=True)
                        df_daily = df_processed['ActualQuantity'].resample('D').sum().fillna(0)
                        if df_daily.empty: st.warning(f"Empty timeseries after aggregation for '{prod_instance}'."); st.session_state.processing_error = True
                        else:
                            st.session_state.processed_product_data = df_daily
                            logging.info(f"Processed data length for {prod_instance}: {len(df_daily)}")
                            seasonal_period = 7
                            if len(df_daily) >= 2 * seasonal_period:
                                try:
                                    decompose_result = seasonal_decompose(df_daily, model='additive', period=seasonal_period)
                                    st.session_state.decomposition_result = decompose_result; logging.info("Decomposition successful.")
                                except Exception as decomp_e: st.warning(f"Decomposition failed: {decomp_e}"); st.session_state.decomposition_result = None
                            else: st.info(f"Data too short ({len(df_daily)} days) for weekly decomposition."); st.session_state.decomposition_result = None
                            st.success(f"Data processed for {prod_instance}!")
                            forecast_point, forecast_lower, forecast_upper = generate_ets_forecast_with_intervals(df_daily, steps=30)
                            if not forecast_point.empty:
                                st.session_state.forecast_lower = forecast_lower; st.session_state.forecast_upper = forecast_upper
                                forecast_df = pd.DataFrame({'ForecastQuantity': forecast_point,'LowerBound': forecast_lower,'UpperBound': forecast_upper})
                                combined_df = pd.merge(df_daily.to_frame(name='ActualQuantity'), forecast_df, left_index=True, right_index=True, how='outer')
                                st.session_state.forecast_df = combined_df; logging.info("Forecast merged.")
                            else: st.warning("Forecast failed."); st.session_state.processing_error = True
        except KeyError as e: st.error(f"Column Error: {e}."); logging.error(f"KeyError: {e}", exc_info=True); st.session_state.processing_error = True
        except Exception as e: st.error(f"Processing error: {e}"); logging.exception("Processing error:"); st.session_state.processing_error = True
    st.session_state.run_forecast = False; st.rerun()

# --- Display Area using Tabs ---
if st.session_state.data_loaded and st.session_state.raw_df is not None:
    tab_titles = ["Raw Data Preview", "Processed Data & Decomposition", "Forecast & Visualization", "Download Results"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1: # (Keep Raw Data Preview as before)
        st.subheader("Uploaded Raw Data Preview")
        st.dataframe(st.session_state.raw_df.head(100))
        st.caption(f"Displaying first 100 rows of '{st.session_state.current_file_name}'.")

    with tab2: # (Keep Processed Data & Decomposition as before)
        st.subheader("Processed Data for Selected Product")
        if st.session_state.processed_product_data is not None:
             st.write(f"Aggregated daily 'ActualQuantity' for '{st.session_state.selected_product_instance}'")
             st.dataframe(st.session_state.processed_product_data.to_frame().style.format("{:.0f}"))
             st.divider(); st.subheader("Time Series Decomposition")
             if st.session_state.decomposition_result is not None:
                 try: fig = st.session_state.decomposition_result.plot(); fig.set_size_inches(10, 8); st.pyplot(fig)
                 except Exception as plot_e: st.error(f"Plotting error: {plot_e}")
             elif not st.session_state.processing_error: st.info("Decomposition not performed.")
        elif st.session_state.processing_error: st.warning("Could not process data.")
        else: st.info("⬅️ Click 'Process & Forecast'.")

    with tab3:
        st.subheader("Forecast Values & Visualization")
        if st.session_state.forecast_df is not None:
            # Forecast Preview Table (Keep as before)
            st.write("Forecast Values Preview (Next 30 Days with 95% CI):")
            forecast_preview = st.session_state.forecast_df[st.session_state.forecast_df['ForecastQuantity'].notna() & st.session_state.forecast_df['ActualQuantity'].isna()]
            if not forecast_preview.empty:
                forecast_preview.index = forecast_preview.index.strftime('%Y-%m-%d')
                st.dataframe(forecast_preview[['LowerBound', 'ForecastQuantity', 'UpperBound']].style.format("{:.2f}"))
            else: st.write("No forecast values generated.")
            st.divider()

            # --- <<< PLOTLY CHART CODE >>> ---
            st.write("Forecast Chart:")
            plot_df = st.session_state.forecast_df.copy()
            # Ensure numeric types for plotting
            for col in ['ActualQuantity', 'ForecastQuantity', 'LowerBound', 'UpperBound']:
                 if col not in plot_df.columns: plot_df[col] = np.nan
                 plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

            fig = go.Figure()

            # Confidence Interval Area (Plot bands first for layering)
            # Need to ensure lower and upper bounds are aligned with the forecast index for plotting
            ci_index = plot_df.index[plot_df['ForecastQuantity'].notna()] # Index where forecast exists
            ci_lower = plot_df.loc[ci_index, 'LowerBound']
            ci_upper = plot_df.loc[ci_index, 'UpperBound']

            fig.add_trace(go.Scatter(
                x=ci_index, y=ci_upper,
                mode='lines', line=dict(width=0), name='Upper Bound CI',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=ci_index, y=ci_lower,
                mode='lines', line=dict(width=0), fillcolor='rgba(255, 165, 0, 0.3)', fill='tonexty', # Orange shade for CI
                name='95% Confidence Interval', showlegend=True # Show CI in legend
            ))

            # Actual Quantity Line
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df['ActualQuantity'],
                mode='lines+markers', name='Actual Quantity', line=dict(color='rgb(31, 119, 180)'), # Blue
                marker=dict(size=4)
            ))

            # Forecast Quantity Line
            fig.add_trace(go.Scatter(
                x=ci_index, y=plot_df.loc[ci_index, 'ForecastQuantity'], # Use ci_index here too
                mode='lines+markers', name='Forecast Quantity', line=dict(color='rgb(214, 39, 40)', dash='dash'), # Red dashed
                marker=dict(size=4)
            ))

            # Update layout
            fig.update_layout(
                title=f'Sales Forecast: {st.session_state.selected_product_instance}',
                xaxis_title='Date',
                yaxis_title='Quantity',
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Interactive chart: Historical actuals, ETS forecast, and 95% confidence interval (shaded).")
            # --- <<< END PLOTLY CHART CODE >>> ---

        elif st.session_state.processing_error: st.warning("Could not generate forecast.")
        else: st.info("⬅️ Click 'Process & Forecast'.")

    with tab4: # (Keep Download logic as before)
         st.subheader("Download Results")
         if st.session_state.forecast_df is not None:
            st.markdown("Download combined historical/forecast data.")
            download_df = st.session_state.forecast_df.round(2)
            excel_data = to_excel(download_df.fillna(''))
            st.download_button(
                label="📥 Download Forecast Data (Excel)", data=excel_data,
                file_name=f"forecast_{st.session_state.selected_product_instance}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
         elif st.session_state.processing_error: st.warning("Cannot download results.")
         else: st.info("⬅️ Click 'Process & Forecast'.")

elif uploaded_file is None and not st.session_state.data_loaded:
    st.info("⬅️ Upload data and make selections.")

# --- Sidebar Info ---
# (Keep as before)
st.sidebar.divider()
st.sidebar.header("Tool Information")
st.sidebar.info("Sales Forecasting Tool - V1e (Plotly Chart)") # Updated version label
if st.session_state.current_file_name: st.sidebar.write(f"**File:** {st.session_state.current_file_name}")
if st.session_state.selected_product_instance: st.sidebar.write(f"**Product:** {st.session_state.selected_product_instance}")
if st.session_state.date_range_start and st.session_state.date_range_end: st.sidebar.write(f"**Dates:** {st.session_state.date_range_start.strftime('%Y-%m-%d')} to {st.session_state.date_range_end.strftime('%Y-%m-%d')}")