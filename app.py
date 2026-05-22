import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Tesla Stock Price Prediction Demo",
    layout="wide"
)

model_files = {
    "ARIMA": "arima_prediction_result.csv",
    "Random Forest": "rf_prediction_result.csv",
    "XGBoost": "xgb_prediction_result.csv",
    "GRU": "gru_prediction_result.csv",
    "LightGBM": "lgb_prediction_result.csv",
    "LSTM": "lstm_prediction_result.csv",
    "Hybrid ARIMA + RF Weighted": "hybrid_arima_rf_weighted_result.csv",
    "Hybrid ARIMA + RF Residual": "hybrid_arima_rf_residual_result.csv",
    "Hybrid ARIMA + XGB Residual": "hybrid_arima_xgb_prediction_result.csv",
}

prediction_column_map = {
    "ARIMA": ("Actual_Close", "Predicted_Close"),
    "Random Forest": ("Actual_Close", "Predicted_Close"),
    "XGBoost": ("Actual_Close", "Predicted_Close"),
    "GRU": ("Actual_Close", "Predicted_Close"),
    "LightGBM": ("Actual_Close", "Predicted_Close"),
    "LSTM": ("Actual_Close", "Predicted_Close"),
    "Hybrid ARIMA + RF Weighted": ("Actual_Next_Close", "Hybrid_Weighted_Pred"),
    "Hybrid ARIMA + RF Residual": ("Actual_Next_Close", "Hybrid_RF_Residual_Pred"),
    "Hybrid ARIMA + XGB Residual": ("Actual_Next_Close", "Hybrid_Pred"),
}

@st.cache_data
def load_raw_data():
    df = pd.read_csv("TESLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

@st.cache_data
def load_prediction_result(file_path):
    path = Path(file_path)
    if path.exists():
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    return None

def calc_metrics(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)

    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    return mae, rmse, mape, r2

def plot_price_history(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Close"], linewidth=2, label="Close")
    ax.set_title("Tesla Historical Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_prediction(df, model_name, actual_col, pred_col):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df[actual_col], linewidth=2, label="Actual")
    ax.plot(df["Date"], df[pred_col], linewidth=2, label=f"{model_name} Predicted")
    ax.set_title(f"{model_name}: Actual vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig

def build_display_df(df, actual_col, pred_col):
    display_df = df.copy()
    rename_map = {
        actual_col: "Actual",
        pred_col: "Predicted"
    }
    display_df = display_df.rename(columns=rename_map)
    keep_cols = ["Date", "Actual", "Predicted"]
    extra_cols = [c for c in display_df.columns if c not in keep_cols]
    return display_df[keep_cols + extra_cols]

st.title("Tesla Stock Price Prediction Demo")
st.markdown(
    """
This demo shows the prediction results of multiple models for Tesla stock price forecasting.

It includes:
- historical price visualization
- model selection
- evaluation metrics
- predicted vs actual comparison
- next-day predicted closing price
- hybrid model comparison
"""
)

raw_df = load_raw_data()

# Top overview
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Tesla Closing Price")
    st.pyplot(plot_price_history(raw_df), width="stretch")

with col2:
    st.subheader("Dataset Overview")
    st.write(f"**Rows:** {len(raw_df)}")
    st.write(f"**Date Range:** {raw_df['Date'].min().date()} to {raw_df['Date'].max().date()}")
    st.write(f"**Latest Close:** {raw_df['Close'].iloc[-1]:.2f}")
    st.write("**Project Focus:** Next-day closing price prediction")
    st.write("**Main Models:** ARIMA, Random Forest, GRU")
    st.write("**Extended Models:** XGBoost, LightGBM, LSTM, Hybrid Models")

st.divider()

# Model selection
st.subheader("Model Prediction Results")
selected_model = st.selectbox(
    "Choose a model",
    list(model_files.keys())
)

pred_df = load_prediction_result(model_files[selected_model])

if pred_df is None:
    st.warning(f"Prediction file for {selected_model} not found: {model_files[selected_model]}")
    st.info("Please run the corresponding model file first to generate prediction results.")
else:
    actual_col, pred_col = prediction_column_map[selected_model]

    if actual_col not in pred_df.columns or pred_col not in pred_df.columns:
        st.error(
            f"Column mismatch in {model_files[selected_model]}. "
            f"Expected columns: {actual_col}, {pred_col}"
        )
    else:
        # Metrics
        mae, rmse, mape, r2 = calc_metrics(pred_df[actual_col], pred_df[pred_col])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{mae:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("MAPE", f"{mape:.4f}%")
        m4.metric("R²", f"{r2:.4f}")

        st.pyplot(plot_prediction(pred_df, selected_model, actual_col, pred_col), width="stretch")

        st.subheader("Next-Day Prediction")
        last_row = pred_df.iloc[-1]
        st.write(f"**Prediction Date:** {last_row['Date'].date()}")
        st.write(f"**Actual Close (test set):** {last_row[actual_col]:.2f}")
        st.write(f"**Predicted Close:** {last_row[pred_col]:.2f}")

        if selected_model == "Hybrid ARIMA + RF Weighted":
            st.subheader("Hybrid Components")
            if "ARIMA_Pred" in pred_df.columns and "RF_Pred" in pred_df.columns:
                st.write(f"**ARIMA Prediction:** {last_row['ARIMA_Pred']:.2f}")
                st.write(f"**Random Forest Prediction:** {last_row['RF_Pred']:.2f}")
                st.write(f"**Hybrid Weighted Prediction:** {last_row[pred_col]:.2f}")

        elif selected_model == "Hybrid ARIMA + RF Residual":
            st.subheader("Hybrid Components")
            if "ARIMA_Pred" in pred_df.columns and "Residual_Pred_By_RF" in pred_df.columns:
                st.write(f"**ARIMA Prediction:** {last_row['ARIMA_Pred']:.2f}")
                st.write(f"**Residual Correction by RF:** {last_row['Residual_Pred_By_RF']:.2f}")
                st.write(f"**Hybrid Residual Prediction:** {last_row[pred_col]:.2f}")

        elif selected_model == "Hybrid ARIMA + XGB Residual":
            st.subheader("Hybrid Components")
            if "ARIMA_Pred" in pred_df.columns and "Residual_Pred_By_XGB" in pred_df.columns:
                st.write(f"**ARIMA Prediction:** {last_row['ARIMA_Pred']:.2f}")
                st.write(f"**Residual Correction by XGBoost:** {last_row['Residual_Pred_By_XGB']:.2f}")
                st.write(f"**Hybrid Residual Prediction:** {last_row[pred_col]:.2f}")

        st.subheader("Prediction Table Preview")
        display_df = build_display_df(pred_df, actual_col, pred_col)
        st.dataframe(display_df.tail(10), width="stretch")

st.divider()

st.subheader("Stage 4 Demo Summary")
st.markdown(
    """
This Streamlit app serves as a simple deployment and demo interface for Tesla stock price prediction.

It allows users to:
- view Tesla historical stock prices
- compare multiple trained models
- inspect evaluation metrics
- visualize actual vs predicted values
- review hybrid model outputs
"""
)