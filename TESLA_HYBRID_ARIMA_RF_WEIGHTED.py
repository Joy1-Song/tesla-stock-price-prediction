import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

import config
from preprocess import get_arima_data, get_tree_data


# Metrics
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Results")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.4f}%")
    print(f"R^2  : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ARIMA predictions
train_arima, test_arima = get_arima_data()

train_series = train_arima["Close"].values
test_series = test_arima["Close"].values
test_dates_arima = test_arima["Date"].values

history = list(train_series)
arima_preds = []

for t in range(len(test_series)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    arima_preds.append(forecast)
    history.append(test_series[t])

arima_preds = np.array(arima_preds)

# Random Forest predictions
X_train, X_test, y_train, y_test, test_dates_rf, feature_cols = get_tree_data()

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=config.RANDOM_SEED,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Align samples by Date
arima_df = pd.DataFrame({
    "Date": pd.to_datetime(test_dates_arima),
    "Actual": test_series,
    "ARIMA_Pred": arima_preds
})

rf_df = pd.DataFrame({
    "Date": pd.to_datetime(test_dates_rf.values),
    "Actual_RF": y_test.values,
    "RF_Pred": rf_preds
})

merged = pd.merge(arima_df, rf_df, on="Date", how="inner")
y_true = merged["Actual"].values
arima_aligned = merged["ARIMA_Pred"].values
rf_aligned = merged["RF_Pred"].values
test_dates = merged["Date"].values
#Weighted ensemble
w_arima = 0.7
w_rf = 0.3

hybrid_pred = w_arima * arima_aligned + w_rf * rf_aligned

# Evaluate
evaluate_model("Pure ARIMA (aligned)", y_true, arima_aligned)
evaluate_model("Pure Random Forest (aligned)", y_true, rf_aligned)
evaluate_model(f"Hybrid ARIMA+RF Weighted ({w_arima:.1f}/{w_rf:.1f})", y_true, hybrid_pred)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_true, label="Actual Next Close", linewidth=2)
plt.plot(test_dates, arima_aligned, label="ARIMA Predicted", linewidth=2)
plt.plot(test_dates, rf_aligned, label="RF Predicted", linewidth=2)
plt.plot(test_dates, hybrid_pred, label="Hybrid Weighted Predicted", linewidth=2)
plt.title("Hybrid ARIMA + Random Forest (Weighted)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save result
result_df = pd.DataFrame({
    "Date": test_dates,
    "Actual_Next_Close": y_true,
    "ARIMA_Pred": arima_aligned,
    "RF_Pred": rf_aligned,
    "Hybrid_Weighted_Pred": hybrid_pred
})
result_df.to_csv("hybrid_arima_rf_weighted_result.csv", index=False)
print("\nSaved: hybrid_arima_rf_weighted_result.csv")