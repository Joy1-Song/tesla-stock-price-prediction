import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

import config
from preprocess import load_raw_data, add_features


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


# Load and preprocess
raw_df = load_raw_data()
feat_df = add_features(raw_df)
feat_df = feat_df.dropna().reset_index(drop=True)

date_to_raw_idx = {d: i for i, d in enumerate(raw_df["Date"])}

# ARIMA one-step-ahead prediction for every row
arima_order = (5, 1, 0)
arima_preds = []

for i in range(len(feat_df)):
    current_date = feat_df.loc[i, "Date"]
    raw_idx = date_to_raw_idx[current_date]

    history = raw_df["Close"].iloc[:raw_idx + 1].values

    model = ARIMA(history, order=arima_order)
    model_fit = model.fit()
    pred_next = model_fit.forecast(steps=1)[0]
    arima_preds.append(pred_next)

feat_df["ARIMA_Pred"] = arima_preds
feat_df["Residual_Target"] = feat_df["Target"] - feat_df["ARIMA_Pred"]
# Split train / test
split_idx = int(len(feat_df) * (1 - config.TEST_SIZE))

train_df = feat_df.iloc[:split_idx].copy()
test_df = feat_df.iloc[split_idx:].copy()

exclude_cols = ["Date", "Target", "ARIMA_Pred", "Residual_Target"]
feature_cols = [col for col in feat_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

y_train_residual = train_df["Residual_Target"]
y_test_actual = test_df["Target"].values

# Train RF on residuals
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=config.RANDOM_SEED,
    n_jobs=-1
)

rf_model.fit(X_train, y_train_residual)
residual_pred_test = rf_model.predict(X_test)
# Final hybrid prediction
arima_pred_test = test_df["ARIMA_Pred"].values
hybrid_pred_test = arima_pred_test + residual_pred_test

# Evaluate
evaluate_model("Pure ARIMA (same aligned samples)", y_test_actual, arima_pred_test)
evaluate_model("Hybrid ARIMA + RF Residual", y_test_actual, hybrid_pred_test)
#Feature importance
importances = pd.Series(
    rf_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nTop 10 Important Features for Residual Correction (Random Forest):")
print(importances.head(10))
#  Plot
plt.figure(figsize=(12, 6))
plt.plot(test_df["Date"].values, y_test_actual, label="Actual Next Close", linewidth=2)
plt.plot(test_df["Date"].values, arima_pred_test, label="Pure ARIMA", linewidth=2)
plt.plot(test_df["Date"].values, hybrid_pred_test, label="Hybrid ARIMA+RF Residual", linewidth=2)
plt.title("Hybrid Model: ARIMA + Random Forest Residual Correction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Save results
result_df = pd.DataFrame({
    "Date": test_df["Date"].values,
    "Actual_Next_Close": y_test_actual,
    "ARIMA_Pred": arima_pred_test,
    "Hybrid_RF_Residual_Pred": hybrid_pred_test,
    "Residual_Pred_By_RF": residual_pred_test
})
result_df.to_csv("hybrid_arima_rf_residual_result.csv", index=False)
print("\nSaved: hybrid_arima_rf_residual_result.csv")