import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("TESLA.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if "Adj Close" in df.columns:
    if np.allclose(df["Adj Close"], df["Close"]):
        df = df.drop(columns=["Adj Close"])


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(data):
    df = data.copy()

    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_5"] = df["Close"].pct_change(5)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["Close_MA5_ratio"] = df["Close"] / (df["MA5"] + 1e-9)
    df["Close_MA10_ratio"] = df["Close"] / (df["MA10"] + 1e-9)

    df["Range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["OC_pct"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-9)
    df["Volatility_5"] = df["Return_1"].rolling(5).std()
    df["Volatility_10"] = df["Return_1"].rolling(10).std()

    df["Volume_change"] = df["Volume"].pct_change(1)
    df["Volume_MA5"] = df["Volume"].rolling(5).mean()
    df["Volume_MA10"] = df["Volume"].rolling(10).mean()
    df["Volume_ratio"] = df["Volume"] / (df["Volume_MA5"] + 1e-9)

    df["RSI14"] = compute_rsi(df["Close"], window=14)

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    obv = [0]
    for i in range(1, len(df)):
        if df.loc[i, "Close"] > df.loc[i - 1, "Close"]:
            obv.append(obv[-1] + df.loc[i, "Volume"])
        elif df.loc[i, "Close"] < df.loc[i - 1, "Close"]:
            obv.append(obv[-1] - df.loc[i, "Volume"])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    pvt = [0]
    for i in range(1, len(df)):
        value = ((df.loc[i, "Close"] - df.loc[i - 1, "Close"]) / (df.loc[i - 1, "Close"] + 1e-9)) * df.loc[i, "Volume"]
        pvt.append(pvt[-1] + value)
    df["PVT"] = pvt

    df["Target"] = df["Close"].shift(-1)
    return df

df_feat = add_features(df)
df_feat = df_feat.dropna().reset_index(drop=True)


feature_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1", "Return_3", "Return_5",
    "MA5", "MA10", "MA20",
    "EMA12", "EMA26",
    "Close_MA5_ratio", "Close_MA10_ratio",
    "Range_pct", "OC_pct",
    "Volatility_5", "Volatility_10",
    "Volume_change", "Volume_MA5", "Volume_MA10", "Volume_ratio",
    "RSI14", "MACD", "MACD_signal", "MACD_hist",
    "OBV", "PVT"
]

X = df_feat[feature_cols]
y = df_feat["Target"]

split_idx = int(len(df_feat) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_dates = df_feat["Date"].iloc[split_idx:]


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

xgb_pred = None
try:
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    evaluate_model("XGBoost", y_test, xgb_pred)

    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Important Features (XGBoost):")
    print(importances.head(10))

    pd.DataFrame({
        "Date": test_dates,
        "Actual_Close": y_test.values,
        "Predicted_Close": xgb_pred
    }).to_csv("xgb_prediction_result.csv", index=False)

except Exception as e:
    print("\nXGBoost not available:", e)

lgb_pred = None
try:
    from lightgbm import LGBMRegressor

    lgb_model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    evaluate_model("LightGBM", y_test, lgb_pred)

    importances = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Important Features (LightGBM):")
    print(importances.head(10))

    pd.DataFrame({
        "Date": test_dates,
        "Actual_Close": y_test.values,
        "Predicted_Close": lgb_pred
    }).to_csv("lgb_prediction_result.csv", index=False)

except Exception as e:
    print("\nLightGBM not available:", e)

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label="Actual Next Close", linewidth=2)

if xgb_pred is not None:
    plt.plot(test_dates, xgb_pred, label="XGBoost Predicted", linewidth=2)

if lgb_pred is not None:
    plt.plot(test_dates, lgb_pred, label="LightGBM Predicted", linewidth=2)

plt.title("Actual vs Predicted Next-Day Tesla Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()