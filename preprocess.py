import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config

# 1. Read raw data
def load_raw_data():
    df = pd.read_csv(config.DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Adj Close" in df.columns:
        if np.allclose(df["Adj Close"], df["Close"]):
            df = df.drop(columns=["Adj Close"])

    return df

# 2. Basic indicator functions
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df.loc[i, "Close"] > df.loc[i - 1, "Close"]:
            obv.append(obv[-1] + df.loc[i, "Volume"])
        elif df.loc[i, "Close"] < df.loc[i - 1, "Close"]:
            obv.append(obv[-1] - df.loc[i, "Volume"])
        else:
            obv.append(obv[-1])
    return obv


def compute_pvt(df):
    pvt = [0]
    for i in range(1, len(df)):
        value = ((df.loc[i, "Close"] - df.loc[i - 1, "Close"]) / (df.loc[i - 1, "Close"] + 1e-9)) * df.loc[i, "Volume"]
        pvt.append(pvt[-1] + value)
    return pvt


# 3. Feature engineering
def add_features(df):
    df = df.copy()

    #Basic features
    if config.USE_BASIC_FEATURES:
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

    #Momentum features
    if config.USE_MOMENTUM_FEATURES:
        df["RSI14"] = compute_rsi(df["Close"], window=14)
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    #Volume features
    if config.USE_VOLUME_FEATURES:
        df["Volume_change"] = df["Volume"].pct_change(1)
        df["Volume_MA5"] = df["Volume"].rolling(5).mean()
        df["Volume_MA10"] = df["Volume"].rolling(10).mean()
        df["Volume_ratio"] = df["Volume"] / (df["Volume_MA5"] + 1e-9)
        df["OBV"] = compute_obv(df)
        df["PVT"] = compute_pvt(df)

    #Candle features
    if config.USE_CANDLE_FEATURES:
        df["Range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
        df["OC_pct"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-9)
        df["Body_pct"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-9)
        df["Upper_shadow_pct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (df["Open"] + 1e-9)
        df["Lower_shadow_pct"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (df["Open"] + 1e-9)
        df["Close_position_pct"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)

    #Volatility features
    if config.USE_VOLATILITY_FEATURES:
        df["Volatility_5"] = df["Close"].pct_change().rolling(5).std()
        df["Volatility_10"] = df["Close"].pct_change().rolling(10).std()
        df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()

        df["BB_Middle"] = df["Close"].rolling(20).mean()
        df["BB_Std"] = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-9)
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    #Extended features
    if config.USE_EXTENDED_FEATURES:
        df["Return_10"] = df["Close"].pct_change(10)
        df["Return_20"] = df["Close"].pct_change(20)

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = true_range.rolling(14).mean()
        df["ATR_14_pct"] = df["ATR_14"] / (df["Close"] + 1e-9)

        low_14 = df["Low"].rolling(14).min()
        high_14 = df["High"].rolling(14).max()
        df["Stoch_K"] = (df["Close"] - low_14) / (high_14 - low_14 + 1e-9) * 100
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        df["Volume_MA20"] = df["Volume"].rolling(20).mean()
        df["Volume_ratio_20"] = df["Volume"] / (df["Volume_MA20"] + 1e-9)
        df["Price_Volume"] = df["Return_1"] * df["Volume_change"]

    #Target
    df["Target"] = df["Close"].shift(-1)

    return df

# 4. Feature list
def get_feature_columns(df):
    exclude_cols = ["Date", "Target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

# 5. Data for ARIMA
def get_arima_data():
    df = load_raw_data()
    df = df[["Date", "Close"]].copy()

    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test

# 6. Data for tree models
def get_tree_data():
    df = load_raw_data()
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df["Target"]

    split_idx = int(len(df) * (1 - config.TEST_SIZE))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    test_dates = df["Date"].iloc[split_idx:]

    return X_train, X_test, y_train, y_test, test_dates, feature_cols

# 7. Sequence creation for DL
def create_sequences(data_df, feature_cols, seq_len):
    X_seq, y_seq, dates = [], [], []

    for i in range(seq_len, len(data_df)):
        X_seq.append(data_df[feature_cols].iloc[i-seq_len:i].values)
        y_seq.append(data_df["Target"].iloc[i])
        dates.append(data_df["Date"].iloc[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), np.array(dates)


# 8. Data for deep learning
def get_dl_data():
    df = load_raw_data()
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(df)

    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    feature_scaler = None
    target_scaler = None

    if config.USE_SCALING_FOR_DL:
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])

        train_df[["Target"]] = target_scaler.fit_transform(train_df[["Target"]])
        test_df[["Target"]] = target_scaler.transform(test_df[["Target"]])

    X_train, y_train, _ = create_sequences(train_df, feature_cols, config.SEQUENCE_LENGTH)
    X_test, y_test, test_dates = create_sequences(test_df, feature_cols, config.SEQUENCE_LENGTH)

    if target_scaler is not None:
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_test_original = y_test.copy()

    return X_train, X_test, y_train, y_test, y_test_original, test_dates, feature_cols, target_scaler