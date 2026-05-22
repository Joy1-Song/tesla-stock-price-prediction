import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 10
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
split_idx = int(len(df_feat) * 0.8)

train_df = df_feat.iloc[:split_idx].copy()
test_df = df_feat.iloc[split_idx:].copy()

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])

train_df[["Target"]] = target_scaler.fit_transform(train_df[["Target"]])
test_df[["Target"]] = target_scaler.transform(test_df[["Target"]])

def create_sequences(data_df, feature_cols, seq_len):
    X_seq, y_seq, dates = [], [], []

    for i in range(seq_len, len(data_df)):
        X_seq.append(data_df[feature_cols].iloc[i-seq_len:i].values)
        y_seq.append(data_df["Target"].iloc[i])
        dates.append(data_df["Date"].iloc[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), np.array(dates)

X_train, y_train, _ = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
X_test, y_test, test_dates = create_sequences(test_df, feature_cols, SEQUENCE_LENGTH)

y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = LSTMRegressor(input_size=len(feature_cols)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_loss = float("inf")
best_state = None
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_state = model.state_dict()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_loss:.6f}")

model.load_state_dict(best_state)

model.eval()
pred_list = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(DEVICE)
        pred = model(X_batch).cpu().numpy()
        pred_list.append(pred)

pred_scaled = np.vstack(pred_list).flatten()
pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

mae = mean_absolute_error(y_test_original, pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
mape = safe_mape(y_test_original, pred_original)
r2 = r2_score(y_test_original, pred_original)

print("\nLSTM Results")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R^2  : {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_original, label="Actual Next Close", linewidth=2)
plt.plot(test_dates, pred_original, label="LSTM Predicted", linewidth=2)
plt.title("LSTM: Actual vs Predicted Next-Day Tesla Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_losses, linewidth=2)
plt.title("LSTM Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

result_df = pd.DataFrame({
    "Date": test_dates,
    "Actual_Close": y_test_original,
    "Predicted_Close": pred_original
})
result_df.to_csv("lstm_prediction_result.csv", index=False)
print("\nSaved: lstm_prediction_result.csv")