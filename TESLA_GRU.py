import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from preprocess import get_dl_data


# Device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# Dataset

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# GRU Model
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
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
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


#  Metrics

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


# Load preprocessed data

X_train, X_test, y_train, y_test, y_test_original, test_dates, feature_cols, target_scaler = get_dl_data()

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


#  Build model

model = GRURegressor(input_size=len(feature_cols)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


#Train
best_loss = float("inf")
best_state = None
train_losses = []

for epoch in range(config.EPOCHS):
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
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {epoch_loss:.6f}")

model.load_state_dict(best_state)


#Predict

model.eval()
pred_list = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(DEVICE)
        pred_batch = model(X_batch).cpu().numpy()
        pred_list.append(pred_batch)

pred_scaled = np.vstack(pred_list).flatten()

if target_scaler is not None:
    pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
else:
    pred_original = pred_scaled.copy()

# Evaluate
mae = mean_absolute_error(y_test_original, pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
mape = safe_mape(y_test_original, pred_original)
r2 = r2_score(y_test_original, pred_original)

print("\nGRU Results")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R^2  : {r2:.4f}")


# lot prediction

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_original, label="Actual Next Close", linewidth=2)
plt.plot(test_dates, pred_original, label="GRU Predicted", linewidth=2)
plt.title("GRU: Actual vs Predicted Next-Day Tesla Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#  Plot training loss

plt.figure(figsize=(10, 5))
plt.plot(train_losses, linewidth=2)
plt.title("GRU Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# Save result
result_df = pd.DataFrame({
    "Date": test_dates,
    "Actual_Close": y_test_original,
    "Predicted_Close": pred_original
})
result_df.to_csv("gru_prediction_result.csv", index=False)
print("\nSaved: gru_prediction_result.csv")