import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

from preprocess import get_arima_data


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


train, test = get_arima_data()

train_series = train["Close"].values
test_series = test["Close"].values
test_dates = test["Date"].values

history = list(train_series)
predictions = []

for t in range(len(test_series)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    predictions.append(forecast[0])
    history.append(test_series[t])

predictions = np.array(predictions)

mae = mean_absolute_error(test_series, predictions)
rmse = np.sqrt(mean_squared_error(test_series, predictions))
mape = safe_mape(test_series, predictions)
r2 = r2_score(test_series, predictions)

print("\nARIMA Results")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R^2  : {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(test_dates, test_series, label="Actual Close", linewidth=2)
plt.plot(test_dates, predictions, label="ARIMA Predicted", linewidth=2)
plt.title("ARIMA: Actual vs Predicted Tesla Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()