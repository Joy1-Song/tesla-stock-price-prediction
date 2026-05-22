import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import get_tree_data


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


X_train, X_test, y_train, y_test, test_dates, feature_cols = get_tree_data()

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = safe_mape(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nXGBoost Results")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R^2  : {r2:.4f}")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 Important Features (XGBoost):")
print(importances.head(10))

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label="Actual Next Close", linewidth=2)
plt.plot(test_dates, pred, label="XGBoost Predicted", linewidth=2)
plt.title("XGBoost: Actual vs Predicted Next-Day Tesla Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()