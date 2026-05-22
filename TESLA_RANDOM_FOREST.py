import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from preprocess import get_tree_data


# 1. Metrics
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


# 2. Load preprocessed tree data
X_train, X_test, y_train, y_test, test_dates, feature_cols = get_tree_data()

# 3. Build model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=config.RANDOM_SEED,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# 4. Evaluate
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = safe_mape(y_test, pred)
r2 = r2_score(y_test, pred)

print("\nRandom Forest Results")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"R^2  : {r2:.4f}")

# 5. Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 Important Features (Random Forest):")
print(importances.head(10))

# 6. Plot
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test.values, label="Actual Next Close", linewidth=2)
plt.plot(test_dates, pred, label="Random Forest Predicted", linewidth=2)
plt.title("Random Forest: Actual vs Predicted Next-Day Tesla Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Save result
result_df = pd.DataFrame({
    "Date": test_dates.values,
    "Actual_Close": y_test.values,
    "Predicted_Close": pred
})
result_df.to_csv("rf_prediction_result.csv", index=False)
print("\nSaved: rf_prediction_result.csv")