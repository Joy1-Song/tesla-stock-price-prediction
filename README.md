# Tesla Stock Price Prediction

A seminar group project on **next-day Tesla closing price prediction** using traditional statistical methods, machine learning, deep learning, and hybrid models.

## 1. Project Summary
This project investigates whether Tesla's next-day closing price can be forecast from daily OHLCV data and engineered technical indicators. We compare three main model families:

- **Traditional statistical model:** ARIMA  
- **Machine learning models:** Random Forest, XGBoost, LightGBM  
- **Deep learning models:** GRU, LSTM  
- **Hybrid models:** ARIMA + Random Forest / XGBoost combinations

The project also includes:
- a structured preprocessing pipeline
- technical analysis and external-event interpretation
- a Streamlit demo for presentation/deployment
- seminar slides and a bilingual speech script

---

## 2. Research Question
**Which type of model is most suitable for Tesla next-day closing price prediction on a small daily stock dataset?**

---

## 3. Main Findings
From the completed experiments:

- **ARIMA** achieved the best overall full-test performance among the core models.
- **Random Forest** was the strongest machine learning model.
- **GRU/LSTM** underperformed on this small daily dataset.
- **Hybrid ARIMA + Random Forest (weighted)** was a useful extension experiment.
- **Residual-correction hybrids** did not outperform the strongest single models.

These results suggest that for limited daily Tesla data, **simpler traditional and tree-based methods outperform deeper sequence models**.

---

## 4. Repository Structure

```text
.
├── app.py
├── config.py
├── preprocess.py
├── TESLA.csv
├── TESLA.py
├── TESLA_ARIMA.py
├── TESLA_RANDOM_FOREST.py
├── TESLA_XGBOOST.py
├── TESLA_GRU.py
├── TESLA_LSTM.py
├── TESLA_HYBRID_ARIMA_RF_WEIGHTED.py
├── TESLA_HYBRID_ARIMA_RF_RESIDUAL.py
├── TESLA_HYBRID_ARIMA_XGB.py
├── arima_prediction_result.csv
├── rf_prediction_result.csv
├── xgb_prediction_result.csv
├── gru_prediction_result.csv
├── lgb_prediction_result.csv
├── lstm_prediction_result.csv
├── hybrid_arima_rf_weighted_result.csv
├── hybrid_arima_rf_residual_result.csv
├── hybrid_arima_xgb_prediction_result.csv
├── requirements.txt
├── README.md
└── docs/
    ├── presentation.pptx
    ├── bilingual_speech_script.docx
    ├── redesigned_presentation_backup.pptx
    └── speech_script_backup.docx
```

---

## 5. Dataset
The repository includes **`TESLA.csv`** in the root directory.

Main columns:
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- optional `Adj Close`

Time range in this dataset: **2021-09-29 to 2022-09-29**

---

## 6. Preprocessing and Feature Engineering
All shared preprocessing is handled in `preprocess.py`, with switches controlled by `config.py`.

### Preprocessing includes
- date parsing and chronological sorting
- train/test split for time-series forecasting
- missing-value handling after rolling indicators
- optional scaling for deep learning models

### Engineered features include
- returns: `Return_1`, `Return_3`, `Return_5`, `Return_10`, `Return_20`
- moving averages: `MA5`, `MA10`, `MA20`, `EMA12`, `EMA26`
- momentum indicators: `RSI14`, `MACD`, `MACD_signal`, `MACD_hist`
- candlestick features: body, shadow, close position, range
- volatility features: rolling volatility, Bollinger Bands, ATR
- volume features: `Volume_MA`, `Volume_ratio`, `OBV`, `PVT`

---

## 7. Environment Setup
Clone the repository and install dependencies.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
```

### Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 8. Configuration
Global settings are stored in `config.py`.

Examples:
- test size
- random seed
- feature switches
- deep learning sequence length
- batch size and epochs

This makes it easy to toggle feature groups and rerun experiments.

---

## 9. How to Run the Models

### ARIMA
```bash
python TESLA_ARIMA.py
```
Output:
- `arima_prediction_result.csv`

### Random Forest
```bash
python TESLA_RANDOM_FOREST.py
```
Output:
- `rf_prediction_result.csv`

### XGBoost
```bash
python TESLA_XGBOOST.py
```
Output:
- `xgb_prediction_result.csv`

### GRU
```bash
python TESLA_GRU.py
```
Output:
- `gru_prediction_result.csv`

### LSTM
```bash
python TESLA_LSTM.py
```
Output:
- `lstm_prediction_result.csv`

### LightGBM / combined tree experiment
```bash
python TESLA.py
```
Output:
- `xgb_prediction_result.csv`
- `lgb_prediction_result.csv`

### Hybrid models
```bash
python TESLA_HYBRID_ARIMA_RF_WEIGHTED.py
python TESLA_HYBRID_ARIMA_RF_RESIDUAL.py
python TESLA_HYBRID_ARIMA_XGB.py
```
Output:
- `hybrid_arima_rf_weighted_result.csv`
- `hybrid_arima_rf_residual_result.csv`
- `hybrid_arima_xgb_prediction_result.csv`

---

## 10. Run the Streamlit Demo
The project includes a simple Streamlit interface for Stage 4.

```bash
streamlit run app.py
```

The app supports:
- Tesla historical price visualization
- model selection
- evaluation metrics (`MAE`, `RMSE`, `MAPE`, `R²`)
- actual vs predicted comparison
- hybrid model result display

Note: the app reads the existing prediction result CSV files in this repository.

---

## 11. Seminar / Presentation Files
The `docs/` folder contains:
- seminar presentation slides
- bilingual speech script
- backup versions of redesigned slides and script

These files support the final seminar presentation and report preparation.

---

## 12. Notes on Model Comparison
A few important interpretation notes:

- Full-test results should be used for the **main model comparison**.
- Some hybrid experiments use **aligned subsets** so that ARIMA and tree-model predictions can be compared on the same dates.
- Therefore, aligned hybrid results should not be directly mixed with full-test single-model rankings.

---

## 13. Suggested Final Story for the Project
A concise way to present the project is:

1. Tesla stock prices are highly volatile and influenced by both technical patterns and external events.  
2. We built a structured preprocessing and feature engineering pipeline.  
3. We compared statistical, machine learning, deep learning, and hybrid approaches.  
4. ARIMA and Random Forest performed best under the current small-sample setting.  
5. External news and macro conditions help explain why prediction remains difficult.  

---

## 14. GitHub Upload Steps
If you are uploading this repository manually:

```bash
git init
git add .
git commit -m "Initial commit: Tesla stock prediction project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

---

## 15. Team
Replace this section with your actual group member names before final submission.

Example:
- Member 1 – preprocessing and feature engineering
- Member 2 – model implementation and evaluation
- Member 3 – Streamlit demo and GitHub organization
- Member 4 – presentation slides and speech script
