# Cab-Booking-System-By-Adnan

Cab booking demand prediction using XGBoost — includes EDA, model comparison across 8 algorithms, and overfitting diagnosis/fix (Train RMSE 0.235, Test RMSE 0.277, R² 0.92).

## Background

Cab booking companies can improve service efficiency and reduce customer wait times by predicting hourly demand in advance. This project uses historical booking data along with weather and time-based features to forecast the number of cab bookings per hour.

## Dataset

| Column | Description |
|---|---|
| datetime | Hourly timestamp |
| season | Spring, Summer, Fall, Winter |
| holiday | Whether the day is a holiday |
| workingday | Whether the day is neither a weekend nor a holiday |
| weather | Clear, Cloudy, Light Rain, Heavy |
| temp | Temperature (°C) |
| atemp | "Feels like" temperature (°C) |
| humidity | Relative humidity |
| windspeed | Wind speed |
| Bookings | Total number of bookings (target) |

## Approach

1. *EDA* — Analyzed booking trends by hour, season, weather, temperature, humidity, and windspeed to understand demand patterns.
2. *Preprocessing* — Extracted year/month/day/hour from datetime, encoded categorical features, capped outliers, and removed invalid holiday/workingday combinations.
3. *Feature Scaling* — Applied StandardScaler, fitting only on training data and transforming the test set separately to avoid data leakage.
4. *Model Comparison* — Evaluated 8 regression models (Linear Regression, Ridge, Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost) using 5-Fold Cross Validation.
5. *Hyperparameter Tuning* — Used GridSearchCV to tune XGBoost, then diagnosed overfitting by comparing train vs. test RMSE (not just relying on CV score).
6. *Fix* — Reduced model complexity and increased regularization to close the train-test gap.

## Results

| Metric | Train | Test |
|---|---|---|
| RMSE | 0.235 | 0.277 |
| R² Score | 0.9447 | 0.9224 |

XGBoost Regressor gave the best performance among all models tested, with a small train-test gap indicating good generalization.

## Key Learnings

- A strong cross-validation score alone doesn't guarantee a reliable model — explicitly comparing train and test performance is essential to catch overfitting.
- Feature scaling must be fit only on training data; fitting a separate scaler on test data causes distribution mismatch.

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

## 🚀 FastAPI

The trained XGBoost model was deployed using FastAPI.

### POST /predict

The API takes cab booking information as JSON and returns the predicted number of bookings.

Example input:
{
  "season": "Summer",
  "holiday": 0,
  "workingday": 1,
  "weather": "Clear + Few clouds",
  "temp": 25,
  "humidity": 60,
  "windspeed": 10,
  "year": 2025,
  "month": 8,
  "day": 15,
  "hour": 18
}

Example response:
{
  "predicted_bookings": 245.67
}

### How it works

Input → Validation → Encoding → Scaling → XGBoost → Prediction → JSON Response

The API uses the same saved encoders and scaler used during model training.
