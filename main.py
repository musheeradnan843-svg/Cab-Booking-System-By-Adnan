import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from db import get_connection

app = FastAPI()

# Load the trained model, scaler, and encoders.
model = joblib.load('xgb_cab_booking_model.joblib')
scaler = joblib.load('scaler.joblib')
season_enc = joblib.load('season_encoder.joblib')
weather_enc = joblib.load('weather_encoder.joblib')

SCALER_COLUMNS = ['season', 'holiday', 'workingday', 'weather', 'temp',
                   'atemp', 'humidity', 'windspeed', 'year', 'month', 'day',
                   'hour', 'Bookings']


def scale_value(column_name, value):
    # We are manually applying the StandardScaler formula to a single column.
    # scaled = (value - mean) / std_deviation this is the formula for standard scaling.
    idx = SCALER_COLUMNS.index(column_name)
    return (value - scaler.mean_[idx]) / scaler.scale_[idx]


def unscale_prediction(scaled_value):
    #The inverse formula is bringing the scaled prediction back to the real booking count.
    idx = SCALER_COLUMNS.index('Bookings')
    return (scaled_value * scaler.scale_[idx]) + scaler.mean_[idx]


class CabBookingInput(BaseModel):
    season: str          # e.g. "Summer" - jo bhi text tumhare CSV mein tha
    holiday: int
    workingday: int
    weather: str         # e.g. "Clear + Few clouds"
    temp: float
    humidity: float
    windspeed: float
    year: int
    month: int
    day: int
    hour: int


@app.post('/predict')
def predict_bookings(data: CabBookingInput):
    # 3. Text categories ko encoder se number mein badla
    season_code = season_enc.transform([data.season])[0]
    weather_code = weather_enc.transform([data.weather])[0]

    # Bring each feature to the same scale as the model was trained on.
    input_df = pd.DataFrame([{
        'season': scale_value('season', season_code),
        'holiday': scale_value('holiday', data.holiday),
        'workingday': scale_value('workingday', data.workingday),
        'weather': scale_value('weather', weather_code),
        'temp': scale_value('temp', data.temp),
        'humidity': scale_value('humidity', data.humidity),
        'windspeed': scale_value('windspeed', data.windspeed),
        'year': scale_value('year', data.year),
        'month': scale_value('month', data.month),
        'day': scale_value('day', data.day),
        'hour': scale_value('hour', data.hour),
    }])

    scaled_prediction = model.predict(input_df)[0]
    real_prediction = unscale_prediction(scaled_prediction)

    # Save this prediction to MySQL for history/tracking purposes.
    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO predictions (season, holiday, workingday, weather, temp, humidity, windspeed, year, month, day, hour, predicted_bookings)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (data.season, data.holiday, data.workingday, data.weather, data.temp, data.humidity, data.windspeed, data.year, data.month, data.day, data.hour, round(float(real_prediction), 2))

    cursor.execute(insert_query, values)   # %s = safe from sql injection.
    conn.commit()    # actually saves the data.
    cursor.close()
    conn.close()

    # The model's output was also scaled, so I converted it back to the real booking count.
    return {'predicted_bookings': round(float(real_prediction), 2)}


@app.get('/')
def index():
    return {'message': 'Cab Booking Prediction API is running'}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# This file converts The Cab Booking project into a prediction API using FastAPI.