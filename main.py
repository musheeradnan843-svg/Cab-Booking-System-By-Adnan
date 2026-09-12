import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('xgb_cab_booking_model.joblib')
scaler = joblib.load('scaler.joblib')
season_enc = joblib.load('season_encoder.joblib')
weather_enc = joblib.load('weather_encoder.joblib')

SCALER_COLUMNS = ['season', 'holiday', 'workingday', 'weather', 'temp',
                   'atemp', 'humidity', 'windspeed', 'year', 'month', 'day',
                   'hour', 'Bookings']


def scale_value(column_name, value):
    # StandardScaler formula manually laga rahe hain ek column ke liye:
    # scaled = (value - mean) / std_deviation ye formula hai
    idx = SCALER_COLUMNS.index(column_name)
    return (value - scaler.mean_[idx]) / scaler.scale_[idx]


def unscale_prediction(scaled_value):
    # Ulta formula - scaled prediction ko wapas real booking count mein laya jara:
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

    # 4. Har feature ko usi scale mein laaye jisme model train hua tha
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

    # 5. Model ka output bhi scaled tha - ise real booking count mein convert kiya
    real_prediction = unscale_prediction(scaled_prediction)

    return {'predicted_bookings': round(float(real_prediction), 2)}


@app.get('/')
def index():
    return {'message': 'Cab Booking Prediction API is running'}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)