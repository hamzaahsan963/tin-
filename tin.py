from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# API key and base URL
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data
def fetch_data(start_date, end_date):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    all_data = {}

    while start_date <= end_date:
        current_end_date = min(start_date + timedelta(days=29), end_date)
        params = {
            "access_key": api_key,
            "base": "USD",
            "symbols": "TIN",
            "start_date": start_date.strftime(date_format),
            "end_date": current_end_date.strftime(date_format)
        }
        response = requests.get(f"{base_url}/timeseries", params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                all_data.update(data.get("rates", {}))
            else:
                return None, data.get('error', {}).get('info')
        else:
            return None, f"Error fetching data: {response.status_code}"

        start_date = current_end_date + timedelta(days=1)

    return all_data if all_data else None, None

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    forecast_plot = None
    arima_plot = None
    performance_metrics = None
    error_message = None
    arima_metrics = None

    if request.method == "POST":
        start_date = request.form.get("start_date")
        prediction_period = request.form.get("prediction_period")

        # Calculate the end date based on selected prediction period
        if prediction_period == "6 Months":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=6 * 30)
        elif prediction_period == "3 Months":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=3 * 30)
        elif prediction_period == "3 Weeks":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=3)
        elif prediction_period == "1 Week":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=1)

        start_date_str = start_date
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Fetch data
        data, error_message = fetch_data(start_date_str, end_date_str)

        if data:
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "ds", "TIN": "y"})
            df = df[["ds", "y"]]

            # Prophet model training and forecasting
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df)

            prediction_days = (end_date - datetime.strptime(start_date_str, "%Y-%m-%d")).days
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)

            # Plot the forecast
            fig1 = model.plot(forecast)
            img = io.BytesIO()
            fig1.savefig(img, format='png')
            img.seek(0)
            forecast_plot = base64.b64encode(img.getvalue()).decode()

            # Evaluate the model
            df_cv = cross_validation(model, initial='14 days', period='7 days', horizon='7 days')
            df_performance = performance_metrics(df_cv)
            performance_metrics = df_performance.to_html()

            # ARIMA model
            result = adfuller(df['y'])
            if result[1] < 0.05:  # The series is stationary
                arima_model = ARIMA(df['y'], order=(5, 1, 0))
                arima_result = arima_model.fit()

                arima_forecast = arima_result.get_forecast(steps=prediction_days)
                arima_conf_int = arima_forecast.conf_int()
                arima_pred = arima_forecast.predicted_mean

                # Plot ARIMA forecast
                plt.figure(figsize=(10, 6))
                plt.plot(df['ds'], df['y'], label='Historical')
                plt.plot(pd.date_range(start=df['ds'].iloc[-1], periods=prediction_days + 1, freq='D')[1:], arima_pred,
                         label='ARIMA Forecast')
                plt.fill_between(pd.date_range(start=df['ds'].iloc[-1], periods=prediction_days + 1, freq='D')[1:],
                                 arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.3)
                plt.legend()
                plt.title('ARIMA Forecast')
                plt.xlabel('Date')
                plt.ylabel('Price')

                img_arima = io.BytesIO()
                plt.savefig(img_arima, format='png')
                img_arima.seek(0)
                arima_plot = base64.b64encode(img_arima.getvalue()).decode()
            else:
                arima_metrics = "The time series is not stationary. ARIMA might not provide reliable predictions."

    return render_template(
        "index.html",
        data=data,
        forecast_plot=forecast_plot,
        performance_metrics=performance_metrics,
        error_message=error_message,
        arima_plot=arima_plot,
        arima_metrics=arima_metrics
    )

if __name__ == "__main__":
    app.run(debug=True)
