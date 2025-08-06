import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from VisualizingData import plot_actual_vs_predicted

def train_and_evaluate_prophet(
    df,
    train_end_date='1995-01-01',
    test_end_date='1995-08-01',
    seasonality_mode='multiplicative',
    seasonality_prior_scale=17,
    changepoint_prior_scale=0.5,
    changepoint_range=0.9,
    n_changepoints=35,
    add_quarterly_seasonality=True,
    quarterly_period=91.25,
    quarterly_fourier_order=8
):
    # Split data
    train_df = df[df['Month'] < train_end_date]
    test_df = df[(df['Month'] >= train_end_date) & (df['Month'] <= test_end_date)]

    # Prepare training data for Prophet
    prophet_train = train_df[['Month', 'Monthlybeerproduction']].rename(columns={
        'Month': 'ds',
        'Monthlybeerproduction': 'y'
    })

    # Initialize model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=changepoint_range,
        n_changepoints=n_changepoints
    )

    # Add quarterly seasonality if required
    if add_quarterly_seasonality:
        model.add_seasonality(name='quarterly', period=quarterly_period, fourier_order=quarterly_fourier_order)

    # Fit model
    model.fit(prophet_train)

    # Create future dataframe and predict
    future = model.make_future_dataframe(periods=8, freq='MS')
    forecast = model.predict(future)

    # Filter predictions to test period
    forecast_test = forecast[(forecast['ds'] >= train_end_date) & (forecast['ds'] <= test_end_date)]

    # Actual and predicted values
    actual = test_df['Monthlybeerproduction'].values
    predicted = forecast_test['yhat'].values

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    print(f"RMSE ({train_end_date}–{test_end_date}): {rmse:.2f}")
    print(f"MAE  ({train_end_date}–{test_end_date}): {mae:.2f}")

    # Plot actual vs predicted
    plot_actual_vs_predicted(
        dates=test_df['Month'],
        actual=actual,
        predicted=predicted,
        title=f'Actual vs Predicted Monthly Beer Production ({train_end_date}–{test_end_date})',
        xlabel='Month',
        ylabel='Monthlybeerproduction'
    )

    # Optionally return the model and metrics
    return model, rmse, mae
