import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_emergency_data(df):
    """
    Prepare daily Emergency Department demand data.
    """

    emergency = (
        df[df["department"] == "Emergency"]
        .groupby("date")["patient_arrivals"]
        .sum()
        .reset_index()
    )

    emergency = emergency.sort_values("date").reset_index(drop=True)

    emergency["day_of_week"] = (
        emergency["date"].dt.dayofweek
    )

    emergency["month"] = (
        emergency["date"].dt.month
    )

    emergency["lag_1"] = (
        emergency["patient_arrivals"].shift(1)
    )

    emergency["lag_7"] = (
        emergency["patient_arrivals"].shift(7)
    )

    emergency["rolling_7"] = (
        emergency["patient_arrivals"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    emergency = emergency.dropna().reset_index(drop=True)

    return emergency


def train_forecasting_model(df):
    """
    Train a Random Forest model to forecast
    Emergency Department patient arrivals.
    """

    data = prepare_emergency_data(df)

    features = [
        "day_of_week",
        "month",
        "lag_1",
        "lag_7",
        "rolling_7",
    ]

    target = "patient_arrivals"

    # Keep the final 30 days for evaluation
    test_size = 30

    train = data.iloc[:-test_size].copy()
    test = data.iloc[-test_size:].copy()

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    evaluation = test[
        ["date", "patient_arrivals"]
    ].copy()

    evaluation["predicted_arrivals"] = (
        predictions.round(1)
    )

    # Retrain on all available historical data
    model.fit(
        data[features],
        data[target],
    )

    forecast = generate_forecast(
        model=model,
        historical=data,
        features=features,
        days=7,
    )

    return {
        "model": model,
        "evaluation": evaluation,
        "forecast": forecast,
        "mae": mae,
        "rmse": rmse,
    }


def generate_forecast(
    model,
    historical,
    features,
    days=7,
):
    """
    Generate a recursive multi-day forecast.
    """

    history = historical[
        ["date", "patient_arrivals"]
    ].copy()

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    forecasts = []

    for _ in range(days):

        next_date = (
            history["date"].iloc[-1]
            + pd.Timedelta(days=1)
        )

        lag_1 = (
            history["patient_arrivals"].iloc[-1]
        )

        lag_7 = (
            history["patient_arrivals"].iloc[-7]
        )

        rolling_7 = (
            history["patient_arrivals"]
            .tail(7)
            .mean()
        )

        row = pd.DataFrame(
            {
                "day_of_week": [
                    next_date.dayofweek
                ],
                "month": [
                    next_date.month
                ],
                "lag_1": [lag_1],
                "lag_7": [lag_7],
                "rolling_7": [rolling_7],
            }
        )

        prediction = float(
            model.predict(row[features])[0]
        )

        prediction = max(
            0,
            round(prediction, 1),
        )

        forecasts.append(
            {
                "date": next_date,
                "predicted_arrivals": prediction,
            }
        )

        new_row = pd.DataFrame(
            {
                "date": [next_date],
                "patient_arrivals": [
                    prediction
                ],
            }
        )

        history = pd.concat(
            [history, new_row],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)