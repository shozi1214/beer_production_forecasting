import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from Visualizing_Actual_and_Predicted_Data import plot_actual_vs_predicted

def train_and_evaluate_rf(
    df,
    train_end_date='1995-01-01',
    test_end_date='1995-08-01',
    n_estimators=50,
    max_depth=10,
    random_state=42,
    min_samples_split=2,
    use_grid_search=False,
    param_grid=None,  # Optional: allows custom grid
    show_feature_importance=False  # New flag to control feature importance plotting
):
    # Split data
    train_df = df[df['Month'] < train_end_date].copy()
    test_df = df[(df['Month'] >= train_end_date) & (df['Month'] <= test_end_date)].copy()

    # Feature engineering: extract date parts
    for d in [train_df, test_df]:
        d['month'] = d['Month'].dt.month
        d['year'] = d['Month'].dt.year
        d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
        d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)

    # Define features and target
    feature_cols = ['month', 'year', 'month_sin', 'month_cos']
    target_col = 'Monthlybeerproduction'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train model
    if use_grid_search:
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=random_state),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print("Best Parameters from GridSearchCV:", grid_search.best_params_)
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            min_samples_split=min_samples_split
        )
        model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print(f"Random Forest RMSE ({train_end_date}–{test_end_date}): {rmse:.2f}")
    print(f"Random Forest MAE  ({train_end_date}–{test_end_date}): {mae:.2f}")

    # Plot actual vs predicted
    plot_actual_vs_predicted(
        dates=test_df['Month'],
        actual=y_test,
        predicted=predictions,
        title=f'Actual vs Predicted Monthly Beer Production (RF) ({train_end_date}–{test_end_date})',
        xlabel='Month',
        ylabel='Monthlybeerproduction'
    )

    # Plot feature importance only if requested
    if show_feature_importance:
        importances = model.feature_importances_
        features = X_train.columns

        plt.figure(figsize=(8, 5))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Importance Score')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.show()

    return model, rmse, mae
