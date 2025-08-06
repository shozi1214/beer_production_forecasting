# beer_production_forecasting
Time series forecasting of Austria’s monthly beer production using Facebook Prophet and Random Forest. Includes feature engineering (cyclical time features), grid search, cross-validation, and performance evaluation via RMSE/MAE.

## 🎯Project Objective
The goal of this project is to build time series forecasting models to predict future beer production values. This involves:

Cleaning and exploring the dataset.

Creating time-based features (like cyclical month encodings).

Training models such as:

Facebook Prophet – a statistical time series model.

Random Forest Regressor – a machine learning model with engineered features.

Evaluating performance using RMSE and MAE.

Visualizing predictions and trends.

## 📂 Dataset: MonthlyBeerAustria.csv
This dataset contains monthly beer production figures in Austria, recorded over multiple years. It includes:

Month: Time column in YYYY-MM format.

Monthlybeerproduction: Total beer production for that month (likely in tonnes, hectoliters, or barrels — unit not explicitly provided, but consistent throughout).
