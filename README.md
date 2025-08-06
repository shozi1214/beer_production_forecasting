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

## 📄 PreparingDataFrame.py
### 🔧 Functions

- **summarize_dataframe(df):**  Returns summary: column names, types, nulls, non-nulls, total rows.

- **remove_spaces_from_column_names(df):**  Removes spaces from column names.

- **convert_columns_to_datetime(df, columns):**  Converts column(s) to datetime format (string or list).

- **drop_duplicates_and_report(df):**  Drops duplicates and returns cleaned DataFrame with message.

## 📄 CleanData.py
This script loads the original CSV file, cleans the data by calling functions from PreparingDataFrame.py, and saves the cleaned dataset as a .pkl file for efficient storage and later use.

## 📄 EDA.py
This script contains functions to visualize beer production data, including:

- Time series plot of monthly beer production

- Monthly seasonality via boxplot

- Heatmap of production by year and month

- Distribution histogram

- Yearly average trend line

These plots help explore trends, seasonality, and distribution patterns in the dataset.
