import pandas as pd
import numpy as np

def remove_spaces_from_column_names(df):
    df = df.copy()
    df.columns = [col.replace(" ", "") for col in df.columns]
    return df
#########

def convert_columns_to_datetime(df, columns):
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    return df

#########
def summarize_dataframe(df):
    summary = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Total Rows': len(df)
    })
    return summary
#########

def drop_duplicates_and_report(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    dropped = initial_count - len(df_cleaned)
    if dropped > 0:
        return df_cleaned, f"{dropped} rows dropped"
    else:
        return df, "No duplicate rows found"
