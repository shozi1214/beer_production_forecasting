import pandas as pd
import numpy as np
from PreparingDataFrame import remove_spaces_from_column_names, convert_columns_to_datetime, summarize_dataframe,drop_duplicates_and_report



df = pd.read_csv('MonthlyBeerAustria.csv')


df_cleaned = remove_spaces_from_column_names(df)
df_cleaned = convert_columns_to_datetime(df_cleaned, 'Month')
df_cleaned,message = drop_duplicates_and_report(df_cleaned)
print(message)

df_cleaned.to_pickle('df_cleaned.pkl')