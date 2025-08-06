import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_beer_production(df):

    plt.figure(figsize=(12, 6))
    plt.plot(df['Month'], df['Monthlybeerproduction'], marker='o', linestyle='-', color='b')
    plt.title('Monthly Beer Production Over Time')
    plt.xlabel('Month')
    plt.ylabel('Monthly Beer Production')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_monthly_seasonality(df):
    df['month'] = df['Month'].dt.month
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='month', y='Monthlybeerproduction', data=df, palette='coolwarm')
    plt.title('Monthly Beer Production Seasonality')
    plt.xlabel('Month')
    plt.ylabel('Monthly Beer Production')
    plt.tight_layout()
    plt.show()

def plot_year_month_heatmap(df):
    df['year'] = df['Month'].dt.year
    df['month'] = df['Month'].dt.month
    pivot_table = df.pivot_table(values='Monthlybeerproduction', index='year', columns='month', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt=".0f")
    plt.title('Heatmap of Average Monthly Beer Production (Year vs Month)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.show()

def plot_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Monthlybeerproduction'], kde=True, color='purple', bins=20)
    plt.title('Distribution of Monthly Beer Production')
    plt.xlabel('Monthly Beer Production')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_yearly_trend(df):
    df['year'] = df['Month'].dt.year
    yearly_avg = df.groupby('year')['Monthlybeerproduction'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_avg['year'], yearly_avg['Monthlybeerproduction'], marker='o', color='green')
    plt.title('Average Yearly Beer Production')
    plt.xlabel('Year')
    plt.ylabel('Average Monthly Beer Production')
    plt.grid(True)
    plt.tight_layout()
    plt.show()