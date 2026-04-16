import pandas as pd

# No API key required - FRED provides this data as a public CSV
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

print(f"Mortgage rate data fetched successfully")
print(f"Date range: {mortgage['date'].min().date()} to {mortgage['date'].max().date()}")
print(f"Total weekly observations: {len(mortgage)}")

# The FRED MORTGAGE30US series is published weekly (every Thursday)
# We must resample to monthly average before joining to MLS transaction data
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .round(2)
    .reset_index()
)

print(f"\nMonthly mortgage rate sample (most recent 6 months):")
print(mortgage_monthly.tail(6).to_string(index=False))


sold = pd.read_csv('../week2/sold_structured.csv')

# Convert CloseDate to datetime before extracting period
sold['CloseDate'] = pd.to_datetime(sold['CloseDate'], errors='coerce')

# Create year_month key from CloseDate to match mortgage monthly data
sold['year_month'] = sold['CloseDate'].dt.to_period('M')

print(f"\nSold dataset loaded: {sold.shape[0]} rows")
print(f"Sold date range: {sold['CloseDate'].min().date()} to {sold['CloseDate'].max().date()}")
print(f"Unique year_month values in sold: {sold['year_month'].nunique()}")


sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')

print(f"\nSold dataset shape BEFORE merge: {sold.shape}")
print(f"Sold dataset shape AFTER merge:  {sold_with_rates.shape}")


null_rates = sold_with_rates['rate_30yr_fixed'].isnull().sum()
print(f"\nRows with null rate_30yr_fixed after merge: {null_rates}")

if null_rates > 0:
    # Show which year_month values failed to match
    unmatched = sold_with_rates[sold_with_rates['rate_30yr_fixed'].isnull()]['year_month'].unique()
    print(f"Unmatched year_month values: {unmatched}")
else:
    print("Validation passed: all rows have a mortgage rate assigned")

# Preview merged result
print("\nSample of merged data:")
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head(10).to_string(index=False))


sold_with_rates.to_csv('sold_with_rates.csv', index=False)
print(f"\nEnriched sold dataset saved to: sold_with_rates.csv")
print(f"Final shape: {sold_with_rates.shape}")