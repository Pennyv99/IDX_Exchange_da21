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


listings = pd.read_csv('../week1/list_combined.csv')

# Convert ListingContractDate to datetime before extracting period
# For listings we key off ListingContractDate, not CloseDate
listings['ListingContractDate'] = pd.to_datetime(listings['ListingContractDate'], errors='coerce')

# Create year_month key from ListingContractDate to match mortgage monthly data
listings['year_month'] = listings['ListingContractDate'].dt.to_period('M')

print(f"\nListings dataset loaded: {listings.shape[0]} rows")
print(f"Listings date range: {listings['ListingContractDate'].min().date()} to {listings['ListingContractDate'].max().date()}")
print(f"Unique year_month values in listings: {listings['year_month'].nunique()}")


listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left')

print(f"\nListings dataset shape BEFORE merge: {listings.shape}")
print(f"Listings dataset shape AFTER merge:  {listings_with_rates.shape}")


null_rates = listings_with_rates['rate_30yr_fixed'].isnull().sum()
print(f"\nRows with null rate_30yr_fixed after merge: {null_rates}")

if null_rates > 0:
    # Show which year_month values failed to match
    unmatched = listings_with_rates[listings_with_rates['rate_30yr_fixed'].isnull()]['year_month'].unique()
    print(f"Unmatched year_month values: {unmatched}")
else:
    print("Validation passed: all rows have a mortgage rate assigned")

# Preview merged result
print("\nSample of merged data:")
print(listings_with_rates[['ListingContractDate', 'year_month', 'ListPrice', 'rate_30yr_fixed']].head(10).to_string(index=False))


listings_with_rates.to_csv('listings_with_rates.csv', index=False)
print(f"\nEnriched listings dataset saved to: listings_with_rates.csv")
print(f"Final shape: {listings_with_rates.shape}")
