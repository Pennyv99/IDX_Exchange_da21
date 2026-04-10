import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np


# Load the combined sold dataset from Week 1
sold = pd.read_csv('../week1/sold_combined.csv')


# Identify number of rows and columns
print("=" * 60)
print("SECTION 1: DATASET UNDERSTANDING")
print("=" * 60)
print(f"Number of rows: {sold.shape[0]}")
print(f"Number of columns: {sold.shape[1]}")

# Review column data types
print("\n--- Column Data Types ---")
print(sold.dtypes.to_string())

# Separate market analysis fields from metadata fields
market_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt', 'CloseDate', 'ListingContractDate',
    'PurchaseContractDate', 'CountyOrParish', 'City', 'PostalCode',
    'MLSAreaMajor', 'PropertyType', 'PropertySubType'
]

metadata_fields = [
    'ListingKey', 'ListingKeyNumeric', 'ListingId', 'MlsStatus',
    'ListAgentFirstName', 'ListAgentLastName', 'ListAgentFullName',
    'ListAgentMlsId', 'BuyerAgentFirstName', 'BuyerAgentLastName',
    'BuyerAgentMlsId', 'CoListAgentFirstName', 'CoListAgentLastName',
    'ListOfficeName', 'BuyerOfficeName', 'CoListOfficeName',
    'BuyerOfficeAOR', 'ElementarySchool', 'MiddleOrJuniorSchool',
    'HighSchool', 'ElementarySchoolDistrict', 'HighSchoolDistrict',
    'MiddleOrJuniorSchoolDistrict', 'SubdivisionName', 'BuilderName',
    'TaxAnnualAmount', 'TaxYear', 'ContractStatusChangeDate',
    'StateOrProvince', 'UnparsedAddress', 'StreetNumberNumeric'
]

# Only keep fields that actually exist in the dataset
market_fields_present = [f for f in market_fields if f in sold.columns]
metadata_fields_present = [f for f in metadata_fields if f in sold.columns]
other_fields = [f for f in sold.columns if f not in market_fields_present and f not in metadata_fields_present]

print("\n--- Market Analysis Fields ---")
print(market_fields_present)

print("\n--- Metadata Fields ---")
print(metadata_fields_present)

print("\n--- Other Fields ---")
print(other_fields)


print("\n" + "=" * 60)
print("SECTION 2: MISSING VALUE ANALYSIS")
print("=" * 60)

# Calculate missing counts and percentages per column
missing_count = sold.isnull().sum()
missing_pct = (missing_count / len(sold) * 100).round(2)
missing_report = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

print("\n--- Full Missing Value Report (sorted by % missing) ---")
print(missing_report.to_string())

# Flag columns with >90% missing values
high_missing_cols = missing_report[missing_report['missing_pct'] > 90].index.tolist()
print(f"\n--- Columns with >90% Missing Values (flagged for review) ---")
print(f"Count: {len(high_missing_cols)}")
print(high_missing_cols)

# Decision: drop high-missing columns but retain core market fields
core_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
    'YearBuilt', 'LotSizeAcres', 'CloseDate', 'ListingContractDate',
    'PurchaseContractDate', 'CountyOrParish', 'City', 'PostalCode',
    'PropertyType', 'PropertySubType', 'MLSAreaMajor',
    'ListOfficeName', 'BuyerOfficeName', 'ListAgentFullName',
    'BuyerAgentFirstName', 'BuyerAgentLastName',
    'Latitude', 'Longitude', 'ListingKey', 'ListingId'
]

# Columns to drop: high missing AND not a core field
cols_to_drop = [c for c in high_missing_cols if c not in core_fields]
cols_to_retain_despite_missing = [c for c in high_missing_cols if c in core_fields]

print(f"\n--- Columns dropped (>90% missing, not core): {len(cols_to_drop)} ---")
print(cols_to_drop)

print(f"\n--- Core columns retained despite >90% missing: {len(cols_to_retain_despite_missing)} ---")
print(cols_to_retain_despite_missing)

sold_clean = sold.drop(columns=cols_to_drop)
print(f"\nDataset shape after dropping high-missing non-core columns: {sold_clean.shape}")


print("\n" + "=" * 60)
print("SECTION 3: SUGGESTED INTERN QUESTIONS")
print("=" * 60)

# Q1: What is the Residential vs. other property type share?
print("\n--- Q1: Property Type Share ---")
if 'PropertyType' in sold_clean.columns:
    prop_type_counts = sold_clean['PropertyType'].value_counts()
    prop_type_pct = (prop_type_counts / len(sold_clean) * 100).round(2)
    prop_type_summary = pd.DataFrame({'count': prop_type_counts, 'pct': prop_type_pct})
    print(prop_type_summary.to_string())

    # Document unique property types found
    print(f"\nUnique property types found: {sold_clean['PropertyType'].unique().tolist()}")

# Q2: What are the median and average close prices?
print("\n--- Q2: Median and Average Close Price ---")
if 'ClosePrice' in sold_clean.columns:
    print(f"Median ClosePrice: ${sold_clean['ClosePrice'].median():,.0f}")
    print(f"Average ClosePrice: ${sold_clean['ClosePrice'].mean():,.0f}")

# Q3: What does the Days on Market distribution look like?
print("\n--- Q3: Days on Market Distribution ---")
if 'DaysOnMarket' in sold_clean.columns:
    dom = sold_clean['DaysOnMarket'].dropna()
    print(f"Min:    {dom.min():.0f} days")
    print(f"Max:    {dom.max():.0f} days")
    print(f"Mean:   {dom.mean():.1f} days")
    print(f"Median: {dom.median():.0f} days")
    print(f"75th percentile: {dom.quantile(0.75):.0f} days")
    print(f"90th percentile: {dom.quantile(0.90):.0f} days")

# Q4: What percentage of homes sold above vs. below list price?
print("\n--- Q4: Sold Above vs. Below List Price ---")
if 'ClosePrice' in sold_clean.columns and 'ListPrice' in sold_clean.columns:
    valid = sold_clean.dropna(subset=['ClosePrice', 'ListPrice'])
    above = (valid['ClosePrice'] > valid['ListPrice']).sum()
    below = (valid['ClosePrice'] < valid['ListPrice']).sum()
    at_list = (valid['ClosePrice'] == valid['ListPrice']).sum()
    total_valid = len(valid)
    print(f"Sold ABOVE list price:  {above:,} ({above/total_valid*100:.1f}%)")
    print(f"Sold BELOW list price:  {below:,} ({below/total_valid*100:.1f}%)")
    print(f"Sold AT list price:     {at_list:,} ({at_list/total_valid*100:.1f}%)")

# Q5: Are there any apparent date consistency issues?
print("\n--- Q5: Date Consistency Issues ---")
date_cols = ['CloseDate', 'ListingContractDate', 'PurchaseContractDate']
existing_date_cols = [c for c in date_cols if c in sold_clean.columns]
for col in existing_date_cols:
    sold_clean[col] = pd.to_datetime(sold_clean[col], errors='coerce')

if 'CloseDate' in sold_clean.columns and 'ListingContractDate' in sold_clean.columns:
    close_before_listing = (sold_clean['CloseDate'] < sold_clean['ListingContractDate']).sum()
    print(f"Records where CloseDate is BEFORE ListingContractDate: {close_before_listing}")

if 'CloseDate' in sold_clean.columns and 'PurchaseContractDate' in sold_clean.columns:
    close_before_purchase = (sold_clean['CloseDate'] < sold_clean['PurchaseContractDate']).sum()
    print(f"Records where CloseDate is BEFORE PurchaseContractDate: {close_before_purchase}")

if 'PurchaseContractDate' in sold_clean.columns and 'ListingContractDate' in sold_clean.columns:
    purchase_before_listing = (sold_clean['PurchaseContractDate'] < sold_clean['ListingContractDate']).sum()
    print(f"Records where PurchaseContractDate is BEFORE ListingContractDate: {purchase_before_listing}")

# Q6: Which counties have the highest median prices?
print("\n--- Q6: Top 10 Counties by Median Close Price ---")
if 'CountyOrParish' in sold_clean.columns and 'ClosePrice' in sold_clean.columns:
    county_median = (
        sold_clean.groupby('CountyOrParish')['ClosePrice']
        .median()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    county_median.columns = ['County', 'Median ClosePrice']
    county_median['Median ClosePrice'] = county_median['Median ClosePrice'].apply(lambda x: f"${x:,.0f}")
    print(county_median.to_string(index=False))

print("\n" + "=" * 60)
print("SECTION 4: NUMERIC DISTRIBUTION SUMMARY")
print("=" * 60)

numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt'
]

existing_numeric = [f for f in numeric_fields if f in sold_clean.columns]

summary_rows = []
for col in existing_numeric:
    s = sold_clean[col].dropna()
    summary_rows.append({
        'Field': col,
        'Count': len(s),
        'Min': s.min(),
        'Max': s.max(),
        'Mean': round(s.mean(), 2),
        'Median': s.median(),
        'P25': s.quantile(0.25),
        'P75': s.quantile(0.75),
        'P90': s.quantile(0.90),
        'P99': s.quantile(0.99)
    })

dist_summary = pd.DataFrame(summary_rows).set_index('Field')
print("\n--- Numeric Distribution Summary ---")
print(dist_summary.to_string())


# Fields to plot
plot_fields = [f for f in ['ClosePrice', 'ListPrice', 'OriginalListPrice',
                             'LivingArea', 'LotSizeAcres', 'BedroomsTotal',
                             'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt']
               if f in sold_clean.columns]

pdf_path = 'week2_distributions.pdf'
log_transform_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea', 'LotSizeAcres'
]

with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
    for col in plot_fields:
        data = sold_clean[col].dropna()

        use_log = col in log_transform_fields and (data > 0).all()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Distribution of {col}' + (' (log scale)' if use_log else ''), fontsize=13)

        plot_data = np.log1p(data) if use_log else data
        xlabel = f'log({col})' if use_log else col

        # Histogram
        axes[0].hist(plot_data, bins=50, color='steelblue', edgecolor='white')
        axes[0].set_title('Histogram')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('Frequency')

        # Boxplot
        axes[1].boxplot(plot_data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', color='navy'),
                        medianprops=dict(color='red', linewidth=2))
        axes[1].set_title('Boxplot')
        axes[1].set_ylabel(xlabel)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
print(f"\nHistograms and boxplots saved to: {pdf_path}")

sold_clean.to_csv('sold_structured.csv', index=False)
print(f"\nFiltered and structured dataset saved to: sold_structured.csv")
print(f"Final dataset shape: {sold_clean.shape}")