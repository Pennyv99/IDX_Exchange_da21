import pandas as pd
import glob

# Get all sold CSV files from the raw folder
all_sold_files = sorted(glob.glob('../week0/raw/CRMLSListing*.csv'))

# Load each monthly file and track individual row counts before concatenation
sold_list = []
total_rows_before_concat = 0

for f in all_sold_files:
    df = pd.read_csv(f)
    total_rows_before_concat += len(df)
    sold_list.append(df)

print(f"Row count BEFORE concatenation: {total_rows_before_concat}")
sold = pd.concat(sold_list, ignore_index=True)
print(f"Row count AFTER concatenation: {len(sold)}")

print(f"Row count BEFORE Residential filter: {len(sold)}")
sold_residential = sold[sold['PropertyType'] == 'Residential']
print(f"Row count AFTER Residential filter: {len(sold_residential)}")

sold_residential.to_csv('list_combined.csv', index=False)

print(f"Saved to list_combined.csv")