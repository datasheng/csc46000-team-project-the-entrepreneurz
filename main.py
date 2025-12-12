import requests
import pandas as pd
from sqlalchemy import create_engine, text

# -------------------------------
# PostgreSQL connection
# -------------------------------
engine = create_engine("postgresql://postgres:password@localhost:5432/Entrepreneur")

# -------------------------------
# SBA 7(a) Loader (sample)
# -------------------------------

csv_files = [
    "foia-7a-fy1991-fy1999-asof-250930_sample.csv",
    "foia-7a-fy2000-fy2009-asof-250930_sample.csv",
    "foia-7a-fy2010-fy2019-asof-250930_sample.csv",
    "foia-7a-fy2020-present-asof-250930_sample.csv"
]


def load_sba_7a_sample(csv_files, sample_size=500):
    dfs = []
    for file in csv_files:
        print(f"Loading first {sample_size} rows from {file}...")
        df = pd.read_csv(file, nrows=sample_size, low_memory=False)
        dfs.append(df)
    sba_sample_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(sba_sample_df)} rows.")
    return sba_sample_df

# -------------------------------
# SBA 7(a) Cleaning
# -------------------------------
def clean_sba(df):
    print("Cleaning SBA 7(a) data...")

    # Standardize column names to snake_case
    df.columns = [c.lower() for c in df.columns]

    # Convert numeric columns and fill missing with mean
    numeric_cols = ['grossapproval', 'sbaguaranteedapproval', 'terminmonths', 'initialinterestrate', 'jobsupported']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    # Drop rows with critical missing values
    df = df.dropna(subset=['grossapproval', 'sbaguaranteedapproval'])

    # Convert date columns
    date_cols = ['approvaldate', 'firstdisbursementdate', 'paidinfulldate', 'chargeoffdate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Derived metric: SBA loan guarantee ratio
    df['sba_ratio'] = df['sbaguaranteedapproval'] / df['grossapproval']

    return df


# -------------------------------
# BDS Cleaning
# -------------------------------
def clean_bds(df):
    print("Cleaning BDS data...")
    df.columns = [c.lower() for c in df.columns]
    numeric_cols = ['emp', 'firm', 'estab']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean()).astype(int)
    return df


# -------------------------------
# CBP Cleaning
# -------------------------------
def clean_cbp(df):
    print("Cleaning CBP data...")
    df.columns = [c.lower() for c in df.columns]
    numeric_cols = ['emp', 'estab', 'payann', 'year']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean()).astype(int)
    return df


# -------------------------------
# FRED Cleaning
# -------------------------------
def clean_fred(df):
    print("Cleaning FRED data...")
    df.columns = [c.lower() for c in df.columns]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = df['value'].fillna(df['value'].mean())
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


# -------------------------------
# BDS Fetcher
# -------------------------------
def fetch_bds_naics(naics="72", start_year=1978, end_year=2022):
    all_records = []
    for year in range(start_year, end_year + 1):
        url = "https://api.census.gov/data/timeseries/bds"
        params = {
            "get": "YEAR,EMP,FIRM,ESTAB",
            "for": "us:1",
            "NAICS": naics,
            "time": year
        }
        r = requests.get(url, params=params)
        if r.status_code != 200 or r.text.startswith("error"):
            print(f"Skipping year {year}: {r.text[:100]}")
            continue
        try:
            data = r.json()
        except:
            print(f"Skipping year {year}: not JSON")
            continue
        df = pd.DataFrame(data[1:], columns=data[0])
        all_records.append(df)
    if not all_records:
        return pd.DataFrame()
    bds_df = pd.concat(all_records, ignore_index=True)
    return clean_bds(bds_df)

# -------------------------------
# CBP Fetcher
# -------------------------------
def fetch_cbp(year=2022, state="06"):
    url = f"https://api.census.gov/data/{year}/cbp"
    params = {
        "get": "EMP,ESTAB,PAYANN,NAICS2017",
        "for": "county:*",
        "in": f"state:{state}"
    }
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["year"] = year
    return clean_cbp(df)

# -------------------------------
# FRED Fetcher
# -------------------------------
def fetch_fred_series(series_id, api_key, limit=1000):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit
    }
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data["observations"])
    return clean_fred(df)

def merge_datasets(sba_df, bds_df, cbp_df, fred_df):
    print("Merging datasets into analysis-ready DataFrame...")

    # Ensure key columns are the same type
    sba_df['approvalfy'] = sba_df['approvalfy'].astype(str)
    bds_df['year'] = bds_df['year'].astype(str)
    cbp_df['year'] = cbp_df['year'].astype(str)

    # Merge SBA → BDS on fiscal year
    merged_df = sba_df.merge(
        bds_df,
        left_on='approvalfy',
        right_on='year',
        how='left',
        suffixes=('_sba', '_bds')
    )

    # Merge CBP on year + state
    merged_df = merged_df.merge(
        cbp_df,
        left_on=['approvalfy', 'projectstate'],
        right_on=['year', 'state'],
        how='left',
        suffixes=('', '_cbp')
    )

    # Merge FRED on year from approvaldate
    merged_df['approval_year'] = merged_df['approvaldate'].dt.year.astype(str)
    fred_df['year'] = fred_df['date'].dt.year.astype(str)
    merged_df = merged_df.merge(
        fred_df[['year','value']],
        left_on='approval_year',
        right_on='year',
        how='left',
        suffixes=('', '_cpi')
    )

    # -------------------------------
    # Fill missing numeric columns with mean
    # -------------------------------
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())

    # Optionally: drop temporary columns
    merged_df.drop(columns=['approval_year', 'year'], inplace=True, errors='ignore')

    print("Merged dataset ready. Missing numeric values filled with column mean.")
    return merged_df


# -------------------------------
# Main pipeline
# -------------------------------
if __name__ == "__main__":

    # Clear old tables
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sba_7a_loans_cleaned;"))
        conn.execute(text("DROP TABLE IF EXISTS bds_cleaned;"))
        conn.execute(text("DROP TABLE IF EXISTS cbp_cleaned;"))
        conn.execute(text("DROP TABLE IF EXISTS fred_cleaned;"))
        conn.execute(text("DROP TABLE IF EXISTS merged_analysis_ready;"))
        conn.execute(text("DROP TABLE IF EXISTS sba_bds_cbp_fred_merged;"))


    # Load and clean datasets
    sba_df = load_sba_7a_sample(csv_files)
    sba_df = clean_sba(sba_df)

    bds_df = fetch_bds_naics(naics="72", start_year=1978, end_year=2022)
    cbp_df = fetch_cbp(year=2022, state="06")
    fred_api_key = "3290178b9dfef23c54c6ddbe214b5edb"
    fred_df = fetch_fred_series("CPIAUCSL", api_key=fred_api_key)

    # Merge all datasets
    merged_df = merge_datasets(sba_df, bds_df, cbp_df, fred_df)

    # Preview merged dataset
    print("\n--- Merged dataset preview ---")
    print(merged_df.head(10))
    print("\nColumns:", merged_df.columns.tolist())
    print("Total rows:", len(merged_df))

    # Save merged dataset to PostgreSQL
    with engine.begin() as conn:
        merged_df.to_sql("merged_analysis_ready", conn, if_exists="replace", index=False)

    # Save to CSV for team members
    merged_csv_path = "merged_analysis_ready_dataset.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged dataset saved to '{merged_csv_path}' and PostgreSQL!")
