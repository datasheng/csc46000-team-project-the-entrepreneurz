import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from os import getenv
from dotenv import load_dotenv

load_dotenv()
# -------------------------------
# CONFIGURATION & MAPPINGS
# -------------------------------
STATE_TO_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
    'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
    'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
    'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
    'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
    'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
    'WI': '55', 'WY': '56', 'DC': '11'
}

# -------------------------------
# DATABASE CONNECTION
# -------------------------------
DB_USER = getenv('user', 'postgres')
DB_PASSWORD = getenv('password', 'your_password')
DB_HOST = getenv('host', 'your-db-instance.rds.amazonaws.com')
DB_PORT = getenv('port', '5432')
DB_NAME = getenv('database', 'postgres')
CENSUS_API_KEY = getenv('census_api_key', 'your_census_api_key').strip()
FRED_API_KEY = getenv('fred_api_key', 'your_fred_api_key').strip()

connection_str = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
try:
    engine = create_engine(connection_str)
except Exception as e:
    print(f"Warning: Database engine could not be created. {e}")
    engine = None

# -------------------------------
# 1. SBA 7(a) LOADER & CLEANER
# -------------------------------
csv_files = [
    "foia-7a-fy1991-fy1999-asof-250930_sample.csv",
    "foia-7a-fy2000-fy2009-asof-250930_sample.csv",
    "foia-7a-fy2010-fy2019-asof-250930_sample.csv",
    "foia-7a-fy2020-present-asof-250930.csv"
]

def load_sba_7a_sample(csv_files, sample_size=500):
    dfs = []
    for file in csv_files:
        if os.path.exists(file):
            print(f"Loading first {sample_size} rows from {file}...")
            # encoding='latin1' is often safer for government CSVs
            df = pd.read_csv(file, encoding='latin1', low_memory=False)
            dfs.append(df)
        else:
            print(f"Warning: File {file} not found. Skipping.")
    
    if not dfs:
        return pd.DataFrame()
        
    sba_sample_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(sba_sample_df)} total SBA rows.")
    return sba_sample_df

def clean_sba(df):
    print("Cleaning SBA 7(a) data...")
    df.columns = [c.lower().replace(" ", "") for c in df.columns]

    # Convert numeric columns
    numeric_cols = ['grossapproval', 'sbaguaranteedapproval', 'terminmonths', 'initialinterestrate', 'jobsupported']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop invalid rows
    df = df.dropna(subset=['grossapproval'])

    # Date Handling
    date_cols = ['approvaldate', 'firstdisbursementdate', 'paidinfulldate', 'chargeoffdate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Ensure Approval Fiscal Year is a string for merging
    if 'approvalfiscalyear' in df.columns:
        df['approvalfy'] = df['approvalfiscalyear'].astype(str)
    
    # Map State to FIPS immediately for later merging
    if 'projectstate' in df.columns:
        df['state_fips'] = df['projectstate'].map(STATE_TO_FIPS)

    # Derived metric: Risk/Guarantee ratio
    if 'sbaguaranteedapproval' in df.columns and 'grossapproval' in df.columns:
        df['sba_ratio'] = df['sbaguaranteedapproval'] / df['grossapproval']

    return df

# -------------------------------
# 2. BDS FETCHER 
# -------------------------------
def fetch_bds_all_sectors(start_year, end_year):
    print(f"Fetching BDS data for years {start_year}-{end_year}...")
    all_records = []
    
    # We only need ages 0 through 9 for our survival analysis.
    target_ages = ['010', '020', '030', '040', '050', '060', '070','080','090' ]
    for year in range(start_year, end_year + 1):
        for age_code in target_ages:
            url = "https://api.census.gov/data/timeseries/bds"
            
            params = {
                "get": "NAICS,YEAR,FIRM,ESTAB,JOB_DESTRUCTION_RATE,FAGE",
                "for": "us:1",
                "NAICS": "*",      # All Industries
                "FAGE": age_code,  # Single Age Group (Prevents Timeout)
                "YEAR": year,
                "key": CENSUS_API_KEY
            }
            
            try:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        # Fix Duplicate Columns (e.g., NAICS appears twice)
                        # We extract headers and data, then normalize
                        headers = data[0]
                        rows = data[1:]
                        df = pd.DataFrame(rows, columns=headers)
                        
                        # Remove duplicate columns if API returns them
                        df = df.loc[:, ~df.columns.duplicated()]
                        
                        all_records.append(df)
                    except ValueError:
                        print(f"  Error {year} (Age {age_code}): Non-JSON response.")
                elif r.status_code == 204:
                    # 204 means data not available for this specific slice
                    pass
                else:
                    print(f"  Failed {year} (Age {age_code}): {r.status_code}")
                    
            except Exception as e:
                print(f"  Connection Error {year}: {e}")

    if not all_records:
        cols = ['naics', 'year', 'firm', 'estab', 'job_destruction_rate', 'fage']
        return pd.DataFrame(columns=cols)

    bds_df = pd.concat(all_records, ignore_index=True)
    bds_df.columns = [c.lower() for c in bds_df.columns]
    
    # Numeric Cleaning
    cols_to_numeric = ['firm', 'estab', 'job_destruction_rate']
    for c in cols_to_numeric:
        if c in bds_df.columns:
            bds_df[c] = pd.to_numeric(bds_df[c], errors='coerce').fillna(0)
    
    if 'year' in bds_df.columns:
        bds_df['year'] = bds_df['year'].astype(str)
            
    print(f"  > Successfully fetched {len(bds_df)} BDS rows.")
    return bds_df
# -------------------------------
# 3. CBP FETCHER (Market Density)
# -------------------------------
def fetch_cbp_dynamic(start_year, end_year):
    print(f"Fetching CBP data for years {start_year}-{end_year}...")
    all_cbp_data = []
    
    for year in range(start_year, end_year + 1):
        # 1. Determine the correct variable name based on the NAICS vintage
        if year < 2008:
            naics_param = "NAICS2002"  # Used for 2002-2007
        elif year < 2012:
            naics_param = "NAICS2007"  # Used for 2008-2011
        elif year < 2017:
            naics_param = "NAICS2012"  # Used for 2012-2016
        else:
            naics_param = "NAICS2017"  # Used for 2017-Present
        
        url = f"https://api.census.gov/data/{year}/cbp"
        
        # We need naics_param to get the code, but we ask for EMP, ESTAB, PAYANN
        params = {
            "get": f"{naics_param},EMP,ESTAB,PAYANN", 
            "for": "state:*",
            "key": CENSUS_API_KEY
        }

        try:
            r = requests.get(url, params=params, timeout=20)
            
            if r.status_code == 200:
                data = r.json()
                df = pd.DataFrame(data[1:], columns=data[0])
                df["year"] = str(year)
                
                # 2. Rename the dynamic column to a static name 'naics_sector'
                if naics_param in df.columns:
                    df.rename(columns={naics_param: 'naics_sector'}, inplace=True)
                
                # 3. Filter for 2-digit sectors only
                # (This keeps the dataset size small and matches your SBA logic)
                df = df[df['naics_sector'].str.len() == 2]
                
                all_cbp_data.append(df)
            else:
                # Still print warning but don't crash
                print(f"  Warning {year}: API returned status {r.status_code}")
                
        except Exception as e:
            print(f"  Skipping {year}: {e}")
            pass 

    if not all_cbp_data:
        return pd.DataFrame(columns=['emp', 'estab', 'payann', 'year', 'state', 'naics_sector'])
    
    full_cbp_df = pd.concat(all_cbp_data, ignore_index=True)
    full_cbp_df.columns = [c.lower() for c in full_cbp_df.columns]
    
    # Type conversion
    if 'year' in full_cbp_df.columns:
        full_cbp_df['year'] = full_cbp_df['year'].astype(str)
    if 'state' in full_cbp_df.columns:
        full_cbp_df['state'] = full_cbp_df['state'].astype(str)

    numeric_cols = ['emp', 'estab', 'payann']
    for col in numeric_cols:
        if col in full_cbp_df.columns:
            full_cbp_df[col] = pd.to_numeric(full_cbp_df[col], errors='coerce').fillna(0)
        
    return full_cbp_df
# -------------------------------
# 4. FRED FETCHER (Macro Drivers)
# -------------------------------
def fetch_fred_series(series_id, api_key):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": 10000 
    }
    try:
        r = requests.get(url, params=params)
        data = r.json()
        if "observations" not in data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data["observations"])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year_str'] = df['date'].dt.year.astype(str)
        
        # Average to yearly level
        df_yearly = df.groupby('year_str')['value'].mean().reset_index()
        df_yearly.rename(columns={'value': f'fred_{series_id.lower()}'}, inplace=True)
        return df_yearly
    except Exception:
        return pd.DataFrame()
# -------------------------------
# 5. MASTER MERGE
# -------------------------------
# -------------------------------
# 5. MASTER MERGE (Type-Safe Version)
# -------------------------------
def merge_datasets(sba_df, bds_df, cbp_df, fred_df):
    print("Merging datasets...")

    # --- SAFETY FIX: FORCE STRING TYPES ---
    # We explicitly convert all join keys to strings to prevent "int64 vs object" errors.
    
    # 1. SBA Keys
    sba_df['approvalfy'] = sba_df['approvalfy'].astype(str).str.replace(".0", "", regex=False)
    if 'projectstate' in sba_df.columns:
         # Ensure State FIPS is string (e.g., "06" not 6)
        sba_df['state_fips'] = sba_df['projectstate'].map(STATE_TO_FIPS).astype(str)

    # 2. BDS Keys
    if 'year' in bds_df.columns:
        bds_df['year'] = bds_df['year'].astype(str)
    
    # 3. CBP Keys
    if 'year' in cbp_df.columns:
        cbp_df['year'] = cbp_df['year'].astype(str)
    if 'state' in cbp_df.columns:
        cbp_df['state'] = cbp_df['state'].astype(str)

    # 4. FRED Keys
    if 'year_str' in fred_df.columns:
        fred_df['year_str'] = fred_df['year_str'].astype(str)
    # -------------------------------------

    # A. SBA -> BDS (Survival Context)
    # Match on Year and 2-Digit Sector Code
    if 'naicscode' in sba_df.columns:
        sba_df['naics_sector'] = sba_df['naicscode'].astype(str).str[:2]
    
    merged = sba_df.merge(
        bds_df,
        left_on=['approvalfy', 'naics_sector'],
        right_on=['year', 'naics'],
        how='left',
        suffixes=('_sba', '_bds')
    )
    
    # B. -> CBP (Local Market Density)
# Ensure columns match (e.g. both are strings)
    merged = merged.merge(
        cbp_df,
        left_on=['approvalfy', 'state_fips', 'naics_sector'], # Add Sector here
        right_on=['year', 'state', 'naics_sector'],           # And here
        how='left',
        suffixes=('', '_cbp')
    )

    # C. -> FRED (Macro Economy)
    merged = merged.merge(
        fred_df,
        left_on='approvalfy',
        right_on='year_str',
        how='left'
    )

    print(f"Merge Complete. Final dataset size: {len(merged)} rows.")
    return merged

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    
    # 1. DEFINE SCOPE (2005 - 2025)
    min_year = 2005
    max_year = 2025
    print(f"--- Starting Pipeline for {min_year}-{max_year} ---")

    # 2. LOAD SBA DATA
    relevant_files = csv_files
    
    sba_df = load_sba_7a_sample(relevant_files, sample_size=5000) # Increased sample size
    
    if not sba_df.empty:
        sba_df = clean_sba(sba_df)
        
        # Filter to relevant years
        sba_df = sba_df[
            (sba_df['approvalfy'].astype(int) >= min_year) & 
            (sba_df['approvalfy'].astype(int) <= max_year)
        ]
        print(f"Filtered SBA Data to {min_year}-{max_year}: {len(sba_df)} loans.")

        # 3. FETCH CONTEXT DATA
        # Note: BDS data lags significantly (2022/23 might be latest available).
        # The fetcher handles missing years gracefully.
        bds_df = fetch_bds_all_sectors(min_year, max_year)
        cbp_df = fetch_cbp_dynamic(min_year, max_year)
        
        # 4. FETCH MACRO DRIVERS (FRED)
        print("Fetching Macro Drivers (Hazard Risks)...")
        
        # A. Inflation
        fred_cpi = fetch_fred_series("CPIAUCSL", api_key=FRED_API_KEY)
        # B. Interest Rates (Cost of Capital)
        fred_rates = fetch_fred_series("DGS10", api_key=FRED_API_KEY)
        # C. Unemployment
        fred_unemp = fetch_fred_series("UNRATE", api_key=FRED_API_KEY)
        # Merge FRED data
        fred_combined = fred_cpi.merge(fred_rates, on='year_str', how='outer') \
                                .merge(fred_unemp, on='year_str', how='outer')
        
        # 5. MERGE EVERYTHING
        final_df = merge_datasets(sba_df, bds_df, cbp_df, fred_combined)
        final_df.to_csv("analysis_dataset.csv", index=False)
        print("Successfully saved 'analysis_dataset.csv'")
        # 6. SAVE TO DB & CSV
        print("Writing final dataset to PostgreSQL and CSV...")
        if engine:
            try:
                final_df.to_sql("sba_analysis_dataset", engine, if_exists="replace", index=False)
                print("Successfully wrote to PostgreSQL table 'sba_analysis_dataset'")
            except Exception as e:
                print(f"DB Write Failed: {e}")
        
        final_df.to_csv("sba_analysis_dataset_5yr.csv", index=False)
        print("Successfully saved 'sba_analysis_dataset_5yr.csv'")
        
    else:
        print("No SBA data loaded. Check that 'foia-7a-fy2020-present...' exists.")