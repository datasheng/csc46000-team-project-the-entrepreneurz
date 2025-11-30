import requests
import pandas as pd

# BDS (Business Dynamics Statistics)

def fetch_bds(naics="72", time="2022"):
    url = "https://api.census.gov/data/timeseries/bds"
    params = {
        "get": "EMP,ESTABS_ENTRY",
        "for": "us:1",
        "NAICS": naics,
        "time": time
    }
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except Exception:
        print("BDS API returned non-JSON response:", r.text)
        raise
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# CBP (County Business Patterns)

def fetch_cbp(year=2022, state="06", county="001", naics="51"):
    url = f"https://api.census.gov/data/{year}/cbp"
    params = {
        "get": "EMP,ESTAB,PAYANN",
        "NAICS2017": naics,
        "for": f"county:{county}",
        "in": f"state:{state}"
    }
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except Exception:
        print("CBP API returned non-JSON response:", r.text)
        raise
    df = pd.DataFrame(data[1:], columns=data[0])
    df["year"] = year
    return df

# SBA (CSV fallback)

def fetch_sba_csv(file_path="sba_7a_loans.csv"):
    """Load publicly available SBA loan data from CSV."""
    df = pd.read_csv(file_path)
    return df

# FRED (Macroeconomic time series)

def fetch_fred_series(series_id, api_key, limit=1000):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit
    }
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except Exception:
        print("FRED API returned non-JSON response:", r.text)
        raise
    df = pd.DataFrame(data["observations"])
    return df

# MAIN PIPELINE

if __name__ == "__main__":

    # BDS example
    print("Fetching BDS...")
    bds_df = fetch_bds(naics="72", time="2022")
    print(bds_df.head(), "\n")

    # CBP example
    print("Fetching CBP...")
    cbp_df = fetch_cbp(year=2022, state="06", county="001", naics="51")
    print(cbp_df.head(), "\n")

    # SBA CSV example
    print("Loading SBA CSV...")
    # Make sure you have downloaded a CSV to this path
    # sba_df = fetch_sba_csv("sba_7a_loans.csv")
    # print(sba_df.head(), "\n")
    print("SBA CSV placeholder. download public CSV to use.\n")

    # FRED example
    print("Fetching FRED CPI series...")
    fred_api_key = "3290178b9dfef23c54c6ddbe214b5edb"
    fred_df = fetch_fred_series("CPIAUCSL", api_key=fred_api_key)
    print(fred_df.head(), "\n")

    print("Pipeline complete. DataFrames ready:")
    print("- bds_df (BDS time series)")
    print("- cbp_df (CBP county-level)")
    print("- sba_df (loan CSV placeholder)")
    print("- fred_df (macroeconomic series)")
