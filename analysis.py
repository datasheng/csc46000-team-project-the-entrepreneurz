"""
Survival Analysis using Kaplan-Meier Estimation for SBA 7(a) Loans
Loads data from PostgreSQL and generates survival probability scenarios.
"""

import pandas as pd
import warnings
from datetime import datetime
import os
from sqlalchemy import create_engine, text
from os import getenv
from dotenv import load_dotenv

from lifelines import KaplanMeierFitter

load_dotenv()
warnings.filterwarnings("ignore")


# -------------------------------
# DATABASE CONNECTION
# -------------------------------

def get_db_engine():
    """Create database engine if credentials are available."""
    DB_USER = getenv('user', 'postgres')
    DB_PASSWORD = getenv('password', 'your_password')
    DB_HOST = getenv('host', 'your-db-instance.rds.amazonaws.com')
    DB_PORT = getenv('port', '5432')
    DB_NAME = getenv('database', 'postgres')
    
    connection_str = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    try:
        engine = create_engine(connection_str)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection established")
        return engine
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None


# -------------------------------
# DATA LOADING
# -------------------------------

def load_prepared_data():
    """Load data from PostgreSQL database."""
    engine = get_db_engine()
    if not engine:
        raise ConnectionError("Cannot connect to PostgreSQL database. Please ensure main.py has been run and database credentials are correct.")
    
    try:
        print("Loading data from PostgreSQL table 'sba_analysis_dataset'...")
        df = pd.read_sql("SELECT * FROM sba_analysis_dataset", engine)
        print(f"✓ Successfully loaded {len(df):,} rows from PostgreSQL")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Failed to load data from database: {e}")


def filter_250k_loans(df, tol=50000):
    """Filter loans around $250k (default tolerance: ±$50k)."""
    df = df.copy()
    df["grossapproval"] = pd.to_numeric(df["grossapproval"], errors="coerce")
    filtered = df[
        (df["grossapproval"] >= 250000 - tol) &
        (df["grossapproval"] <= 250000 + tol)
    ]
    print(f"Filtered to {len(filtered):,} loans around $250k (±${tol:,})")
    return filtered


# -------------------------------
# SURVIVAL DATA PREPARATION
# -------------------------------

def prepare_survival_data(df, max_months=60):
    """Prepare survival analysis data from loan dates."""
    df = df.copy()
    
    # Convert date columns
    for c in ["approvaldate", "paidinfulldate", "chargeoffdate", "asofdate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Event indicator: 1 if chargeoff occurred, 0 if censored
    df["event"] = df["chargeoffdate"].notna().astype(int)
    df["start"] = df["approvaldate"]
    
    # Censoring date: use asofdate if available, otherwise current date
    censor = df["asofdate"] if "asofdate" in df.columns else pd.Timestamp.now()
    censor = censor.fillna(pd.Timestamp.now())
    
    # End time: chargeoff date if available, otherwise paid in full date, otherwise censor date
    df["end"] = df["chargeoffdate"].fillna(
        df["paidinfulldate"].fillna(censor)
    )
    
    # Calculate time in months
    df["time"] = (df["end"] - df["start"]).dt.days / 30.44
    df = df[df["time"] >= 0]  # Remove invalid negative times
    
    # Censor at max_months
    df["time"] = df["time"].clip(upper=max_months)
    df.loc[df["time"] >= max_months, "event"] = 0
    
    print(f"Prepared survival data: {df['event'].sum():,} events, {len(df) - df['event'].sum():,} censored")
    return df


# -------------------------------
# KAPLAN-MEIER ESTIMATION
# -------------------------------

def km_table(df, label, time_points=[12, 24, 36, 48, 60]):
    """
    Generate Kaplan-Meier survival table with detailed statistics.
    
    Returns DataFrame with survival probabilities, failure rates, confidence intervals,
    and population statistics at specified time points.
    """
    if len(df) == 0:
        print(f"Warning: No data for {label}")
        return pd.DataFrame()
    
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], df["event"], label=label)
    
    rows = []
    for t in time_points:
        # Get survival probability at time t
        t_eval = min(t, kmf.timeline.max()) if len(kmf.timeline) > 0 else t
        s = float(kmf.predict(t_eval)) if len(kmf.timeline) > 0 else 1.0
        
        # Get confidence intervals
        if len(kmf.confidence_interval_) > 0:
            ci = kmf.confidence_interval_
            ci_filtered = ci.loc[ci.index <= t_eval]
            if len(ci_filtered) > 0:
                lo, hi = ci_filtered.iloc[-1]
            else:
                lo, hi = s, s
        else:
            lo, hi = s, s
        
        # Calculate statistics
        at_risk = int((df["time"] >= t).sum())
        events_up_to_t = int(df[(df["event"] == 1) & (df["time"] <= t)].shape[0])
        censored_up_to_t = int(df[(df["event"] == 0) & (df["time"] <= t)].shape[0])
        
        # Calculate hazard rate (events per month in this period)
        if t == time_points[0]:
            prev_t = 0
        else:
            prev_t = time_points[time_points.index(t) - 1]
        
        events_in_period = int(df[(df["event"] == 1) & 
                                  (df["time"] > prev_t) & 
                                  (df["time"] <= t)].shape[0])
        months_in_period = t - prev_t
        hazard_rate = events_in_period / (at_risk * months_in_period) if at_risk > 0 and months_in_period > 0 else 0.0
        
        rows.append({
            "time_months": t,
            "survival_prob": round(s, 4),
            "failure_prob": round(1 - s, 4),
            "failure_rate_pct": round((1 - s) * 100, 2),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "at_risk": at_risk,
            "events": events_up_to_t,
            "censored": censored_up_to_t,
            "hazard_rate": round(hazard_rate, 6),
            "group": label
        })
    
    return pd.DataFrame(rows)


# -------------------------------
# EXPORT FUNCTIONS
# -------------------------------

def export(df, name, description):
    """Export DataFrame to CSV with metadata header."""
    os.makedirs("analysis_results", exist_ok=True)
    path = f"analysis_results/{name}.csv"
    
    with open(path, "w") as f:
        f.write(f"# {description}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total observations: {len(df)}\n\n")
        df.to_csv(f, index=False)
    
    print(f"✓ Saved {path} ({len(df)} rows)")


# -------------------------------
# MAIN ANALYSIS
# -------------------------------

def main():
    """Main analysis function: load data, perform survival analysis, export results."""
    print("=" * 70)
    print("SBA 7(a) Loan Survival Analysis - Kaplan-Meier Estimation")
    print("=" * 70)
    
    # Load data from PostgreSQL
    df = load_prepared_data()
    
    # Filter to ~$250k loans
    df = filter_250k_loans(df)
    
    # Prepare survival data
    df = prepare_survival_data(df, max_months=60)
    
    print("\n" + "=" * 70)
    print("Generating Survival Scenarios...")
    print("=" * 70)
    
    # Scenario 1: Overall Survival
    print("\n1. Overall Survival (All ~$250k loans)")
    overall = km_table(df, "Overall")
    export(overall, "overall_survival.csv", 
           "Overall survival probabilities for all SBA 7(a) loans around $250k")
    
    # Scenario 2: Low Interest Rate
    if "initialinterestrate" in df.columns:
        print("\n2. Low Interest Rate Scenario")
        median_rate = df["initialinterestrate"].median()
        low_interest = df[df["initialinterestrate"] < median_rate].copy()
        low_km = km_table(low_interest, "Low Interest Rate")
        export(low_km, "low_interest_rate_survival.csv",
               f"Survival probabilities for loans with interest rate < median ({median_rate:.2f}%)")
    else:
        print("\n2. Low Interest Rate Scenario - SKIPPED (column not found)")
        low_km = pd.DataFrame()
    
    # Scenario 3: High Interest Rate
    if "initialinterestrate" in df.columns:
        print("\n3. High Interest Rate Scenario")
        median_rate = df["initialinterestrate"].median()
        high_interest = df[df["initialinterestrate"] >= median_rate].copy()
        high_km = km_table(high_interest, "High Interest Rate")
        export(high_km, "high_interest_rate_survival.csv",
               f"Survival probabilities for loans with interest rate ≥ median ({median_rate:.2f}%)")
    else:
        print("\n3. High Interest Rate Scenario - SKIPPED (column not found)")
        high_km = pd.DataFrame()
    
    # Scenario 4: Low SBA Guarantee
    if "sba_ratio" in df.columns:
        print("\n4. Low SBA Guarantee Scenario")
        median_ratio = df["sba_ratio"].median()
        low_sba = df[df["sba_ratio"] < median_ratio].copy()
        low_sba_km = km_table(low_sba, "Low SBA Guarantee")
        export(low_sba_km, "low_sba_guarantee_survival.csv",
               f"Survival probabilities for loans with SBA guarantee ratio < median ({median_ratio:.3f})")
    else:
        print("\n4. Low SBA Guarantee Scenario - SKIPPED (column not found)")
        low_sba_km = pd.DataFrame()
    
    # Scenario 5: High SBA Guarantee
    if "sba_ratio" in df.columns:
        print("\n5. High SBA Guarantee Scenario")
        median_ratio = df["sba_ratio"].median()
        high_sba = df[df["sba_ratio"] >= median_ratio].copy()
        high_sba_km = km_table(high_sba, "High SBA Guarantee")
        export(high_sba_km, "high_sba_guarantee_survival.csv",
               f"Survival probabilities for loans with SBA guarantee ratio ≥ median ({median_ratio:.3f})")
    else:
        print("\n5. High SBA Guarantee Scenario - SKIPPED (column not found)")
        high_sba_km = pd.DataFrame()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nGenerated 5 CSV files in 'analysis_results/' directory:")
    print("  1. overall_survival.csv")
    print("  2. low_interest_rate_survival.csv")
    print("  3. high_interest_rate_survival.csv")
    print("  4. low_sba_guarantee_survival.csv")
    print("  5. high_sba_guarantee_survival.csv")
    print("\nEach file contains:")
    print("  - Survival probabilities at 12, 24, 36, 48, 60 months")
    print("  - Failure rates and confidence intervals")
    print("  - At-risk population, events, and censored observations")
    print("  - Hazard rates for each time period")


if __name__ == "__main__":
    main()

