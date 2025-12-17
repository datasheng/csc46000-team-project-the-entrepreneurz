"""
Visualization Module for SBA Loan Survival Analysis
Creates publication-quality visualizations from Kaplan-Meier survival analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


# -------------------------------
# DATA LOADING
# -------------------------------

def load_survival_csv(filepath):
    """Load a survival analysis CSV file, skipping comment lines."""
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time_months'):
                return pd.read_csv(filepath, skiprows=i)
    return pd.read_csv(filepath)


def load_all_scenarios(results_dir="analysis_results"):
    """Load all survival analysis CSV files."""
    results_path = Path(results_dir)
    scenarios = {}
    csv_files = {
        "overall": "overall_survival.csv",
        "low_interest": "low_interest_rate_survival.csv",
        "high_interest": "high_interest_rate_survival.csv",
        "low_sba": "low_sba_guarantee_survival.csv",
        "high_sba": "high_sba_guarantee_survival.csv",
    }
    
    for key, filename in csv_files.items():
        for ext in ["", ".csv"]:
            filepath = results_path / f"{filename}{ext}"
            if filepath.exists():
                scenarios[key] = load_survival_csv(filepath)
                print(f"✓ Loaded {filename}")
                break
    
    return scenarios


# -------------------------------
# VISUALIZATION FUNCTIONS
# -------------------------------

def plot_survival_curves(scenarios, save_path="analysis_results/survival_curves.png"):
    """Plot survival curves for all scenarios with confidence intervals."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = {
        "Overall": "#1f77b4", "Low Interest Rate": "#2ca02c", "High Interest Rate": "#d62728",
        "Low SBA Guarantee": "#ff7f0e", "High SBA Guarantee": "#9467bd"
    }
    
    for key, df in scenarios.items():
        if df.empty:
            continue
        group_name = df["group"].iloc[0] if "group" in df.columns else key
        df_sorted = df.sort_values("time_months")
        color = colors.get(group_name, "#000000")
        
        ax.plot(df_sorted["time_months"], df_sorted["survival_prob"], 
               marker='o', linewidth=3, markersize=10, label=group_name, color=color, zorder=3)
        ax.fill_between(df_sorted["time_months"], df_sorted["ci_low"], df_sorted["ci_high"],
                       alpha=0.2, color=color, zorder=1)
    
    ax.set_xlabel("Time Since Loan Approval (Months)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Survival Probability", fontsize=13, fontweight='bold')
    ax.set_title("Kaplan-Meier Survival Curves: SBA 7(a) Loan Survival by Scenario", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0.90, 1.005])
    ax.set_xlim([0, 65])
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_failure_curves(scenarios, save_path="analysis_results/failure_curves.png"):
    """Plot cumulative failure probability curves."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = {
        "Overall": "#1f77b4", "Low Interest Rate": "#2ca02c", "High Interest Rate": "#d62728",
        "Low SBA Guarantee": "#ff7f0e", "High SBA Guarantee": "#9467bd"
    }
    
    for key, df in scenarios.items():
        if df.empty:
            continue
        group_name = df["group"].iloc[0] if "group" in df.columns else key
        df_sorted = df.sort_values("time_months")
        color = colors.get(group_name, "#000000")
        ax.plot(df_sorted["time_months"], df_sorted["failure_rate_pct"], 
               marker='s', linewidth=3, markersize=10, label=group_name, color=color, zorder=3)
    
    ax.set_xlabel("Time Since Loan Approval (Months)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Cumulative Failure Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title("Cumulative Loan Failure Rate Over Time by Scenario", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 5])
    ax.set_xlim([0, 65])
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_scenario_comparison(scenarios, save_path="analysis_results/scenario_comparison.png"):
    """Create side-by-side comparison of survival probabilities."""
    all_data = [df for df in scenarios.values() if not df.empty]
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    pivot_data = combined_df.pivot(index="group", columns="time_months", values="survival_prob")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(pivot_data.index))
    width = 0.15
    time_points = [12, 24, 36, 48, 60]
    colors_time = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    for i, t in enumerate(time_points):
        if t in pivot_data.columns:
            offset = (i - len(time_points)/2) * width + width/2
            ax.bar(x + offset, pivot_data[t], width, label=f'{t} months', 
                  alpha=0.85, color=colors_time[i], edgecolor='black', linewidth=1)
    
    ax.set_xlabel("Scenario", fontsize=13, fontweight='bold')
    ax.set_ylabel("Survival Probability", fontsize=13, fontweight='bold')
    ax.set_title("Survival Probability Comparison Across Scenarios and Time Points", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index, rotation=15, ha='right')
    ax.legend(title="Time Point", frameon=True, shadow=True)
    ax.set_ylim([0.90, 1.01])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_heatmap_survival(scenarios, save_path="analysis_results/survival_heatmap.png"):
    """Create heatmap showing survival probabilities across scenarios and time."""
    all_data = [df for df in scenarios.values() if not df.empty]
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    pivot_data = combined_df.pivot(index="group", columns="time_months", values="survival_prob")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0.90, vmax=1.0)
    
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)
    
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.4f}', 
                          ha="center", va="center", color="black", fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Survival Probability', fontsize=12, fontweight='bold')
    ax.set_title("Survival Probability Heatmap: Scenarios vs Time", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Time (Months)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Scenario", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def _plot_comparison_panels(low_df, high_df, title, low_color, high_color, diff_order, save_path):
    """Helper function to create 4-panel comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    # Survival curves
    axes[0, 0].plot(low_df["time_months"], low_df["survival_prob"], 
                   marker='o', linewidth=3, markersize=10, label=low_df["group"].iloc[0], color=low_color, zorder=3)
    axes[0, 0].fill_between(low_df["time_months"], low_df["ci_low"], low_df["ci_high"],
                           alpha=0.2, color=low_color, zorder=1)
    axes[0, 0].plot(high_df["time_months"], high_df["survival_prob"], 
                   marker='s', linewidth=3, markersize=10, label=high_df["group"].iloc[0], color=high_color, zorder=3)
    axes[0, 0].fill_between(high_df["time_months"], high_df["ci_low"], high_df["ci_high"],
                           alpha=0.2, color=high_color, zorder=1)
    axes[0, 0].set_xlabel("Time (Months)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Survival Probability", fontsize=12, fontweight='bold')
    axes[0, 0].set_title("Survival Curves", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim([0.90, 1.005])
    axes[0, 0].set_xlim([0, 65])
    axes[0, 0].legend(loc='best', frameon=True, shadow=True)
    
    # Failure rates
    axes[0, 1].plot(low_df["time_months"], low_df["failure_rate_pct"], 
                   marker='o', linewidth=3, markersize=10, label=low_df["group"].iloc[0], color=low_color)
    axes[0, 1].plot(high_df["time_months"], high_df["failure_rate_pct"], 
                   marker='s', linewidth=3, markersize=10, label=high_df["group"].iloc[0], color=high_color)
    axes[0, 1].set_xlabel("Time (Months)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Cumulative Failure Rate (%)", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Failure Rate Comparison", fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim([0, 5])
    axes[0, 1].set_xlim([0, 65])
    axes[0, 1].legend(loc='upper left', frameon=True, shadow=True)
    
    # Hazard rates
    axes[1, 0].plot(low_df["time_months"], low_df["hazard_rate"] * 1000, 
                   marker='o', linewidth=3, markersize=10, label=low_df["group"].iloc[0], color=low_color)
    axes[1, 0].plot(high_df["time_months"], high_df["hazard_rate"] * 1000, 
                   marker='s', linewidth=3, markersize=10, label=high_df["group"].iloc[0], color=high_color)
    axes[1, 0].set_xlabel("Time (Months)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Hazard Rate (per 1000 loans/month)", fontsize=12, fontweight='bold')
    axes[1, 0].set_title("Hazard Rate Over Time", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlim([0, 65])
    axes[1, 0].legend(loc='best', frameon=True, shadow=True)
    
    # Survival difference
    survival_diff = (low_df["survival_prob"].values - high_df["survival_prob"].values) * diff_order
    axes[1, 1].bar(low_df["time_months"], survival_diff * 100, 
                  color='steelblue' if diff_order > 0 else 'purple', alpha=0.7, edgecolor='black', linewidth=1)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel("Time (Months)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Survival Difference (%)", fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f"Survival Advantage: {low_df['group'].iloc[0]} vs {high_df['group'].iloc[0]}", 
                        fontsize=13, fontweight='bold')
    axes[1, 1].set_xlim([0, 65])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_interest_rate_comparison(scenarios, save_path="analysis_results/interest_rate_comparison.png"):
    """Comprehensive comparison of low vs high interest rate scenarios."""
    if "low_interest" not in scenarios or "high_interest" not in scenarios:
        return
    low_df = scenarios["low_interest"].sort_values("time_months")
    high_df = scenarios["high_interest"].sort_values("time_months")
    if low_df.empty or high_df.empty:
        return
    _plot_comparison_panels(low_df, high_df, "Interest Rate Impact on Loan Survival",
                           "#2ca02c", "#d62728", 1, save_path)


def plot_sba_guarantee_comparison(scenarios, save_path="analysis_results/sba_guarantee_comparison.png"):
    """Comprehensive comparison of low vs high SBA guarantee scenarios."""
    if "low_sba" not in scenarios or "high_sba" not in scenarios:
        return
    low_df = scenarios["low_sba"].sort_values("time_months")
    high_df = scenarios["high_sba"].sort_values("time_months")
    if low_df.empty or high_df.empty:
        return
    _plot_comparison_panels(low_df, high_df, "SBA Guarantee Ratio Impact on Loan Survival",
                           "#ff7f0e", "#9467bd", -1, save_path)


def plot_at_risk_population(scenarios, save_path="analysis_results/at_risk_population.png"):
    """Plot number of loans at risk over time."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = {
        "Overall": "#1f77b4", "Low Interest Rate": "#2ca02c", "High Interest Rate": "#d62728",
        "Low SBA Guarantee": "#ff7f0e", "High SBA Guarantee": "#9467bd"
    }
    
    for key, df in scenarios.items():
        if df.empty:
            continue
        group_name = df["group"].iloc[0] if "group" in df.columns else key
        df_sorted = df.sort_values("time_months")
        color = colors.get(group_name, "#000000")
        ax.plot(df_sorted["time_months"], df_sorted["at_risk"], 
               marker='o', linewidth=3, markersize=10, label=group_name, color=color, zorder=3)
    
    ax.set_xlabel("Time Since Loan Approval (Months)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Loans at Risk", fontsize=13, fontweight='bold')
    ax.set_title("At-Risk Population Over Time by Scenario", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 65])
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.set_yscale('log')
    ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def plot_event_counts(scenarios, save_path="analysis_results/event_counts.png"):
    """Plot cumulative event (chargeoff) counts over time."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = {
        "Overall": "#1f77b4", "Low Interest Rate": "#2ca02c", "High Interest Rate": "#d62728",
        "Low SBA Guarantee": "#ff7f0e", "High SBA Guarantee": "#9467bd"
    }
    
    for key, df in scenarios.items():
        if df.empty:
            continue
        group_name = df["group"].iloc[0] if "group" in df.columns else key
        df_sorted = df.sort_values("time_months")
        color = colors.get(group_name, "#000000")
        ax.plot(df_sorted["time_months"], df_sorted["events"], 
               marker='s', linewidth=3, markersize=10, label=group_name, color=color, zorder=3)
    
    ax.set_xlabel("Time Since Loan Approval (Months)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Cumulative Chargeoff Events", fontsize=13, fontweight='bold')
    ax.set_title("Cumulative Loan Chargeoffs Over Time by Scenario", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 65])
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


def create_summary_table(scenarios, save_path="analysis_results/summary_statistics.png"):
    """Create visual summary statistics table at 60 months."""
    all_data = [df for df in scenarios.values() if not df.empty]
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    summary_60 = combined_df[combined_df["time_months"] == 60].copy()
    
    if summary_60.empty:
        summary_60 = combined_df.groupby("group").last().reset_index()
    
    summary_60 = summary_60[["group", "survival_prob", "failure_rate_pct", 
                             "at_risk", "events", "censored"]].copy()
    summary_60.columns = ["Scenario", "Survival Prob", "Failure Rate (%)", 
                          "At Risk", "Events", "Censored"]
    summary_60["Survival Prob"] = summary_60["Survival Prob"].apply(lambda x: f"{x:.4f}")
    summary_60["Failure Rate (%)"] = summary_60["Failure Rate (%)"].apply(lambda x: f"{x:.2f}%")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_60.values, colLabels=summary_60.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    for i in range(len(summary_60.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_60) + 1):
        for j in range(len(summary_60.columns)):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else '#ffffff')
    
    plt.title("Summary Statistics at 60 Months (or Latest Available)", 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {save_path}")
    plt.close()


# -------------------------------
# MAIN FUNCTION
# -------------------------------

def generate_all_visualizations(results_dir="analysis_results", output_dir="analysis_results"):
    """Generate all visualizations from survival analysis results."""
    print("=" * 70)
    print("Generating Survival Analysis Visualizations")
    print("=" * 70)
    
    scenarios = load_all_scenarios(results_dir)
    if not scenarios:
        print("⚠ No data files found. Please run analysis.py first.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Creating Visualizations...")
    print("=" * 70)
    
    plot_survival_curves(scenarios, f"{output_dir}/survival_curves.png")
    plot_failure_curves(scenarios, f"{output_dir}/failure_curves.png")
    plot_scenario_comparison(scenarios, f"{output_dir}/scenario_comparison.png")
    plot_heatmap_survival(scenarios, f"{output_dir}/survival_heatmap.png")
    plot_interest_rate_comparison(scenarios, f"{output_dir}/interest_rate_comparison.png")
    plot_sba_guarantee_comparison(scenarios, f"{output_dir}/sba_guarantee_comparison.png")
    plot_at_risk_population(scenarios, f"{output_dir}/at_risk_population.png")
    plot_event_counts(scenarios, f"{output_dir}/event_counts.png")
    create_summary_table(scenarios, f"{output_dir}/summary_statistics.png")
    
    print("\n" + "=" * 70)
    print("All visualizations generated successfully!")
    print("=" * 70)
    print(f"\nGenerated 9 visualization files in '{output_dir}/' directory")


if __name__ == "__main__":
    generate_all_visualizations()
