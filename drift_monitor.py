"""
Drift Monitor — detects distribution shift between the training baseline
(California) and new incoming data using Population Stability Index (PSI).

This operationalizes the project's core thesis: models trained on one
distribution degrade when the input distribution shifts. This module
flags that shift before it silently damages predictions.
"""

import numpy as np
import pandas as pd


def calculate_psi(baseline: np.ndarray, incoming: np.ndarray, bins: int = 10) -> float:
    """
    Compute the Population Stability Index between a baseline and incoming
    distribution for a single feature.

    PSI = sum( (incoming% - baseline%) * ln(incoming% / baseline%) )
    """
    # Define bin edges from the baseline distribution's range
    min_val = min(baseline.min(), incoming.min())
    max_val = max(baseline.max(), incoming.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Get proportion of samples in each bin
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    incoming_counts, _ = np.histogram(incoming, bins=bin_edges)

    baseline_pct = baseline_counts / len(baseline)
    incoming_pct = incoming_counts / len(incoming)

    # Avoid division by zero / log(0) with a tiny epsilon
    epsilon = 1e-6
    baseline_pct = np.clip(baseline_pct, epsilon, None)
    incoming_pct = np.clip(incoming_pct, epsilon, None)

    psi = np.sum((incoming_pct - baseline_pct) * np.log(incoming_pct / baseline_pct))
    return float(psi)


def interpret_psi(psi: float) -> str:
    """Map a PSI value to a human-readable risk level."""
    if psi < 0.1:
        return "STABLE — no significant drift"
    elif psi < 0.25:
        return "MODERATE — monitor closely"
    else:
        return "SEVERE — drift detected, model may be unreliable"


def monitor_drift(baseline_df: pd.DataFrame, incoming_df: pd.DataFrame,
                  features: list = None) -> pd.DataFrame:
    """
    Compute PSI for each feature and return a drift report DataFrame.
    """
    if features is None:
        features = baseline_df.columns.tolist()

    report = []
    for feat in features:
        psi = calculate_psi(
            baseline_df[feat].values,
            incoming_df[feat].values
        )
        report.append({
            "feature": feat,
            "psi": round(psi, 4),
            "status": interpret_psi(psi)
        })

    report_df = pd.DataFrame(report).sort_values("psi", ascending=False)
    return report_df.reset_index(drop=True)


if __name__ == "__main__":
    # Demonstration: compare California (baseline) vs Puerto Rico (shifted)
    from folktables import ACSDataSource, ACSIncome

    print("Loading California (baseline) and Puerto Rico (incoming) data...")
    source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

    ca_data = source.get_data(states=['CA'], download=True)
    pr_data = source.get_data(states=['PR'], download=True)

    ca_features, _, _ = ACSIncome.df_to_numpy(ca_data)
    pr_features, _, _ = ACSIncome.df_to_numpy(pr_data)

    ca_df = pd.DataFrame(ca_features, columns=ACSIncome.features)
    pr_df = pd.DataFrame(pr_features, columns=ACSIncome.features)

    print("\nComputing drift report (California -> Puerto Rico)...\n")
    report = monitor_drift(ca_df, pr_df)
    print(report.to_string(index=False))

    max_psi = report['psi'].max()
    print(f"\nMax PSI across features: {max_psi:.4f}")
    print(f"Overall assessment: {interpret_psi(max_psi)}")