#!/usr/bin/env python3
"""
Analysis script to evaluate autoresearch experiment results.
This would be used after running experiments to determine next steps.
"""

import pandas as pd
import os

def analyze_results():
    """Analyze the results.tsv file and suggest next steps."""

    if not os.path.exists('results.tsv'):
        print("No results.tsv found. Please run experiments first.")
        return

    # Read the results
    df = pd.read_csv('results.tsv', sep='\t')

    if len(df) < 2:  # Need at least baseline and one experiment
        print("Need at least 2 runs to analyze improvements.")
        return

    print("=== Autoresearch Experiment Analysis ===")
    print(f"Total experiments run: {len(df)}")

    # Show recent results
    recent = df.tail(5)
    print("\nRecent Results:")
    print(recent[['commit', 'val_bpb', 'memory_gb', 'status']])

    # Check for improvements
    baseline = df.iloc[0]  # First run (baseline)
    latest = df.iloc[-1]   # Most recent run

    print(f"\n=== Comparison ===")
    print(f"Baseline (commit {baseline['commit']}): val_bpb = {baseline['val_bpb']:.6f}")
    print(f"Latest (commit {latest['commit']}): val_bpb = {latest['val_bpb']:.6f}")

    improvement = baseline['val_bpb'] - latest['val_bpb']
    if improvement > 0:
        print(f"Improvement: +{improvement:.6f} (better)")
    elif improvement < 0:
        print(f"Regression: {improvement:.6f} (worse)")
    else:
        print("No change in performance")

if __name__ == "__main__":
    analyze_results()