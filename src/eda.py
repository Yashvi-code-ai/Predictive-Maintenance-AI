"""Reproducible exploratory data analysis for the AI4I dataset.

Run: python -m src.eda
Requires data/ai4i2020.csv. Figures and a text summary are saved under outputs/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import (
    IDENTIFIER_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN, load_dataset, clean_dataset,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    raw = load_dataset()
    df, report = clean_dataset(raw)
    report_path = OUTPUT_DIR / "eda_summary.txt"
    report_path.write_text(
        "AI4I 2020 EDA summary\n"
        f"Rows before cleaning: {report['rows_before_cleaning']}\n"
        f"Rows after cleaning: {report['rows_after_cleaning']}\n"
        f"Duplicates removed: {report['exact_duplicate_rows_removed']}\n"
        f"Missing-target rows removed: {report['rows_with_missing_target_removed']}\n"
        f"Target distribution: {report['target_distribution']}\n"
        f"Missing values by column: {report['missing_values_by_column']}\n"
        f"IQR outlier counts (reported, not dropped): {report['iqr_outlier_counts_by_numeric_feature']}\n",
        encoding="utf-8",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET_COLUMN, ax=ax)
    ax.set(title="Machine failure class balance", xlabel="Machine failure (0 = no, 1 = yes)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    numeric = df.select_dtypes(include="number").drop(
        columns=[TARGET_COLUMN, *IDENTIFIER_COLUMNS, *LEAKAGE_COLUMNS], errors="ignore"
    )
    if not numeric.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numeric sensor-feature correlations")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "numeric_correlation.png", dpi=160)
        plt.close(fig)

    print(f"EDA outputs written to {OUTPUT_DIR.resolve()}")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
