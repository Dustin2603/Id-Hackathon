#!/usr/bin/env python3
"""
scripts/prepare_serving_dataset.py

Create a compact serving parquet containing only the columns used by the app.
This script downcasts numeric dtypes, converts low-cardinality objects to
categories and writes an optimized parquet suitable for deployment.

Usage:
  python scripts/prepare_serving_dataset.py --input dataset_enriched.parquet --output dataset_serving.parquet

Do NOT commit the output parquet into the repository; keep it for deployment only.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Columns to keep: union of feature lists used by the app + metadata + scores
STUFF_FEATURES = [
    "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
    "VertBreak", "EffectiveVelo", "SpeedDrop",
    "RelHeight", "RelSide", "Extension",
    "VertRelAngle", "HorzRelAngle",
    "AutoPitchType_Num", "PitcherThrows_Num", "BatterSide_Num",
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
]

LOCATION_FEATURES = [
    "PlateLocHeight", "PlateLocSide", "in_strike_zone",
    "Balls", "Strikes",
    "AutoPitchType_Num", "PitcherThrows_Num", "BatterSide_Num",
]

PITCHING_FEATURES = [
    "RelSpeed", "SpinRate", "SpinAxis", "InducedVertBreak", "HorzBreak",
    "VertBreak", "EffectiveVelo", "SpeedDrop",
    "RelHeight", "RelSide", "Extension",
    "VertRelAngle", "HorzRelAngle",
    "PlateLocHeight", "PlateLocSide", "in_strike_zone",
    "Balls", "Strikes",
    "AutoPitchType_Num", "PitcherThrows_Num", "BatterSide_Num",
    "velo_diff_vs_fb", "ivb_diff_vs_fb", "hb_diff_vs_fb",
    "effective_distance", "perceived_velo_factor",
]

# Scores and metadata the app may need
SCORE_COLS = ["Stuff+", "Location+", "Pitching+"]
METADATA_COLS = ["pitcher_anon_id", "year", "AutoPitchType", "altitude_category"]

KEEP_COLS = list({
    *STUFF_FEATURES,
    *LOCATION_FEATURES,
    *PITCHING_FEATURES,
    *SCORE_COLS,
    *METADATA_COLS,
})


def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    # Downcast floats
    float_cols = df.select_dtypes(include=[np.floating]).columns
    for c in float_cols:
        try:
            df[c] = pd.to_numeric(df[c], downcast='float')
        except Exception:
            try:
                df[c] = df[c].astype(np.float32)
            except Exception:
                pass

    # Downcast ints
    int_cols = df.select_dtypes(include=['int64', 'int32', 'Int64']).columns
    for c in int_cols:
        try:
            df[c] = pd.to_numeric(df[c], downcast='integer')
        except Exception:
            pass

    # Convert object/string cols to category when cardinality is reasonably small
    obj_cols = df.select_dtypes(include=['object']).columns
    n_rows = len(df)
    for c in obj_cols:
        try:
            nunique = df[c].nunique(dropna=False)
            if nunique == 0:
                continue
            if (nunique / max(1, n_rows)) < 0.5 or nunique < 1000:
                df[c] = df[c].astype('category')
        except Exception:
            continue

    return df


def build_serving_dataset(input_path: Path, output_path: Path, sample_frac: float = None):
    print(f"Reading {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Original shape: {df.shape}")

    # Keep only columns that exist and are required
    cols_to_keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols_to_keep].copy()
    print(f"Kept columns: {len(cols_to_keep)}; shape after column filtering: {df.shape}")

    # Optional sampling (to reduce size for very large datasets)
    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"Sampled to frac={sample_frac}; new shape: {df.shape}")

    # Downcast data
    df = _downcast_df(df)
    print("Downcasted dtypes:")
    print(df.dtypes)

    # Write parquet
    print(f"Writing optimized serving parquet to {output_path}")
    df.to_parquet(output_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Prepare a compact serving parquet optimized for Streamlit deployment")
    parser.add_argument("--input", required=True, help="Input parquet path (dataset_enriched.parquet or dataset_scores_final_v2.parquet)")
    parser.add_argument("--output", required=True, help="Output optimized parquet path (dataset_serving.parquet)")
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional fractional sampling to reduce output size (0-1)")
    args = parser.parse_args()

    build_serving_dataset(Path(args.input), Path(args.output), sample_frac=args.sample_frac)


if __name__ == '__main__':
    main()
