import pandas as pd
from pathlib import Path
import sys

# 🔹 Always set BASE to the project root (one level above src/)
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "synthetic_online_retail_data.csv"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def run():
    # ✅ Check raw file exists
    if not RAW.exists():
        sys.exit(f"❌ Raw data not found at {RAW}. Please place 'synthetic_online_retail_data.csv' in data/raw/.")

    # Load raw dataset
    df = pd.read_csv(RAW, parse_dates=['order_date'])

    # ✅ Compute revenue
    df['total_revenue'] = df['quantity'] * df['price']

    # ✅ Compute profit if missing
    if 'profit' not in df.columns:
        df['profit'] = df['total_revenue'] * 0.2  # assume 20% margin
    else:
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(df['total_revenue'] * 0.2)

    # ✅ Aggregate monthly per category
    df['ds'] = df['order_date'].dt.to_period('M').dt.to_timestamp()
    monthly = (
        df.groupby(['ds', 'category_name'])
        .agg(
            quantity=('quantity', 'sum'),
            y_revenue=('total_revenue', 'sum'),
            profit=('profit', 'sum')
        )
        .reset_index()
        .rename(columns={'category_name': 'category'})
    )

    # Save processed file
    monthly.to_csv(PROC / 'monthly_by_category.csv', index=False)
    print(f"✅ Saved processed file to: {PROC / 'monthly_by_category.csv'}")

if __name__ == '__main__':
    run()
