import pandas as pd
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "synthetic_online_retail_data.csv"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def run():

    if not RAW.exists():
        sys.exit(f"❌ Raw data not found at {RAW}. Please place 'synthetic_online_retail_data.csv' in data/raw/.")

 
    df = pd.read_csv(RAW, parse_dates=['order_date'])


    df['total_revenue'] = df['quantity'] * df['price']


    if 'profit' not in df.columns:
        df['profit'] = df['total_revenue'] * 0.2 
    else:
        df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(df['total_revenue'] * 0.2)


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


    monthly.to_csv(PROC / 'monthly_by_category.csv', index=False)
    print(f"✅ Saved processed file to: {PROC / 'monthly_by_category.csv'}")

if __name__ == '__main__':
    run()
