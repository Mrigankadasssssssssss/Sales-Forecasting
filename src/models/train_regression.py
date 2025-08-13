import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

BASE = Path(__file__).resolve().parent.parent.parent
PROC = BASE / "data" / "processed"
MODEL_DIR = BASE / "model_store"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df = pd.read_csv(PROC / 'monthly_by_category.csv', parse_dates=['ds'])
    df = df.sort_values(['category','ds'])
    df['prev_revenue'] = df.groupby('category')['y_revenue'].shift(1)
    df = df.dropna()

    X = df[['y_revenue', 'prev_revenue']]
    y = df['profit']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    m = RandomForestRegressor(n_estimators=200, random_state=42)
    m.fit(X_scaled, y)

    joblib.dump(m, MODEL_DIR / 'profit_regressor.pkl')
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    print('✅ Saved profit regressor and scaler')

if __name__ == '__main__':
    train()
