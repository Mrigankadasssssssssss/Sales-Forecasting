
import pandas as pd
from pathlib import Path
from prophet import Prophet
import joblib

BASE = Path(__file__).resolve().parent.parent.parent
PROC = BASE / "data" / "processed"
MODEL_DIR = BASE / "model_store"
MODEL_DIR.mkdir(exist_ok=True)

def train(category):
    df = pd.read_csv(PROC / 'monthly_by_category.csv', parse_dates=['ds'])
    series = df[df['category']==category][['ds','y_revenue']].rename(columns={'y_revenue':'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(series)
    joblib.dump(m, MODEL_DIR / f'prophet_{category}.pkl')
    print('Saved model for', category)

if __name__ == '__main__':
    df = pd.read_csv(PROC / 'monthly_by_category.csv', parse_dates=['ds'])
    cats = df['category'].unique()[:]
    for c in cats:
        train(c)
