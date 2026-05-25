"""Train and save a Random Forest model (quick sample) for the Streamlit app.

Saves: ../models/random_forest_arrdelay.joblib

Run from the app directory:
    python train_random_forest.py
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import save_model helper if available
try:
    from model_utils import save_model
except Exception:
    # If running from workspace root, try to import from app package
    sys.path.append(str(Path(__file__).resolve().parent))
    from model_utils import save_model


def main():
    data_path = Path('../data/processed_flight_data.csv')
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    features = [
        'DepDelay', 'CRSElapsedTime', 'Distance', 'DepHour',
        'IsWeekend', 'IsRushHour', 'Origin_Freq', 'Dest_Freq',
        'Month', 'DayOfWeek'
    ]
    target = 'ArrDelay'

    print('Loading data (columns subset) ...')
    df = pd.read_csv(data_path, usecols=[*features, target])
    df = df.dropna()
    n_rows = len(df)
    print(f'Total rows available: {n_rows}')

    sample_size = min(50000, n_rows)
    if n_rows > sample_size:
        print(f'Sampling {sample_size} rows for faster training...')
        df = df.sample(n=sample_size, random_state=42)

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print('Training RandomForestRegressor (n_estimators=100) ...')
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print('Evaluating model...')
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse:.3f}, R2: {r2:.4f}')

    print('Saving model to ../models/random_forest_arrdelay.joblib')
    model_dir = Path('../models')
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(rf, feature_names=features, scaler=None, model_name='random_forest_arrdelay', model_dir=model_dir)

    print('Done.')


if __name__ == '__main__':
    main()
