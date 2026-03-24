from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = [
    'LST',
    'NDVI',
    'NDBI',
    'Population_Density',
    'Building_Density',
    'Road_Density',
    'Humidity',
    'Wind_Speed',
    'Rainfall',
]


def generate_synthetic_data(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    lst = np.clip(rng.normal(35, 5, n_rows), 20, 52)
    ndvi = np.clip(rng.beta(2.2, 3.5, n_rows), 0, 1)
    ndbi = np.clip(rng.beta(3.0, 2.0, n_rows), 0, 1)
    pop_density = np.clip(rng.normal(9000, 4000, n_rows), 500, 25000)
    building_density = np.clip(rng.normal(0.45, 0.2, n_rows), 0.05, 0.95)
    road_density = np.clip(rng.normal(7.0, 2.5, n_rows), 1.0, 18.0)
    humidity = np.clip(rng.normal(58, 16, n_rows), 15, 100)
    wind_speed = np.clip(rng.normal(10, 4, n_rows), 0.5, 30)
    rainfall = np.clip(rng.gamma(2.2, 6.0, n_rows), 0, 120)

    heat_score = (
        0.42 * (lst - 20) / (52 - 20)
        + 0.18 * ndbi
        + 0.12 * (pop_density / 25000)
        + 0.12 * building_density
        + 0.08 * (road_density / 18)
        - 0.18 * ndvi
        - 0.05 * (wind_speed / 30)
        - 0.07 * (rainfall / 120)
        - 0.04 * (humidity / 100)
        + rng.normal(0, 0.03, n_rows)
    )

    q1, q2 = np.quantile(heat_score, [0.40, 0.75])
    heat_risk_level = np.where(heat_score < q1, 0, np.where(heat_score < q2, 1, 2))

    return pd.DataFrame(
        {
            'LST': lst,
            'NDVI': ndvi,
            'NDBI': ndbi,
            'Population_Density': pop_density,
            'Building_Density': building_density,
            'Road_Density': road_density,
            'Humidity': humidity,
            'Wind_Speed': wind_speed,
            'Rainfall': rainfall,
            'Heat_Risk_Level': heat_risk_level,
        }
    )


def train_and_save_model(model_path: Path, dataset_path: Path, n_rows: int = 1000) -> RandomForestClassifier:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data(n_rows=n_rows)
    x_train = df[FEATURE_COLUMNS]
    y_train = df['Heat_Risk_Level']

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(x_train, y_train)

    df.to_csv(dataset_path, index=False)
    joblib.dump(model, model_path)
    return model
