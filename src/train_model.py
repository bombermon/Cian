import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np
from xgboost import XGBRegressor

# Настройки
model_names = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}
test_size = 0.2

# Загрузка данных
df = pd.read_csv("data/processed/processed_raw_2025-05-19_19-58.csv")
if 'price' not in df.columns:
    raise ValueError("Missing 'price' column in data.")
if 'url_id' not in df.columns:
    raise ValueError("Missing 'url_id' column to sort data.")
df.sort_values(by='url_id', inplace=True)

# Подготовка признаков
y = df['price']
X = df.drop(columns=['price'])
X = pd.get_dummies(X)
X, y = X.align(y, join='inner', axis=0)

# Разделение на train/test по времени
split_index = int((1 - test_size) * len(df))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Каталог моделей
os.makedirs('models', exist_ok=True)

# Обучение и оценка моделей
for name, model in model_names.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Sample predictions: {y_pred[:5]}")

    # Сохранение модели
    model_path = os.path.join('models', f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
