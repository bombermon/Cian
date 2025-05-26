import pandas as pd
import numpy as np
import os
import joblib
import yaml
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_and_evaluate(csv_path, model_name, test_size):
    df = pd.read_csv(csv_path)
    if 'price' not in df.columns:
        raise ValueError("Missing 'price' column in data.")
    if 'url_id' not in df.columns:
        raise ValueError("Missing 'url_id' column.")

    df.sort_values(by='url_id', inplace=True)

    y = df['price']
    X = df.drop(columns=['price'])
    X = pd.get_dummies(X)
    X, y = X.align(y, join='inner', axis=0)

    split_index = int((1 - test_size) * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    logging.info(f"Model: {model_name}")
    logging.info(f"MSE: {mse:.2f}")
    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"R2: {r2:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"Model coefficients: {model.coef_}")
    logging.info(f"Sample predictions: {y_pred[:5]}")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_name)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train_and_evaluate(
        csv_path=params["processed_csv"],
        model_name=params["model_name"],
        test_size=params["test_size"]
    )
