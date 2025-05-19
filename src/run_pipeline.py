import argparse
import os
import datetime
import pandas as pd
import joblib
import logging
from cianparser import CianParser
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# Настройка логирования
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def collect_data(output_dir: str) -> str:
    """
    Collects real estate data from cian.ru and saves it as CSV.

    Args:
        output_dir (str): Directory to save the raw data.

    Returns:
        str: Path to the saved raw CSV file.
    """
    parser = CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = os.path.join(output_dir, f"raw_{t}.csv")
    data = parser.get_flats(
        deal_type="sale",
        rooms=(1, 2, 3),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 50,
            "object_type": "secondary",
            "min_price": 1000000,
            "max_price": 25000000,
        }
    )
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logging.info(f"Raw data saved to {csv_path}")
    return csv_path


def preprocess_data(input_csv: str, output_dir: str) -> str:
    """
    Preprocesses the raw data CSV.

    Args:
        input_csv (str): Path to the raw data file.
        output_dir (str): Directory to save the processed data.

    Returns:
        str: Path to the processed CSV file.
    """
    df = pd.read_csv(input_csv)
    df.dropna(inplace=True)
    df['url_id'] = df.index

    # Добавляем признаки первого и последнего этажа
    df['first_floor'] = df['floor'] == 1
    df['last_floor'] = df['floor'] == df['floors_count']

    columns_to_drop = [
        'author', 'commissions', 'author_type', 'url', 'location',
        'house_number', 'street', 'residential_complex', 'ID',
        'price_per_month', 'comissions', 'accommodation_type',
        'deal_type', 'underground', 'district'
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    df = df.dropna(subset=['price', 'total_meters'])
    df = df[(df['price'] > 100_000) & (df['price'] < 100_000_000)]
    df = df[(df['total_meters'] > 10) & (df['total_meters'] < 100)]

    processed_path = os.path.join(output_dir, f"processed_{os.path.basename(input_csv)}")
    df.to_csv(processed_path, index=False)
    logging.info(f"Processed data saved to {processed_path}")
    logging.info(f"Sample data:\n{df.head()}")
    return processed_path


def train_and_evaluate(csv_path: str, model_name: str, test_size: float) -> None:
    """
    Trains a model and evaluates it using test data sorted by url_id.

    Args:
        csv_path (str): Path to processed data.
        model_name (str): Name for the saved model file.
        test_size (float): Fraction of data to be used for testing.
    """
    df = pd.read_csv(csv_path)
    if 'price' not in df.columns:
        raise ValueError("Missing 'price' column in data.")
    if 'url_id' not in df.columns:
        raise ValueError("Missing 'url_id' column to sort data.")

    df.sort_values(by='url_id', inplace=True)  # Use url_id to sort chronologically

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

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', model_name)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


def main():
    """
    Main function to execute full pipeline via CLI.
    """
    parser = argparse.ArgumentParser(description="Run ML pipeline.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to save (e.g., model.pkl)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size (0.1 - 0.5)")

    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    logging.info("Pipeline started.")
    raw_csv = collect_data("data/raw")
    processed_csv = preprocess_data(raw_csv, "data/processed")
    train_and_evaluate(processed_csv, args.model_name, args.test_size)
    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()
