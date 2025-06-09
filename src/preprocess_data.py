import pandas as pd
import os
import yaml
import logging

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df.dropna(inplace=True)
    df['url_id'] = df.index
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

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"Preprocessed data saved to {output_csv}")
    logging.info(f"Sample:\n{df.head()}")

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    preprocess(params["input_csv"], params["processed_csv"])
