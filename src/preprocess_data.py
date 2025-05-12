import pandas as pd
import os

raw_dir = '../data/raw'
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

for file in os.listdir(raw_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(raw_dir, file))
        df = df.dropna()

        # Добавляем столбец 'url_id', если его нет, с уникальными идентификаторами
        if 'url_id' not in df.columns:
            df['url_id'] = df.index

        columns_to_drop = ['author', 'commissions', 'author_type', 'url', 'location', 
                           'house_number', 'street', 'residential_complex', 'ID', 
                           'price_per_month', 'comissions', 'accommodation_type', 
                           'deal_type', 'underground', 'residential_complex', 'district']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        df = df.dropna(subset=['price', 'total_meters'])
        df = df[(df['price'] > 100_000) & (df['price'] < 100_000_000)]
        df = df[(df['total_meters'] > 10) & (df['total_meters'] < 100)]

        processed_file = os.path.join(processed_dir, f"processed_{file}")
        df.to_csv(processed_file, index=False)
        print(f"Processed data saved to {processed_file}")
