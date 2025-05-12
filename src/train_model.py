import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os


processed_dir = '../data/processed'


dataframes = [pd.read_csv(os.path.join(processed_dir, f)) for f in os.listdir(processed_dir) if f.endswith('.csv')]

df = pd.concat(dataframes, ignore_index=True).fillna(False)

if 'price' not in df.columns:
    raise ValueError("Column 'price' is missing in the processed data.")

y = df['price']
X = df.drop(columns=['price'])

X = pd.get_dummies(X)

X, y = X.align(y, join='inner', axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

model_path = '../models/linear_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
