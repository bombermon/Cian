# app.py (Backend entry point)
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import logging
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
LOG_FOLDER = 'logs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///realestate.db'
db = SQLAlchemy(app)

# Logging configuration
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, 'server.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

@app.before_request
def log_request():
    logging.info(f"Request: {request.method} {request.path} | IP: {request.remote_addr}")

# Database model
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), unique=True, nullable=False)

# Initialize database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real Estate Estimator</title>
    </head>
    <body>
        <h1>Heuristic Price Estimation</h1>
        <form id="heuristicForm">
            <label for="area">Enter area (m²):</label>
            <input type="number" id="area" name="area" required>
            <button type="submit">Estimate</button>
        </form>
        <p id="result"></p>
        <script>
            document.getElementById('heuristicForm').addEventListener('submit', function(event) {
                event.preventDefault();
                const area = document.getElementById('area').value;
                fetch('/heuristic', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ area: area })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').textContent =
                        data.heuristic_price ? `Estimated Price: ${data.heuristic_price.toLocaleString()} RUB` : `Error: ${data.error}`;
                });
            });
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Save to DB
    if not Dataset.query.filter_by(filename=filename).first():
        new_entry = Dataset(filename=filename)
        db.session.add(new_entry)
        db.session.commit()

    logging.info(f"File uploaded: {filename}")
    return jsonify({"message": "File uploaded successfully", "filename": filename})

@app.route('/preprocess/<filename>', methods=['POST'])
def preprocess_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        df.dropna(inplace=True)
        df = pd.get_dummies(df)
        df.to_csv(filepath, index=False)
        logging.info(f"Preprocessed file: {filename}, shape: {df.shape}")
        return jsonify({"message": "Data preprocessed", "shape": df.shape})
    except Exception as e:
        logging.error(f"Preprocessing error for {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train/<filename>', methods=['POST'])
def train_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(filepath)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        model_path = os.path.join(MODEL_FOLDER, filename.replace('.csv', '_model.pkl'))
        joblib.dump(model, model_path)
        logging.info(f"Model trained for {filename}, MSE: {mse:.2f}")
        return jsonify({"message": "Model trained", "mse": mse})
    except Exception as e:
        logging.error(f"Training error for {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<filename>', methods=['POST'])
def predict(filename):
    model_path = os.path.join(MODEL_FOLDER, filename.replace('.csv', '_model.pkl'))
    try:
        model = joblib.load(model_path)
        input_data = request.get_json()
        df = pd.DataFrame([input_data])
        preds = model.predict(df)
        logging.info(f"Prediction made for {filename}: {preds.tolist()}")
        return jsonify({"prediction": preds.tolist()})
    except Exception as e:
        logging.error(f"Prediction error for {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/heuristic', methods=['POST'])
def heuristic_price():
    data = request.get_json()
    area = data.get("area")
    try:
        price = float(area) * 300000
        logging.info(f"Heuristic prediction: area={area}, price={price}")
        return jsonify({"heuristic_price": price})
    except Exception as e:
        logging.error(f"Heuristic error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/files/<filename>', methods=['GET'])
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/models', methods=['GET'])
def list_models():
    models = os.listdir(MODEL_FOLDER)
    return jsonify({"models": models})

# API documentation endpoint
@app.route('/docs', methods=['GET'])
def docs():
    return redirect("https://petstore.swagger.io/?url=http://localhost:5000/openapi.json", code=302)

@app.route('/openapi.json')
def openapi():
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Real Estate ML API",
            "version": "1.0.0"
        },
        "paths": {
            "/upload": {
                "post": {
                    "summary": "Upload a CSV or Excel file",
                    "responses": {"200": {"description": "File uploaded"}}
                }
            },
            "/preprocess/{filename}": {
                "post": {
                    "summary": "Preprocess uploaded file",
                    "responses": {"200": {"description": "Data preprocessed"}}
                }
            },
            "/train/{filename}": {
                "post": {
                    "summary": "Train ML model",
                    "responses": {"200": {"description": "Model trained"}}
                }
            },
            "/predict/{filename}": {
                "post": {
                    "summary": "Predict from input data",
                    "responses": {"200": {"description": "Prediction result"}}
                }
            },
            "/files/{filename}": {
                "get": {
                    "summary": "Download file",
                    "responses": {"200": {"description": "File content"}}
                }
            },
            "/models": {
                "get": {
                    "summary": "List all trained models",
                    "responses": {"200": {"description": "Model list"}}
                }
            },
            "/heuristic": {
                "post": {
                    "summary": "Heuristic price estimation (300k per m^2)",
                    "responses": {"200": {"description": "Heuristic price returned"}}
                }
            }
        }
    }
    return jsonify(spec)

if __name__ == '__main__':
    app.run(debug=True)
