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

app = Flask(__name__, template_folder='templates')
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
    return render_template('index.html')

@app.route('/predict/<filename>', methods=['POST'])
def predict(filename):
    model_path = os.path.join(MODEL_FOLDER, 'linear_model.pkl')
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


@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        model_path = os.path.join(MODEL_FOLDER, 'latest_model.pkl')
        if not os.path.exists(model_path):
            return render_template('index.html', error="Модель не найдена. Пожалуйста, обучите модель.")
        model = joblib.load(model_path)
        data = {
            'floor': int(request.form['floor']),
            'floors_count': int(request.form['floors_count']),
            'rooms_count': int(request.form['rooms_count']),
            'total_meters': float(request.form['total_meters'])
        }
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

       
        millions = int(prediction) // 1_000_000
        thousands = (int(prediction) % 1_000_000) // 1000
        price_text = f"{millions} млн {thousands} тыс"

        return render_template('index.html', prediction=price_text)
    except Exception as e:
        logging.error(f"Form prediction error: {str(e)}")
        return render_template('index.html', error="Ошибка предсказания. Проверьте входные данные.")


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
            "/predict/{filename}": {
                "post": {
                    "summary": "Predict from input data",
                    "responses": {"200": {"description": "Prediction result"}}
                }
            },
            "/models": {
                "get": {
                    "summary": "List all trained models",
                    "responses": {"200": {"description": "Model list"}}
                }
            }
        }
    }
    return jsonify(spec)

if __name__ == '__main__':
    app.run(debug=True)
