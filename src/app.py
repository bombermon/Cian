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
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth
from dotenv import load_dotenv
import os

# Загрузка .env
load_dotenv()

# Flask
app = Flask(__name__, template_folder='templates')
CORS(app)

# Авторизация по токену
auth = HTTPTokenAuth(scheme='Bearer')
VALID_TOKEN = os.getenv("API_TOKEN")  # добавьте API_TOKEN в .env

@auth.verify_token
def verify_token(token):
    return token == VALID_TOKEN



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

# Route for documentation 
@app.route('/docs', methods=['GET'])
def docs():
    return redirect(url_for('openapi_ui'), code=302)

# Call swagger
@app.route('/swagger')
def openapi_ui():
    return redirect(f"https://petstore.swagger.io/?url={request.url_root}openapi.json", code=302)


@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    model_name = request.args.get('model', 'XGBoost.pkl')
    model_path = os.path.join(MODEL_FOLDER, model_name)

    try:
        model = joblib.load(model_path)

        if request.is_json:
            data = request.get_json()
            features = data.get("features")
            if not features or len(features) != 4:
                raise ValueError("Неверный формат входных данных, ожидаются 4 признака в массиве features")
            
            # Создаем DataFrame с нужными колонками
            input_data = {
                'floor': features[0],
                'floors_count': features[1],
                'rooms_count': features[2],
                'total_meters': features[3]
            }
        else:
            input_data = {
                'floor': int(request.form['floor']),
                'floors_count': int(request.form['floors_count']),
                'rooms_count': int(request.form['rooms_count']),
                'total_meters': float(request.form['total_meters'])
            }

        df = pd.DataFrame([input_data])
        logging.info(f"Input data for prediction: {input_data}")

        df['first_floor'] = df['floor'] == 1
        df['last_floor'] = df['floor'] == df['floors_count']

        prediction = float(model.predict(df)[0])

        if request.is_json:
            return jsonify({"prediction": prediction})
        else:
            millions = int(prediction) // 1_000_000
            thousands = (int(prediction) % 1_000_000) // 1000
            price_text = f"{millions} млн {thousands} тыс"
            return render_template('index.html', prediction=price_text)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        else:
            return render_template('index.html', error="Ошибка предсказания. Проверьте входные данные.")



@app.route('/models', methods=['GET'])
def list_models():
    models = os.listdir(MODEL_FOLDER)
    return jsonify({"models": models})



@app.route('/openapi.json')
def openapi():
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Real Estate ML API",
            "version": "1.0.0",
            "description": "API для предсказания цены недвижимости на основе параметров квартиры.\n\n"
                           "Можно использовать POST-запросы с параметрами квартиры для получения предсказанной стоимости."
        },
        "paths": {
            "/predict/{filename}": {
                "post": {
                    "summary": "Сделать предсказание по JSON-данным",
                    "description": "Возвращает предсказанную цену квартиры на основе параметров, переданных в теле запроса.",
                    "parameters": [
                        {
                            "name": "filename",
                            "in": "path",
                            "required": True,
                            "description": "Имя файла, для которого делается предсказание (используется только в логах).",
                            "schema": {"type": "string"}
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "floor": {"type": "integer", "example": 5},
                                        "floors_count": {"type": "integer", "example": 12},
                                        "rooms_count": {"type": "integer", "example": 2},
                                        "total_meters": {"type": "number", "format": "float", "example": 56.3}
                                    },
                                    "required": ["floor", "floors_count", "rooms_count", "total_meters"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Успешный ответ с предсказанной ценой.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prediction": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "example": [7800000.0]
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Ошибка при предсказании",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "error": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models": {
                "get": {
                    "summary": "Получить список доступных моделей",
                    "description": "Возвращает список всех сохранённых файлов моделей в папке `/models`.",
                    "responses": {
                        "200": {
                            "description": "Список файлов моделей.",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "models": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "example": ["latest_model.pkl", "model_2025_05_01.pkl"]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return jsonify(spec)


if __name__ == '__main__':
    app.run(debug=True)
