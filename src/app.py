# app.py (Backend entry point)
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import ssl
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import shutil
import logging

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///realestate.db'
db = SQLAlchemy(app)

# Logging for security monitoring
logging.basicConfig(filename='security.log', level=logging.INFO)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database models
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), unique=True, nullable=False)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
with app.app_context():
    db.create_all()

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "User already exists"}), 400
    hashed_pw = generate_password_hash(data['password'], method='sha256')
    new_user = User(username=data['username'], password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        login_user(user)
        logging.info(f"User {user.username} logged in.")
        return jsonify({"message": "Logged in"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/logout')
@login_required
def logout():
    logging.info(f"User {current_user.username} logged out.")
    logout_user()
    return jsonify({"message": "Logged out"})

@app.route('/upload', methods=['POST'])
@login_required
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

    return jsonify({"message": "File uploaded successfully", "filename": filename})

@app.route('/preprocess/<filename>', methods=['POST'])
@login_required
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
        return jsonify({"message": "Data preprocessed", "shape": df.shape})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train/<filename>', methods=['POST'])
@login_required
def train_model(filename):
    return jsonify({"message": f"Training skipped (stub). File: {filename}", "mse": None, "status": "stub"})

@app.route('/predict/<filename>', methods=['POST'])
@login_required
def predict(filename):
    return jsonify({"message": "Prediction skipped (stub)", "status": "stub"})

@app.route('/files/<filename>', methods=['GET'])
@login_required
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/models', methods=['GET'])
@login_required
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
            }
        }
    }
    return jsonify(spec)

# Резервное копирование (простой cron-like вызов)
@app.route('/backup', methods=['POST'])
@login_required
def backup():
    try:
        shutil.make_archive("backup", 'zip', root_dir='uploads')
        return jsonify({"message": "Backup created"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # SSL context (можно использовать self-signed certs для dev)
    ssl_context = ('cert.pem', 'key.pem') if os.path.exists('cert.pem') and os.path.exists('key.pem') else None
    app.run(debug=True, ssl_context=ssl_context)
