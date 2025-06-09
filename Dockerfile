# Используем официальный Python образ
FROM python:3.10-slim

# Установка зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Копируем весь проект
COPY . .

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Команда запуска через Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "src.app:app"]
