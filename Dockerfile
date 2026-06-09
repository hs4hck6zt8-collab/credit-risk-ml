FROM python:3.10-slim

WORKDIR /app

# Системные зависимости для LightGBM и CatBoost
RUN apt-get update && apt-get install -y\
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Открываем порт Streamlit
EXPOSE 8501

# Запуск приложения
CMD ["streamlit", "run", "app/app.py", \
    "--server.port=8501", \
    "--server.headless=true"]