# Используем официальный образ Python
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y build-essential libpq-dev

# Установка рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY . /app

# Установка зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Открытие порта Streamlit (по умолчанию 8501)
EXPOSE 8501

# Команда запуска приложения
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]