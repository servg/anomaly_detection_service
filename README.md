# anomaly_detection_service
Итоговая работа на курсе Python для продвинутых специалистов. Машинное обучение
# Запуск
docker-compose up --build

Приложение будет доступно на http://localhost:8501, 
PostgreSQL на порту 5432. 
Так же доступен PGAdmin на порту 5050

# Конфигурация подключения к PostgreSQL
DB_CONFIG = {
    'user': 'anomaly_user',
    'password': 'password123',
    'host': 'postgres_container',
    'port': '5432',
    'database': 'anomaly_db'
}

Если мы хотим использовать датасет SKAB  нужно использовать скрипт transform.ipynb
Этот датасет уже преобразован и сохранен как test2.csv

Результаты детекции не идеальны тк система универсальна и может принимать на вход данные с произвольным числом метрик, аномалии выявляются с помощью Isolation Forest

