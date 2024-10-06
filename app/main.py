
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
import plotly.graph_objects as go
from sqlalchemy import create_engine
import json
from io import StringIO

# Конфигурация подключения к PostgreSQL
DB_CONFIG = {
    'user': 'anomaly_user',
    'password': 'password123',
    'host': 'postgres_container',
    'port': '5432',
    'database': 'anomaly_db'
}

def get_db_engine(config):
    uri = f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    engine = create_engine(uri)
    return engine

# Загрузка данных
@st.cache
def load_data(file):
    df = pd.read_csv(file, parse_dates=['timestamp'])
    df = df.set_index('timestamp').sort_index()
    df = df.resample('1T').mean()
    df = df.fillna(method='ffill')
    return df

# Предобработка данных
def preprocess_data(df, metrics):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[metrics] = scaler.fit_transform(df[metrics])
    return df_scaled

# Обнаружение аномалий с помощью Isolation Forest
def detect_anomalies_iforest(df, metrics, contamination=0.01):
    features = df[metrics]
    model = IForest(contamination=contamination, random_state=42)
    model.fit(features)
    labels = model.predict(features)  # 0 - нормальное, 1 - аномалия
    df['anomaly_iforest'] = labels
    return df, model

# Визуализация с аномалиями
def plot_anomalies(df, metric):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[metric], mode='lines', name='Нормальные данные'))
    anomalies = df[df['anomaly_iforest'] == 1]
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[metric], mode='markers', name='Аномалии', marker=dict(color='red', size=5)))
    fig.update_layout(title=f'Обнаружение аномалий с помощью Isolation Forest для {metric}',
                      xaxis_title='Время',
                      yaxis_title=metric)
    st.plotly_chart(fig)

# Генерация отчета
def generate_anomaly_report(anomalies, metric):
    report = anomalies[[metric, 'anomaly_iforest']].copy()
    report = report.rename(columns={'anomaly_iforest': 'Anomaly'})
    return report

# Основная функция приложения
def main():
    st.title("Сервис обнаружения аномалий в ИТ-системах")

    menu = ["Загрузка данных", "Анализ данных", "Обнаружение аномалий", "Отчеты"]
    choice = st.sidebar.selectbox("Меню", menu)

    if choice == "Загрузка данных":
        st.subheader("Загрузка данных телеметрии")
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными телеметрии", type=['csv'])
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.write("Данные успешно загружены:")
            st.write(data.head())
            # Показать график
            st.line_chart(data)
            # Сохранение в PostgreSQL
            if st.button("Сохранить в базу данных"):
                engine = get_db_engine(DB_CONFIG)
                data.to_sql('telemetry', engine, if_exists='replace')
                st.success("Данные сохранены в PostgreSQL")

    elif choice == "Анализ данных":
        st.subheader("Анализ временных рядов")
        engine = get_db_engine(DB_CONFIG)
        query = "SELECT * FROM telemetry"
        data = pd.read_sql(query, engine, parse_dates=['timestamp'], index_col='timestamp')
        st.write("Данные:")
        st.write(data.head())

        metrics = st.multiselect("Выберите метрики для анализа", data.columns.tolist())
        if metrics:
            for metric in metrics:
                st.write(f"**{metric}**")
                fig, ax = plt.subplots(figsize=(12,4))
                sns.lineplot(x=data.index, y=data[metric], ax=ax)
                plt.title(f'Time Series of {metric}')
                plt.xlabel('Timestamp')
                plt.ylabel(metric)
                st.pyplot(fig)
                st.write(data[metric].describe())

    elif choice == "Обнаружение аномалий":
        st.subheader("Обнаружение аномалий в данных")
        engine = get_db_engine(DB_CONFIG)
        query = "SELECT * FROM telemetry"
        data = pd.read_sql(query, engine, parse_dates=['timestamp'], index_col='timestamp')
        st.write("Данные:")
        st.write(data.head())

        metrics = st.multiselect("Выберите метрики для обнаружения аномалий", data.columns.tolist())
        contamination = st.slider("Предполагаемая доля аномалий (contamination)", 0.0, 0.1, 0.01, 0.001)

        if metrics:
            df_scaled = preprocess_data(data, metrics)
            df_anomaly, model = detect_anomalies_iforest(df_scaled, metrics, contamination=contamination)
            df_anomaly_original = data.loc[df_anomaly.index]
            df_anomaly_original['anomaly_iforest'] = df_anomaly['anomaly_iforest']
            st.write("Обнаруженные аномалии:")
            st.write(df_anomaly_original[df_anomaly_original['anomaly_iforest'] == 1])

            # Визуализация
            for metric in metrics:
                plot_anomalies(df_anomaly_original, metric)

            # Сохранение модели и аномалий в базу данных или файлах
            if st.button("Сохранить результаты"):
                anomalies = df_anomaly_original[df_anomaly_original['anomaly_iforest'] == 1]
                anomalies.to_sql('anomalies', engine, if_exists='replace')
                st.success("Результаты аномалий сохранены в PostgreSQL")

    elif choice == "Отчеты":
        st.subheader("Просмотр и загрузка отчетов о аномалиях")
        engine = get_db_engine(DB_CONFIG)
        query = "SELECT * FROM anomalies"
        anomalies = pd.read_sql(query, engine, parse_dates=['timestamp'], index_col='timestamp')
        st.write("Обнаруженные аномалии:")
        st.write(anomalies)

        if not anomalies.empty:
            if st.button("Скачать отчет"):
                csv = anomalies.to_csv(index=True)
                st.download_button(label="Скачать CSV", data=csv, file_name='anomaly_report.csv', mime='text/csv')
        else:
            st.info("Нет доступных отчетов для скачивания.")

if __name__ == '__main__':
    main()