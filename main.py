# Google Colab
!pip install -q yfinance catboost optuna \

# ЗАДАНИЕ 1: ЗАГРУЗКА, ОЧИСТКА И ВИЗУАЛИЗАЦИЯ ДАННЫХ

import os, glob, warnings, datetime, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

crypto_tk = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
stock_tk  = ['MSFT', 'GOOGL', 'META']
tickers = crypto_tk + stock_tk
START, END = "2024-01-01", "2025-01-01"

# Загрузка данных
data = yf.download(tickers, start=START, end=END, interval="1h", group_by='ticker', progress=False)
data.ffill(inplace=True)
data.bfill(inplace=True)

# Очистка выбросов с логированием
def clean_outliers(series, window=168):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z = (series - rolling_mean) / rolling_std
    outliers = np.abs(z) > 3
    print(f"Обработано выбросов: {outliers.sum()} из {len(series)} значений")
    series[outliers] = np.nan
    return series.ffill().bfill()

for tk in tickers:
    if tk in data.columns.levels[0]:
        series = data[(tk, 'Close')]
        data[(tk, 'Close')] = clean_outliers(series)

# Сохранение данных
os.makedirs("data", exist_ok=True)
data.to_csv("data/trading_data.csv")
print("Данные сохранены в папке /data")

# Визуализация нормализованных графиков
def plot_scaled_data(ticker_list, title):
    plt.figure(figsize=(14, 5))
    for tk in ticker_list:
        if tk not in data.columns.levels[0]: continue
        close = data.xs(tk, axis=1, level=0)['Close']
        norm = MinMaxScaler().fit_transform(close.values.reshape(-1, 1)).ravel()
        plt.plot(close.index, norm, label=tk)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

plot_scaled_data(stock_tk, "Нормализованные акции")
plot_scaled_data(crypto_tk, "Нормализованные криптовалюты")