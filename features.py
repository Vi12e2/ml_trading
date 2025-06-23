import os
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
    
from config import (
    crypto_tk, stock_tk, index_tk,
    START_H, END_H, START_D,
    RAW_DIR, DATA_LAKE_DIR,
    logger
)
from ta.volatility import BollingerBands, AverageTrueRange

# Загрузка данных с yahoo finance
def yf_chunked(tickers: list[str], start: str, end: str,
               interval: str = "1h", step_days: int = 700) -> pd.DataFrame:

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    cur = start_dt
    chunks = []

    while cur < end_dt:
        nxt = min(cur + pd.Timedelta(days=step_days), end_dt)
        logger.info("YF %s: %s - %s", interval, cur.date(), nxt.date())
        df = yf.download(
            tickers=" ".join(tickers),
            start = cur.strftime("%Y-%m-%d"),
            end   = (nxt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval = interval,
            group_by = "ticker",
            auto_adjust=True,
            progress=False
        )
        if df is None or df.empty:
            raise RuntimeError(f"Yahoo вернул пустой DataFrame при загрузке {cur.date()}–{nxt.date()}")
        chunks.append(df)
        cur = nxt

    df_all = pd.concat(chunks).sort_index()
    # убираем дублированные индексы
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    # убираем tz-метку
    if df_all.index.tz is not None:
        df_all.index = df_all.index.tz_localize(None)
    return df_all

def make_multi(df_raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:

    # DataFrame в MultiIndex
    frames = []
    for tk in tickers:
        if tk in df_raw.columns.get_level_values(0):
            df_tk = df_raw[tk].copy()
            if "Adj Close" in df_tk.columns:
                df_tk.drop(columns=["Adj Close"], inplace=True)
            df_tk.columns = pd.MultiIndex.from_product([[tk], df_tk.columns])
            frames.append(df_tk.astype(np.float32))
        else:
            logger.warning("%s: отсутствует в загруженных данных", tk)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()

# Загрузка часовых данных
all_tickers = crypto_tk + stock_tk + index_tk

logger.info("Загрузка часовых данных")
par_h = yf_chunked(all_tickers, START_H, END_H, interval="1h", step_days=400)
hourly = make_multi(par_h, all_tickers)

# Загрузка дневных данных
logger.info("Загрузка дневных данных")
par_d = yf.download(
    tickers=" ".join(all_tickers),
    start=START_D,
    end=END_H,
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    progress=False
)
# MultiIndex
par_d.index = par_d.index.tz_localize(None)
daily = make_multi(par_d, all_tickers)

# Индексы для в FE
index_df = pd.concat({
    # "^GSPC": hourly.xs("^GSPC", axis=1, level=0)[["Close", "Volume"]],
    "^IXIC": hourly.xs("^IXIC", axis=1, level=0)[["Close", "Volume"]],
}, axis=1)

# Сохранение в Parquet/CSV, обработка выбросов
class Par_:
    def __init__(
        self,
        crypto: list[str],
        stocks: list[str],
        indexes: list[str],
        start: str,
        end:   str,
        local_dir:    str = RAW_DIR,
        parquet_name: str = "trading.parquet",
        csv_name:     str = "trading_data.csv"
    ):
        # Сохранение тикеров в том же порядке
        self.tickers = crypto + stocks + indexes
        self.start   = start
        self.end     = end
        self.local_dir    = local_dir
        self.parquet_path = os.path.join(local_dir, parquet_name)
        self.csv_path     = os.path.join(local_dir, csv_name)

    @staticmethod
    def _zscore(s: pd.Series, window: int = 65, thr: float = 3.0) -> pd.Series:
        # выбросы по z-score
        c = s.copy()
        # if len(c.dropna()) < window:
        #     return c.ffill()
        mu = c.rolling(window, min_periods=1).mean()
        sd = c.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-6)
        mask = ((c - mu).abs() / sd) > thr
        c[mask] = np.nan
        return c.ffill().fillna(mu)

    def load(self) -> pd.DataFrame:

        os.makedirs(self.local_dir, exist_ok=True)
        if os.path.exists(self.parquet_path):
            df_par = pd.read_parquet(self.parquet_path, engine="pyarrow")
            logger.info("Parquet загружен из кеша: %s", self.parquet_path)
        else:
            try:
                logger.info("Parquet не найден, используем часовые данные из памяти")
                df_par = hourly.copy()
                if df_par.empty:
                    logger.info("hourly пустой, скачиваем через yf_chunked")
                    df_par = yf_chunked(self.tickers, self.start, self.end, interval="1h")
            except Exception as e:
                logger.warning("Ошибка при чтении hourly: %s. Попытка загрузки yf.download", e)
                df_par = yf.download(
                    tickers=self.tickers,
                    start=self.start,
                    end=self.end,
                    interval="1h",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False
                )

            df_par.to_parquet(self.parquet_path, engine="pyarrow")
            logger.info("Parquet сохранён: %s", self.parquet_path)

        assert isinstance(df_par.columns, pd.MultiIndex), "Ожидается MultiIndex в parquet"
        frames = []
        for tk in self.tickers:
            if tk not in df_par.columns.get_level_values(0):
                logger.warning("%s: нет данных в parquet", tk)
                continue
            df_tk = df_par[tk].copy()
            # убираем "Adj Close", если есть
            if "Adj Close" in df_tk.columns:
                df_tk.drop(columns=["Adj Close"], inplace=True)
            # применяем z-score к Close
            df_tk["Close"] = Par_._zscore(df_tk["Close"], window=65, thr=3.0)

            df_tk.columns = pd.MultiIndex.from_product([[tk], df_tk.columns])
            frames.append(df_tk)

        if not frames:
            raise RuntimeError("Не удалось загрузить данные ни одного тикера")
        df = pd.concat(frames, axis=1)
        df.index.name = "Date"

        # Сохраняем в CSV если его ещё нет
        if not os.path.exists(self.csv_path):
            df.to_csv(self.csv_path)
            logger.info("CSV сохранён: %s", self.csv_path)

        return df

    @staticmethod
    def outlier_analysis(df: pd.DataFrame, ticker: str, window: int = 65, thr: float = 3.0):

        s = df.xs(ticker, axis=1, level=0)["Close"].dropna()
        if len(s) < window:
            logger.warning("%s: недостаточно данных для анализа выбросов", ticker)
            return
        cleaned = Par_._zscore(s, window, thr)
        mask    = s != cleaned
        if not mask.any():
            logger.info("%s: выбросов не найдено", ticker)
            return
        fig = go.Figure([
            go.Scatter(x=s.index, y=s, name=ticker, mode="lines"),
            go.Scatter(x=s[mask].index, y=s[mask],
                       name="Выбросы", mode="markers",
                       marker=dict(color="red", size=8))
        ])
        fig.update_layout(
            title=f"{ticker}: выбросы (z>{thr}, window={window})",
            hovermode="x unified"
        )
        fig.show()
        total, cnt = len(s), int(mask.sum())
        logger.info(
            "%s: всего баров %d, выбросов %d (%.2f%%)",
            ticker, total, cnt, cnt / total * 100
        )

# Визуализация нормализованных графиков и анализ выбросов
def plot_norm(tickers: list[str], title: str):

    fig = go.Figure()
    for tk in tickers:
        try:
            s = raw.xs(tk, axis=1, level=0)["Close"].dropna()
            if s.empty:
                logger.info("%s: нет данных для нормализации", tk)
                continue
            fig.add_trace(go.Scatter(
                x=s.index, y=s / s.iloc[0],
                mode="lines", name=tk
            ))
        except Exception as e:
            logger.warning("Ошибка при нормализации %s: %s", tk, e)
    fig.update_layout(title=title, hovermode="x unified")
    fig.show()

# plot_norm(stock_tk,  "Нормализованные акции")
# plot_norm(crypto_tk, "Нормализованные криптовалюты")
# plot_norm(index_tk,  "Нормализованные индексы")

def plot_outliers(raw_df: pd.DataFrame, tickers: list[str], window: int = 65, thr: float = 3.0):
    for tk in tickers:
        Par_.outlier_analysis(raw_df, tk, window, thr)
        
# Признаки
class FE:

    def __init__(self, tickers, index_df, lake=DATA_LAKE_DIR, base_win=15, thr=3, vol_window=35, min_win=5, max_win=35):
        self.tickers    = tickers
        self.index_df   = index_df
        # self.daily_full = daily_df.copy()
        # self.prev_daily = self.daily_full.shift(1)  # сдвиг на 1 день назад
        self.lake       = lake
        self.base_win   = base_win
        self.thr        = thr
        self.vol_window = vol_window
        self.min_win    = min_win
        self.max_win    = max_win
        self.logger     = logging.getLogger(self.__class__.__name__)
        os.makedirs(self.lake, exist_ok=True)

    @staticmethod
    def _zscore(s: pd.Series, window: int, thr: float) -> pd.Series:
        # выбросы и заполнение
        c = s.copy()
        mu = c.rolling(window, min_periods=1).mean()
        sd = c.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-6)
        mask = ((c - mu).abs() / sd) > thr
        c[mask] = np.nan
        return c.ffill().fillna(mu)

    @staticmethod
    def _rm_out(df: pd.DataFrame, win: int, thr: float) -> pd.DataFrame:
        res = df.copy()
        for col in res.columns:
            mu = res[col].rolling(win, min_periods=1).mean()
            sd = res[col].rolling(win, min_periods=1).std().replace(0, np.nan).fillna(1e-6)
            mask = ((res[col] - mu).abs() / sd) > thr
            res.loc[mask, col] = np.nan
        return res

    def _make(self, df: pd.DataFrame) -> pd.DataFrame:

        tk = df.name
        # если данных меньше, чем базовое окно, признаков не создаётся
        if len(df) < self.base_win:
            return pd.DataFrame()
        # DataFrame для признаков с теми же метками дат
        f = pd.DataFrame(index=df.index)
        # Adjusted Close
        if "Adj Close" in df.columns:
            df = df.drop(columns=["Adj Close"])
        # OHLCV
        f['open'] = df['Open']
        f['high'] = df['High']
        f['low'] = df['Low']
        f['close'] = df['Close']
        f['volume'] = df['Volume']
        # SMA
        for w in (7, 35, 65, 120):
            f[f'sma{w}'] = df['Close'].rolling(w, min_periods=1).mean()
        # Истинный диапазон (True Range) для ATR
        tr = pd.concat([
            (df['High'] - df['Low']).abs(),
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low'] - df['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        for e in (15, 65):
            f[f'atr{e}'] = tr.rolling(e, min_periods=1).mean()
        # Bollinger Bands
        bb = BollingerBands(df['Close'], window=90, window_dev=2)
        f['bb_up'] = bb.bollinger_hband()
        f['bb_low'] = bb.bollinger_lband()
        f['bb_w'] = f['bb_up'] - f['bb_low']
        # дополнительные данные по биржевым индексам, здесь Nasdaq
        for ix in self.index_df.columns.get_level_values(0).unique():
            c_ix = self.index_df[ix,'Close'].reindex(df.index).ffill().fillna(self.index_df[ix,'Close'].median())
            v_ix = self.index_df[ix,'Volume'].reindex(df.index).ffill().fillna(self.index_df[ix,'Volume'].median())
            f[f'{ix}_close'] = c_ix
            f[f'{ix}_vol'] = v_ix
        # заполнение NaN ffill и медианой
        f = f.ffill().fillna(f.median())
        # скользящее ATR для динамического окна очистки выбросов
        atr_dyn = AverageTrueRange(df['High'], df['Low'], df['Close'], self.vol_window).average_true_range()
        atr_med = atr_dyn.median() or 1e-6
        dyn_win = int(np.clip((atr_dyn/atr_med).median() * self.base_win, self.min_win, self.max_win))
        # выбросы по окну динамическому окну dyn_win и заполнение ffill и медианой
        feats = FE._rm_out(f, dyn_win, self.thr).ffill().fillna(f.median())
        # масштабирование и замена бесконечностей на 0
        med = feats.median().replace(np.nan, 0)
        iqr = (feats.quantile(0.75) - feats.quantile(0.25)).replace(0,1)
        feats = ((feats - med) / iqr).replace([np.inf,-np.inf],0).astype(np.float32)
        return feats

    def _process_ticker(self, tk: str, mi: pd.DataFrame):
        df_tk = mi.xs(tk, axis=1, level=0).dropna()
        if df_tk.empty:
            self.logger.warning(f"{tk}: нет данных для FE")
            return
        df_tk.name = tk
        feats = self._make(df_tk)
        if feats.empty:
            self.logger.info(f"{tk}: мало баров (<{self.base_win}) для признаков")
            return
        path = os.path.join(self.lake, tk)
        os.makedirs(path, exist_ok=True)
        feats.to_parquet(os.path.join(path, "features.parquet"))

    def run(self, mi):
        os.makedirs(self.lake, exist_ok=True)
        for tk in self.tickers:
            self._process_ticker(tk, mi)

    def load_features_for_ml(self, ticker):
        fn = os.path.join(self.lake, ticker, "features.parquet")
        if os.path.exists(fn):
            df = pd.read_parquet(fn)
            df.index.name = "ds"
            df.name = ticker
            return df
        else:
            self.logger.warning(f"features.parquet не найден для {ticker}")
            return pd.DataFrame()

    @staticmethod
    def create_ml_target(df, horizon=H):
        df = df.copy()
        df['future'] = df['close'].shift(-horizon)
        df['target'] = (df['future'] > df['close']).astype(int)
        return df.dropna(subset=['future', 'target'])


def plot_feature_file_counts(lake_dir: str, tickers: list[str]):
# Визуализация количества накопленных файлов
    counts = {
        tk: len(os.listdir(os.path.join(DATA_LAKE_DIR, tk)))
        for tk in crypto_tk + stock_tk
    }
    fig = px.bar(
        x=list(counts.values()),
        y=list(counts.keys()),
        orientation="h",
        title="Количество накопленных файлов (features.parquet) по тикерам",
        labels={"x": "Файлов", "y": "Тикер"},
        width=600, height=400
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    fig.show()

par_obj = Par_(crypto_tk, stock_tk, index_tk, START_H, END_H)
raw = par_obj.load()
fe = FE(
    tickers=crypto_tk + stock_tk,
    index_df=index_df,
    # daily_df=daily,
    lake=DATA_LAKE_DIR,
    base_win=15,
    thr=3.0,
    vol_window=35,
    min_win=5,
    max_win=35
)
fe.run(hourly)





