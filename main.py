# -*- coding: utf-8 -*-
"""1_3
Ссылка с выходными данными (законченной оптимизацией Optuna)
https://colab.research.google.com/drive/1sM5rxaDxw5LGxsvKYi8P78nXiBSeTLgY?usp=sharing

Если не подключаться к диску, будет скачивать с yahoo
"""

import os, warnings, time, logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    force=True
)
logger = logging.getLogger("StrategyTester")

!pip -q install yfinance ta backtesting optuna plotly pyarrow watchdog swifter scikit-learn

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
from ta.momentum import RSIIndicator
from ta.trend     import SMAIndicator, MACD
from ta.volatility import AverageTrueRange
from watchdog.events    import FileSystemEventHandler
from watchdog.observers import Observer

try:
    import swifter
    apply_fn = lambda ser, fn: ser.swifter.apply(fn)
except ImportError:
    apply_fn = lambda ser, fn: ser.apply(fn)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RAW_DIR        = "data"
DATA_LAKE_DIR  = "data_lake"
DRIVE_ROOT     = "/content/drive/MyDrive/1yh"
DRIVE_RAW      = os.path.join(DRIVE_ROOT, "raw_data")
DRIVE_PARQ     = os.path.join(DRIVE_ROOT, "trading.parquet")
DRIVE_LAKE     = os.path.join(DRIVE_ROOT, "data_lake")

try:
    # Google Drive (Colab)
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

    if os.path.exists(DRIVE_PARQ):
        # чтение trading.parquet
        RAW_DIR = DRIVE_ROOT
        logger.info("Чтение из %s", DRIVE_PARQ)
    elif os.path.isdir(DRIVE_RAW):
        RAW_DIR = DRIVE_RAW
        logger.info("Чтение из %s", DRIVE_RAW)
    else:
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("Локальный RAW_DIR: %s", RAW_DIR)

    if os.path.isdir(DRIVE_LAKE):
        DATA_LAKE_DIR = DRIVE_LAKE
        logger.info("Сохранение data lake в %s", DRIVE_LAKE)
    else:
        os.makedirs(DATA_LAKE_DIR, exist_ok=True)
        logger.info("Локальный DATA_LAKE_DIR: %s", DATA_LAKE_DIR)

except Exception as e:
    # logger.warning(" %s", e)
    os.makedirs(RAW_DIR,       exist_ok=True)
    os.makedirs(DATA_LAKE_DIR, exist_ok=True)

# Задания 1–2

class Par_:
    def __init__(self, crypto, stocks, start, end,
                 local_dir=RAW_DIR,
                 parquet_name="trading.parquet",
                 csv_name="trading_data.csv"):
        self.tickers      = crypto + stocks
        self.start        = start
        self.end          = end
        self.local_dir    = local_dir
        self.parquet_path = os.path.join(local_dir, parquet_name)
        self.csv_path     = os.path.join(local_dir, csv_name)

    @staticmethod
    def _zscore(s, window=65, thr=3):
        c = s.copy()
        if len(c.dropna()) < window:
            return c.ffill().bfill()
        mu   = c.rolling(window).mean()
        sd   = c.rolling(window).std().replace(0, np.nan).ffill().bfill().fillna(1e-6)
        mask = (c - mu).abs() / sd > thr
        c[mask] = np.nan
        return c.ffill().bfill()

    def load(self):
        os.makedirs(self.local_dir, exist_ok=True)
        if os.path.exists(self.parquet_path):
            df_par = pd.read_parquet(self.parquet_path, engine="pyarrow")
            logger.info("Загружено из кеша: %s", self.parquet_path)
        else:
            logger.info("Parquet не найден, загрузка из yfinance")
            for i in (1, 2):
                try:
                    df_par = yf.download(
                        tickers     = self.tickers,
                        start       = self.start,
                        end         = self.end,
                        interval    = "1h",
                        group_by    = "ticker",
                        auto_adjust = True,
                        progress    = False
                    )
                    if not df_par.empty:
                        logger.info("Данные загружены (попытка %d)", i)
                        break
                except Exception as e:
                    logger.warning("Ошибка yfinance (попытка %d): %s", i, e)
            else:
                raise RuntimeError("Не удалось скачать данные.")
            df_par.to_parquet(self.parquet_path)
            logger.info("Parquet сохранён: %s", self.parquet_path)
        # DataFrame по тикерам
        assert isinstance(df_par.columns, pd.MultiIndex)
        frames = []
        for tk in self.tickers:
            if tk not in df_par.columns.get_level_values(0):
                continue
            df_tk = df_par[tk].drop(columns=[c for c in ("Adj Close",) if c in df_par[tk].columns])
            df_tk["Close"] = self._zscore(df_tk["Close"])
            df_tk.columns  = pd.MultiIndex.from_product([[tk], df_tk.columns])
            frames.append(df_tk)
        df = pd.concat(frames, axis=1)
        df.index.name = "Date"
        os.makedirs(self.local_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            df.to_csv(self.csv_path)
            logger.info("CSV сохранён: %s", self.csv_path)
        return df

    @staticmethod
    def outlier_analysis(df, ticker, window=65, thr=3):
        s = df.xs(ticker, axis=1, level=0)["Close"].dropna()
        if len(s) < window:
            logger.warning("%s: недостаточно данных для outlier_analysis", ticker)
            return
        cleaned = Par_._zscore(s, window, thr)
        mask    = s != cleaned
        if not mask.any():
            logger.info("%s: выбросов не найдено", ticker)
            return
        fig = go.Figure([
            go.Scatter(x=s.index, y=s, name="Цена"),
            go.Scatter(x=s[mask].index, y=s[mask],
                       mode="markers", name="Выбросы",
                       marker=dict(color="red", size=8))
        ])
        fig.update_layout(
            title=f"{ticker}: выбросы (z>{thr}, window={window})",
            hovermode="x unified"
        )
        fig.show()
        total, cnt = len(s), int(mask.sum())
        logger.info(
            "%s: всего %d, выбросов %d (%.2f%%)",
            ticker, total, cnt, cnt/total*100
        )
        logger.debug("Примеры выбросов %s:\n%s", ticker, s[mask].tail())


crypto_tk = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD"]
stock_tk  = ["GOOGL","NVDA","MSFT","AAPL"]
START, END = "2024-01-01","2025-01-01"

raw = Par_(crypto_tk, stock_tk, START, END).load()

# Нормализованные графики

def plot_norm(tickers, title):
    fig = go.Figure()
    for tk in tickers:
        s = raw.xs(tk, axis=1, level=0)["Close"].dropna()
        if s.empty:
            logger.info("%s: нет данных для нормализации", tk)
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s / s.iloc[0],
            mode="lines", name=tk
        ))
    fig.update_layout(title=title, hovermode="x unified")
    fig.show()

plot_norm(stock_tk,  "Нормализованные акции")
plot_norm(crypto_tk, "Нормализованные криптовалюты")

for tk in crypto_tk + stock_tk:
    Par_.outlier_analysis(raw, tk, window=65, thr=3)

# Метрики, стратегии с ATR-trailing-stop

def compute_metrics(cum, rets, trades):
    days      = (cum.index[-1] - cum.index[0]).days / 365.25
    total_ret = (cum.iloc[-1] - 1) * 100
    cagr      = cum.iloc[-1] ** (1 / days) - 1
    vol       = rets.std() * np.sqrt(252) * 100
    sharpe    = rets.mean()/rets.std()*np.sqrt(252) if rets.std() else np.nan
    max_dd    = ((cum/cum.cummax()) - 1).min() * 100
    down      = rets[rets<0]
    sortino   = rets.mean()/down.std()*np.sqrt(252) if len(down) and down.std() else np.nan
    pnl       = trades.dropna()
    win_rate  = pnl[pnl>0].count()/pnl.count()*100 if pnl.count() else 0
    gross_win = pnl[pnl>0].sum()
    gross_loss= -pnl[pnl<0].sum()
    pf        = gross_win/gross_loss if gross_loss else np.inf
    calmar    = (total_ret/100)/(max_dd/-100) if max_dd else np.nan
    return {
        "Total Return [%]": total_ret, "CAGR [%]": cagr*100,
        "Volatility [%]": vol,       "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,    "Max. Drawdown [%]": max_dd,
        "Calmar Ratio": calmar,      "Win Rate [%]": win_rate,
        "Profit Factor": pf,         "# Trades": pnl.count()
    }

# Базовый класс с динамическим ATR трейлинг-стопом

class _Base_ATR(Strategy):
    atr_window = 35
    dyn_window = 65


    def _init_atr(self):
        h = pd.Series(self.data.High,  index=self.data.index)
        l = pd.Series(self.data.Low,   index=self.data.index)
        c = pd.Series(self.data.Close, index=self.data.index)
        # индикатор ATR
        self.atr = self.I(
            lambda c_: AverageTrueRange(h, l, c_, self.atr_window).average_true_range(),
            c
        )
        self.entry_extreme = None

    def _apply_trail(self, price):
        if pd.isna(self.atr[-1]) or self.atr[-1] == 0 or not self.position:
            return False

        atr_s = pd.Series(self.atr)
        base = atr_s.rolling(self.dyn_window).mean().replace(0, np.nan).ffill().bfill()
        dyn_mult = np.clip(atr_s / base, 0.5, 2.5)
        level = dyn_mult.iloc[-1] * self.atr[-1]

        if self.position.is_long:
            self.entry_extreme = max(self.entry_extreme or price, price)
            if price < self.entry_extreme - level:
                self.position.close()
                return True
        else:
            self.entry_extreme = min(self.entry_extreme or price, price)
            if price > self.entry_extreme + level:
                self.position.close()
                return True
        return False

# Стратегии

class SMA_MACD_RSI(_Base_ATR):
    ss, sl, thr = 18, 39, 48
    min_hold_bars = 2

    def init(self):
        c = pd.Series(self.data.Close, index=self.data.index)
        self.sma_s = self.I(lambda s: s.rolling(self.ss).mean(), c)
        self.sma_l = self.I(lambda s: s.rolling(self.sl).mean(), c)
        macd_raw   = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        self.macd  = self.I(lambda s: pd.Series(s).ewm(span=9).mean(), macd_raw)
        self.rsi    = self.I(lambda s: RSIIndicator(s, 14).rsi(), c)

        self._init_atr()
        self.bars_since_trade = self.min_hold_bars

    def next(self):
        price = self.data.Close[-1]
        self.bars_since_trade += 1

        # 1) ATR-трейлинг
        if self._apply_trail(price):
            self.bars_since_trade = 0
            return

        # 2) холд-тайм после последней сделки
        if self.bars_since_trade < self.min_hold_bars:
            return

        # 3) сигналы
        s, l = self.sma_s[-1], self.sma_l[-1]
        diff = self.macd[-1] - self.macd[-2]
        r    = self.rsi[-1]
        long_sig  = (s > l) and (diff > 0) and (r < 100 - self.thr)
        short_sig = (s < l) and (diff < 0) and (r > self.thr)

        # 4) если нет позиции — входим
        if not self.position:
            if long_sig:
                self.buy();  self.entry_extreme = price; self.bars_since_trade = 0
            elif short_sig:
                self.sell(); self.entry_extreme = price; self.bars_since_trade = 0

        # 5) если в позиции — выходим, когда сигнал пропал
        elif self.position.is_long and not long_sig:
            self.position.close(); self.bars_since_trade = 0
        elif self.position.is_short and not short_sig:
            self.position.close(); self.bars_since_trade = 0

# Возврат к среднему

class MeanRev(_Base_ATR):
    w, z_thr = 36, 3.0

    def init(self):
        c = pd.Series(self.data.Close, index=self.data.index)
        # z-score
        def zs(s):
            mu = s.rolling(self.w).mean()
            sd = s.rolling(self.w).std().replace(0, np.nan).ffill().bfill().fillna(1e-6)
            return (s - mu) / sd
        self.zscore = self.I(zs, c)
        self._init_atr()

    def next(self):
        price = self.data.Close[-1]
        if self._apply_trail(price):
            return

        zv = self.zscore[-1]
        long_sig  = zv < -self.z_thr
        short_sig = zv >  self.z_thr

        # вход
        if not self.position:
            if long_sig:
                self.buy()
                self.entry_extreme = price
            elif short_sig:
                self.sell()
                self.entry_extreme = price

        # выход при откате к 50% от порога
        elif self.position.is_long and zv >= -self.z_thr * 0.5:
            self.position.close()
        elif self.position.is_short and zv <=  self.z_thr * 0.5:
            self.position.close()


class SMACross(_Base_ATR):
    ss, sl = 18, 35

    def init(self):
        c = pd.Series(self.data.Close, index=self.data.index)
        self.fast = self.I(lambda s: SMAIndicator(pd.Series(s), self.ss).sma_indicator(), c)
        self.slow = self.I(lambda s: SMAIndicator(pd.Series(s), self.sl).sma_indicator(), c)
        self._init_atr()

    def next(self):
        price = self.data.Close[-1]
        if self._apply_trail(price):
            return

        cross_up   = crossover(self.fast, self.slow)
        cross_down = crossover(self.slow, self.fast)

        # выход по обратному пересечению
        if self.position.is_long and cross_down:
            self.position.close()
        elif self.position.is_short and cross_up:
            self.position.close()

        # вход
        if not self.position:
            if cross_up:
                self.buy()
                self.entry_extreme = price
            elif cross_down:
                self.sell()
                self.entry_extreme = price


class MACD_(_Base_ATR):
    fast, slow, signal = 18, 35, 30

    def init(self):
        c = pd.Series(self.data.Close, index=self.data.index)
        raw = c.ewm(span=self.fast).mean() - c.ewm(span=self.slow).mean()
        self.macd = self.I(lambda s: pd.Series(s).ewm(span=self.signal).mean(), raw)
        self._init_atr()

    def next(self):
        price = self.data.Close[-1]
        if self._apply_trail(price):
            return

        prev, curr = self.macd[-2], self.macd[-1]
        long_sig  = (curr > 0) and (prev <= 0)
        short_sig = (curr < 0) and (prev >= 0)

        # выход при обратном пересечении
        if self.position.is_long and short_sig:
            self.position.close()
        elif self.position.is_short and long_sig:
            self.position.close()

        # вход
        if not self.position:
            if long_sig:
                self.buy()
                self.entry_extreme = price
            elif short_sig:
                self.sell()
                self.entry_extreme = price


class Candle(_Base_ATR):
    min_hold_bars = 10

    def init(self):
        self.bars_since_trade = self.min_hold_bars
        self._init_atr()

    def next(self):
        price = self.data.Close[-1]
        self.bars_since_trade += 1

        # ATR-стоп
        if self._apply_trail(price):
            self.bars_since_trade = 0
            return

        # свечные сигналы
        o, c = self.data.Open, self.data.Close
        bull = (c[-1] > o[-1] and c[-2] < o[-2])
        bear = (c[-1] < o[-1] and c[-2] > o[-2])

        if self.bars_since_trade < self.min_hold_bars:
            return

        if self.position:
            if self.position.is_long and bear:
                self.position.close()
                self.sell()
                self.bars_since_trade = 0
            elif self.position.is_short and bull:
                self.position.close()
                self.buy()
                self.bars_since_trade = 0
        else:
            if bull:
                self.buy()
                self.bars_since_trade = 0
            elif bear:
                self.sell()
                self.bars_since_trade = 0

class BuyHold(Strategy):
    def init(self): pass
    def next(self):
        if not self.position: self.buy()


min_bars_map = {
    SMA_MACD_RSI: lambda: max(SMA_MACD_RSI.ss, SMA_MACD_RSI.sl,
                              MACD_.fast, MACD_.slow, MACD_.signal,
                              _Base_ATR.atr_window, _Base_ATR.dyn_window),
    MeanRev:      lambda: max(MeanRev.w,
                              _Base_ATR.atr_window, _Base_ATR.dyn_window),
    SMACross:     lambda: max(SMACross.ss, SMACross.sl,
                              _Base_ATR.atr_window, _Base_ATR.dyn_window),
    MACD_:        lambda: max(MACD_.fast, MACD_.slow, MACD_.signal,
                              _Base_ATR.atr_window, _Base_ATR.dyn_window),
    Candle:       lambda: max(2,
                              _Base_ATR.atr_window, _Base_ATR.dyn_window),
    BuyHold:      lambda: 1
}

class Tester:
    def __init__(self, cls, n_splits=2):
        self.cls      = cls
        self.n_splits = n_splits
        self.logger   = logging.getLogger(f"Tester:{cls.__name__}")

    def _space(self, trial):
        if self.cls is SMA_MACD_RSI:
            ss  = trial.suggest_int("ss", 6, 18, step = 2)
            sl  = trial.suggest_int("sl", ss + 10, 39, step = 2)
            thr = trial.suggest_int("thr", 30, 48, step = 2)
            return {"ss": ss, "sl": sl, "thr": thr}

        elif self.cls is MeanRev:
            w    = trial.suggest_int("w", 6, 36, step = 2)
            z_thr = trial.suggest_float("z_thr", 1.0, 3.0, step=0.1)
            return {"w": w, "z_thr": z_thr}

        elif self.cls is SMACross:
            ss = trial.suggest_int("ss", 6, 18, step = 2)
            sl = trial.suggest_int("sl", ss + 10, 35, step = 2)
            if sl / ss < 1.5:
                raise optuna.TrialPruned()
            return {"ss": ss, "sl": sl}

        elif self.cls is MACD_:
            fast   = trial.suggest_int("fast", 6, 18, step = 2)
            slow   = trial.suggest_int("slow", fast+5, 35, step=2)
            signal = trial.suggest_int("signal", 2, min(slow-1, 30), step=2)
            return {"fast": fast, "slow": slow, "signal": signal}

        else:
            # Candle и BuyHold не имеют параметров
            return {}

    def best(self, df, n_trials=18):
        req = max(min_bars_map[self.cls](), 2*self.n_splits)
        if len(df) < req:
            raise ValueError(f"Недостаточно данных: {len(df)} < {req}")
        tscv=TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial):
            params = self._space(trial)

            sortino_scores = []
            for tr_idx, te_idx in tscv.split(df):
                dtr, dte = df.iloc[tr_idx], df.iloc[te_idx]
                if len(dtr) < req or len(dte) < 1:
                    sortino_scores.append(-10)
                    continue
                res = Backtest(
                    dte,
                    self.cls,
                    cash=100000,
                    commission=0.002,
                    exclusive_orders=True
                ).run(**params)
                equity = res._equity_curve["Equity"]
                rets   = equity.pct_change().dropna()
                downside = rets[rets < 0]
                if len(downside) and downside.std() > 0:
                    sortino = rets.mean() / downside.std() * np.sqrt(252)
                else:
                    sortino = -10
                sortino_scores.append(sortino)

            return float(np.mean(sortino_scores))

        study=optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=8,seed=42),
            pruner=MedianPruner()
        )
        study.optimize(objective,n_trials=n_trials,n_jobs=4)
        return study.best_params

# запуск тестирования

bt_classes=[SMA_MACD_RSI,MeanRev,SMACross,MACD_,Candle,BuyHold]

def split_sets(df):
    i1,i2=int(len(df)*0.6),int(len(df)*0.8)
    return df[:i1],df[i1:i2],df[i2:]

def eval_bt(df,cls,params):
    res=Backtest(df,cls,cash=100000,commission=0.002,exclusive_orders=True).run(**params)
    ec=res._equity_curve; rets=ec["Equity"].pct_change().dropna()
    cum=ec["Equity"]/ec["Equity"].iloc[0]
    trades=res._trades["PnL"].dropna() if not res._trades.empty else pd.Series()
    m=compute_metrics( cum, rets, trades)
    m.update({"name":cls.__name__,"dates":ec.index,"equity":ec["Equity"]})
    return m

for tk in stock_tk:
    df_tk = raw.xs(tk,axis=1,level=0).dropna()
    if df_tk.empty:
        logger.warning("%s: нет данных", tk)
        continue

    tr, va, te = split_sets(df_tk)
    valid = []
    for C in bt_classes:
        if len(tr) >= min_bars_map[C]():
            valid.append(C)
        else:
            logger.warning("%s: пропускаем %s (len(tr)<req)", tk, C.__name__)
    if not valid:
        logger.warning("%s: ни одной стратегии не подошло", tk); continue

    params_list, val_stats = [], []
    for C in valid:
        p = Tester(C, n_splits=2).best(tr, n_trials=18)
        params_list.append((C,p))
        val_stats .append(eval_bt(va, C, p))

    df_val = pd.DataFrame(val_stats).sort_values("Sharpe Ratio",ascending=False).reset_index(drop=True)

    # Валидационная диаграмма
    fig = px.bar(df_val, x="name",
                 y=["Sharpe Ratio","Total Return [%]","Max. Drawdown [%]",
                    "Sortino Ratio","Win Rate [%]","Volatility [%]"],
                 barmode="group",
                 title=f"{tk}: Валидация {va.index[0].date()}–{va.index[-1].date()}")
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title_text=None, margin=dict(b=40))
    fig.show()

    # Таблица с метриками
    cols = ["name","Total Return [%]","CAGR [%]","Volatility [%]",
            "Sharpe Ratio","Sortino Ratio","Max. Drawdown [%]",
            "Calmar Ratio","Win Rate [%]","Profit Factor","# Trades"]
    tbl = df_val[cols].round(4)
    fig = go.Figure(data=[go.Table(
        header=dict(values=tbl.columns, fill_color="#f3f5f7", align="left"),
        cells=dict(values=[tbl[c] for c in cols], fill_color="#f6faff", align="left")
    )])
    fig.update_layout(margin=dict(l=30,r=40,t=40,b=40))
    fig.show()

    # Валидационный и тестовый период
    wf = []
    for C,p in params_list:
        wf.append(eval_bt(va, C, p))
        wf.append(eval_bt(te, C, p))

    logger.info("Найденные параметры для %s:", tk)
    for C,p in params_list:
        logger.info("  %s: %s", C.__name__, p)

    uniq = list(dict.fromkeys([m["name"] for m in wf]))
    palette = px.colors.qualitative.Plotly
    cmap = {n: palette[i%len(palette)] for i,n in enumerate(uniq)}

    fig = go.Figure()
    seen = set()
    for m in wf:
        fig.add_trace(go.Scatter(
            x=m["dates"], y=m["equity"]/m["equity"].iloc[0],
            mode="lines", name=m["name"], legendgroup=m["name"],
            showlegend=m["name"] not in seen,
            line=dict(color=cmap[m["name"]])
        ))
        seen.add(m["name"])
    fig.update_layout(
        title=(f"{tk}: Валидация и тест Val {va.index[0].date()}–{va.index[-1].date()} "
               f"| Test {te.index[0].date()}–{te.index[-1].date()}"),
        hovermode="x unified",
        legend=dict(traceorder="normal")
    )
    fig.show()

# Задание 3. Обработка даннах

class FE:
    def __init__(self,
                 tickers,
                 lake="data_lake",
                 base_win=65,
                 thr=3,
                 vol_window=65,
                 min_win=5,
                 max_win=100):
        self.tickers    = tickers
        self.lake       = lake
        self.base_win   = base_win
        self.thr        = thr
        self.vol_window = vol_window
        self.min_win    = min_win
        self.max_win    = max_win
        self.logger     = logging.getLogger(self.__class__.__name__)

    def _outliers(self, df, win):

        df2 = df.copy()
        for col in df2.columns:
            s  = df2[col]
            mu = s.rolling(win).mean()
            sd = (
                s.rolling(win).std()
                 .replace(0, np.nan)
                 .ffill().bfill()
                 .fillna(1e-6)
            )
            mask = (s - mu).abs() / sd > self.thr
            df2.loc[mask, col] = np.nan
        return df2

    def _make(self, df):

        if len(df) < self.base_win:
            return pd.DataFrame()
        f = pd.DataFrame(index=df.index)
        f["close"] = df["Close"]
        f["r1"]    = df["Close"].pct_change()
        f["r5"]    = df["Close"].pct_change(5)
        f["sma24"] = SMAIndicator(df["Close"], 24).sma_indicator().ffill().bfill()
        f["sma120"] = SMAIndicator(df["Close"], 120).sma_indicator().ffill().bfill()
        f["vol5"]  = f["r1"].rolling(5).std().ffill().bfill()
        f["rsi14"] = RSIIndicator(df["Close"], 14).rsi().ffill().bfill()
        f["macd"]  = MACD(df["Close"]).macd_diff().ffill().bfill()
        f["atr65"] = (
            AverageTrueRange(
                df["High"], df["Low"], df["Close"], self.vol_window
            ).average_true_range()
             .ffill().bfill()
        )
        return f

    def _process_ticker(self, tk, mi):
        df_tk = mi.xs(tk, axis=1, level=0).dropna()
        if df_tk.empty:
            self.logger.warning(f"{tk}: нет данных для FE")
            return
        # выбросы по динамическому окну
        atr = AverageTrueRange(
            df_tk["High"], df_tk["Low"], df_tk["Close"], self.vol_window
        ).average_true_range()
        atr_med = atr.median() or 1e-6
        dyn_win = int(np.clip(
            (atr / atr_med).median() * self.base_win,
            self.min_win, self.max_win
        ))

        feats = self._make(df_tk)
        if feats.empty:
            self.logger.info(f"{tk}: мало баров (<{self.base_win}) для признаков")
            return

        # выбросы по динамическому окну
        feats = self._outliers(feats, dyn_win)

        # заполнение пропусков
        feats = feats.ffill().bfill().fillna(feats.median())

        # robust-scaling: (x - median)/IQR
        med = feats.median()
        iqr = feats.quantile(0.75) - feats.quantile(0.25)
        iqr = iqr.replace(0, 1)
        feats = (feats - med) / iqr

        # сохранение по дням
        path = os.path.join(self.lake, tk)
        os.makedirs(path, exist_ok=True)
        for day, chunk in feats.resample("D"):
            fn = os.path.join(path, f"{day.date()}.parquet")
            if chunk.empty:
                continue
            if chunk.isnull().any().any():
                self.logger.warning(f"{tk} {day.date()}: NaN пропущено")
                continue
            if not os.path.exists(fn):
                chunk.to_parquet(fn)
                # self.logger.info(f"Сохранён {fn}")

    def run(self, mi):
        os.makedirs(self.lake, exist_ok=True)
        for tk in self.tickers:
            self._process_ticker(tk, mi)

class New_Data(FileSystemEventHandler):
    def __init__(self, eng):
        self.eng = eng

    def on_created(self, event):
        # запуск FE, когда появляется новый parquet-файл
        if event.src_path.endswith(".parquet"):
            mi = pd.read_parquet(event.src_path)
            self.eng.run(mi)

# Запуск FE и Watchdog
fe = FE(crypto_tk + stock_tk, base_win=65, thr=3,
                     vol_window=65, min_win=5, max_win=100)
fe.run(raw)

observer = Observer()
observer.schedule(New_Data(fe), path = RAW_DIR, recursive=False)
observer.daemon = True
observer.start()

# Диаграмма накопления файлов
counts = {
    tk: len(os.listdir(os.path.join("data_lake", tk)))
    for tk in crypto_tk + stock_tk
}
fig = px.bar(
    x=list(counts.values()),
    y=list(counts.keys()),
    orientation="h",
    title="Накопление файлов (дней)",
    labels={"x": "Дней", "y": "Тикер"},
    width=600, height=500
)
fig.update_traces(text=list(counts.values()),
                  textposition="outside")
fig.update_layout(xaxis_title=None,
                  yaxis_title=None)
fig.show()
