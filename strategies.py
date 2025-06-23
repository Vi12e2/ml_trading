import numpy as np
import pandas as pd
import torch
import logging

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import AverageTrueRange

from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from catboost import CatBoostClassifier, Pool

from fastai.losses import CrossEntropyLossFlat
from fastai.callback.tracker import EarlyStoppingCallback

from tsai.all import (
    get_ts_dls, ts_learner, accuracy,
    InceptionTime, TSTPlus,
    MiniRocketFeaturesPlus, get_minirocket_features
)

from neuralforecast import NeuralForecast
from neuralforecast.models.patchtst import PatchTST
from neuralforecast.models.nhits import NHITS

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig, DiscreteCQL

from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

import plotly.express as px
import plotly.graph_objects as go

from config import *
from features import raw, index_df, FE


# fe = FE(
#     tickers=crypto_tk + stock_tk,
#     index_df=index_df,
#     lake=DATA_LAKE_DIR,
#     base_win=15,
#     thr=3.0,
#     vol_window=35,
#     min_win=5,
#     max_win=35
# )

# Стратегия с динамическим ATR trailing stop

class _Base_ATR(Strategy):
    atr_window = ATR_W
    dyn_window = ATR_DYN

    def _init_atr(self):

        h = pd.Series(self.data.High, index=self.data.index)
        l = pd.Series(self.data.Low,  index=self.data.index)
        c = pd.Series(self.data.Close,index=self.data.index)

        self.atr = self.I(
            lambda c_: AverageTrueRange(h, l, c_, self.atr_window).average_true_range(),
            c
        )
        self.entry_extreme = None

    def _apply_trail(self, price: float) -> bool:

        if pd.isna(self.atr[-1]) or self.atr[-1] == 0 or not self.position:
            return False

        atr_s = pd.Series(self.atr)
        base = atr_s.rolling(self.dyn_window).mean().replace(0, np.nan).ffill().fillna(1e-6)
        dyn_mult = np.clip(atr_s / base, 0.8, 3.5)
        level = dyn_mult.iloc[-1] * self.atr[-1]

        if self.position.is_long:
            self.entry_extreme = max(self.entry_extreme or price, price)
            if price < self.entry_extreme - level:
                self.position.close()
                self.entry_extreme = None
                return True
        else:
            self.entry_extreme = min(self.entry_extreme or price, price)
            if price > self.entry_extreme + level:
                self.position.close()
                self.entry_extreme = None
                return True
        return False


# СТРАТЕГИИ SMA_MACD_RSI, MeanRev, SMACross, MACD_, BuyHold

class SMA_MACD_RSI(_Base_ATR):
    ss, sl, thr = 18, 39, 48   # короткая SMA, длинная SMA, порог RSI
    min_hold_bars = 2          # минимальное время удержания позиции

    def init(self):
        # подготовка индикаторов на всей истории
        c = pd.Series(self.data.Close, index=self.data.index)
        # короткая и длинная SMA
        self.sma_s = self.I(lambda s: s.rolling(self.ss).mean(), c)
        self.sma_l = self.I(lambda s: s.rolling(self.sl).mean(), c)
        # MACD: raw = EMA12 - EMA26, сглаживание сигнальной EMA9
        macd_raw = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        self.macd  = self.I(lambda s: pd.Series(s).ewm(span=9).mean(), macd_raw)
        # RSI(14)) для оценки перекупленности/перепроданности
        self.rsi   = self.I(lambda s: RSIIndicator(s, 14).rsi(), c)
        # ATR-трейлинг-стоп из _Base_ATR
        self._init_atr()
        # Счетчик баров с момента последней сделки
        self.bars_since_trade = self.min_hold_bars

    def next(self):
        price = self.data.Close[-1]
        self.bars_since_trade += 1

        # Попытка срабатывания ATR-трейлинг-стопа
        if self._apply_trail(price):
            self.bars_since_trade = 0
            return

        # min_hold_bars с последней сделки
        if self.bars_since_trade < self.min_hold_bars:
            return

        # Сигналы: пересечение SMA, изменение MACD, условия по RSI
        s = self.sma_s[-1]
        l = self.sma_l[-1]
        diff = self.macd[-1] - self.macd[-2]
        r = self.rsi[-1]
        long_sig  = (s > l) and (diff > 0) and (r < 100 - self.thr)
        short_sig = (s < l) and (diff < 0) and (r > self.thr)

        # Вход в позицию, если нет открытой
        if not self.position:
            if long_sig:
                self.buy()
                self.entry_extreme = price
                self.bars_since_trade = 0
            elif short_sig:
                self.sell()
                self.entry_extreme = price
                self.bars_since_trade = 0

        # Выход если уже в позиции когда сигнал пропал
        elif self.position.is_long and not long_sig:
            self.position.close()
            self.bars_since_trade = 0
        elif self.position.is_short and not short_sig:
            self.position.close()
            self.bars_since_trade = 0


class MeanRev(_Base_ATR):
    w, z_thr = 36, 3.0   # окно для Z-score, порог отклонения

    def init(self):

        c = pd.Series(self.data.Close, index=self.data.index)

        def zs(s: pd.Series) -> pd.Series:
            # скользящее среднее и std с min_periods=1, без bfill
            mu = s.rolling(self.w, min_periods=1).mean()
            sd = (
                s.rolling(self.w, min_periods=1).std()
                 .replace(0, np.nan)
                 .fillna(1e-6)    # остатки константой
            )
            return (s - mu) / sd
        self.zscore = self.I(zs, c)
        self._init_atr()

    def next(self):
        price = self.data.Close[-1]
        # Попытка срабатывания ATR-трейл-стопа
        if self._apply_trail(price):
            return

        zv = self.zscore[-1]
        long_sig  = zv < -self.z_thr  # перепроданность
        short_sig = zv >  self.z_thr  # перекупленность

        # Вход при сигнале
        if not self.position:
            if long_sig:
                self.buy()
                self.entry_extreme = price
            elif short_sig:
                self.sell()
                self.entry_extreme = price

        # Выходы при откате к 50% от порога
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

        # Выходы по обратному пересечению
        if self.position.is_long and cross_down:
            self.position.close()
        elif self.position.is_short and cross_up:
            self.position.close()

        # Входы
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

        # Выход при обратном пересечении
        if self.position.is_long and short_sig:
            self.position.close()
        elif self.position.is_short and long_sig:
            self.position.close()

        # Входы
        if not self.position:
            if long_sig:
                self.buy()
                self.entry_extreme = price
            elif short_sig:
                self.sell()
                self.entry_extreme = price


class BuyHold(Strategy):

    def init(self): pass

    def next(self):
        if not self.position:
            self.buy()

# Минимальное количество баров для стратегий
min_bars_map = {
    SMA_MACD_RSI: lambda: max(SMA_MACD_RSI.ss, SMA_MACD_RSI.sl, MACD_.fast, MACD_.slow,
                              MACD_.signal, _Base_ATR.atr_window, _Base_ATR.dyn_window),
    # MeanRev: lambda: max(MeanRev.w, _Base_ATR.atr_window, _Base_ATR.dyn_window),
    # SMACross: lambda: max(SMACross.ss, SMACross.sl, _Base_ATR.atr_window, _Base_ATR.dyn_window),
    MACD_: lambda: max(MACD_.fast, MACD_.slow, MACD_.signal, _Base_ATR.atr_window, _Base_ATR.dyn_window),
    BuyHold: lambda: 1
}


# Метрики
def compute_metrics(cum: pd.Series, rets: pd.Series, trades: pd.Series) -> dict:

    # период наблюдения
    period = cum.index[-1] - cum.index[0]
    # полные дни и остатки секунд переведённые в доли дня
    days = period.days + period.seconds / 86400
    # Total Return
    total_ret = (cum.iloc[-1] - 1) * 100
    # CAGR
    years = days / 365.25
    cagr = (cum.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    # число часовых баров в годовую шкалу
    hours_observed = len(rets)
    # сколько часов эквивалентно годовой торговле
    hours_per_year = hours_observed / days * 365.25
    # Волатильность и Sharpe на часовых доходностях
    vol = rets.std() * np.sqrt(hours_per_year) * 100
    sharpe = (
        rets.mean() / rets.std() * np.sqrt(hours_per_year)
        if rets.std() > 0 else np.nan
    )
    # Sortino Ratio
    down = rets[rets < 0]
    sortino = (
        rets.mean() / down.std() * np.sqrt(hours_per_year)
        if len(down) > 0 and down.std() > 0 else np.nan
    )
    # Max Drawdown
    max_dd = ((cum / cum.cummax()) - 1).min() * 100
    # Calmar
    calmar = (cagr) / (-max_dd / 100) if max_dd < 0 else np.nan

    # Метрики по PnL
    pnl = trades.dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = (len(wins) / len(pnl) * 100) if len(pnl) > 0 else 0
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = gross_win / gross_loss if gross_loss != 0 else np.inf
    num_trades = len(pnl)

    return {
        "Total Return [%]": total_ret,
        "CAGR [%]": cagr * 100,
        "Volatility [%]": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max. Drawdown [%]": max_dd,
        "Calmar Ratio": calmar,
        "Win Rate [%]": win_rate,
        "Profit Factor": profit_factor,
        "# Trades": num_trades
    }

# КЛАСС Tester для оптимизации черерез OPTUNA

class Tester:
    def __init__(self, cls, n_splits: int = 2):
        self.cls      = cls
        self.n_splits = n_splits
        self.logger   = logging.getLogger(f"Tester:{cls.__name__}")

    def _space(self, trial: optuna.trial.Trial) -> dict:

        # Пространство поиска гиперпараметров

        if self.cls is SMA_MACD_RSI:
            ss  = trial.suggest_int("ss", 6, 18, step=2)
            sl  = trial.suggest_int("sl", ss + 10, 39, step=2)
            thr = trial.suggest_int("thr", 30, 48, step=2)
            return {"ss": ss, "sl": sl, "thr": thr}

        # elif self.cls is MeanRev:
        #     w     = trial.suggest_int("w", 6, 36, step=2)
        #     z_thr = trial.suggest_float("z_thr", 1.0, 3.0, step=0.1)
        #     return {"w": w, "z_thr": z_thr}

        # elif self.cls is SMACross:
        #     ss = trial.suggest_int("ss", 6, 18, step=2)
        #     sl = trial.suggest_int("sl", ss + 10, 35, step=2)
        #     if sl / ss < 1.5:
        #         raise optuna.TrialPruned()
        #     return {"ss": ss, "sl": sl}

        # elif self.cls is MACD_:
        #     fast   = trial.suggest_int("fast", 6, 18, step=2)
        #     slow   = trial.suggest_int("slow", fast + 5, 35, step=2)
        #     signal = trial.suggest_int("signal", 2, min(slow - 1, 30), step=2)
        #     return {"fast": fast, "slow": slow, "signal": signal}

        else:
            # Candle и BuyHold не имеют гиперпараметров
            return {}

    def best(self, df: pd.DataFrame, n_trials: int = 9) -> dict:

        # Лучшие параметры для стратегии с использованеим n_splits.
        req = max(min_bars_map[self.cls](), 2 * self.n_splits)
        if len(df) < req:
            raise ValueError(f"Недостаточно данных: {len(df)} < {req}")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial: optuna.trial.Trial) -> float:
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
                    cash=100_000,
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

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=9, seed=42),
            pruner=MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=4)
        return study.best_params


# РАЗБИЕНИЕ НА TRAIN/VAL/TEST И Backtest

def split_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    i1 = int(len(df) * 0.6)
    i2 = int(len(df) * 0.8)
    return df[:i1], df[i1:i2], df[i2:]

def eval_bt(df: pd.DataFrame, cls, params: dict) -> dict:

    res = Backtest(
        df,
        cls,
        cash=100_000,
        commission=0.002,
        exclusive_orders=True
    ).run(**params)

    ec = res._equity_curve
    equity = ec["Equity"]
    cum = equity / equity[0]
    rets = cum.pct_change().dropna()
    trades = res._trades["PnL"].dropna() if not res._trades.empty else pd.Series(dtype=float)

    m = compute_metrics(cum, rets, trades)
    m.update({"name": cls.__name__, "dates": cum.index, "equity": cum.values})
    return m

# bt_classes=[SMA_MACD_RSI, BuyHold]

# for tk in stock_tk:
#     df_tk = raw.xs(tk,axis=1,level=0).dropna()
#     if df_tk.empty:
#         logger.warning("%s: нет данных", tk)
#         continue

#     tr, va, te = split_sets(df_tk)
#     valid = []
#     for C in bt_classes:
#         if len(tr) >= min_bars_map[C]():
#             valid.append(C)
#         else:
#             logger.warning("%s: пропускаем %s (len(tr)<req)", tk, C.__name__)
#     if not valid:
#         logger.warning("%s: ни одной стратегии не подошло", tk); continue

#     params_list, val_stats = [], []
#     for C in valid:
#         p = Tester(C, n_splits=2).best(tr, n_trials=9)
#         params_list.append((C,p))
#         val_stats .append(eval_bt(va, C, p))

#     df_val = pd.DataFrame(val_stats).sort_values("Sharpe Ratio",ascending=False).reset_index(drop=True)

#     # Валидационная диаграмма
#     fig = px.bar(df_val, x="name",
#                  y=["Sharpe Ratio","Total Return [%]","Max. Drawdown [%]",
#                     "Sortino Ratio","Win Rate [%]","Volatility [%]"],
#                  barmode="group",
#                  title=f"{tk}: Валидация {va.index[0].date()}–{va.index[-1].date()}")
#     fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title_text=None, margin=dict(b=40))
#     fig.show()

#     # Таблица с метриками
#     cols = ["name","Total Return [%]","CAGR [%]","Volatility [%]",
#             "Sharpe Ratio","Sortino Ratio","Max. Drawdown [%]",
#             "Calmar Ratio","Win Rate [%]","Profit Factor","# Trades"]
#     tbl = df_val[cols].round(4)
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=tbl.columns, fill_color="#f3f5f7", align="left"),
#         cells=dict(values=[tbl[c] for c in cols], fill_color="#f6faff", align="left")
#     )])
#     fig.update_layout(margin=dict(l=30,r=40,t=40,b=40))
#     fig.show()

#     # Валидационный и тестовый период
#     wf = []
#     for C,p in params_list:
#         wf.append(eval_bt(va, C, p))
#         wf.append(eval_bt(te, C, p))

#     logger.info("Найденные параметры для %s:", tk)
#     for C,p in params_list:
#         logger.info("  %s: %s", C.__name__, p)

#     uniq = list(dict.fromkeys([m["name"] for m in wf]))
#     palette = px.colors.qualitative.Plotly
#     cmap = {n: palette[i%len(palette)] for i,n in enumerate(uniq)}

#     fig = go.Figure()
#     seen = set()
#     for m in wf:
#         fig.add_trace(go.Scatter(
#             x=m["dates"], y=m["equity"]/m["equity"][0],
#             mode="lines", name=m["name"], legendgroup=m["name"],
#             showlegend=m["name"] not in seen,
#             line=dict(color=cmap[m["name"]])
#         ))
#         seen.add(m["name"])
#     fig.update_layout(
#         title=(f"{tk}: Валидация и тест Val {va.index[0].date()}–{va.index[-1].date()} "
#                f"| Test {te.index[0].date()}–{te.index[-1].date()}"),
#         hovermode="x unified",
#         legend=dict(traceorder="normal")
#     )
#     fig.show()


#      PATCHTST и NHITS (Neuralforecast)  _______________

def generate_all_nf_signals(fe: FE, stock_tk: list[str], H: int):
    
    signals_ptst_val,  signals_ptst_test  = {}, {}
    signals_nhits_val, signals_nhits_test = {}, {}

    # Конфигурация PatchTST (не использует признаки)
    common_cfg_patch = dict(
        input_size                = SEQ_LEN,
        stat_exog_list            = None,
        hist_exog_list            = None,
        futr_exog_list            = None,
        dropout                   = 0.3,
        max_steps                 = 7,
        batch_size                = 128,
        dataloader_kwargs         = {"num_workers": 2},
        accelerator               = "auto",
        devices                   = 1,
        start_padding_enabled     = True,
        optimizer_kwargs          = {"lr": 1e-4, "weight_decay": 5e-4},
    )
    # Конфигурация NHITS с применением признаков из FE
    common_cfg_nhits = dict(
        input_size                = SEQ_LEN,
        hist_exog_list            = None,
        stack_types               = ["identity"] * 3,
        n_blocks                  = [1, 1, 1],
        mlp_units                 = [[1024, 1024]] * 3,
        n_pool_kernel_size        = [2, 2, 1],
        n_freq_downsample         = [4, 2, 1],
        dropout_prob_theta        = 0.1,
        max_steps                 = 7,
        batch_size                = 128,
        dataloader_kwargs         = {"num_workers": 2},
        accelerator               = "auto",
        devices                   = 1,
        start_padding_enabled     = True,
        optimizer_kwargs          = {"lr": 5e-4, "weight_decay": 1e-4},
    )

    for tk in stock_tk:
        # price и true_sig
        price_df = (
            raw.xs(tk, axis=1, level=0)["Close"]
            .dropna().to_frame("y")
            .reset_index().rename(columns={"Date": "ds"})
        )
        price_df["unique_id"] = tk
        price_df["true_sig"]  = (price_df["y"].shift(-1) > price_df["y"]).astype(int)
        price_df = price_df.iloc[:-1]

        # Загрузка признаков из FE
        feats = fe.load_features_for_ml(tk).copy()
        feats.index.name = "ds"
        feats = feats.reset_index()
        feats["unique_id"] = tk

        # только метки где есть и цена и признаки
        df_all = pd.merge(price_df, feats, on=["unique_id","ds"], how="inner")
        df_all = df_all.sort_values("ds").reset_index(drop=True)

        # 60/20/20 на df_all
        n  = len(df_all)
        i1 = int(0.6 * n)
        i2 = int(0.8 * n)
        train_df = df_all.iloc[:i1].copy()
        val_df   = df_all.iloc[i1:i2].copy()
        test_df  = df_all.iloc[i2:].copy()

        val_h, test_h = len(val_df), len(test_df)

        # PATCHTST

        model_pt = PatchTST(h=val_h + test_h, **common_cfg_patch)
        nf_pt    = NeuralForecast(models=[model_pt], freq="H")
        nf_pt.fit(train_df, val_size=val_h + test_h)

        fut_all = nf_pt.make_future_dataframe().reset_index(drop=True)
        pr_pt    = nf_pt.predict(futr_df=fut_all)["PatchTST"].values
        pv, pt   = pr_pt[:val_h], pr_pt[val_h:]

        signals_ptst_val[tk]  = pd.Series((pv >  val_df["y"].values).astype(int), index=val_df["ds"])
        signals_ptst_test[tk] = pd.Series((pt >  test_df["y"].values).astype(int), index=test_df["ds"])
        logger.info(
            "PatchTST %s: VA acc=%.4f, TE acc=%.4f",
            tk,
            accuracy_score(val_df["true_sig"],  signals_ptst_val[tk]),
            accuracy_score(test_df["true_sig"], signals_ptst_test[tk]),
        )

        # NHIST

        # Модель и обучениe
        exogs = [c for c in df_all.columns if c not in ("unique_id","ds","y","true_sig")]
        cfg_nh = common_cfg_nhits.copy()
        cfg_nh["hist_exog_list"] = exogs
        model_nh = NHITS(h=val_h + test_h, **cfg_nh)
        nf_nh    = NeuralForecast(models=[model_nh], freq="H")
        nf_nh.fit(train_df, val_size=val_h + test_h)
        # val+test
        fut_vt = (nf_nh.make_future_dataframe().reset_index(drop=True).merge(df_all[["unique_id","ds"]+exogs].iloc[i1:],
                        on=["unique_id","ds"], how="left"))
        pr_vt = nf_nh.predict(futr_df=fut_vt)["NHITS"].values
        pv_nh, pt_nh = pr_vt[:val_h], pr_vt[val_h:]
        signals_nhits_val[tk]  = pd.Series((pv_nh >  val_df["y"].values).astype(int), index=val_df["ds"])
        signals_nhits_test[tk] = pd.Series((pt_nh >  test_df["y"].values).astype(int), index=test_df["ds"])

        logger.info(
            "NHITS %s: VA acc=%.4f, TE acc=%.4f",
            tk,
            accuracy_score(val_df["true_sig"],  signals_nhits_val[tk]),
            accuracy_score(test_df["true_sig"], signals_nhits_test[tk])
        )

    return signals_ptst_val, signals_ptst_test, signals_nhits_val, signals_nhits_test

#                   CATBOOST _________________

def generate_all_cb_signals(fe: FE, stock_tk: list[str], H: int):
    
    signals_cb_val, signals_cb_test = {}, {}
    fixed_iter = 350
    early_stop = 8
    # TimeSeriesSplit для кросс-валидации подбора гиперпараметров
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = {}

    for tk in stock_tk:
        # загрузка признаков
        df_feat = fe.load_features_for_ml(tk)
        if df_feat.empty:
            signals_cb_val[tk]  = pd.Series(dtype=int)
            signals_cb_test[tk] = pd.Series(dtype=int)
            continue

        df_ml = FE.create_ml_target(df_feat, H)
        X_ml, y_ml = df_ml.drop(columns=['future','target']), df_ml['target']
        # train/val/test по 60/20/20
        n = len(X_ml)
        i1, i2 = int(0.6 * n), int(0.8 * n)
        X_tr, X_va, X_te = X_ml.iloc[:i1], X_ml.iloc[i1:i2], X_ml.iloc[i2:]
        y_tr, y_va, y_te = y_ml.iloc[:i1], y_ml.iloc[i1:i2], y_ml.iloc[i2:]

        # class_weights
        classes = np.unique(y_tr)
        cw_arr = compute_class_weight('balanced', classes=classes, y=y_tr)
        class_weights = dict(zip(classes, cw_arr))

        # подбор гиперпараметров с Optuna
        def cb_objective(trial):
            params = {
                'depth':               trial.suggest_int('depth', 7, 8),
                'learning_rate':       trial.suggest_loguniform('learning_rate', 1e-4, 1e-3),
                'l2_leaf_reg':         trial.suggest_loguniform('l2_leaf_reg', 1e-3, 1e-2),
                'subsample':           trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel':   trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.6, 1.0),
                'iterations':          fixed_iter,
                'random_seed':         42,
                'loss_function':       'Logloss',
                'eval_metric':         'Accuracy',
                'verbose':             False,
                'class_weights':       class_weights
            }
            scores = []
            for tr_idx, val_idx in tscv.split(X_tr):
                model = CatBoostClassifier(**params)
                model.fit(
                    Pool(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]),
                    eval_set=Pool(X_tr.iloc[val_idx], y_tr.iloc[val_idx]),
                    early_stopping_rounds=early_stop,
                    use_best_model=True
                )
                preds = model.predict(X_tr.iloc[val_idx])
                scores.append(accuracy_score(y_tr.iloc[val_idx], preds))
            return np.mean(scores)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_warmup_steps=20)
        )
        study.optimize(cb_objective, n_trials=50, n_jobs=4)
        best_params[tk] = study.best_params
        logger.info("CatBoost %s: Лучшие параметры %s", tk, best_params[tk])

        # Обучение на train и val
        model = CatBoostClassifier(
            **best_params[tk],
            iterations=fixed_iter,
            random_seed=42,
            loss_function='Logloss',
            eval_metric='Accuracy',
            class_weights=class_weights,
            verbose=False
        )
        model.fit(
            Pool(X_tr, y_tr),
            eval_set=Pool(X_va, y_va),
            early_stopping_rounds=early_stop,
            use_best_model=True
        )

        # Предсказания
        s_va = pd.Series(model.predict(X_va), index=X_va.index).astype(int)
        s_te = pd.Series(model.predict(X_te), index=X_te.index).astype(int)

        signals_cb_val[tk]  = s_va
        signals_cb_test[tk] = s_te

        logger.info(
            "CatBoost %s: VA acc=%.4f, TE acc=%.4f",
            tk,
            accuracy_score(y_va, s_va),
            accuracy_score(y_te, s_te)
        )

    return signals_cb_val, signals_cb_test

#      МОДЕЛИ TSAI:   InceptionTime,  TSTPlus,  MiniRocket+SGD

def generate_all_tsai_signals(fe: FE, stock_tk: list[str], H: int):  
    
    signals_ts_i_val     = {}
    signals_ts_i_test    = {}
    signals_ts_tstp_val  = {}
    signals_ts_tstp_test = {}
    signals_ts_mr_val    = {}
    signals_ts_mr_test   = {}

    signals_ts_i_tr     = {}
    signals_ts_tstp_tr  = {}

    threshold = 0.5

    for tk in stock_tk:
        df_feat = fe.load_features_for_ml(tk)
        if df_feat.empty:
            for d in (signals_ts_i_tr, signals_ts_i_val, signals_ts_i_test,
                    signals_ts_tstp_tr, signals_ts_tstp_val, signals_ts_tstp_test,
                    signals_ts_mr_val, signals_ts_mr_test):
                d[tk] = pd.Series(dtype=int)
            continue

        # датасет с метками на H горизонт 
        df_nn = FE.create_ml_target(df_feat, H)
        # список признаков без future, target
        feat_cols = [c for c in df_nn.columns if c not in ("future","target")]
        arr, ys = df_nn[feat_cols].values, df_nn["target"].values
        
        # последовательные окна длины SEQ_LEN для TS-моделей
        Xs, ys2 = [], []
        for i in range(len(arr)-SEQ_LEN):
            Xs.append(arr[i:i+SEQ_LEN].T)
            ys2.append(ys[i+SEQ_LEN])
        Xs, ys2 = np.stack(Xs), np.array(ys2)

        n = len(ys2)
        i1, i2 = int(0.6*n), int(0.8*n)
        train_idx = list(range(i1))
        val_idx   = list(range(i1, i2))
        test_idx  = list(range(i2, n))
        idx       = df_nn.index[SEQ_LEN:]

        # метки их частоты в обучающем наборе
        classes, counts = np.unique(ys2[train_idx], return_counts=True)

        # расчет весов
        cw = compute_class_weight("balanced", classes=classes, y=ys2[train_idx])

        # перевод весов в тензор для fastai
        weight_tensor = torch.tensor(cw, dtype=torch.float32)

        # общие настройки моделей
        common_ts_kwargs = dict(
            c_out=2,
            loss_func=CrossEntropyLossFlat(weight=weight_tensor),
            metrics=accuracy,
            cbs=[EarlyStoppingCallback(patience=3)]
        )

        # Обучение модели и получение сигналов на train/val/test
        def train_ts(model_cls, arch_cfg, lr, wd):

            dls = get_ts_dls(Xs, ys2, splits=(train_idx,val_idx), bs=128)
            learn = ts_learner(
                dls,
                model_cls,
                arch_config=arch_cfg,
                **common_ts_kwargs
            ).to_fp16()
            learn.fit_one_cycle(50, lr, wd=wd)

            # train
            p_tr,_ = learn.get_preds(ds_idx=0)
            probs_tr = p_tr.softmax(dim=1)[:,1].cpu().numpy()
            s_tr = pd.Series((probs_tr>threshold).astype(int), index=idx[train_idx])

            # val
            p_val,_ = learn.get_preds(ds_idx=1)
            probs_val = p_val.softmax(dim=1)[:,1].cpu().numpy()
            s_val = pd.Series((probs_val>threshold).astype(int), index=idx[val_idx])

            # test
            dls_te = get_ts_dls(Xs, ys2, splits=(list(range(i2)), list(range(i2,n))), bs=128)
            p_te,_ = learn.get_preds(dl=dls_te.valid)
            probs_te = p_te.softmax(dim=1)[:,1].cpu().numpy()
            s_te = pd.Series((probs_te>threshold).astype(int), index=idx[test_idx])

            return s_tr, s_val, s_te

        # InceptionTime
        s_i_tr, s_i_val, s_i_te = train_ts(InceptionTime, {}, lr=1e-3, wd=1e-3)
        signals_ts_i_tr[tk], signals_ts_i_val[tk], signals_ts_i_test[tk] = s_i_tr, s_i_val, s_i_te

        logger.info(f"InceptionTime {tk}: VA acc={accuracy_score(ys2[val_idx],s_i_val):.4f}, "
                    f"TE acc={accuracy_score(ys2[test_idx],s_i_te):.4f}")

        # TSTPlus
        arch_t = dict(n_layers=3, d_model=128, n_heads=8, d_ff=256, ks=20, dropout=0.1)
        s_t_tr, s_t_val, s_t_te = train_ts(TSTPlus, arch_t, lr=1e-3, wd=1e-4)
        signals_ts_tstp_tr[tk], signals_ts_tstp_val[tk], signals_ts_tstp_test[tk] = s_t_tr, s_t_val, s_t_te

        logger.info(f"TSTPlus {tk}: VA acc={accuracy_score(ys2[val_idx],s_t_val):.4f}, "
                    f"TE acc={accuracy_score(ys2[test_idx],s_t_te):.4f}")

        # MiniRocket + SGD
        feat_model = MiniRocketFeaturesPlus(
            c_in=Xs.shape[1], seq_len=SEQ_LEN,
            num_features=25000, max_dilations_per_kernel=64, kernel_size=14
        ).cpu().float()
        X_feats = get_minirocket_features(Xs.astype(np.float32), feat_model)
        if X_feats.ndim==3 and X_feats.shape[2]==1:
            X_feats = X_feats.squeeze(-1)
        X_feats = X_feats.cpu().numpy()

        X_tr, X_va, X_te = X_feats[train_idx], X_feats[val_idx], X_feats[test_idx]
        y_tr, y_va, y_te = ys2[train_idx], ys2[val_idx], ys2[test_idx]

        clf = SGDClassifier(
            loss='log_loss', penalty='l2', class_weight='balanced',
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=12, tol=1e-4, max_iter=5000,
            random_state=42, alpha=1e-3  
        )
        clf.fit(X_tr, y_tr)
        s_mr_val = pd.Series((clf.predict_proba(X_va)[:,1]>threshold)
        .astype(int), index=idx[val_idx])
        s_mr_te  = pd.Series((clf.predict_proba(X_te)[:,1]>threshold)
        .astype(int), index=idx[test_idx])
        signals_ts_mr_val[tk], signals_ts_mr_test[tk] = s_mr_val, s_mr_te

        logger.info(f"MiniRocket {tk}: VA acc={accuracy_score(y_va,s_mr_val):.4f}, "
                    f"TE acc={accuracy_score(y_te,s_mr_te):.4f}")
        
    return (signals_ts_i_val, signals_ts_i_test, signals_ts_tstp_val, signals_ts_tstp_test, 
            signals_ts_mr_val, signals_ts_mr_test, signals_ts_i_tr, signals_ts_tstp_tr)


#                   RL агент CQL        ________________

# Расчет SMA_MACD_RSI для использования в CQL и ансамбле
best_params_sma = {
    "GOOGL": {"ss": 18, "sl": 38, "thr": 40}
}

def compute_sm_rsi_signals(price: pd.Series, ss: int, sl: int, thr: float):

    sma_s    = price.rolling(ss).mean()
    sma_l    = price.rolling(sl).mean()
    macd_raw = price.ewm(span=12).mean() - price.ewm(span=26).mean()
    macd     = macd_raw.ewm(span=9).mean()
    hist = (macd_raw - macd).fillna(0)
    rsi      = RSIIndicator(price, window=14).rsi()
    return ((sma_s > sma_l) & (hist > 0) & (rsi < 100 - thr)).astype(int)

# Построение среды CQL
def make_env(
    features: pd.DataFrame,
    price_cols: list[str] = ['open','high','low','close','volume'],
    max_episode_steps: int = 144,
    reward_scaling: float = 1
):
    price_array = features[price_cols].to_numpy(dtype=np.float32)
    tech_cols   = [c for c in features.columns if c not in price_cols + ['future','target']]
    tech_array  = features[tech_cols].to_numpy(dtype=np.float32)

    env = StockTradingEnv(
        {
            'price_array':        price_array,
            'tech_array':         tech_array,
            'turbulence_array':   np.zeros((len(features),1), dtype=np.float32),
            'stock_dim':          1,
            'tech_indicator_list': tech_cols,
            'if_train':           False
        },
        initial_account=100_000,
        buy_cost_pct=0.002,
        sell_cost_pct=0.002,
        reward_scaling=reward_scaling,
        max_stock=100
    )
    env = TimeLimit(env, max_episode_steps)

    env.observation_space = Box(-np.inf, np.inf,
                                shape=env.observation_space.shape,
                                dtype=np.float32)
    env.action_space      = Box(env.action_space.low,
                                env.action_space.high,
                                shape=env.action_space.shape,
                                dtype=np.float32)
    return env

# Генерация сигналов из обученного DiscreteCQL агента
def generate_cql_signals(
    cql_model: DiscreteCQL,
    features_pred: pd.DataFrame,
    max_episode_steps: int = 144
):
    env = make_env(features_pred, max_episode_steps=max_episode_steps)
    obs, _ = env.reset()
    signals: list[int] = []

    for _ in range(len(features_pred)):

        action_disc = cql_model.sample_action(obs[np.newaxis])[0]
        signals.append(int(action_disc))

        # конвертация в непрерывное действие для среды
        action_cont = np.array([2 * int(action_disc) - 1], dtype=np.float32)
        obs, _, term, to, _ = env.step(action_cont)
        if term or to:
            obs, _ = env.reset()

    return pd.Series(signals, index=features_pred.index)

# Сбор obs, actions, rewards, terminated, truncated
def collect_transitions_from_policy(
    features: pd.DataFrame,
    signal: pd.Series,
    price_cols: list[str] = ['open','high','low','close','volume'],
    max_episode_steps: int = 144
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    env = make_env(features, max_episode_steps=max_episode_steps)
    obs, _ = env.reset()

    obs_buf, act_buf, rew_buf = [], [], []
    term_buf, to_buf = [], []

    for dt in features.index:

        a_disc = int(signal.loc[dt])
        a_cont = np.array([2 * a_disc - 1], dtype=np.float32)

        next_obs, reward, terminated, truncated, _ = env.step(a_cont)

        obs_buf.append(obs)
        act_buf.append(np.array([a_disc], dtype=np.int64))
        rew_buf.append(reward)
        term_buf.append(terminated)
        to_buf.append(truncated)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    return (
        np.stack(obs_buf),
        np.vstack(act_buf).squeeze(-1),
        np.array(rew_buf,   dtype=np.float32),
        np.array(term_buf,  dtype=bool),
        np.array(to_buf,    dtype=bool),
    )


def generate_all_cql_signals(fe: FE, stock_tk: list[str], H: int):  
  
    signals_cql_val = {}
    signals_cql_test = {}

    for tk in stock_tk:

        feats = fe.load_features_for_ml(tk)
        if feats.empty:
            signals_cql_val [tk] = pd.Series(dtype=int)
            signals_cql_test[tk] = pd.Series(dtype=int)
            continue
        
        # train/val/test индексы
        df_ml     = FE.create_ml_target(feats, H)
        env_feats = df_ml.drop(columns=["future","target"])
        y_all     = df_ml["target"].to_numpy()
        n, i1, i2 = len(df_ml), int(0.6*len(df_ml)), int(0.8*len(df_ml))
        tr_feats, va_feats, te_feats = (
            env_feats.iloc[:i1],
            env_feats.iloc[i1:i2],
            env_feats.iloc[i2:]
        )
        y_va, y_te = y_all[i1:i2], y_all[i2:]

        # SMA_MACD_RSI сигналы на train
        price_tr = feats.loc[tr_feats.index, "close"]
        params   = best_params_sma.get(tk, {
            "ss": SMA_MACD_RSI.ss,
            "sl": SMA_MACD_RSI.sl,
            "thr": SMA_MACD_RSI.thr
        })
        sm_rsi = compute_sm_rsi_signals(
            price_tr, ss=params["ss"], sl=params["sl"], thr=params["thr"])

        base_signals = {
            'SMRSI': sm_rsi.astype(int),
            'Inc_' : signals_ts_i_tr[tk].reindex(tr_feats.index).fillna(0).astype(int),
            'TSTP' : signals_ts_tstp_tr[tk].reindex(tr_feats.index).fillna(0).astype(int),
        }

        va_preds, te_preds = [], []

        for name, sig_tr in base_signals.items():

            obs, acts, rews, terms, to = collect_transitions_from_policy(tr_feats, sig_tr)

            dataset = MDPDataset(
                observations=obs,
                actions=acts,
                rewards=rews,
                terminals=terms,
                timeouts=to
            )

            # Конфигурация DiscreteCQL
            cfg = DiscreteCQLConfig(
                learning_rate=1e-3,
                batch_size=128,
                gamma=0.99,
                n_critics=2,
                target_update_interval=200,
                alpha=1.0
            )
            discrete_cql = cfg.create()
            logger.info(f"{tk} DiscreteCQL - {name}: подготовка на {len(tr_feats)} примерах")
            discrete_cql.fit(dataset, n_steps=600, n_steps_per_epoch=200)

            # Генерация сигналов
            s_va = generate_cql_signals(discrete_cql, va_feats)
            s_te = generate_cql_signals(discrete_cql, te_feats)
            logger.info(
                f"{tk} DiscreteCQL - {name}: "
                f"VA acc={accuracy_score(y_va, s_va):.4f}, "
                f"TE acc={accuracy_score(y_te, s_te):.4f}"
            )

            va_preds.append(s_va.rename(name))
            te_preds.append(s_te.rename(name))

        # Голосование большинством
        df_va = pd.concat(va_preds, axis=1).fillna(0).astype(int)
        df_te = pd.concat(te_preds, axis=1).fillna(0).astype(int)

        vote_va = (df_va.sum(axis=1) >= (len(df_va.columns)//2 + 1)).astype(int)
        vote_te = (df_te.sum(axis=1) >= (len(df_te.columns)//2 + 1)).astype(int)

        signals_cql_val [tk] = vote_va
        signals_cql_test[tk] = vote_te
        
    return signals_cql_val, signals_cql_test


#        ВЕСА ДЛЯ АНСАМБЛЯ       _____________________________

weights = {
    'SMRSI':      0.10,
    'Cat_b':      0.10,
    'Inc_':       0.15,
    'TSTPlus':    0.20,
    'Mini_r':     0.10,
    'PatchTST':   0.10,
    'NHITS':      0.15,
    'CQL':        0.10
}

# Общая стратегия без ATR trailing stop
class Com(Strategy):
    _signals: pd.Series = pd.Series(dtype=int)

    def init(self):
        pass

    def next(self):
        price = self.data.Close[-1]

        if False:
            return

        dt  = self.data.index[-1]
        sig = type(self)._signals.get(dt, 0)
        if sig and not self.position:
            self.buy()
            self.entry_extreme = price
        elif not sig and self.position:
            self.position.close()
            self.entry_extreme = price

# Ансамбль с применением ATR trailing stop
class Ensemble_dyn_atr(_Base_ATR):
    _signals: pd.Series = pd.Series(dtype=float)

    def init(self):
        self._init_atr()

    def next(self):
        price    = self.data.Close[-1]
        if self._apply_trail(price):
            return
        dt       = self.data.index[-1]
        strength = float(type(self)._signals.get(dt, 0.0))
        if strength > 0 and not self.position:

            self.buy() 
            self.entry_extreme = price
        elif strength == 0 and self.position:
            self.position.close()
            self.entry_extreme = price

# Ансамбль без ATR trailing stop
class Ensemble(Strategy):
    _signals: pd.Series = pd.Series(dtype=float)

    def init(self):
        pass

    def next(self):
        price = self.data.Close[-1]
        dt    = self.data.index[-1]
        strength = float(type(self)._signals.get(dt, 0.0))
        if strength > 0 and not self.position:
            self.buy()
            self.entry_extreme = price
        elif strength == 0 and self.position:
            self.position.close()
            self.entry_extreme = price


WARMUP = 120  # разогрев технических индикаторов в признаках

for tk in stock_tk:

    # Разбивка на train, val, test
    df_tk = raw.xs(tk, axis=1, level=0).dropna()
    tr, va, te = split_sets(df_tk)

    # CatBoost
    cb_val  = signals_cb_val.get(tk, pd.Series(0, index=va.index))
    cb_test = signals_cb_test.get(tk, pd.Series(0, index=te.index))
    cb_sig  = pd.concat([cb_val, cb_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)

    # TSAI: InceptionTime, TSTPlus, MiniRocket
    ti_val,  ti_test  = signals_ts_i_val[tk],     signals_ts_i_test[tk]
    tt_val,  tt_test  = signals_ts_tstp_val[tk], signals_ts_tstp_test[tk]
    mr_val,  mr_test  = signals_ts_mr_val[tk],   signals_ts_mr_test[tk]
    ti_sig = pd.concat([ti_val, ti_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)
    tt_sig = pd.concat([tt_val, tt_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)
    mr_sig = pd.concat([mr_val, mr_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)

    # PatchTST
    pt_val, pt_test = signals_ptst_val[tk], signals_ptst_test[tk]
    pt_sig = pd.concat([pt_val, pt_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)

    # NHITS
    nh_val, nh_test = signals_nhits_val[tk], signals_nhits_test[tk]
    nh_sig = pd.concat([nh_val, nh_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)

    # CQL
    cq_val, cq_test = signals_cql_val[tk], signals_cql_test[tk]
    cq_sig = pd.concat([cq_val, cq_test]).reindex(df_tk.index).ffill().fillna(0).astype(int)

    # весь ряд цен (Close) по тикеру
    price = df_tk["Close"]

    # SMA_MACD_RSI
    params = best_params_sma.get(tk, {
        "ss":  SMA_MACD_RSI.ss,
        "sl":  SMA_MACD_RSI.sl,
        "thr": SMA_MACD_RSI.thr
    })
    sm_rsi = compute_sm_rsi_signals(
        price,
        ss = params["ss"],
        sl = params["sl"],
        thr= params["thr"]
    )

    # Формирование сигналов
    sig_df = pd.concat({
        'SMRSI':    sm_rsi.astype(int),
        'BuyHold':  pd.Series(1.0, index=df_tk.index),
        'Cat_b':    cb_sig.astype(int),
        'Inc_':     ti_sig.astype(int),
        'TSTPlus':  tt_sig.astype(int),
        'Mini_r':   mr_sig.astype(int),
        'PatchTST': pt_sig.astype(int),
        'NHITS':    nh_sig.astype(int),
        'CQL':      cq_sig.astype(int),
    }, axis=1).fillna(0.0).astype(int)


    # Сигнал каждой стратегии умножаем на ее вес и суммируем:
    raw_vote = sum(sig_df[col] * weights.get(col, 0.0) for col in sig_df.columns)

    # Итоговая доля от суммы всех голосов:
    pct_vote = raw_vote / sum(weights.values())

    # Порог отбора в ансамбле
    threshold = 0.4
    sig_df['Ensemble'] = pct_vote.where(pct_vote >= threshold, 0.0)
    sig_df['Ens_dyn_atr'] = pct_vote.where(pct_vote >= threshold, 0.0)
    sig_df = sig_df.shift(1).ffill().fillna(0.0)

    wf = []; val_stats = []; test_stats = []
    periods = [
        ("Val",  pd.concat([tr.iloc[-WARMUP:], va]), va),
        ("Test", pd.concat([va.iloc[-WARMUP:], te]),  te)
    ]


    # header = Div(text=f"<h2>{tk}: Ensemble валидация и тест</h2>", width=1200)
    # show(header)
    for name, series in sig_df.items():
        for period_name, df_bt, df_main in periods:
            sig_bt = series.reindex(df_bt.index).fillna(0)

            if name == 'Ensemble':
                Ensemble._signals = sig_bt.astype(float)
                Str = Ensemble
            elif name == 'Ens_dyn_atr':
                Ensemble_dyn_atr._signals = sig_bt.astype(float)
                Str = Ensemble_dyn_atr
            elif name == 'BuyHold':
                Str = BuyHold
            else:
                Com._signals = sig_bt.astype(int)
                Str = Com

            bt = Backtest(
                df_bt, Str,
                cash=100_000, commission=0.002, exclusive_orders=True
            )
            res = bt.run()
            # Equity на период df_main
            eq = res._equity_curve["Equity"].loc[df_main.index]
            eq_norm = eq / eq.iloc[0]
            # Trades на том же индексе
            trades = res._trades["PnL"]
            if isinstance(trades.index, pd.DatetimeIndex):
                trades = trades.loc[df_main.index]

            # Метрики
            m = compute_metrics(
                eq_norm,
                eq_norm.pct_change().dropna(),
                trades.dropna()
            )
            m.update(name=name, period=period_name)
            if period_name == "Val":
                val_stats.append(m)
            else:
                test_stats.append(m)

            wf.append({
                "name": name,
                "period": period_name,
                "dates": eq_norm.index,
                "equity": eq_norm.values
            })
            # Все сделки для этого периода
            # print(f"\n  {name} — {period_name}")
            # trades = res._trades.copy()
            # if trades.empty:
            #     print("  No trades\n")
            # else:
            #     trades = trades[['EntryTime','ExitTime','Size','EntryPrice','ExitPrice','PnL']]
            #     print(trades.to_string(index=False), "\n")
            # if name == 'Ensemble':
            #     bokeh_plot = bt.plot(plot_volume=True, plot_width=1200)

    # Итоговые данные валидации и теста
    df_val  = pd.DataFrame(val_stats)
    df_test = pd.DataFrame(test_stats)


    fig = px.bar(
        df_val,
        x="name",
        y=[
            "Sharpe Ratio",
            "Total Return [%]",
            "Max. Drawdown [%]",
            "Sortino Ratio",
            "Win Rate [%]",
            "Volatility [%]"
        ],
        barmode="group",
        title=f"{tk}: Валидация {va.index[0].date()}–{va.index[-1].date()}"
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title_text=None)
    fig.show()


    # Таблица метрик
    cols = [
        "name", "Total Return [%]", "CAGR [%]", "Volatility [%]",
        "Sharpe Ratio", "Sortino Ratio", "Max. Drawdown [%]",
        "Calmar Ratio", "Win Rate [%]", "Profit Factor", "# Trades"
    ]
    tbl = df_val[cols].round(4)
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=tbl.columns,
                fill_color="#f3f5f7",
                align="left"
            ),
            cells=dict(
                values=[tbl[c] for c in cols],
                fill_color="#f6faff",
                align="left"
            )
        )]
    )
    fig.update_layout(
        title_text=f"{tk}:  Метрики Валидации",
        margin=dict(l=30, r=40, t=50)
    )
    fig.show()


    fig = px.bar(
        df_test, x="name",
        y=[
            "Sharpe Ratio",
            "Total Return [%]",
            "Max. Drawdown [%]",
            "Sortino Ratio",
            "Win Rate [%]",
            "Volatility [%]"
        ],
        barmode="group",
        title=f"{tk}: Тест {te.index[0].date()}–{te.index[-1].date()}"
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title_text=None)
    fig.show()


    # Таблица метрик
    cols = [
        "name", "Total Return [%]", "CAGR [%]", "Volatility [%]",
        "Sharpe Ratio", "Sortino Ratio", "Max. Drawdown [%]",
        "Calmar Ratio", "Win Rate [%]", "Profit Factor", "# Trades"
    ]
    tbl = df_test[cols].round(4)
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=tbl.columns,
                fill_color="#f3f5f7",
                align="left"
            ),
            cells=dict(
                values=[tbl[c] for c in cols],
                fill_color="#f6faff",
                align="left"
            )
        )]
    )

    fig.update_layout(
        title_text=f"{tk}:  Метрики Теста",
        margin=dict(l=30, r=40, t=50)
    )
    fig.show()


    uniq = list(dict.fromkeys(rec["name"] for rec in wf))
    palette = px.colors.qualitative.Plotly
    cmap    = {n: palette[i % len(palette)] for i,n in enumerate(uniq)}

    fig = go.Figure()
    seen = set()
    for m in wf:
        fig.add_trace(go.Scatter(
            x=m["dates"],
            y=m["equity"],
            mode="lines",
            name=m["name"],
            legendgroup=m["name"],
            showlegend=m["name"] not in seen,
            line=dict(color=cmap[m["name"]])
        ))
        seen.add(m["name"])
    fig.update_layout(
        title=(f"{tk}: валидация {va.index[0].date()}–{va.index[-1].date()} | "
              f"тест {te.index[0].date()}–{te.index[-1].date()}"),
        hovermode="x unified",
        legend=dict(traceorder="normal")
    )
    fig.show()
    
    """
    1. Лучший вариант ансамбля: Ensemble без применения динамического ATR trailing stop
    
    2. В этом варианте Ensemble с порогом голосования 0.4 метрики:
        Total Return: 14.1,  Win Rate: 50.0   на валидации
        Total Return: 27.8,  Win Rate: 62.5   на тесте
        
    3. В варианте с порогом голосования 0.5 метрики:
        Total Return: 12.0,  Win Rate: 83.3   на валидации
        Total Return: 22.1,  Win Rate: 88.9   на тесте
    """
