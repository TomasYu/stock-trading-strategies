#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20周线趋势跟踪策略 - 三种优化方案对比回测

方案A（基准）：回踩20周线买入 + 跌破20周线立即止损
方案B：回踩20周线买入 + 连续2周跌破才止损
方案C：放量突破20周线买入 + 止损设在突破前低点
方案D：回踩买入 + MACD周线金叉 + 10周线在20周线上方
"""

import akshare as ak
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置参数 ============
INITIAL_CAPITAL = 40000
MAX_SINGLE_POSITION = 0.60
MONTHLY_LOSS_LIMIT = 0.06
MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
PROFIT_HALF_EXIT = 0.20
PROFIT_ALL_EXIT = 0.30
ADD_POSITION_TRIGGER = 0.10
WEAK_HOLD_WEEKS = 12
WEAK_HOLD_RETURN = 0.05


def get_weekly_data(symbol, start_date, end_date, is_index=False):
    """获取周线数据"""
    try:
        if is_index:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高',
                                     'low': '最低', 'close': '收盘', 'volume': '成交量'})
        else:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                      start_date=start_date.replace('-', ''),
                                      end_date=end_date.replace('-', ''),
                                      adjust="qfq")
            df = df.rename(columns={'日期': '日期', '开盘': '开盘', '最高': '最高',
                                     '最低': '最低', '收盘': '收盘', '成交量': '成交量'})

        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
        df = df.sort_values('日期').reset_index(drop=True)

        df.set_index('日期', inplace=True)
        weekly = df.resample('W-FRI').agg({
            '开盘': 'first', '最高': 'max', '最低': 'min',
            '收盘': 'last', '成交量': 'sum'
        }).dropna()
        weekly.reset_index(inplace=True)

        # 均线
        weekly['MA10'] = weekly['收盘'].rolling(window=10).mean()
        weekly['MA20'] = weekly['收盘'].rolling(window=MA_PERIOD).mean()
        weekly['MA20_slope'] = weekly['MA20'] - weekly['MA20'].shift(1)
        weekly['VOL_MA5'] = weekly['成交量'].rolling(window=5).mean()

        # MACD（周线级别：12, 26, 9）
        ema12 = weekly['收盘'].ewm(span=12, adjust=False).mean()
        ema26 = weekly['收盘'].ewm(span=26, adjust=False).mean()
        weekly['DIF'] = ema12 - ema26
        weekly['DEA'] = weekly['DIF'].ewm(span=9, adjust=False).mean()
        weekly['MACD'] = 2 * (weekly['DIF'] - weekly['DEA'])

        # 近N周最低价（用于方案C止损）
        weekly['LOW_8W'] = weekly['最低'].rolling(window=8).min()

        return weekly
    except Exception as e:
        print(f"  获取 {symbol} 数据失败: {e}")
        return None


class Position:
    """持仓"""
    def __init__(self, symbol, name, buy_price, shares, buy_date, buy_week_idx, stop_price=None):
        self.symbol = symbol
        self.name = name
        self.buy_price = buy_price
        self.avg_price = buy_price
        self.shares = shares
        self.buy_date = buy_date
        self.buy_week_idx = buy_week_idx
        self.added = False
        self.half_sold = False
        self.cost = buy_price * shares
        self.stop_price = stop_price          # 方案C用的固定止损价
        self.weeks_below_ma20 = 0             # 方案B用的连续跌破周数


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital, strategy_name):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_log = []
        self.weekly_equity = []
        self.monthly_start_equity = initial_capital
        self.current_month = None
        self.monthly_paused = False
        self.strategy_name = strategy_name

    def get_total_equity(self, current_prices):
        equity = self.cash
        for sym, pos in self.positions.items():
            if sym in current_prices:
                equity += current_prices[sym] * pos.shares
        return equity

    def get_position_ratio(self, current_prices):
        equity = self.get_total_equity(current_prices)
        if equity <= 0:
            return 0
        position_value = sum(current_prices.get(sym, 0) * pos.shares
                           for sym, pos in self.positions.items())
        return position_value / equity

    def check_monthly_loss(self, current_equity, current_date):
        month_key = current_date.strftime('%Y-%m')
        if self.current_month != month_key:
            self.current_month = month_key
            self.monthly_start_equity = current_equity
            self.monthly_paused = False
            return False
        if self.monthly_start_equity > 0:
            monthly_return = (current_equity - self.monthly_start_equity) / self.monthly_start_equity
            if monthly_return < -MONTHLY_LOSS_LIMIT:
                self.monthly_paused = True
                return True
        return self.monthly_paused

    def buy(self, symbol, name, price, shares, date, week_idx, reason="", stop_price=None):
        cost = price * shares
        if cost > self.cash:
            shares = int(self.cash / price / 100) * 100
            if shares <= 0:
                return
            cost = price * shares

        self.cash -= cost
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_cost = pos.avg_price * pos.shares + cost
            pos.shares += shares
            pos.avg_price = total_cost / pos.shares
            pos.cost = total_cost
            pos.added = True
        else:
            self.positions[symbol] = Position(symbol, name, price, shares, date, week_idx, stop_price)

        self.trade_log.append({
            '日期': date, '操作': '买入', '标的': name, '代码': symbol,
            '价格': price, '数量': shares, '金额': cost, '原因': reason
        })

    def sell(self, symbol, price, shares, date, reason=""):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        sell_shares = min(shares, pos.shares)
        revenue = price * sell_shares
        self.cash += revenue
        profit = (price - pos.avg_price) * sell_shares
        self.trade_log.append({
            '日期': date, '操作': '卖出', '标的': pos.name, '代码': symbol,
            '价格': price, '数量': sell_shares, '金额': revenue,
            '盈亏': round(profit, 2), '原因': reason
        })
        pos.shares -= sell_shares
        if pos.shares <= 0:
            del self.positions[symbol]

    def sell_all(self, symbol, price, date, reason=""):
        if symbol in self.positions:
            self.sell(symbol, price, self.positions[symbol].shares, date, reason)


def calc_buy_shares(engine, price, stop_loss_pct, current_prices):
    """计算买入股数"""
    equity = engine.get_total_equity(current_prices)
    max_amount = equity * MAX_SINGLE_POSITION
    stop_loss_pct = max(stop_loss_pct, 0.01)
    risk_amount = equity * 0.03
    risk_based_shares = int(risk_amount / (price * stop_loss_pct) / 100) * 100
    max_shares = int(max_amount / price / 100) * 100
    available_cash = engine.cash - equity * 0.10
    cash_shares = int(available_cash / price / 100) * 100
    shares = min(risk_based_shares, max_shares, cash_shares)
    return shares if shares >= 100 else 0


def had_volume_breakout(data, current_idx, lookback=8):
    """检查过去N周是否有放量突破20周线"""
    start = max(0, current_idx - lookback)
    for i in range(start, current_idx):
        row = data.iloc[i]
        if (not pd.isna(row['VOL_MA5']) and not pd.isna(row['MA20'])
                and row['成交量'] > row['VOL_MA5'] * VOLUME_MULTIPLIER
                and row['收盘'] > row['MA20']):
            return True
    return False


def get_strength(data, current_date, weeks=10):
    """计算近N周相对涨幅"""
    subset = data[data['日期'] <= current_date].tail(weeks + 1)
    if len(subset) >= 2:
        return (subset.iloc[-1]['收盘'] - subset.iloc[0]['收盘']) / subset.iloc[0]['收盘']
    return 0


# ============ 方案A：基准（回踩买 + 立即止损） ============
def strategy_a_signals(engine, etf_data, current_date, prev_date, week_idx,
                       index_above_ma20, current_prices):
    """方案A：回踩20周线买入 + 跌破20周线立即止损"""
    # 卖出
    for sym in list(engine.positions.keys()):
        if sym not in engine.positions or sym not in current_prices:
            continue
        pos = engine.positions[sym]
        price = current_prices[sym]
        data = etf_data[sym][1]
        matched = data[data['日期'] == current_date]
        if len(matched) == 0:
            continue
        row = matched.iloc[0]
        ma20 = row['MA20']
        if pd.isna(ma20):
            continue

        # 止损：跌破20周线
        if price < ma20:
            engine.sell_all(sym, price, current_date, "跌破20周线止损")
            continue

        # 止盈逻辑
        deviation = (price - ma20) / ma20
        if deviation > PROFIT_ALL_EXIT:
            engine.sell_all(sym, price, current_date, f"远离20周线{deviation:.1%}全部止盈")
            continue
        profit_pct = (price - pos.avg_price) / pos.avg_price
        if profit_pct >= PROFIT_HALF_EXIT and not pos.half_sold:
            half = (pos.shares // 2 // 100) * 100
            if half >= 100:
                engine.sell(sym, price, half, current_date, f"盈利{profit_pct:.1%}卖一半")
                pos.half_sold = True
            continue
        hold_weeks = week_idx - pos.buy_week_idx
        if hold_weeks >= WEAK_HOLD_WEEKS and profit_pct < WEAK_HOLD_RETURN:
            engine.sell_all(sym, price, current_date, f"持有{hold_weeks}周涨幅{profit_pct:.1%}弱势换股")

    # 买入
    if index_above_ma20 and not engine.monthly_paused and len(engine.positions) < 2:
        candidates = []
        for sym, (name, data) in etf_data.items():
            if sym in engine.positions:
                continue
            matched = data[data['日期'] == current_date]
            if len(matched) == 0:
                continue
            row = matched.iloc[0]
            ma20, slope, price = row['MA20'], row['MA20_slope'], row['收盘']
            if pd.isna(ma20) or pd.isna(slope):
                continue
            if slope < 0 or price < ma20:
                continue
            distance = (price - ma20) / ma20
            if distance > 0.03:
                continue
            idx = matched.index[0]
            if not had_volume_breakout(data, idx):
                continue
            strength = get_strength(data, current_date)
            candidates.append((sym, name, price, ma20, strength - distance, distance))

        candidates.sort(key=lambda x: x[4], reverse=True)
        for sym, name, price, ma20, score, distance in candidates:
            if len(engine.positions) >= 2:
                break
            shares = calc_buy_shares(engine, price, distance, current_prices)
            if shares >= 100:
                engine.buy(sym, name, price, shares, current_date, week_idx,
                          f"回踩20周线{distance:.1%}买入")


# ============ 方案B：连续2周跌破才止损 ============
def strategy_b_signals(engine, etf_data, current_date, prev_date, week_idx,
                       index_above_ma20, current_prices):
    """方案B：回踩买入 + 连续2周收盘跌破20周线才止损"""
    # 卖出
    for sym in list(engine.positions.keys()):
        if sym not in engine.positions or sym not in current_prices:
            continue
        pos = engine.positions[sym]
        price = current_prices[sym]
        data = etf_data[sym][1]
        matched = data[data['日期'] == current_date]
        if len(matched) == 0:
            continue
        row = matched.iloc[0]
        ma20 = row['MA20']
        if pd.isna(ma20):
            continue

        # 连续2周跌破才止损
        if price < ma20:
            pos.weeks_below_ma20 += 1
            if pos.weeks_below_ma20 >= 2:
                engine.sell_all(sym, price, current_date, "连续2周跌破20周线止损")
                continue
        else:
            pos.weeks_below_ma20 = 0

        # 止盈
        deviation = (price - ma20) / ma20
        if deviation > PROFIT_ALL_EXIT:
            engine.sell_all(sym, price, current_date, f"远离20周线{deviation:.1%}全部止盈")
            continue
        profit_pct = (price - pos.avg_price) / pos.avg_price
        if profit_pct >= PROFIT_HALF_EXIT and not pos.half_sold:
            half = (pos.shares // 2 // 100) * 100
            if half >= 100:
                engine.sell(sym, price, half, current_date, f"盈利{profit_pct:.1%}卖一半")
                pos.half_sold = True
            continue
        hold_weeks = week_idx - pos.buy_week_idx
        if hold_weeks >= WEAK_HOLD_WEEKS and profit_pct < WEAK_HOLD_RETURN:
            engine.sell_all(sym, price, current_date, f"持有{hold_weeks}周涨幅{profit_pct:.1%}弱势换股")

    # 买入（和方案A一样）
    if index_above_ma20 and not engine.monthly_paused and len(engine.positions) < 2:
        candidates = []
        for sym, (name, data) in etf_data.items():
            if sym in engine.positions:
                continue
            matched = data[data['日期'] == current_date]
            if len(matched) == 0:
                continue
            row = matched.iloc[0]
            ma20, slope, price = row['MA20'], row['MA20_slope'], row['收盘']
            if pd.isna(ma20) or pd.isna(slope):
                continue
            if slope < 0 or price < ma20:
                continue
            distance = (price - ma20) / ma20
            if distance > 0.03:
                continue
            idx = matched.index[0]
            if not had_volume_breakout(data, idx):
                continue
            strength = get_strength(data, current_date)
            candidates.append((sym, name, price, ma20, strength - distance, distance))

        candidates.sort(key=lambda x: x[4], reverse=True)
        for sym, name, price, ma20, score, distance in candidates:
            if len(engine.positions) >= 2:
                break
            shares = calc_buy_shares(engine, price, distance, current_prices)
            if shares >= 100:
                engine.buy(sym, name, price, shares, current_date, week_idx,
                          f"回踩20周线{distance:.1%}买入")


# ============ 方案C：突破买入 + 前低止损 ============
def strategy_c_signals(engine, etf_data, current_date, prev_date, week_idx,
                       index_above_ma20, current_prices):
    """方案C：放量突破20周线买入 + 止损设在突破前8周最低点"""
    # 卖出
    for sym in list(engine.positions.keys()):
        if sym not in engine.positions or sym not in current_prices:
            continue
        pos = engine.positions[sym]
        price = current_prices[sym]
        data = etf_data[sym][1]
        matched = data[data['日期'] == current_date]
        if len(matched) == 0:
            continue
        row = matched.iloc[0]
        ma20 = row['MA20']
        if pd.isna(ma20):
            continue

        # 止损：跌破固定止损价（突破前8周低点）
        if pos.stop_price and price < pos.stop_price:
            engine.sell_all(sym, price, current_date,
                          f"跌破止损价{pos.stop_price:.3f}")
            continue

        # 额外止损：跌破20周线也走
        if price < ma20:
            engine.sell_all(sym, price, current_date, "跌破20周线止损")
            continue

        # 止盈
        deviation = (price - ma20) / ma20
        if deviation > PROFIT_ALL_EXIT:
            engine.sell_all(sym, price, current_date, f"远离20周线{deviation:.1%}全部止盈")
            continue
        profit_pct = (price - pos.avg_price) / pos.avg_price
        if profit_pct >= PROFIT_HALF_EXIT and not pos.half_sold:
            half = (pos.shares // 2 // 100) * 100
            if half >= 100:
                engine.sell(sym, price, half, current_date, f"盈利{profit_pct:.1%}卖一半")
                pos.half_sold = True
            continue
        hold_weeks = week_idx - pos.buy_week_idx
        if hold_weeks >= WEAK_HOLD_WEEKS and profit_pct < WEAK_HOLD_RETURN:
            engine.sell_all(sym, price, current_date, f"持有{hold_weeks}周涨幅{profit_pct:.1%}弱势换股")

    # 买入：放量突破20周线
    if index_above_ma20 and not engine.monthly_paused and len(engine.positions) < 2:
        candidates = []
        for sym, (name, data) in etf_data.items():
            if sym in engine.positions:
                continue
            matched = data[data['日期'] == current_date]
            prev_matched = data[data['日期'] == prev_date]
            if len(matched) == 0 or len(prev_matched) == 0:
                continue
            row = matched.iloc[0]
            prev_row = prev_matched.iloc[0]
            ma20, slope = row['MA20'], row['MA20_slope']
            price, vol, vol_ma5 = row['收盘'], row['成交量'], row['VOL_MA5']
            prev_price, prev_ma20 = prev_row['收盘'], prev_row['MA20']
            low_8w = row['LOW_8W']

            if pd.isna(ma20) or pd.isna(slope) or pd.isna(vol_ma5) or pd.isna(prev_ma20) or pd.isna(low_8w):
                continue

            # 条件1：20周线走平或向上
            if slope < 0:
                continue

            # 条件2：本周突破（上周在20周线下方或附近，本周站上）
            prev_above = prev_price > prev_ma20 * 1.02  # 上周已经明显在上方就不算突破
            current_above = price > ma20
            if prev_above or not current_above:
                continue

            # 条件3：放量
            if vol < vol_ma5 * VOLUME_MULTIPLIER:
                continue

            # 止损价 = 前8周最低价
            stop_price = low_8w * 0.98  # 再留2%缓冲
            stop_loss_pct = (price - stop_price) / price

            strength = get_strength(data, current_date)
            candidates.append((sym, name, price, stop_price, stop_loss_pct, strength))

        candidates.sort(key=lambda x: x[5], reverse=True)
        for sym, name, price, stop_price, stop_loss_pct, strength in candidates:
            if len(engine.positions) >= 2:
                break
            shares = calc_buy_shares(engine, price, stop_loss_pct, current_prices)
            if shares >= 100:
                engine.buy(sym, name, price, shares, current_date, week_idx,
                          f"放量突破20周线，止损{stop_price:.3f}", stop_price=stop_price)


# ============ 方案D：回踩 + MACD金叉 + MA10>MA20 ============
def strategy_d_signals(engine, etf_data, current_date, prev_date, week_idx,
                       index_above_ma20, current_prices):
    """方案D：回踩买入 + MACD周线金叉确认 + 10周线在20周线上方"""
    # 卖出（和方案B一样用连续2周）
    for sym in list(engine.positions.keys()):
        if sym not in engine.positions or sym not in current_prices:
            continue
        pos = engine.positions[sym]
        price = current_prices[sym]
        data = etf_data[sym][1]
        matched = data[data['日期'] == current_date]
        if len(matched) == 0:
            continue
        row = matched.iloc[0]
        ma20 = row['MA20']
        if pd.isna(ma20):
            continue

        # 连续2周跌破才止损
        if price < ma20:
            pos.weeks_below_ma20 += 1
            if pos.weeks_below_ma20 >= 2:
                engine.sell_all(sym, price, current_date, "连续2周跌破20周线止损")
                continue
        else:
            pos.weeks_below_ma20 = 0

        # 额外止损：MACD死叉
        ma10 = row['MA10']
        if not pd.isna(ma10) and ma10 < ma20 and price < ma20:
            engine.sell_all(sym, price, current_date, "MA10下穿MA20+跌破均线止损")
            continue

        # 止盈
        deviation = (price - ma20) / ma20
        if deviation > PROFIT_ALL_EXIT:
            engine.sell_all(sym, price, current_date, f"远离20周线{deviation:.1%}全部止盈")
            continue
        profit_pct = (price - pos.avg_price) / pos.avg_price
        if profit_pct >= PROFIT_HALF_EXIT and not pos.half_sold:
            half = (pos.shares // 2 // 100) * 100
            if half >= 100:
                engine.sell(sym, price, half, current_date, f"盈利{profit_pct:.1%}卖一半")
                pos.half_sold = True
            continue
        hold_weeks = week_idx - pos.buy_week_idx
        if hold_weeks >= WEAK_HOLD_WEEKS and profit_pct < WEAK_HOLD_RETURN:
            engine.sell_all(sym, price, current_date, f"持有{hold_weeks}周涨幅{profit_pct:.1%}弱势换股")

    # 买入：回踩 + MACD金叉 + MA10 > MA20
    if index_above_ma20 and not engine.monthly_paused and len(engine.positions) < 2:
        candidates = []
        for sym, (name, data) in etf_data.items():
            if sym in engine.positions:
                continue
            matched = data[data['日期'] == current_date]
            prev_matched = data[data['日期'] == prev_date]
            if len(matched) == 0 or len(prev_matched) == 0:
                continue
            row = matched.iloc[0]
            prev_row = prev_matched.iloc[0]
            ma20, ma10, slope = row['MA20'], row['MA10'], row['MA20_slope']
            price = row['收盘']
            dif, dea = row['DIF'], row['DEA']
            prev_dif, prev_dea = prev_row['DIF'], prev_row['DEA']

            if any(pd.isna(v) for v in [ma20, ma10, slope, dif, dea, prev_dif, prev_dea]):
                continue

            # 条件1：20周线向上
            if slope < 0:
                continue
            # 条件2：股价在20周线上方
            if price < ma20:
                continue
            # 条件3：回踩20周线附近（5%以内，比方案A放宽一点）
            distance = (price - ma20) / ma20
            if distance > 0.05:
                continue
            # 条件4：MA10 > MA20（短期趋势也向上）
            if ma10 < ma20:
                continue
            # 条件5：MACD金叉或DIF在零轴上方
            macd_ok = (dif > dea and prev_dif <= prev_dea) or (dif > 0)
            if not macd_ok:
                continue
            # 条件6：之前有过放量突破
            idx = matched.index[0]
            if not had_volume_breakout(data, idx):
                continue

            strength = get_strength(data, current_date)
            candidates.append((sym, name, price, ma20, strength - distance, distance))

        candidates.sort(key=lambda x: x[4], reverse=True)
        for sym, name, price, ma20, score, distance in candidates:
            if len(engine.positions) >= 2:
                break
            shares = calc_buy_shares(engine, price, distance, current_prices)
            if shares >= 100:
                engine.buy(sym, name, price, shares, current_date, week_idx,
                          f"MACD确认+回踩20周线{distance:.1%}买入")


def run_single_backtest(strategy_func, strategy_name, etf_data, index_data, start_date, end_date):
    """运行单个策略回测"""
    engine = BacktestEngine(INITIAL_CAPITAL, strategy_name)
    all_weeks = index_data[index_data['MA20'].notna()].reset_index(drop=True)

    for week_idx in range(1, len(all_weeks)):
        week = all_weeks.iloc[week_idx]
        prev_week = all_weeks.iloc[week_idx - 1]
        current_date = week['日期']
        prev_date = prev_week['日期']

        index_above_ma20 = week['收盘'] > week['MA20']

        current_prices = {}
        for sym, (name, data) in etf_data.items():
            matched = data[data['日期'] == current_date]
            if len(matched) > 0:
                current_prices[sym] = matched.iloc[0]['收盘']

        current_equity = engine.get_total_equity(current_prices)
        engine.check_monthly_loss(current_equity, current_date)

        # 执行策略
        strategy_func(engine, etf_data, current_date, prev_date, week_idx,
                      index_above_ma20, current_prices)

        # 加仓逻辑（通用）
        for sym in list(engine.positions.keys()):
            if sym not in engine.positions:
                continue
            pos = engine.positions[sym]
            if pos.added or sym not in current_prices:
                continue
            price = current_prices[sym]
            profit_pct = (price - pos.avg_price) / pos.avg_price
            if profit_pct >= ADD_POSITION_TRIGGER:
                data = etf_data[sym][1]
                matched = data[data['日期'] == current_date]
                if len(matched) == 0:
                    continue
                row = matched.iloc[0]
                ma20 = row['MA20']
                if pd.isna(ma20):
                    continue
                distance = (price - ma20) / ma20
                if distance <= 0.06:
                    equity = engine.get_total_equity(current_prices)
                    current_val = price * pos.shares
                    max_add = equity * MAX_SINGLE_POSITION - current_val
                    if max_add > 0:
                        add_shares = min(
                            int(max_add / price / 100) * 100,
                            int((engine.cash - equity * 0.10) / price / 100) * 100
                        )
                        if add_shares >= 100:
                            engine.buy(sym, pos.name, price, add_shares, current_date, week_idx,
                                     f"盈利{profit_pct:.1%}加仓")

        equity = engine.get_total_equity(current_prices)
        engine.weekly_equity.append({
            '日期': current_date,
            '净值': equity,
            '收益率': (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            '持仓数': len(engine.positions),
            '仓位比': engine.get_position_ratio(current_prices)
        })

    return engine


def analyze_engine(engine):
    """分析回测结果，返回统计字典"""
    if not engine.weekly_equity:
        return None

    eq_df = pd.DataFrame(engine.weekly_equity)
    final = eq_df.iloc[-1]['净值']
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    weeks = len(eq_df)
    years = weeks / 52
    annual_ret = (final / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 and final > 0 else 0

    running_max = eq_df['净值'].cummax()
    drawdown = (eq_df['净值'] - running_max) / running_max
    max_dd = drawdown.min()

    trade_df = pd.DataFrame(engine.trade_log)
    sells = trade_df[trade_df['操作'] == '卖出'] if len(trade_df) > 0 else pd.DataFrame()
    if len(sells) > 0 and '盈亏' in sells.columns:
        wins = sells[sells['盈亏'] > 0]
        losses = sells[sells['盈亏'] <= 0]
        win_rate = len(wins) / len(sells)
        avg_win = wins['盈亏'].mean() if len(wins) > 0 else 0
        avg_loss = losses['盈亏'].mean() if len(losses) > 0 else 0
        total_pnl = sells['盈亏'].sum()
    else:
        win_rate = avg_win = avg_loss = total_pnl = 0

    avg_pos = eq_df['仓位比'].mean()
    buy_count = len(trade_df[trade_df['操作'] == '买入']) if len(trade_df) > 0 else 0
    sell_count = len(sells)

    return {
        '策略': engine.strategy_name,
        '最终资金': final,
        '总收益率': total_ret,
        '年化收益': annual_ret,
        '最大回撤': max_dd,
        '胜率': win_rate,
        '盈亏比': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
        '平均盈利': avg_win,
        '平均亏损': avg_loss,
        '交易次数': buy_count + sell_count,
        '平均仓位': avg_pos,
        '已实现盈亏': total_pnl,
        'equity_df': eq_df
    }


def main():
    start_date = '2020-01-01'
    end_date = '2025-12-31'

    etf_list = [
        ('510300', '沪深300ETF'), ('510500', '中证500ETF'),
        ('159915', '创业板ETF'), ('512000', '券商ETF'),
        ('512480', '半导体ETF'), ('510880', '红利ETF'),
        ('512010', '医药ETF'), ('515030', '新能源车ETF'),
    ]

    print(f"\n{'='*70}")
    print(f"  20周线趋势跟踪策略 - 四种方案对比回测")
    print(f"  资金: {INITIAL_CAPITAL:,.0f}元 | 周期: {start_date} ~ {end_date}")
    print(f"{'='*70}\n")

    # 获取数据
    print("获取上证指数数据...")
    index_data = get_weekly_data('sh000001', start_date, end_date, is_index=True)
    if index_data is None or len(index_data) < MA_PERIOD + 5:
        print("上证指数数据不足")
        return

    etf_data = {}
    for symbol, name in etf_list:
        print(f"获取 {name}({symbol})...")
        data = get_weekly_data(symbol, start_date, end_date)
        if data is not None and len(data) > MA_PERIOD + 5:
            etf_data[symbol] = (name, data)

    print(f"\n有效标的: {len(etf_data)}个，开始回测...\n")

    # 跑四个方案
    strategies = [
        (strategy_a_signals, "A:基准(回踩+立即止损)"),
        (strategy_b_signals, "B:回踩+连续2周止损"),
        (strategy_c_signals, "C:突破买入+前低止损"),
        (strategy_d_signals, "D:MACD确认+多重过滤"),
    ]

    results = []
    for func, name in strategies:
        print(f"  运行 {name}...")
        engine = run_single_backtest(func, name, etf_data, index_data, start_date, end_date)
        stats = analyze_engine(engine)
        if stats:
            results.append(stats)
            # 打印交易记录
            print(f"\n  [{name}] 交易记录:")
            for t in engine.trade_log:
                pnl = f" 盈亏:{t['盈亏']:+,.0f}" if '盈亏' in t and pd.notna(t.get('盈亏')) else ""
                print(f"    {t['日期'].strftime('%Y-%m-%d')} {t['操作']} {t['标的']} "
                      f"{t['数量']}股 @{t['价格']:.3f}{pnl}  [{t['原因']}]")
            print()

    # 上证指数同期
    idx_start_val = index_data[index_data['日期'] >= pd.to_datetime(start_date)].iloc[0]['收盘']
    idx_end_val = index_data.iloc[-1]['收盘']
    index_return = (idx_end_val - idx_start_val) / idx_start_val

    # ====== 打印对比表 ======
    print(f"\n{'='*70}")
    print(f"  对比结果（上证同期: {index_return:.2%}）")
    print(f"{'='*70}")
    header = f"{'指标':<14} | " + " | ".join(f"{r['策略']:<22}" for r in results)
    print(header)
    print("-" * len(header))

    rows = [
        ('最终资金', lambda r: f"{r['最终资金']:>12,.0f}元"),
        ('总收益率', lambda r: f"{r['总收益率']:>12.2%}  "),
        ('年化收益', lambda r: f"{r['年化收益']:>12.2%}  "),
        ('最大回撤', lambda r: f"{r['最大回撤']:>12.2%}  "),
        ('超额收益', lambda r: f"{r['总收益率']-index_return:>12.2%}  "),
        ('胜率', lambda r: f"{r['胜率']:>12.2%}  "),
        ('盈亏比', lambda r: f"{r['盈亏比']:>12.2f}  "),
        ('交易次数', lambda r: f"{r['交易次数']:>12}笔 "),
        ('平均仓位', lambda r: f"{r['平均仓位']:>12.2%}  "),
        ('已实现盈亏', lambda r: f"{r['已实现盈亏']:>12,.0f}元"),
    ]
    for label, fmt in rows:
        line = f"{label:<14} | " + " | ".join(f"{fmt(r):<22}" for r in results)
        print(line)

    # ====== 画图 ======
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('20周线趋势跟踪策略 - 四种方案对比回测（4万资金）', fontsize=16, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    ax1 = axes[0]

    # 上证指数
    idx_weekly = index_data.set_index('日期').resample('W-FRI').last().dropna()
    idx_norm = idx_weekly['收盘'] / idx_start_val
    ax1.plot(idx_norm.index, idx_norm.values, 'k--', linewidth=1.5, label='上证指数', alpha=0.5)

    for i, r in enumerate(results):
        eq_df = r['equity_df']
        norm = eq_df['净值'] / INITIAL_CAPITAL
        ax1.plot(eq_df['日期'], norm, color=colors[i], linewidth=2, label=r['策略'])

    ax1.set_ylabel('归一化净值')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('净值曲线对比')

    # 回撤对比
    ax2 = axes[1]
    for i, r in enumerate(results):
        eq_df = r['equity_df']
        rm = eq_df['净值'].cummax()
        dd = (eq_df['净值'] - rm) / rm
        ax2.plot(eq_df['日期'], dd, color=colors[i], linewidth=1.5, label=r['策略'], alpha=0.7)

    ax2.set_ylabel('回撤')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('回撤对比')

    plt.tight_layout()
    output_path = '/Users/caoxinyu/gits/stock-trading-strategies/backtest_compare.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {output_path}")

    # 推荐
    best = max(results, key=lambda r: r['总收益率'] / max(abs(r['最大回撤']), 0.01))
    print(f"\n{'='*70}")
    print(f"  推荐方案: {best['策略']}")
    print(f"  理由: 收益回撤比最优（收益{best['总收益率']:.2%} / 回撤{best['最大回撤']:.2%}）")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
