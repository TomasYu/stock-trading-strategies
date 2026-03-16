#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20周线趋势跟踪策略 - 30万资金版

相比4万版本的升级：
1. 持仓数量：最多5只（分散风险）
2. 单只仓位上限：30%（更分散）
3. 金字塔加仓：最多加仓2次（盈利10%加一次，盈利20%再加一次）
4. 标的池扩大：30只ETF覆盖更多行业
5. 板块分散：同一板块最多2只
6. 空仓资金年化2%计息
7. 仓位动态调整：账户盈利超10%后，单笔风险从3%提到4%
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
INITIAL_CAPITAL = 300000
MAX_POSITIONS = 5                # 最多持仓5只
MAX_SINGLE_POSITION = 0.30       # 单只上限30%
MAX_TOTAL_POSITION = 0.90        # 总仓位上限90%
MONTHLY_LOSS_LIMIT = 0.06        # 月亏6%停手
MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
MAX_STOP_LOSS = 0.20
ADD_POSITION_TRIGGER1 = 0.10     # 第一次加仓：盈利10%
ADD_POSITION_TRIGGER2 = 0.20     # 第二次加仓：盈利20%
WEAK_HOLD_WEEKS = 12
WEAK_HOLD_RETURN = 0.05
CASH_ANNUAL_RATE = 0.02
BASE_RISK_PCT = 0.03             # 基础单笔风险3%
BOOSTED_RISK_PCT = 0.04          # 账户盈利10%后提高到4%
ACCOUNT_BOOST_THRESHOLD = 0.10   # 账户盈利10%触发

# 板块分类
SECTOR_MAP = {
    # 宽基
    '510300': '宽基', '510500': '宽基', '159915': '宽基',
    '512100': '宽基', '562000': '宽基',
    # 金融
    '512000': '金融', '512800': '金融',
    # 科技
    '512480': '科技', '159995': '科技', '515880': '科技',
    '515050': '科技',
    # 消费
    '512690': '消费', '159766': '消费',
    # 医药
    '512010': '医药', '512170': '医药',
    # 新能源
    '515030': '新能源', '516160': '新能源', '515790': '新能源',
    # 周期
    '512400': '周期', '515220': '周期', '515210': '周期',
    # 其他
    '510880': '红利', '512660': '军工', '512980': '传媒',
    '159869': '传媒', '512580': '环保', '159865': '农业',
    '513050': '海外', '512200': '地产', '159611': '公用',
}
MAX_SAME_SECTOR = 2  # 同板块最多2只


def get_weekly(symbol, start_date, end_date, data_type='etf'):
    try:
        sd = start_date.replace('-', '')
        ed = end_date.replace('-', '')
        if data_type == 'index':
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高',
                                     'low': '最低', 'close': '收盘', 'volume': '成交量'})
        else:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                      start_date=sd, end_date=ed, adjust="qfq")
            df = df.rename(columns={'日期': '日期', '开盘': '开盘', '最高': '最高',
                                     '最低': '最低', '收盘': '收盘', '成交量': '成交量'})

        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))]
        df = df.sort_values('日期').reset_index(drop=True)

        df.set_index('日期', inplace=True)
        weekly = df.resample('W-FRI').agg({
            '开盘': 'first', '最高': 'max', '最低': 'min',
            '收盘': 'last', '成交量': 'sum'
        }).dropna()
        weekly.reset_index(inplace=True)

        weekly['MA20'] = weekly['收盘'].rolling(window=MA_PERIOD).mean()
        weekly['MA20_slope'] = weekly['MA20'] - weekly['MA20'].shift(1)
        weekly['VOL_MA5'] = weekly['成交量'].rolling(window=5).mean()
        weekly['LOW_8W'] = weekly['最低'].rolling(window=8).min()

        return weekly
    except:
        return None


class Position:
    def __init__(self, symbol, name, buy_price, shares, buy_date, buy_week_idx, stop_price=None):
        self.symbol = symbol
        self.name = name
        self.buy_price = buy_price
        self.avg_price = buy_price
        self.shares = shares
        self.buy_date = buy_date
        self.buy_week_idx = buy_week_idx
        self.cost = buy_price * shares
        self.stop_price = stop_price
        self.max_price = buy_price
        self.add_count = 0       # 已加仓次数（最多2次）


class BacktestEngine:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_log = []
        self.weekly_equity = []
        self.monthly_start_equity = initial_capital
        self.current_month = None
        self.monthly_paused = False
        self.cash_interest_earned = 0

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
        pos_val = sum(current_prices.get(s, 0) * p.shares for s, p in self.positions.items())
        return pos_val / equity

    def get_single_position_ratio(self, symbol, current_prices):
        equity = self.get_total_equity(current_prices)
        if equity <= 0 or symbol not in self.positions:
            return 0
        return current_prices.get(symbol, 0) * self.positions[symbol].shares / equity

    def get_sector_count(self, sector):
        """统计当前持仓中某板块的数量"""
        count = 0
        for sym in self.positions:
            if SECTOR_MAP.get(sym, '其他') == sector:
                count += 1
        return count

    def get_risk_pct(self, current_prices):
        """动态风险比例：账户盈利超10%则提高到4%"""
        equity = self.get_total_equity(current_prices)
        account_return = (equity - self.initial_capital) / self.initial_capital
        if account_return >= ACCOUNT_BOOST_THRESHOLD:
            return BOOSTED_RISK_PCT
        return BASE_RISK_PCT

    def accrue_cash_interest(self):
        if self.cash > 0:
            interest = self.cash * CASH_ANNUAL_RATE / 52
            self.cash += interest
            self.cash_interest_earned += interest

    def check_monthly_loss(self, current_equity, current_date):
        month_key = current_date.strftime('%Y-%m')
        if self.current_month != month_key:
            self.current_month = month_key
            self.monthly_start_equity = current_equity
            self.monthly_paused = False
            return False
        if self.monthly_start_equity > 0:
            if (current_equity - self.monthly_start_equity) / self.monthly_start_equity < -MONTHLY_LOSS_LIMIT:
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
            pos.add_count += 1
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
        profit_pct = (price - pos.avg_price) / pos.avg_price
        self.trade_log.append({
            '日期': date, '操作': '卖出', '标的': pos.name, '代码': symbol,
            '价格': price, '数量': sell_shares, '金额': revenue,
            '盈亏': round(profit, 2), '收益率': f"{profit_pct:.1%}", '原因': reason
        })
        pos.shares -= sell_shares
        if pos.shares <= 0:
            del self.positions[symbol]

    def sell_all(self, symbol, price, date, reason=""):
        if symbol in self.positions:
            self.sell(symbol, price, self.positions[symbol].shares, date, reason)


def calc_buy_shares(engine, price, stop_loss_pct, current_prices):
    equity = engine.get_total_equity(current_prices)
    max_amount = equity * MAX_SINGLE_POSITION
    stop_loss_pct = max(stop_loss_pct, 0.01)
    risk_pct = engine.get_risk_pct(current_prices)
    risk_amount = equity * risk_pct
    risk_shares = int(risk_amount / (price * stop_loss_pct) / 100) * 100
    max_shares = int(max_amount / price / 100) * 100
    # 保留10%现金
    available = engine.cash - equity * 0.10
    # 总仓位不超过90%
    total_pos_val = sum(current_prices.get(s, 0) * p.shares for s, p in engine.positions.items())
    max_total_add = equity * MAX_TOTAL_POSITION - total_pos_val
    total_shares = int(max_total_add / price / 100) * 100

    cash_shares = int(available / price / 100) * 100
    shares = min(risk_shares, max_shares, cash_shares, total_shares)
    return shares if shares >= 100 else 0


def get_strength(data, current_date, weeks=10):
    subset = data[data['日期'] <= current_date].tail(weeks + 1)
    if len(subset) >= 2:
        return (subset.iloc[-1]['收盘'] - subset.iloc[0]['收盘']) / subset.iloc[0]['收盘']
    return 0


def run_backtest(etf_data, index_data, start_date, end_date):
    engine = BacktestEngine(INITIAL_CAPITAL)
    all_weeks = index_data[index_data['MA20'].notna()].reset_index(drop=True)

    for week_idx in range(1, len(all_weeks)):
        week = all_weeks.iloc[week_idx]
        prev_week = all_weeks.iloc[week_idx - 1]
        current_date = week['日期']
        prev_date = prev_week['日期']

        index_above_ma20 = week['收盘'] > week['MA20']

        engine.accrue_cash_interest()

        current_prices = {}
        for sym, (name, data) in etf_data.items():
            matched = data[data['日期'] == current_date]
            if len(matched) > 0:
                current_prices[sym] = matched.iloc[0]['收盘']

        current_equity = engine.get_total_equity(current_prices)
        engine.check_monthly_loss(current_equity, current_date)

        for sym, pos in engine.positions.items():
            if sym in current_prices:
                pos.max_price = max(pos.max_price, current_prices[sym])

        # ====== 卖出 ======
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

            hold_weeks = week_idx - pos.buy_week_idx
            profit_pct = (price - pos.avg_price) / pos.avg_price

            # 止损1：跌破固定止损价
            if pos.stop_price and price < pos.stop_price:
                engine.sell_all(sym, price, current_date,
                              f"跌破止损价{pos.stop_price:.3f}（持有{hold_weeks}周）")
                continue

            # 止损2：跌破20周线
            if price < ma20:
                engine.sell_all(sym, price, current_date,
                              f"跌破20周线{ma20:.3f}（持有{hold_weeks}周，收益{profit_pct:.1%}）")
                continue

            # 弱势换股
            if hold_weeks >= WEAK_HOLD_WEEKS and profit_pct < WEAK_HOLD_RETURN:
                engine.sell_all(sym, price, current_date,
                              f"持有{hold_weeks}周涨幅{profit_pct:.1%}弱势换股")
                continue

        # ====== 买入 ======
        if index_above_ma20 and not engine.monthly_paused and len(engine.positions) < MAX_POSITIONS:
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

                if any(pd.isna(v) for v in [ma20, slope, vol_ma5, prev_ma20, low_8w]):
                    continue

                if slope < 0 or price <= ma20:
                    continue
                if prev_price > prev_ma20 * 1.02:
                    continue
                if vol < vol_ma5 * VOLUME_MULTIPLIER:
                    continue

                stop_price = low_8w * 0.98
                stop_loss_pct = (price - stop_price) / price
                if stop_loss_pct > MAX_STOP_LOSS:
                    continue

                # 板块检查
                sector = SECTOR_MAP.get(sym, '其他')
                if engine.get_sector_count(sector) >= MAX_SAME_SECTOR:
                    continue

                strength = get_strength(data, current_date)
                candidates.append((sym, name, price, stop_price, stop_loss_pct, strength, sector))

            candidates.sort(key=lambda x: x[5], reverse=True)
            for sym, name, price, stop_price, stop_loss_pct, strength, sector in candidates:
                if len(engine.positions) >= MAX_POSITIONS:
                    break
                # 再次检查板块（可能前面循环中已经加了同板块的）
                if engine.get_sector_count(sector) >= MAX_SAME_SECTOR:
                    continue

                shares = calc_buy_shares(engine, price, stop_loss_pct, current_prices)
                if shares >= 100:
                    engine.buy(sym, name, price, shares, current_date, week_idx,
                             f"放量突破，止损{stop_price:.3f}（{stop_loss_pct:.1%}）",
                             stop_price=stop_price)

        # ====== 金字塔加仓（最多2次） ======
        for sym in list(engine.positions.keys()):
            if sym not in engine.positions or sym not in current_prices:
                continue
            pos = engine.positions[sym]
            if pos.add_count >= 2:
                continue
            price = current_prices[sym]
            profit_pct = (price - pos.avg_price) / pos.avg_price

            trigger = ADD_POSITION_TRIGGER1 if pos.add_count == 0 else ADD_POSITION_TRIGGER2
            if profit_pct < trigger:
                continue

            data = etf_data[sym][1]
            matched = data[data['日期'] == current_date]
            if len(matched) == 0:
                continue
            row = matched.iloc[0]
            ma20 = row['MA20']
            if pd.isna(ma20):
                continue
            distance = (price - ma20) / ma20
            if distance > 0.06:
                continue

            # 检查单只仓位上限
            single_ratio = engine.get_single_position_ratio(sym, current_prices)
            if single_ratio >= MAX_SINGLE_POSITION:
                continue

            equity = engine.get_total_equity(current_prices)
            current_val = price * pos.shares
            max_add = equity * MAX_SINGLE_POSITION - current_val
            if max_add <= 0:
                continue

            # 金字塔递减：第2次加仓量比第1次少
            scale = 1.0 if pos.add_count == 0 else 0.5
            available = engine.cash - equity * 0.10
            add_shares = min(
                int(max_add / price / 100) * 100,
                int(available * scale / price / 100) * 100
            )
            if add_shares >= 100:
                engine.buy(sym, pos.name, price, add_shares, current_date, week_idx,
                         f"第{pos.add_count+1}次加仓（盈利{profit_pct:.1%}）")

        # 记录
        equity = engine.get_total_equity(current_prices)
        engine.weekly_equity.append({
            '日期': current_date,
            '净值': equity,
            '收益率': (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            '持仓数': len(engine.positions),
            '仓位比': engine.get_position_ratio(current_prices)
        })

    return engine


def print_results(engine, index_data, start_date, end_date):
    if not engine.weekly_equity:
        return

    eq_df = pd.DataFrame(engine.weekly_equity)
    final = eq_df.iloc[-1]['净值']
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    weeks = len(eq_df)
    years = weeks / 52
    annual_ret = (final / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 and final > 0 else 0

    running_max = eq_df['净值'].cummax()
    drawdown = (eq_df['净值'] - running_max) / running_max
    max_dd = drawdown.min()
    max_dd_date = eq_df.iloc[drawdown.idxmin()]['日期']

    idx_start = index_data[index_data['日期'] >= pd.to_datetime(start_date)].iloc[0]['收盘']
    idx_end = index_data.iloc[-1]['收盘']
    index_ret = (idx_end - idx_start) / idx_start

    trade_df = pd.DataFrame(engine.trade_log)
    sells = trade_df[trade_df['操作'] == '卖出'] if len(trade_df) > 0 else pd.DataFrame()
    if len(sells) > 0 and '盈亏' in sells.columns:
        wins = sells[sells['盈亏'] > 0]
        losses = sells[sells['盈亏'] <= 0]
        win_rate = len(wins) / len(sells)
        avg_win = wins['盈亏'].mean() if len(wins) > 0 else 0
        avg_loss = losses['盈亏'].mean() if len(losses) > 0 else 0
        total_pnl = sells['盈亏'].sum()
        max_win = sells['盈亏'].max()
        max_loss = sells['盈亏'].min()
    else:
        win_rate = avg_win = avg_loss = total_pnl = max_win = max_loss = 0

    avg_pos = eq_df['仓位比'].mean()

    print(f"\n{'='*65}")
    print(f"  20周线趋势跟踪策略 · 30万资金版回测结果")
    print(f"  资金: {INITIAL_CAPITAL:,.0f}元 | 周期: {start_date} ~ {end_date}")
    print(f"{'='*65}")
    print(f"  初始资金:       {INITIAL_CAPITAL:>12,.0f} 元")
    print(f"  最终资金:       {final:>12,.0f} 元")
    print(f"  盈利金额:       {final-INITIAL_CAPITAL:>12,.0f} 元")
    print(f"  总收益率:       {total_ret:>11.2%}")
    print(f"  年化收益率:     {annual_ret:>11.2%}")
    print(f"  最大回撤:       {max_dd:>11.2%}  ({max_dd_date.strftime('%Y-%m-%d')})")
    print(f"  上证同期:       {index_ret:>11.2%}")
    print(f"  超额收益:       {total_ret - index_ret:>11.2%}")
    print(f"  空仓利息:       {engine.cash_interest_earned:>12,.0f} 元")
    print(f"{'='*65}")
    print(f"  胜率:           {win_rate:>11.2%}")
    print(f"  盈亏比:         {abs(avg_win/avg_loss) if avg_loss != 0 else 0:>11.2f}")
    print(f"  平均盈利:       {avg_win:>12,.0f} 元")
    print(f"  平均亏损:       {avg_loss:>12,.0f} 元")
    print(f"  最大单笔盈利:   {max_win:>12,.0f} 元")
    print(f"  最大单笔亏损:   {max_loss:>12,.0f} 元")
    print(f"  已实现盈亏:     {total_pnl:>12,.0f} 元")
    print(f"  交易次数:       {len(trade_df):>9} 笔")
    print(f"  平均仓位:       {avg_pos:>11.2%}")
    print(f"  回测周数:       {weeks:>9} 周（{years:.1f}年）")
    print(f"{'='*65}")

    # 交易明细
    print(f"\n  交易明细")
    print(f"  {'-'*60}")
    for _, row in trade_df.iterrows():
        date_str = row['日期'].strftime('%Y-%m-%d')
        pnl_str = ""
        if '盈亏' in row and pd.notna(row.get('盈亏')):
            pnl_str = f"  盈亏:{row['盈亏']:+,.0f}({row.get('收益率','')})"
        print(f"  {date_str} {row['操作']} {row['标的']} "
              f"{row['数量']}股 @{row['价格']:.3f} 金额:{row['金额']:,.0f}{pnl_str}")
        print(f"    → {row['原因']}")

    # 画图
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('20周线趋势跟踪策略 · 30万资金版', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    norm = eq_df['净值'] / INITIAL_CAPITAL
    ax1.plot(eq_df['日期'], norm, 'b-', linewidth=2, label='策略净值')

    idx_weekly = index_data.set_index('日期').resample('W-FRI').last().dropna()
    idx_norm = idx_weekly['收盘'] / idx_start
    ax1.plot(idx_norm.index, idx_norm.values, 'r--', linewidth=1.5, label='上证指数', alpha=0.6)

    for _, row in trade_df.iterrows():
        eq_row = eq_df[eq_df['日期'] >= row['日期']].head(1)
        if len(eq_row) > 0:
            y = eq_row.iloc[0]['净值'] / INITIAL_CAPITAL
            if row['操作'] == '买入':
                ax1.scatter(row['日期'], y, color='red', marker='^', s=60, zorder=5)
            else:
                ax1.scatter(row['日期'], y, color='green', marker='v', s=60, zorder=5)

    ax1.set_ylabel('归一化净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('净值曲线（红△买入 绿▽卖出）')

    ax2 = axes[1]
    ax2.fill_between(eq_df['日期'], drawdown, 0, color='red', alpha=0.3)
    ax2.plot(eq_df['日期'], drawdown, 'r-', linewidth=1)
    ax2.set_ylabel('回撤')
    ax2.set_title('回撤曲线')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(eq_df['日期'], eq_df['仓位比'], 0, color='blue', alpha=0.3)
    ax3.plot(eq_df['日期'], eq_df['仓位比'], 'b-', linewidth=1)
    ax3.set_ylabel('仓位占比')
    ax3.set_title('仓位变化')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = '/Users/caoxinyu/gits/stock-trading-strategies/backtest_30w.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {path}")


def main():
    start_date = '2020-01-01'
    end_date = '2025-12-31'

    etf_pool = [
        ('510300', '沪深300ETF'), ('510500', '中证500ETF'),
        ('159915', '创业板ETF'), ('512000', '券商ETF'),
        ('512480', '半导体ETF'), ('510880', '红利ETF'),
        ('512010', '医药ETF'), ('515030', '新能源车ETF'),
        ('512800', '银行ETF'), ('512660', '军工ETF'),
        ('515880', '通信ETF'), ('159995', '芯片ETF'),
        ('512690', '酒ETF'), ('512170', '医疗ETF'),
        ('516160', '新能源ETF'), ('512980', '传媒ETF'),
        ('515050', '5GETF'), ('159766', '旅游ETF'),
        ('512400', '有色ETF'), ('159611', '电力ETF'),
        ('515220', '煤炭ETF'), ('512580', '环保ETF'),
        ('159865', '养殖ETF'), ('512100', '中证1000ETF'),
        ('513050', '中概互联ETF'), ('515790', '光伏ETF'),
        ('159869', '游戏ETF'), ('562000', '中证2000ETF'),
        ('512200', '房地产ETF'), ('515210', '钢铁ETF'),
    ]

    print(f"\n{'='*65}")
    print(f"  20周线趋势跟踪策略 · 30万资金版回测")
    print(f"  资金: {INITIAL_CAPITAL:,.0f}元 | 周期: {start_date} ~ {end_date}")
    print(f"{'='*65}")
    print(f"\n  升级项：5只持仓 | 30%单只上限 | 金字塔2次加仓")
    print(f"  板块分散 | 动态风险 | 空仓计息\n")

    print("获取上证指数...")
    index_data = get_weekly('sh000001', start_date, end_date, 'index')
    if index_data is None:
        return

    etf_data = {}
    print(f"获取ETF（{len(etf_pool)}只）...")
    for code, name in etf_pool:
        d = get_weekly(code, start_date, end_date, 'etf')
        if d is not None and len(d) > MA_PERIOD + 5:
            etf_data[code] = (name, d)
    print(f"  有效: {len(etf_data)}只")
    print(f"\n开始回测...")

    engine = run_backtest(etf_data, index_data, start_date, end_date)
    print_results(engine, index_data, start_date, end_date)


if __name__ == '__main__':
    main()
