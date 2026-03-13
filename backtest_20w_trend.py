#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20周线趋势跟踪策略回测（4万小资金版）

策略规则：
- 大盘过滤：上证指数在20周线上方才允许开仓
- 选股：ETF/沪深300成分股，20周线拐头向上，放量突破后回踩20周线买入
- 建仓：回踩20周线附近（距离3%以内）且缩量时买入，一次到位
- 加仓：盈利10%后可加仓1次，止损移到成本价
- 止盈：盈利20%卖一半，剩余跟踪20周线；远离20周线30%全部卖出
- 止损：周线收盘跌破20周线清仓
- 仓位：单只最多60%，最多持2只，至少留10%现金
- 月度亏损上限6%停手
"""

import akshare as ak
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False


# ============ 配置参数 ============
INITIAL_CAPITAL = 40000      # 初始资金4万
MAX_SINGLE_POSITION = 0.60   # 单只最大仓位60%
MAX_TOTAL_POSITION = 0.90    # 总仓位上限90%
MONTHLY_LOSS_LIMIT = 0.06    # 月度亏损上限6%
PULLBACK_THRESHOLD = 0.03    # 回踩20周线距离阈值3%
PROFIT_HALF_EXIT = 0.20      # 盈利20%卖一半
PROFIT_ALL_EXIT = 0.30       # 远离20周线30%全部卖出
ADD_POSITION_TRIGGER = 0.10  # 盈利10%触发加仓
WEAK_HOLD_WEEKS = 12         # 持有超过12周涨幅不到5%换股
WEAK_HOLD_RETURN = 0.05      # 弱势持仓涨幅阈值
MA_PERIOD = 20               # 20周均线周期
VOLUME_MULTIPLIER = 1.5      # 放量倍数


def get_weekly_data(symbol, start_date, end_date, is_index=False):
    """获取周线数据"""
    try:
        if is_index:
            # 获取指数日线数据
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高',
                                     'low': '最低', 'close': '收盘', 'volume': '成交量'})
        else:
            # 获取个股/ETF日线数据
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                      start_date=start_date.replace('-', ''),
                                      end_date=end_date.replace('-', ''),
                                      adjust="qfq")
            df = df.rename(columns={'日期': '日期', '开盘': '开盘', '最高': '最高',
                                     '最低': '最低', '收盘': '收盘', '成交量': '成交量'})

        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
        df = df.sort_values('日期').reset_index(drop=True)

        # 转为周线
        df.set_index('日期', inplace=True)
        weekly = df.resample('W-FRI').agg({
            '开盘': 'first',
            '最高': 'max',
            '最低': 'min',
            '收盘': 'last',
            '成交量': 'sum'
        }).dropna()

        weekly.reset_index(inplace=True)

        # 计算20周均线
        weekly['MA20'] = weekly['收盘'].rolling(window=MA_PERIOD).mean()
        # 均线斜率（本周MA20 vs 上周MA20）
        weekly['MA20_slope'] = weekly['MA20'] - weekly['MA20'].shift(1)
        # 5周平均成交量
        weekly['VOL_MA5'] = weekly['成交量'].rolling(window=5).mean()

        return weekly
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None


def get_index_weekly(start_date, end_date):
    """获取上证指数周线数据"""
    return get_weekly_data('sh000001', start_date, end_date, is_index=True)


class Position:
    """持仓信息"""
    def __init__(self, symbol, name, buy_price, shares, buy_date, buy_week_idx):
        self.symbol = symbol
        self.name = name
        self.buy_price = buy_price      # 初始买入均价
        self.avg_price = buy_price      # 当前持仓均价
        self.shares = shares            # 持股数量
        self.buy_date = buy_date        # 首次买入日期
        self.buy_week_idx = buy_week_idx
        self.added = False              # 是否已加仓
        self.half_sold = False          # 是否已卖出一半
        self.cost = buy_price * shares  # 总成本


class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}           # symbol -> Position
        self.trade_log = []           # 交易记录
        self.weekly_equity = []       # 每周净值
        self.monthly_start_equity = initial_capital  # 月初净值
        self.current_month = None
        self.monthly_paused = False   # 当月是否已停手

    def get_total_equity(self, current_prices):
        """计算当前总净值"""
        equity = self.cash
        for sym, pos in self.positions.items():
            if sym in current_prices:
                equity += current_prices[sym] * pos.shares
        return equity

    def get_position_ratio(self, current_prices):
        """当前仓位占比"""
        equity = self.get_total_equity(current_prices)
        if equity <= 0:
            return 0
        position_value = sum(current_prices.get(sym, 0) * pos.shares
                           for sym, pos in self.positions.items())
        return position_value / equity

    def check_monthly_loss(self, current_equity, current_date):
        """检查月度亏损"""
        month_key = current_date.strftime('%Y-%m')
        if self.current_month != month_key:
            # 新的月份，重置
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

    def buy(self, symbol, name, price, shares, date, week_idx, reason=""):
        """买入"""
        cost = price * shares
        if cost > self.cash:
            shares = int(self.cash / price / 100) * 100
            if shares <= 0:
                return
            cost = price * shares

        self.cash -= cost
        if symbol in self.positions:
            # 加仓
            pos = self.positions[symbol]
            total_cost = pos.avg_price * pos.shares + cost
            pos.shares += shares
            pos.avg_price = total_cost / pos.shares
            pos.cost = total_cost
            pos.added = True
        else:
            self.positions[symbol] = Position(symbol, name, price, shares, date, week_idx)

        self.trade_log.append({
            '日期': date, '操作': '买入', '标的': name, '代码': symbol,
            '价格': price, '数量': shares, '金额': cost, '原因': reason
        })

    def sell(self, symbol, price, shares, date, reason=""):
        """卖出"""
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
        """清仓"""
        if symbol in self.positions:
            self.sell(symbol, price, self.positions[symbol].shares, date, reason)


def run_backtest(etf_list, start_date, end_date):
    """
    运行回测

    etf_list: [(代码, 名称), ...]
    """
    print(f"\n{'='*60}")
    print(f"  20周线趋势跟踪策略回测")
    print(f"  资金: {INITIAL_CAPITAL:,.0f}元 | 周期: {start_date} ~ {end_date}")
    print(f"{'='*60}\n")

    # 获取上证指数数据
    print("正在获取上证指数数据...")
    index_data = get_index_weekly(start_date, end_date)
    if index_data is None or len(index_data) < MA_PERIOD + 5:
        print("上证指数数据不足，退出")
        return

    # 获取ETF数据
    etf_data = {}
    for symbol, name in etf_list:
        print(f"正在获取 {name}({symbol}) 数据...")
        data = get_weekly_data(symbol, start_date, end_date)
        if data is not None and len(data) > MA_PERIOD + 5:
            etf_data[symbol] = (name, data)
        else:
            print(f"  {name} 数据不足，跳过")

    if not etf_data:
        print("没有可用的标的数据，退出")
        return

    print(f"\n有效标的: {len(etf_data)}个")
    print("开始回测...\n")

    engine = BacktestEngine(INITIAL_CAPITAL)

    # 获取所有周的日期（以上证指数为基准）
    all_weeks = index_data[index_data['MA20'].notna()].reset_index(drop=True)

    for week_idx in range(1, len(all_weeks)):
        week = all_weeks.iloc[week_idx]
        prev_week = all_weeks.iloc[week_idx - 1]
        current_date = week['日期']

        # 上证指数状态
        index_above_ma20 = week['收盘'] > week['MA20']
        index_ma20_up = week['MA20_slope'] > 0

        # 当前所有标的价格
        current_prices = {}
        for sym, (name, data) in etf_data.items():
            matched = data[data['日期'] == current_date]
            if len(matched) > 0:
                current_prices[sym] = matched.iloc[0]['收盘']

        # 计算当前净值
        current_equity = engine.get_total_equity(current_prices)

        # 检查月度亏损限制
        engine.check_monthly_loss(current_equity, current_date)

        # ====== 卖出逻辑（优先处理） ======
        symbols_to_check = list(engine.positions.keys())
        for sym in symbols_to_check:
            if sym not in engine.positions:
                continue
            pos = engine.positions[sym]
            if sym not in current_prices:
                continue

            price = current_prices[sym]
            sym_data = etf_data[sym][1]
            matched = sym_data[sym_data['日期'] == current_date]
            if len(matched) == 0:
                continue
            row = matched.iloc[0]
            ma20 = row['MA20']

            if pd.isna(ma20):
                continue

            # 止损：周线收盘跌破20周线
            if price < ma20:
                engine.sell_all(sym, price, current_date, "跌破20周线止损")
                continue

            # 止盈：远离20周线30%以上全部卖出
            deviation = (price - ma20) / ma20
            if deviation > PROFIT_ALL_EXIT:
                engine.sell_all(sym, price, current_date, f"远离20周线{deviation:.1%}，全部止盈")
                continue

            # 止盈：盈利20%卖一半
            profit_pct = (price - pos.avg_price) / pos.avg_price
            if profit_pct >= PROFIT_HALF_EXIT and not pos.half_sold:
                half_shares = (pos.shares // 2 // 100) * 100
                if half_shares >= 100:
                    engine.sell(sym, price, half_shares, current_date, f"盈利{profit_pct:.1%}，卖出一半")
                    pos.half_sold = True
                continue

            # 弱势换股：持有超过12周涨幅不到5%
            hold_weeks = week_idx - pos.buy_week_idx
            if hold_weeks >= WEAK_HOLD_WEEKS and profit_pct < WEAK_HOLD_RETURN:
                engine.sell_all(sym, price, current_date,
                              f"持有{hold_weeks}周涨幅仅{profit_pct:.1%}，弱势换股")
                continue

        # ====== 买入逻辑 ======
        # 条件：大盘在20周线上方，当月未停手，持仓数量<2
        if (index_above_ma20 and not engine.monthly_paused
                and len(engine.positions) < 2):

            candidates = []
            for sym, (name, data) in etf_data.items():
                if sym in engine.positions:
                    continue

                matched = data[data['日期'] == current_date]
                prev_matched = data[data['日期'] == prev_week['日期']]
                if len(matched) == 0 or len(prev_matched) == 0:
                    continue

                row = matched.iloc[0]
                prev_row = prev_matched.iloc[0]
                ma20 = row['MA20']
                ma20_slope = row['MA20_slope']
                vol = row['成交量']
                vol_ma5 = row['VOL_MA5']
                price = row['收盘']
                prev_price = prev_row['收盘']
                prev_ma20 = prev_row['MA20']

                if pd.isna(ma20) or pd.isna(ma20_slope) or pd.isna(vol_ma5):
                    continue

                # 条件1：20周线走平或拐头向上
                if ma20_slope < 0:
                    continue

                # 条件2：股价在20周线上方
                if price < ma20:
                    continue

                # 条件3：回踩20周线附近（距离3%以内）
                distance = (price - ma20) / ma20
                if distance > PULLBACK_THRESHOLD:
                    continue

                # 条件4：之前有过放量突破（前几周有过放量站上20周线的动作）
                had_breakout = False
                lookback_start = max(0, matched.index[0] - 8)
                lookback_end = matched.index[0]
                for lb_idx in range(lookback_start, lookback_end):
                    lb_row = data.iloc[lb_idx]
                    if (not pd.isna(lb_row['VOL_MA5']) and not pd.isna(lb_row['MA20'])
                            and lb_row['成交量'] > lb_row['VOL_MA5'] * VOLUME_MULTIPLIER
                            and lb_row['收盘'] > lb_row['MA20']):
                        had_breakout = True
                        break

                if not had_breakout:
                    continue

                # 条件5：本周缩量（回踩缩量更健康）
                is_shrink = vol < vol_ma5

                # 计算近10周涨幅（相对强度）
                lookback_10 = data[data['日期'] <= current_date].tail(11)
                if len(lookback_10) >= 2:
                    strength = (lookback_10.iloc[-1]['收盘'] - lookback_10.iloc[0]['收盘']) / lookback_10.iloc[0]['收盘']
                else:
                    strength = 0

                score = strength + (0.02 if is_shrink else 0) - distance
                candidates.append((sym, name, price, ma20, score, distance))

            # 按评分排序，取最优
            candidates.sort(key=lambda x: x[4], reverse=True)

            for sym, name, price, ma20, score, distance in candidates:
                if len(engine.positions) >= 2:
                    break

                # 计算仓位
                equity = engine.get_total_equity(current_prices)
                max_amount = equity * MAX_SINGLE_POSITION
                # 止损空间 = 当前价到20周线的距离
                stop_loss_pct = max(distance, 0.01)  # 至少1%
                # 基于风险的仓位：单笔亏损不超过3%总资金
                risk_amount = equity * 0.03
                risk_based_shares = int(risk_amount / (price * stop_loss_pct) / 100) * 100

                # 取较小值
                max_shares = int(max_amount / price / 100) * 100
                shares = min(risk_based_shares, max_shares)

                # 检查现金是否足够，且保留10%现金
                available_cash = engine.cash - equity * 0.10
                cash_shares = int(available_cash / price / 100) * 100
                shares = min(shares, cash_shares)

                if shares >= 100:
                    engine.buy(sym, name, price, shares, current_date, week_idx,
                             f"回踩20周线{distance:.1%}买入")

        # ====== 加仓逻辑 ======
        for sym in list(engine.positions.keys()):
            if sym not in engine.positions:
                continue
            pos = engine.positions[sym]
            if pos.added:
                continue
            if sym not in current_prices:
                continue

            price = current_prices[sym]
            profit_pct = (price - pos.avg_price) / pos.avg_price

            # 盈利超过10%，且价格回踩20周线附近
            if profit_pct >= ADD_POSITION_TRIGGER:
                sym_data = etf_data[sym][1]
                matched = sym_data[sym_data['日期'] == current_date]
                if len(matched) == 0:
                    continue
                row = matched.iloc[0]
                ma20 = row['MA20']
                if pd.isna(ma20):
                    continue

                distance = (price - ma20) / ma20
                # 加仓也要求贴近20周线
                if distance <= PULLBACK_THRESHOLD * 2:  # 加仓允许稍远一点（6%以内）
                    equity = engine.get_total_equity(current_prices)
                    current_pos_value = price * pos.shares
                    max_add = equity * MAX_SINGLE_POSITION - current_pos_value
                    if max_add > 0:
                        add_shares = min(
                            int(max_add / price / 100) * 100,
                            int((engine.cash - equity * 0.10) / price / 100) * 100
                        )
                        if add_shares >= 100:
                            engine.buy(sym, pos.name, price, add_shares, current_date, week_idx,
                                     f"盈利{profit_pct:.1%}加仓")

        # 记录每周净值
        equity = engine.get_total_equity(current_prices)
        engine.weekly_equity.append({
            '日期': current_date,
            '净值': equity,
            '收益率': (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            '持仓数': len(engine.positions),
            '仓位比': engine.get_position_ratio(current_prices)
        })

    # ====== 输出结果 ======
    print_results(engine, index_data, all_weeks, start_date, end_date)
    plot_results(engine, index_data, start_date, end_date)

    return engine


def print_results(engine, index_data, all_weeks, start_date, end_date):
    """打印回测结果"""
    if not engine.weekly_equity:
        print("没有产生交易数据")
        return

    equity_df = pd.DataFrame(engine.weekly_equity)
    final_equity = equity_df.iloc[-1]['净值']
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    weeks = len(equity_df)
    years = weeks / 52

    # 年化收益
    if years > 0 and final_equity > 0:
        annual_return = (final_equity / INITIAL_CAPITAL) ** (1 / years) - 1
    else:
        annual_return = 0

    # 最大回撤
    equity_series = equity_df['净值']
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()
    max_dd_date = equity_df.iloc[drawdown.idxmin()]['日期'] if len(drawdown) > 0 else None

    # 上证指数同期收益
    idx_start = index_data[index_data['日期'] >= pd.to_datetime(start_date)].iloc[0]['收盘'] if len(index_data) > 0 else 1
    idx_end = index_data.iloc[-1]['收盘'] if len(index_data) > 0 else 1
    index_return = (idx_end - idx_start) / idx_start

    # 交易统计
    trade_df = pd.DataFrame(engine.trade_log)
    sell_trades = trade_df[trade_df['操作'] == '卖出'] if len(trade_df) > 0 else pd.DataFrame()
    if len(sell_trades) > 0 and '盈亏' in sell_trades.columns:
        win_trades = sell_trades[sell_trades['盈亏'] > 0]
        lose_trades = sell_trades[sell_trades['盈亏'] <= 0]
        win_rate = len(win_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
        total_profit = sell_trades['盈亏'].sum()
        avg_win = win_trades['盈亏'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['盈亏'].mean() if len(lose_trades) > 0 else 0
    else:
        win_rate = 0
        total_profit = 0
        avg_win = 0
        avg_loss = 0

    # 平均仓位
    avg_position = equity_df['仓位比'].mean()

    print(f"\n{'='*60}")
    print(f"  回测结果")
    print(f"{'='*60}")
    print(f"  初始资金:     {INITIAL_CAPITAL:>12,.0f} 元")
    print(f"  最终资金:     {final_equity:>12,.0f} 元")
    print(f"  总收益率:     {total_return:>11.2%}")
    print(f"  年化收益率:   {annual_return:>11.2%}")
    print(f"  最大回撤:     {max_drawdown:>11.2%}  ({max_dd_date.strftime('%Y-%m-%d') if max_dd_date else 'N/A'})")
    print(f"  上证同期:     {index_return:>11.2%}")
    print(f"  超额收益:     {total_return - index_return:>11.2%}")
    print(f"{'='*60}")
    print(f"  交易次数:     {len(trade_df):>8} 笔")
    print(f"  卖出次数:     {len(sell_trades):>8} 笔")
    print(f"  胜率:         {win_rate:>11.2%}")
    print(f"  平均盈利:     {avg_win:>12,.0f} 元")
    print(f"  平均亏损:     {avg_loss:>12,.0f} 元")
    print(f"  盈亏比:       {abs(avg_win/avg_loss) if avg_loss != 0 else float('inf'):>11.2f}")
    print(f"  已实现盈亏:   {total_profit:>12,.0f} 元")
    print(f"  平均仓位:     {avg_position:>11.2%}")
    print(f"  回测周数:     {weeks:>8} 周（{years:.1f}年）")
    print(f"{'='*60}")

    # 打印交易记录
    if len(trade_df) > 0:
        print(f"\n{'='*60}")
        print(f"  交易记录")
        print(f"{'='*60}")
        for _, row in trade_df.iterrows():
            date_str = row['日期'].strftime('%Y-%m-%d')
            pnl_str = f"  盈亏:{row['盈亏']:+,.0f}" if '盈亏' in row and pd.notna(row.get('盈亏')) else ""
            print(f"  {date_str} {row['操作']} {row['标的']} "
                  f"{row['数量']}股 @{row['价格']:.3f} "
                  f"金额:{row['金额']:,.0f}{pnl_str}  [{row['原因']}]")


def plot_results(engine, index_data, start_date, end_date):
    """绘制回测结果图表"""
    if not engine.weekly_equity:
        return

    equity_df = pd.DataFrame(engine.weekly_equity)
    output_dir = '/Users/caoxinyu/gits/stock-trading-strategies'

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('20周线趋势跟踪策略回测（4万资金版）', fontsize=16, fontweight='bold')

    # 图1：净值曲线 vs 上证指数
    ax1 = axes[0]
    # 策略净值归一化
    strategy_normalized = equity_df['净值'] / INITIAL_CAPITAL
    ax1.plot(equity_df['日期'], strategy_normalized, 'b-', linewidth=2, label='策略净值')

    # 上证指数归一化
    idx_matched = index_data[index_data['日期'] >= pd.to_datetime(start_date)]
    if len(idx_matched) > 0:
        idx_base = idx_matched.iloc[0]['收盘']
        # 转为周线
        idx_weekly = idx_matched.set_index('日期').resample('W-FRI').last().dropna()
        idx_normalized = idx_weekly['收盘'] / idx_base
        ax1.plot(idx_normalized.index, idx_normalized.values, 'r--', linewidth=1.5,
                label='上证指数', alpha=0.7)

    # 标记买卖点
    trade_df = pd.DataFrame(engine.trade_log)
    if len(trade_df) > 0:
        buys = trade_df[trade_df['操作'] == '买入']
        sells = trade_df[trade_df['操作'] == '卖出']

        for _, row in buys.iterrows():
            eq_row = equity_df[equity_df['日期'] >= row['日期']].head(1)
            if len(eq_row) > 0:
                y_val = eq_row.iloc[0]['净值'] / INITIAL_CAPITAL
                ax1.scatter(row['日期'], y_val, color='red', marker='^', s=80, zorder=5)

        for _, row in sells.iterrows():
            eq_row = equity_df[equity_df['日期'] >= row['日期']].head(1)
            if len(eq_row) > 0:
                y_val = eq_row.iloc[0]['净值'] / INITIAL_CAPITAL
                ax1.scatter(row['日期'], y_val, color='green', marker='v', s=80, zorder=5)

    ax1.set_ylabel('归一化净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('策略净值 vs 上证指数（红色三角=买入，绿色三角=卖出）')

    # 图2：回撤曲线
    ax2 = axes[1]
    running_max = equity_df['净值'].cummax()
    drawdown = (equity_df['净值'] - running_max) / running_max
    ax2.fill_between(equity_df['日期'], drawdown, 0, color='red', alpha=0.3)
    ax2.plot(equity_df['日期'], drawdown, 'r-', linewidth=1)
    ax2.set_ylabel('回撤')
    ax2.set_title('回撤曲线')
    ax2.grid(True, alpha=0.3)

    # 图3：仓位变化
    ax3 = axes[2]
    ax3.fill_between(equity_df['日期'], equity_df['仓位比'], 0, color='blue', alpha=0.3)
    ax3.plot(equity_df['日期'], equity_df['仓位比'], 'b-', linewidth=1)
    ax3.set_ylabel('仓位占比')
    ax3.set_title('仓位变化')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'backtest_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {output_path}")


if __name__ == '__main__':
    # 回测标的：常见宽基和行业ETF
    etf_list = [
        ('510300', '沪深300ETF'),
        ('510500', '中证500ETF'),
        ('159915', '创业板ETF'),
        ('512000', '券商ETF'),
        ('512480', '半导体ETF'),
        ('510880', '红利ETF'),
        ('512010', '医药ETF'),
        ('515030', '新能源车ETF'),
    ]

    # 回测区间：2020-01-01 ~ 2025-12-31（约6年）
    start_date = '2020-01-01'
    end_date = '2025-12-31'

    engine = run_backtest(etf_list, start_date, end_date)
