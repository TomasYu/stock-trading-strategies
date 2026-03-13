#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20周线趋势跟踪策略 - 选股器

按策略规则扫描ETF和个股，找出满足买入条件的标的。

买入条件（全部满足）：
1. 20周线走平或拐头向上（斜率≥0）
2. 本周收盘站上20周线
3. 上周收盘在20周线下方或附近（≤MA20×1.02）
4. 本周放量（成交量 > 5周均量 × 1.5）

额外限制：
- 大盘（上证指数）须在20周线上方
- 止损幅度（当前价到前8周低点）不超过20%
"""

import akshare as ak
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
MAX_STOP_LOSS = 0.20


def get_weekly(symbol, start_date, end_date, data_type='etf'):
    """获取周线数据"""
    try:
        sd = start_date.replace('-', '')
        ed = end_date.replace('-', '')
        if data_type == 'index':
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={'date': '日期', 'open': '开盘', 'high': '最高',
                                     'low': '最低', 'close': '收盘', 'volume': '成交量'})
        elif data_type == 'etf':
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                      start_date=sd, end_date=ed, adjust="qfq")
            df = df.rename(columns={'日期': '日期', '开盘': '开盘', '最高': '最高',
                                     '最低': '最低', '收盘': '收盘', '成交量': '成交量'})
        elif data_type == 'stock':
            df = ak.stock_zh_a_hist(symbol=symbol, period='daily',
                                     start_date=sd, end_date=ed, adjust='qfq')
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
    except Exception as e:
        return None


def check_index():
    """检查大盘状态"""
    print("检查上证指数...")
    data = get_weekly('sh000001', '2024-01-01', '2026-12-31', 'index')
    if data is None or len(data) < MA_PERIOD + 2:
        print("  获取上证指数失败")
        return False, None

    last = data.iloc[-1]
    above = last['收盘'] > last['MA20']
    slope = last['MA20_slope']

    print(f"  收盘: {last['收盘']:.2f}  MA20: {last['MA20']:.2f}  "
          f"偏离: {(last['收盘']-last['MA20'])/last['MA20']*100:.1f}%  "
          f"斜率: {slope:.2f}")
    if above:
        print(f"  ✓ 上证在20周线上方，可以操作")
    else:
        print(f"  ✗ 上证在20周线下方，不宜开仓")
    return above, data


def scan_single(symbol, name, data_type='etf'):
    """扫描单只标的"""
    data = get_weekly(symbol, '2024-01-01', '2026-12-31', data_type)
    if data is None or len(data) < MA_PERIOD + 2:
        return None

    last = data.iloc[-1]
    prev = data.iloc[-2]

    if any(pd.isna(v) for v in [last['MA20'], last['MA20_slope'], last['VOL_MA5'], last['LOW_8W'],
                                  prev['MA20']]):
        return None

    # 四个条件
    c1_slope_up = last['MA20_slope'] >= 0
    c2_above_ma = last['收盘'] > last['MA20']
    c3_prev_below = prev['收盘'] <= prev['MA20'] * 1.02
    c4_volume = last['成交量'] > last['VOL_MA5'] * VOLUME_MULTIPLIER

    score = sum([c1_slope_up, c2_above_ma, c3_prev_below, c4_volume])

    # 计算止损相关
    stop_price = last['LOW_8W'] * 0.98
    stop_loss_pct = (last['收盘'] - stop_price) / last['收盘'] if last['收盘'] > 0 else 0
    distance = (last['收盘'] - last['MA20']) / last['MA20'] * 100
    vol_ratio = last['成交量'] / last['VOL_MA5'] if last['VOL_MA5'] > 0 else 0

    return {
        '代码': symbol,
        '名称': name,
        '收盘价': last['收盘'],
        'MA20': last['MA20'],
        '偏离%': distance,
        '斜率': last['MA20_slope'],
        '量比': vol_ratio,
        '前8周低': last['LOW_8W'],
        '止损价': stop_price,
        '止损幅度%': stop_loss_pct * 100,
        '条件1_均线向上': c1_slope_up,
        '条件2_站上均线': c2_above_ma,
        '条件3_上周在下方': c3_prev_below,
        '条件4_放量': c4_volume,
        '满足条件数': score,
        '可买入': score == 4 and stop_loss_pct <= MAX_STOP_LOSS,
        '日期': last['日期'].strftime('%Y-%m-%d'),
    }


def main():
    print(f"\n{'='*70}")
    print(f"  20周线趋势跟踪策略 · 选股扫描")
    print(f"  扫描日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*70}\n")

    # 第一步：检查大盘
    index_ok, _ = check_index()
    print()

    # 第二步：扫描标的池
    # ETF池
    etf_pool = [
        ('510300', '沪深300ETF'), ('510500', '中证500ETF'),
        ('159915', '创业板ETF'), ('512000', '券商ETF'),
        ('512480', '半导体ETF'), ('510880', '红利ETF'),
        ('512010', '医药ETF'), ('515030', '新能源车ETF'),
        ('512800', '银行ETF'), ('512200', '房地产ETF'),
        ('515790', '光伏ETF'), ('159869', '游戏ETF'),
        ('512660', '军工ETF'), ('515880', '通信ETF'),
        ('159995', '芯片ETF'), ('512690', '酒ETF'),
        ('512170', '医疗ETF'), ('516160', '新能源ETF'),
        ('512980', '传媒ETF'), ('515050', '5GETF'),
        ('159766', '旅游ETF'), ('562000', '中证2000ETF'),
        ('512400', '有色ETF'), ('515210', '钢铁ETF'),
        ('159611', '电力ETF'), ('515220', '煤炭ETF'),
        ('512580', '环保ETF'), ('159865', '养殖ETF'),
        ('512100', '中证1000ETF'), ('513050', '中概互联ETF'),
    ]

    # 热门个股池（沪深300部分权重股 + 科创板热门）
    stock_pool = [
        ('600519', '贵州茅台'), ('000858', '五粮液'), ('600036', '招商银行'),
        ('601318', '中国平安'), ('000333', '美的集团'), ('600900', '长江电力'),
        ('002594', '比亚迪'), ('601888', '中国中免'), ('300750', '宁德时代'),
        ('600809', '山西汾酒'), ('002475', '立讯精密'), ('000001', '平安银行'),
        ('600276', '恒瑞医药'), ('002714', '牧原股份'), ('601012', '隆基绿能'),
        ('300014', '亿纬锂能'), ('002049', '紫光国微'), ('300124', '汇川技术'),
        ('688981', '中芯国际'), ('688111', '金山办公'), ('688036', '传音控股'),
        ('603259', '药明康德'), ('002371', '北方华创'), ('300661', '圣邦股份'),
        ('601899', '紫金矿业'), ('600030', '中信证券'), ('002236', '大华股份'),
        ('000568', '泸州老窖'), ('600309', '万华化学'), ('601225', '陕西煤业'),
    ]

    results = []

    print(f"扫描ETF（{len(etf_pool)}只）...")
    for i, (code, name) in enumerate(etf_pool):
        r = scan_single(code, name, 'etf')
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  已扫描 {i+1}/{len(etf_pool)}...")

    print(f"\n扫描个股（{len(stock_pool)}只）...")
    for i, (code, name) in enumerate(stock_pool):
        r = scan_single(code, name, 'stock')
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  已扫描 {i+1}/{len(stock_pool)}...")

    print(f"\n共扫描 {len(results)} 只标的\n")

    if not results:
        print("没有获取到有效数据")
        return

    # 分类输出
    # 1. 满足全部条件的（买入信号）
    buy_signals = [r for r in results if r['可买入']]
    # 2. 满足3个条件的（接近信号）
    near_signals = [r for r in results if r['满足条件数'] == 3 and not r['可买入']]
    # 3. 满足全部条件但止损幅度超20%的
    too_risky = [r for r in results if r['满足条件数'] == 4 and r['止损幅度%'] > MAX_STOP_LOSS * 100]

    # ====== 买入信号 ======
    print(f"{'='*70}")
    if not index_ok:
        print(f"  ⚠ 大盘在20周线下方，以下信号仅供参考，建议不开仓")
    print(f"  买入信号（4/4条件满足 + 止损幅度≤20%）：{'无' if not buy_signals else ''}")
    print(f"{'='*70}")

    if buy_signals:
        for r in sorted(buy_signals, key=lambda x: x['量比'], reverse=True):
            print(f"\n  ★ {r['名称']}（{r['代码']}）")
            print(f"    收盘: {r['收盘价']:.2f}  MA20: {r['MA20']:.2f}  偏离: {r['偏离%']:+.1f}%")
            print(f"    量比: {r['量比']:.2f}x  斜率: {r['斜率']:.2f}")
            print(f"    止损价: {r['止损价']:.2f}  止损幅度: {r['止损幅度%']:.1f}%")
            print(f"    前8周最低: {r['前8周低']:.2f}")
            # 算仓位
            capital = 40000
            risk = capital * 0.03
            shares_calc = risk / (r['收盘价'] * r['止损幅度%'] / 100) if r['止损幅度%'] > 0 else 0
            if r['收盘价'] >= 1:
                shares = int(shares_calc / 100) * 100
            else:
                shares = int(shares_calc)
            amount = shares * r['收盘价']
            ratio = amount / capital * 100
            print(f"    → 4万资金建议: 买入{shares}股，金额{amount:,.0f}元（仓位{ratio:.1f}%）")

    # ====== 接近信号 ======
    print(f"\n{'='*70}")
    print(f"  接近信号（3/4条件满足，持续关注）：{'无' if not near_signals else ''}")
    print(f"{'='*70}")

    if near_signals:
        for r in sorted(near_signals, key=lambda x: x['偏离%']):
            missing = []
            if not r['条件1_均线向上']:
                missing.append('均线还在下行')
            if not r['条件2_站上均线']:
                missing.append('未站上均线')
            if not r['条件3_上周在下方']:
                missing.append('非突破行为（已在上方）')
            if not r['条件4_放量']:
                missing.append('未放量')
            print(f"\n  ○ {r['名称']}（{r['代码']}）收盘:{r['收盘价']:.2f}  MA20:{r['MA20']:.2f}  "
                  f"偏离:{r['偏离%']:+.1f}%  量比:{r['量比']:.2f}x")
            print(f"    缺少: {', '.join(missing)}")

    # ====== 满足条件但风险太高 ======
    if too_risky:
        print(f"\n{'='*70}")
        print(f"  信号触发但止损幅度超20%（风险过大，不建议）：")
        print(f"{'='*70}")
        for r in too_risky:
            print(f"\n  △ {r['名称']}（{r['代码']}）收盘:{r['收盘价']:.2f}  "
                  f"止损幅度:{r['止损幅度%']:.1f}%（超过20%上限）")

    # ====== 汇总 ======
    print(f"\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")
    print(f"  大盘状态: {'✓ 可操作' if index_ok else '✗ 不宜开仓'}")
    print(f"  扫描标的: {len(results)} 只")
    print(f"  买入信号: {len(buy_signals)} 只")
    print(f"  接近信号: {len(near_signals)} 只")
    print(f"  风险过高: {len(too_risky)} 只")
    print(f"  数据日期: {results[0]['日期'] if results else 'N/A'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
