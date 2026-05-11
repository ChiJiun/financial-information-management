#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20年回測交易策略 - 驗證是否打敗 Buy & Hold
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 設定繪圖風格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

print('=' * 70)
print('正在下載 20 年歷史數據 (SPY)...')
print('=' * 70)

# 下載數據
ticker = 'SPY'
start_date = '2005-01-01'
end_date = '2024-12-31'
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if len(data) == 0:
    print('錯誤：無法下載數據，請檢查網路連線。')
    exit()

print(f'成功下載 {len(data)} 筆交易數據')
print(f'時間範圍：{start_date} 至 {end_date}')

# 資料預處理 - yfinance 新版本使用 MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(1, axis=1)  # 移除 'SPY' 層級

data = data.dropna(subset=['Close'])
data['Returns'] = data['Close'].pct_change()

# --- 技術指標實作 ---
print('\n計算技術指標...')

# 1. EMA (指數移動平均)
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

# 2. RSI (相對強弱指標)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# 3. MACD
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# --- 交易策略邏輯 ---
print('執行交易策略回測...')

# 進場條件：
# 1. 均線多頭排列 (EMA20 > EMA50)
# 2. RSI > 50 (多頭強勢區)
# 3. MACD > Signal (動能向上)
# 延遲一天進場 (使用 shift(1))

conditions = (
    (data['EMA_20'] > data['EMA_50']) & 
    (data['RSI'] > 50) & 
    (data['MACD'] > data['Signal_Line'])
)

data['Signal'] = conditions.astype(int)
data['Position'] = data['Signal'].shift(1)  # 延遲一天進場

# 計算策略報酬
data['Strategy_Returns'] = data['Position'] * data['Returns']

# 計算累積報酬 (淨值)
data['Cumulative_BuyHold'] = (1 + data['Returns']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

# --- 績效分析 ---
total_days = len(data)
strategy_total_ret = data['Cumulative_Strategy'].iloc[-1] - 1
buyhold_total_ret = data['Cumulative_BuyHold'].iloc[-1] - 1

# 年化報酬率 (CAGR)
years = 20
cagr_strategy = (data['Cumulative_Strategy'].iloc[-1]) ** (1/years) - 1
cagr_buyhold = (data['Cumulative_BuyHold'].iloc[-1]) ** (1/years) - 1

# 波動度與夏普比率 (假設無風險利率 2%)
vol_strategy = data['Strategy_Returns'].std() * np.sqrt(252)
vol_buyhold = data['Returns'].std() * np.sqrt(252)
sharpe_strategy = (cagr_strategy - 0.02) / vol_strategy if vol_strategy != 0 else 0
sharpe_buyhold = (cagr_buyhold - 0.02) / vol_buyhold if vol_buyhold != 0 else 0

# 最大回撤 (Max Drawdown)
def get_max_drawdown(cum_ret):
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret - peak) / peak
    return drawdown.min()

mdd_strategy = get_max_drawdown(data['Cumulative_Strategy'])
mdd_buyhold = get_max_drawdown(data['Cumulative_BuyHold'])

# 交易天數統計
trading_days = data['Position'].sum()
total_trading_days = len(data) - 1
market_participation = trading_days / total_trading_days * 100

# --- 輸出結果 ---
print('\n' + '=' * 70)
print(f'回測結果報告：{ticker} ({start_date} 至 {end_date})')
print('=' * 70)

print(f'\n【核心績效指標】')
print('-' * 70)
print(f'{"指標":<25} | {"策略":>12} | {"Buy & Hold":>12} | {"勝負":>8}')
print('-' * 70)

winner_total = "策略勝" if strategy_total_ret > buyhold_total_ret else "BH 勝"
winner_cagr = "策略勝" if cagr_strategy > cagr_buyhold else "BH 勝"
winner_sharpe = "策略勝" if sharpe_strategy > sharpe_buyhold else "BH 勝"
winner_mdd = "策略勝" if mdd_strategy > mdd_buyhold else "BH 勝"  # 回撤越小越好（絕對值小）

print(f'{"總報酬率":<25} | {strategy_total_ret:>11.2%} | {buyhold_total_ret:>11.2%} | {winner_total:>8}')
print(f'{"年化報酬率 (CAGR)":<25} | {cagr_strategy:>11.2%} | {cagr_buyhold:>11.2%} | {winner_cagr:>8}')
print(f'{"夏普比率":<25} | {sharpe_strategy:>11.2f} | {sharpe_buyhold:>11.2f} | {winner_sharpe:>8}')
print(f'{"最大回撤":<25} | {mdd_strategy:>11.2%} | {mdd_buyhold:>11.2%} | {winner_mdd:>8}')
print(f'{"市場參與率":<25} | {market_participation:>11.1f}% | {"N/A":>12} | {"-":>8}')
print('-' * 70)

print(f'\n【交易統計】')
print(f'  - 總交易天數：{total_trading_days:,} 天')
print(f'  - 持倉天數：{int(trading_days):,} 天')
print(f'  - 空倉天數：{int(total_trading_days - trading_days):,} 天')

# 判斷最終結果
if strategy_total_ret > buyhold_total_ret:
    print(f'\n✅ 恭喜！技術分析策略打敗了 Buy & Hold！')
    print(f'   超額報酬：{(strategy_total_ret - buyhold_total_ret)*100:.2f}%')
else:
    print(f'\n❌ Buy & Hold 表現較佳')
    print(f'   落後幅度：{(buyhold_total_ret - strategy_total_ret)*100:.2f}%')

# 繪圖
print('\n正在生成視覺化圖表...')
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# 圖 1: 累積報酬比較
ax1 = axes[0]
ax1.plot(data.index, data['Cumulative_BuyHold'], label='Buy & Hold', color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax1.plot(data.index, data['Cumulative_Strategy'], label='Technical Strategy', color='#2E86AB', linewidth=2.5)
ax1.set_title(f'20-Year Backtest: Strategy vs. Buy & Hold ({ticker})', fontsize=16, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative Return (Growth of $1)', fontsize=12)
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# 添加績效標籤
ax1.text(0.02, 0.95, f'Strategy Total: {strategy_total_ret:.2%}', transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.text(0.02, 0.88, f'B&H Total: {buyhold_total_ret:.2%}', transform=ax1.transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# 圖 2: 技術指標與交易訊號
ax2 = axes[1]
ax2.plot(data.index, data['Close'], label='SPY Close Price', color='black', linewidth=1.5, alpha=0.7)
ax2.plot(data.index, data['EMA_20'], label='EMA 20', color='#E63946', linewidth=1.2)
ax2.plot(data.index, data['EMA_50'], label='EMA 50', color='#457B9D', linewidth=1.2)

# 標記交易訊號區域
signal_colors = np.where(data['Position'] == 1, 'rgba(46, 134, 171, 0.2)', 'rgba(200, 200, 200, 0)')
for i in range(1, len(data)):
    if data['Position'].iloc[i] == 1:
        ax2.axvspan(data.index[i-1], data.index[i], color='green', alpha=0.15)

ax2.set_title('Technical Indicators & Trading Signals', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Price', fontsize=12)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_result.png', dpi=150, bbox_inches='tight')
print('圖表已儲存為 backtest_result.png')

plt.show()

print('\n' + '=' * 70)
print('回測執行完畢！')
print('=' * 70)
