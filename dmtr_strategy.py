"""
=============================================================================
雙動能趨勢輪動策略 (Dual-Momentum Trend Rotation, DMTR)
=============================================================================
策略說明:
    本策略結合「絕對動能」與「相對動能」，在不使用槓桿的情況下，
    透過嚴格的擇時機制打敗 Buy & Hold。

核心邏輯:
    1. 趨勢過濾：價格 > 200日均線 (確保長期多頭)
    2. 動能確認：短期動能 (3個月) 或 長期動能 (12個月) 為正
    3. 部位控制：Long Only (0% 或 100%)，無槓桿

適用標的:
    - 美股：SPY (S&P 500 ETF)
    - 台股：0050.TW (台灣 50 ETF)

回測期間:
    - 美股：2001-2025 (25年)
    - 台股：2003-2025 (因 0050 於 2003 年上市)

作者：量化策略研究
日期：2025
=============================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 設定繪圖風格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei']  # 支援中文
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 資料下載與預處理 (Data Preparation)
# =============================================================================
def get_data(ticker, start_date='2001-01-01'):
    """
    下載股票數據並進行預處理：
    - 處理 yfinance 多層級索引問題
    - 向前填充缺值 (Forward Fill)
    - 確保價格欄位為數值型態
    
    參數:
        ticker: 股票代號 (如 'SPY', '0050.TW')
        start_date: 起始日期
    
    返回:
        DataFrame: 包含 OHLCV 數據的 DataFrame
    """
    print(f"正在下載 {ticker} 數據...")
    df = yf.download(ticker, start=start_date, end='2025-12-31', progress=False)
    
    if df.empty:
        print(f"警告：{ticker} 無數據返回")
        return None
    
    # 處理 yfinance 新版本的多層級索引問題
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    
    # 只保留必要欄位
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols]
    
    # 處理缺值：向前填充 (休市時價格不變)
    df = df.ffill()
    # 若開頭仍有缺值，向後填充
    df = df.bfill()
    
    # 確保 Close 價格為數值，並移除無法修復的缺值
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    
    print(f"  ✓ {ticker} 數據下載完成：{len(df)} 筆交易記錄 ({df.index[0].date()} ~ {df.index[-1].date()})")
    
    return df


# =============================================================================
# 2. 技術指標實作 (Technical Indicators)
# =============================================================================
def calculate_indicators(df):
    """
    計算 DMTR 策略所需的技術指標：
    
    1. SMA_200: 200日簡單移動平均線 (長期趨勢過濾器)
       公式：SMA = Sum(Close, 200) / 200
    
    2. ROC_3m: 3個月動能 (Rate of Change)
       公式：ROC = (Close_today - Close_63days_ago) / Close_63days_ago
       說明：63交易日 ≈ 3個月
    
    3. ROC_12m: 12個月動能
       公式：ROC = (Close_today - Close_252days_ago) / Close_252days_ago
       說明：252交易日 ≈ 12個月
    
    4. Vol_20: 20日滾動波動率 (年化)
       公式：Vol = Std(Returns, 20) * sqrt(252)
    
    參數:
        df: 包含價格數據的 DataFrame
    
    返回:
        DataFrame: 新增指標欄位的 DataFrame
    """
    df = df.copy()
    
    # 1. 200日均線 - 長期趨勢過濾器
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. 動能指標 (Rate of Change)
    # 3個月約 63 交易日，12個月約 252 交易日
    df['ROC_3m'] = df['Close'].pct_change(periods=63)
    df['ROC_12m'] = df['Close'].pct_change(periods=252)
    
    # 3. 波動率計算 (用於風險評估)
    df['Returns'] = df['Close'].pct_change()
    df['Vol_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # 移除因滾動計算產生的 NaN 值
    df = df.dropna()
    
    print(f"  ✓ 指標計算完成，有效數據筆數：{len(df)}")
    
    return df


# =============================================================================
# 3. 交易訊號產生 (Signal Generation)
# =============================================================================
def generate_signals(df):
    """
    產生 DMTR 策略的交易訊號：
    
    進場條件 (Hold = 1):
        1. 價格 > SMA_200 (處於長期多頭趨勢)
        2. (ROC_3m > 0) OR (ROC_12m > 0) (至少一個動能為正)
    
    出場條件 (Hold = 0):
        1. 價格 < SMA_200 (跌破長期趨勢)
        2. 或動能極度惡化 (選用條件)
    
    執行規則:
        - T+1 執行：今天的訊號明天才執行，避免未來函數
    
    參數:
        df: 包含指標的 DataFrame
    
    返回:
        DataFrame: 新增 Signal 和 Strategy_Position 欄位
    """
    df = df.copy()
    
    # 初始化訊號欄位
    df['Signal'] = 0
    
    # 條件 1: 價格在 200 日均線之上 (長期多頭趨勢)
    cond_trend = df['Close'] > df['SMA_200']
    
    # 條件 2: 動能保護 (短期或長期動能為正)
    # 這確保即使在均線之上，如果動能極度疲軟也不會進場
    cond_momentum = (df['ROC_3m'] > 0) | (df['ROC_12m'] > 0)
    
    # 組合條件：同時滿足趨勢和動能條件
    df.loc[cond_trend & cond_momentum, 'Signal'] = 1
    
    # 延遲一天執行訊號 (T+1)，避免使用未來數據
    df['Strategy_Position'] = df['Signal'].shift(1)
    df['Strategy_Position'] = df['Strategy_Position'].fillna(0)
    
    # 統計訊號分佈
    hold_days = (df['Strategy_Position'] == 1).sum()
    cash_days = (df['Strategy_Position'] == 0).sum()
    print(f"  ✓ 訊號產生完成：持有天數 {hold_days} ({hold_days/len(df)*100:.1f}%), 空手天數 {cash_days} ({cash_days/len(df)*100:.1f}%)")
    
    return df


# =============================================================================
# 4. 報酬計算 (Return Calculation)
# =============================================================================
def calculate_returns(df, transaction_cost=0.001):
    """
    計算策略報酬與基準報酬：
    
    - Market_Return: Buy & Hold 的日報酬
    - Strategy_Return: 策略的日報酬 (考慮交易成本)
    - Cumulative_*: 累積報酬曲線
    
    參數:
        df: 包含部位訊號的 DataFrame
        transaction_cost: 單邊交易成本 (預設 0.1%)
    
    返回:
        DataFrame: 包含報酬和累積報酬的 DataFrame
    """
    df = df.copy()
    
    # 市場報酬 (Buy & Hold)
    df['Market_Return'] = df['Close'].pct_change()
    
    # 策略報酬：今天持有的部位 * 今天的市場報酬
    df['Strategy_Return'] = df['Strategy_Position'] * df['Market_Return']
    
    # 扣除交易成本 (當部位改變時)
    df['Position_Change'] = df['Strategy_Position'].diff().abs()
    df['Transaction_Cost'] = df['Position_Change'] * transaction_cost
    df['Strategy_Return'] = df['Strategy_Return'] - df['Transaction_Cost']
    
    # 計算累積報酬 (Net Value)
    # 假設初始淨值為 1
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    
    print(f"  ✓ 報酬計算完成 (已扣除 {transaction_cost*100:.2f}% 交易成本)")
    
    return df


# =============================================================================
# 5. 績效分析 (Performance Analysis)
# =============================================================================
def analyze_performance(df, name):
    """
    計算並輸出關鍵績效指標：
    
    - 總報酬 (Total Return)
    - 年化報酬 (CAGR)
    - 最大回撤 (Max Drawdown)
    - 夏普比率 (Sharpe Ratio)
    - 卡瑪比率 (Calmar Ratio)
    
    參數:
        df: 包含累積報酬的 DataFrame
        name: 市場名稱 (如 'SPY', '0050.TW')
    
    返回:
        dict: 包含各項績效指標的字典
    """
    # 計算總報酬
    total_return_s = df['Cumulative_Strategy'].iloc[-1] - 1
    total_return_bh = df['Cumulative_Market'].iloc[-1] - 1
    
    # 計算年化報酬 (CAGR)
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr_s = (df['Cumulative_Strategy'].iloc[-1]) ** (1/years) - 1
    cagr_bh = (df['Cumulative_Market'].iloc[-1]) ** (1/years) - 1
    
    # 計算最大回撤 (Max Drawdown)
    cum_max_s = df['Cumulative_Strategy'].cummax()
    drawdown_s = (df['Cumulative_Strategy'] - cum_max_s) / cum_max_s
    max_dd_s = drawdown_s.min()
    
    cum_max_bh = df['Cumulative_Market'].cummax()
    drawdown_bh = (df['Cumulative_Market'] - cum_max_bh) / cum_max_bh
    max_dd_bh = drawdown_bh.min()
    
    # 計算夏普比率 (假設無風險利率 2%)
    rf = 0.02
    excess_ret_s = df['Strategy_Return'] - rf/252
    excess_ret_bh = df['Market_Return'] - rf/252
    
    sharpe_s = np.sqrt(252) * excess_ret_s.mean() / df['Strategy_Return'].std() if df['Strategy_Return'].std() != 0 else 0
    sharpe_bh = np.sqrt(252) * excess_ret_bh.mean() / df['Market_Return'].std() if df['Market_Return'].std() != 0 else 0
    
    # 計算卡瑪比率 (Calmar Ratio = CAGR / |MaxDD|)
    calmar_s = cagr_s / abs(max_dd_s) if max_dd_s != 0 else 0
    calmar_bh = cagr_bh / abs(max_dd_bh) if max_dd_bh != 0 else 0
    
    # 輸出績效報告
    print("\n" + "="*70)
    print(f"{name} 績效分析報告 ({df.index[0].date()} ~ {df.index[-1].date()})")
    print("="*70)
    print(f"{'指標':<20} | {'DMTR 策略':>12} | {'Buy & Hold':>12} | {'勝負':>8}")
    print("-"*70)
    
    def compare(val_s, val_bh, lower_is_better=False):
        if lower_is_better:
            return "✅ 勝" if val_s > val_bh else "❌ 輸"  # 負得少比較好
        else:
            return "✅ 勝" if val_s > val_bh else "❌ 輸"
    
    print(f"{'總報酬':<20} | {total_return_s:>11.2%} | {total_return_bh:>11.2%} | {compare(total_return_s, total_return_bh)}")
    print(f"{'年化報酬 (CAGR)':<20} | {cagr_s:>11.2%} | {cagr_bh:>11.2%} | {compare(cagr_s, cagr_bh)}")
    print(f"{'最大回撤':<20} | {max_dd_s:>11.2%} | {max_dd_bh:>11.2%} | {compare(max_dd_s, max_dd_bh, lower_is_better=True)}")
    print(f"{'夏普比率':<20} | {sharpe_s:>11.2f} | {sharpe_bh:>11.2f} | {compare(sharpe_s, sharpe_bh)}")
    print(f"{'卡瑪比率':<20} | {calmar_s:>11.2f} | {calmar_bh:>11.2f} | {compare(calmar_s, calmar_bh)}")
    print("="*70)
    
    # 牛熊市分析
    print("\n牛熊市表現分析:")
    print("-"*70)
    
    # 定義牛市/熊市：以 200 日均線斜率判斷
    df['SMA_Slope'] = df['SMA_200'].pct_change(periods=20)  # 20 日斜率
    bull_mask = df['SMA_Slope'] > 0
    bear_mask = df['SMA_Slope'] <= 0
    
    bull_ret_s = df.loc[bull_mask, 'Strategy_Return'].sum()
    bull_ret_bh = df.loc[bull_mask, 'Market_Return'].sum()
    bear_ret_s = df.loc[bear_mask, 'Strategy_Return'].sum()
    bear_ret_bh = df.loc[bear_mask, 'Market_Return'].sum()
    
    print(f"{'市場狀態':<15} | {'天數佔比':>10} | {'策略報酬':>12} | {'B&H 報酬':>12}")
    print("-"*55)
    print(f"{'牛市':<15} | {bull_mask.sum()/len(df)*100:>9.1f}% | {bull_ret_s:>11.2%} | {bull_ret_bh:>11.2%}")
    print(f"{'熊市':<15} | {bear_mask.sum()/len(df)*100:>9.1f}% | {bear_ret_s:>11.2%} | {bear_ret_bh:>11.2%}")
    
    return {
        'market': name,
        'cagr_s': cagr_s, 'cagr_bh': cagr_bh,
        'max_dd_s': max_dd_s, 'max_dd_bh': max_dd_bh,
        'sharpe_s': sharpe_s, 'sharpe_bh': sharpe_bh,
        'calmar_s': calmar_s, 'calmar_bh': calmar_bh,
        'total_s': total_return_s, 'total_bh': total_return_bh,
        'bull_ret_s': bull_ret_s, 'bull_ret_bh': bull_ret_bh,
        'bear_ret_s': bear_ret_s, 'bear_ret_bh': bear_ret_bh
    }


# =============================================================================
# 6. 視覺化圖表 (Visualization)
# =============================================================================
def plot_results(df, title, save_path=None):
    """
    繪製策略績效對比圖表：
    
    圖 1: 累積報酬曲線 (Strategy vs Buy & Hold)
    圖 2: 價格走勢與持倉狀態 (標示持有/空手期間)
    圖 3: 動能指標 (ROC 3M & 12M)
    
    參數:
        df: 包含所有數據的 DataFrame
        title: 圖表標題
        save_path: 可選的儲存路徑
    """
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'DMTR 雙動能趨勢輪動策略 vs Buy & Hold - {title}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # --- 圖 1: 累積報酬對比 ---
    ax1 = axs[0]
    ax1.plot(df.index, df['Cumulative_Strategy'], label='DMTR 策略', color='#2E86AB', linewidth=2.5)
    ax1.plot(df.index, df['Cumulative_Market'], label='Buy & Hold', color='#A23B72', linestyle='--', linewidth=2)
    ax1.set_ylabel('累積報酬 (Cumulative Return)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('圖 1: 累積報酬成長曲線對比', fontsize=14, fontweight='bold')
    ax1.set_ylim(bottom=0.5)  # 從 0.5 開始顯示
    
    # --- 圖 2: 價格走勢與持倉狀態 ---
    ax2 = axs[1]
    ax2.plot(df.index, df['Close'], label='收盤價', color='gray', alpha=0.4, linewidth=1)
    
    # 標記持有期間 (藍色點)
    hold_mask = df['Strategy_Position'] == 1
    ax2.scatter(df.index[hold_mask], df['Close'][hold_mask], 
                color='#2E86AB', s=15, label='持有部位 (Long)', alpha=0.7, edgecolors='none')
    
    # 標記空手期間 (橘色點)
    cash_mask = df['Strategy_Position'] == 0
    ax2.scatter(df.index[cash_mask], df['Close'][cash_mask], 
                color='#F18F01', s=15, label='空手觀望 (Cash)', alpha=0.7, edgecolors='none')
    
    # 繪製 200 日均線
    ax2.plot(df.index, df['SMA_200'], label='200 日均線 (SMA200)', color='black', linestyle=':', linewidth=1.5)
    
    ax2.set_ylabel('價格 (Price)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('圖 2: 價格走勢、200 日均線與持倉狀態', fontsize=14, fontweight='bold')
    
    # --- 圖 3: 動能指標 ---
    ax3 = axs[2]
    ax3.plot(df.index, df['ROC_3m'], label='3 月動能 (ROC 3M)', color='green', linewidth=1.2, alpha=0.8)
    ax3.plot(df.index, df['ROC_12m'], label='12 月動能 (ROC 12M)', color='red', linewidth=1.2, alpha=0.8)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='-')
    
    # 填充正負區域
    ax3.fill_between(df.index, df['ROC_3m'], 0, where=(df['ROC_3m'] > 0), 
                     color='green', alpha=0.15)
    ax3.fill_between(df.index, df['ROC_3m'], 0, where=(df['ROC_3m'] < 0), 
                     color='red', alpha=0.15)
    
    ax3.set_ylabel('動能 (Momentum)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_title('圖 3: 動能指標變化 (進出場關鍵訊號)', fontsize=14, fontweight='bold')
    
    # 格式化 X 軸日期
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=0)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 圖表已儲存至：{save_path}")
    
    plt.show()


# =============================================================================
# 7. 主程式執行 (Main Execution)
# =============================================================================
def main():
    """
    主程式入口：
    1. 下載數據
    2. 計算指標
    3. 產生訊號
    4. 計算報酬
    5. 績效分析
    6. 繪製圖表
    """
    print("="*70)
    print("DMTR 雙動能趨勢輪動策略 - 回測系統")
    print("="*70)
    
    results = {}
    
    # -------------------------------------------------------------------------
    # 處理美股 SPY
    # -------------------------------------------------------------------------
    print("\n[1/2] 處理美股 SPY (S&P 500)...")
    spy_data = get_data('SPY', start_date='2001-01-01')
    
    if spy_data is not None:
        spy_data = calculate_indicators(spy_data)
        spy_data = generate_signals(spy_data)
        spy_data = calculate_returns(spy_data, transaction_cost=0.001)  # 0.1% 交易成本
        results['SPY'] = analyze_performance(spy_data, '美股 SPY (S&P 500)')
        plot_results(spy_data, 'SPY (US Market)', save_path='/workspace/spy_performance.png')
    else:
        print("❌ 無法下載 SPY 數據，跳過美股回測")
    
    # -------------------------------------------------------------------------
    # 處理台股 0050.TW
    # -------------------------------------------------------------------------
    print("\n[2/2] 處理台股 0050.TW (台灣 50)...")
    tw_data = get_data('0050.TW', start_date='2001-01-01')
    
    if tw_data is not None:
        tw_data = calculate_indicators(tw_data)
        tw_data = generate_signals(tw_data)
        tw_data = calculate_returns(tw_data, transaction_cost=0.001)  # 0.1% 交易成本
        results['TW0050'] = analyze_performance(tw_data, '台股 0050.TW (台灣 50)')
        plot_results(tw_data, '0050.TW (Taiwan Market)', save_path='/workspace/tw0050_performance.png')
    else:
        print("❌ 無法下載 0050.TW 數據，跳過台股回測")
    
    # -------------------------------------------------------------------------
    # 最終總結
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("最終結論：策略是否全面打敗 Buy & Hold？")
    print("="*70)
    
    all_win = True
    
    for market, res in results.items():
        win_cagr = res['cagr_s'] > res['cagr_bh']
        win_dd = res['max_dd_s'] > res['max_dd_bh']  # 負得少比較好
        win_sharpe = res['sharpe_s'] > res['sharpe_bh']
        
        print(f"\n[{market}]")
        print(f"  年化報酬 (CAGR): {'✅ 勝' if win_cagr else '❌ 輸'} (策略 {res['cagr_s']:.2%} vs B&H {res['cagr_bh']:.2%})")
        print(f"  最大回撤 (MaxDD): {'✅ 勝' if win_dd else '❌ 輸'} (策略 {res['max_dd_s']:.2%} vs B&H {res['max_dd_bh']:.2%})")
        print(f"  夏普比率 (Sharpe): {'✅ 勝' if win_sharpe else '❌ 輸'} (策略 {res['sharpe_s']:.2f} vs B&H {res['sharpe_bh']:.2f})")
        
        if not (win_cagr and win_dd and win_sharpe):
            all_win = False
    
    print("\n" + "="*70)
    if all_win:
        print("🎉 恭喜！DMTR 策略在所有市場中全面打敗 Buy & Hold！")
        print("   - 更高的年化報酬")
        print("   - 更小的最大回撤")
        print("   - 更好的風險調整後報酬 (Sharpe)")
    else:
        print("⚠️  策略在某些指標上未超越 Buy & Hold，請參考詳細報告分析原因")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
