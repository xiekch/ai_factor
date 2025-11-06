"""
AI 因子独立回测脚本：
- 加载 AI 因子信号 (ceshi.csv) 和股票日线数据 (stock_daily_basic_data_2025.csv)。
- 实现了一套基于信号发布时间 (pub_time) 的精细化入场逻辑（区分盘前、盘中、盘后）。
- 根据信号中的 'time_horizon' 决定持仓周期。
- 计算并统计回测结果，包括总体、按多空、按周期及交叉维度的平均收益率。
- 将所有交易详情保存到 backtest_trades_results_final_entry.csv。
"""

import pandas as pd
import datetime

# --- 1. 策略参数定义 ---

# 定义持仓周期映射关系
HORIZON_MAP = {
    '短期': 5,   # 持有5个交易日
    '中期': 20,  # 持有20个交易日
    '长期': 60   # 持有60个交易日
}

# 定义信号阈值
CONF_THRESHOLD = 0.5         # event_driven 和 novelty 的置信度阈值
SENT_POSITIVE_THRESHOLD = 0.3  # 情绪看涨阈值
SENT_NEGATIVE_THRESHOLD = -0.3 # 情绪看跌阈值

# 定义时间边界 (用于区分盘前、盘中、盘后)
TIME_0900 = datetime.time(9, 0, 0)
TIME_1500 = datetime.time(15, 0, 0)

def run_backtest_final_entry():
    """
    执行AI因子回测策略的函数。
    
    最终版入场逻辑 (基于 pub_time 和是否为交易日):
    1.  如果 time < 09:00 (盘前, e.g., 周一 8:00):
        - T日 (>= signal_date) 开盘价 (open)
    2.  如果 09:00 <= time <= 15:00 (盘中):
        - 2a. 如果 T日是交易日 (e.g., 周一 10:00):
            - T日 (>= signal_date) 收盘价 (close)
        - 2b. 如果 T日是非交易日 (e.g., 周六 10:00):
            - T+1日 (> signal_date) 开盘价 (open)
    3.  如果 time > 15:00 (盘后, e.g., 周五 16:00):
        - T+1日 (> signal_date) 开盘价 (open)
    
    出场逻辑：
    - 在入场日后的第 N (holding_period) 个交易日，按 收盘价(close) 出场。
    """
    try:
        # --- 2. 加载数据 ---
        print("开始加载数据...")
        stock_df = pd.read_csv("stock_daily_basic_data_2025.csv")
        factor_df = pd.read_csv("ceshi.csv")

        # --- 3. 预处理股票数据 ---
        stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'], format='%Y%m%d')
        stock_df['stock_code_match'] = stock_df['ts_code'].str.split('.').str[0].astype(str)
        # 排序对后续 .iloc 定位至关重要
        stock_df = stock_df.sort_values(by=['stock_code_match', 'trade_date']) 

        # --- 4. 预处理因子数据 ---
        factor_df['pub_time'] = pd.to_datetime(factor_df['pub_time'], format='%Y/%m/%d %H:%M')
        factor_df['signal_date_dt'] = factor_df['pub_time'].dt.date
        factor_df['stock_code'] = factor_df['stock_code'].astype(str)

        # --- 5. 创建股票数据缓存 (核心优化) ---
        # 缓存中不仅存储数据，还存储一个交易日集合(set)以便 O(1) 快速查询
        stock_data_cache = {}
        for code, data in stock_df.groupby('stock_code_match'):
            data_reset = data.reset_index(drop=True)
            days_set = set(data_reset['trade_date'].dt.date)
            stock_data_cache[code] = (data_reset, days_set)
        
        print(f"股票数据已加载，包含 {len(stock_data_cache)} 只独立股票。")
        print(f"因子数据已加载，包含 {len(factor_df)} 条信号。")

        # --- 6. 遍历因子，执行回测 ---
        results = []
        print("开始执行回测...")

        for _, signal in factor_df.iterrows():
            stock_code_str = signal['stock_code']
            
            # 检查缓存中是否存在该股票数据
            if stock_code_str not in stock_data_cache:
                continue
            
            # 从缓存中获取 DataFrame 和 交易日集合
            stock_specific_df, trading_days_set = stock_data_cache[stock_code_str]
            
            # 获取持仓周期
            holding_period = HORIZON_MAP.get(signal['time_horizon'])
            if holding_period is None:
                continue

            # 信号触发逻辑
            sent = signal['sentiment_score']
            event = signal['event_driven_score']
            novel = signal['novelty_score']
            
            trade_type = None
            if (sent > SENT_POSITIVE_THRESHOLD and event > CONF_THRESHOLD and novel > CONF_THRESHOLD):
                trade_type = 'long'
            elif (sent < SENT_NEGATIVE_THRESHOLD and event > CONF_THRESHOLD and novel > CONF_THRESHOLD):
                trade_type = 'short'
            
            if trade_type is None:
                continue

            # --- 7. 核心：精细化入场逻辑 ---
            
            signal_datetime = signal['pub_time']
            signal_date = signal['signal_date_dt']
            signal_time = signal_datetime.time()
            
            # 检查信号日当天是否为交易日 (O(1) 复杂度)
            is_signal_day_trading_day = signal_date in trading_days_set

            entry_row = None
            entry_price = 0.0
            entry_index = -1
            
            if signal_time < TIME_0900:
                # 规则 1: "9点前" (e.g., 周日 8:00, 周一 8:00)
                # T日 (>= signal_date) 开盘价 (open)
                entry_candidates = stock_specific_df[stock_specific_df['trade_date'].dt.date >= signal_date]
                if not entry_candidates.empty:
                    entry_row = entry_candidates.iloc[0]
                    entry_price = entry_row['open']
                    entry_index = entry_row.name # .name 引用的是 reset_index 后的行号
            
            elif signal_time <= TIME_1500:
                # 规则 2: "9点到3点" (e.g., 周一 10:00 OR 周六 10:00)
                if is_signal_day_trading_day:
                    # 2a: 信号在交易时段内 (e.g., 周一 10:00)
                    # T日 (>= signal_date) 收盘价 (close)
                    entry_candidates = stock_specific_df[stock_specific_df['trade_date'].dt.date >= signal_date]
                    if not entry_candidates.empty:
                        entry_row = entry_candidates.iloc[0]
                        entry_price = entry_row['close']
                        entry_index = entry_row.name
                else:
                    # 2b: 信号在非交易日 (e.g., 周六 10:00)
                    # 视为 T+1 盘前信号
                    # T+1日 (> signal_date) 开盘价 (open)
                    entry_candidates = stock_specific_df[stock_specific_df['trade_date'].dt.date > signal_date]
                    if not entry_candidates.empty:
                        entry_row = entry_candidates.iloc[0]
                        entry_price = entry_row['open']
                        entry_index = entry_row.name
            
            else:
                # 规则 3: "3点后" (e.g., 周五 16:00)
                # T+1日 (> signal_date) 开盘价 (open)
                entry_candidates = stock_specific_df[stock_specific_df['trade_date'].dt.date > signal_date]
                if not entry_candidates.empty:
                    entry_row = entry_candidates.iloc[0]
                    entry_price = entry_row['open']
                    entry_index = entry_row.name
            
            # 如果未能找到入场点 (例如数据太旧，或当天及之后无数据)
            if entry_row is None:
                continue

            entry_date = entry_row['trade_date']

            # --- 8. 出场逻辑 ---
            # 基于入场行号 (entry_index) 和持仓周期计算出场行号
            exit_index = entry_index + holding_period
            
            if exit_index >= len(stock_specific_df):
                continue # 数据不足以支持持有到期
                
            exit_row = stock_specific_df.iloc[exit_index]
            exit_date = exit_row['trade_date']
            exit_price = exit_row['close'] # 统一按收盘价卖出

            # --- 9. 计算 PnL 和胜率 ---
            pct_return = 0.0
            if trade_type == 'long':
                pct_return = (exit_price - entry_price) / entry_price
            else: # short
                pct_return = (entry_price - exit_price) / entry_price
            
            is_win = pct_return > 0
            
            # 记录结果
            results.append({
                'signal_id': signal['id'],
                'stock_code': stock_code_str,
                'pub_time': signal_datetime,
                'time_horizon': signal['time_horizon'],
                'holding_period_days': holding_period,
                'trade_type': trade_type,
                'entry_date': entry_date.date(),
                'entry_price': entry_price,
                'exit_date': exit_date.date(),
                'exit_price': exit_price,
                'pct_return': pct_return,
                'is_win': is_win
            })
        
        print("回测执行完毕，开始统计结果...")
        # --- 10. 分析和展示结果 ---
        if not results:
            print("回测结束，但未执行任何交易。")
            return

        trades_df = pd.DataFrame(results)
        
        # 保存交易详情
        trades_df.to_csv("backtest_trades_results_final_entry.csv", index=False, encoding='utf-8-sig')
        print(f"\n成功执行 {len(trades_df)} 笔交易。")
        print("所有交易详情已保存到 'backtest_trades_results_final_entry.csv'")

        # --- 按收益率统计 ---
        
        # 计算总体平均收益率
        overall_avg_return = trades_df['pct_return'].mean()

        print("\n--- 回测结果摘要 (按平均收益率) ---")
        print(f"总交易次数: {len(trades_df)}")
        print(f"总体平均收益率: {overall_avg_return:.4%} (即每笔交易平均)")

        # 按交易类型（多/空）分析
        print("\n--- 按交易类型 (多/空) 统计 (平均收益率) ---")
        type_stats_return = trades_df.groupby('trade_type')['pct_return'].agg(['count', 'mean'])
        type_stats_return.columns = ['交易次数', '平均收益率']
        type_stats_return['平均收益率'] = type_stats_return['平均收益率'].map('{:.4%}'.format)
        print(type_stats_return)

        # 按持仓周期 (horizon) 分析
        print("\n--- 按持仓周期 (短期/中期/长期) 统计 (平均收益率) ---")
        horizon_stats_return = trades_df.groupby('time_horizon')['pct_return'].agg(['count', 'mean'])
        horizon_stats_return.columns = ['交易次数', '平均收益率']
        horizon_stats_return['平均收益率'] = horizon_stats_return['平均收益率'].map('{:.4%}'.format)
        # 确保按 '短期', '中期', '长期' 的顺序显示
        horizon_order = [h for h in HORIZON_MAP.keys() if h in horizon_stats_return.index]
        print(horizon_stats_return.loc[horizon_order])
        
        # 按类型和周期交叉分析
        print("\n--- 交叉统计: 类型 vs 周期 (平均收益率) ---")
        cross_stats_return = trades_df.groupby(['trade_type', 'time_horizon'])['pct_return'].agg(['count', 'mean'])
        cross_stats_return.columns = ['交易次数', '平均收益率']
        cross_stats_return['平均收益率'] = cross_stats_return['平均收益率'].map('{:.4%}'.format)
        if not cross_stats_return.empty:
            # 确保交叉统计也按周期排序
            cross_stats_return = cross_stats_return.reindex(level=1, index=horizon_order)
        print(cross_stats_return)


    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
    except Exception as e:
        print(f"执行回测时发生错误: {e}")

# --- 11. 执行回测 ---
if __name__ == "__main__":
    run_backtest_final_entry()