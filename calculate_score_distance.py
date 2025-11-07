#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算两个CSV文件中相同ID评分的曼哈顿距离
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import os

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    加载并清理CSV数据，处理重复记录
    """
    df = pd.read_csv(file_path)
    
    # 检查是否有重复的ID
    duplicate_ids = df[df.duplicated('id', keep=False)]
    if not duplicate_ids.empty:
        print(f"警告: 文件 {file_path} 中发现重复ID，将保留最后一条记录")
        print(f"重复ID数量: {len(duplicate_ids['id'].unique())}")
        
        # 保留每个ID的最后一条记录
        df = df.drop_duplicates('id', keep='last')
    
    return df

def calculate_manhattan_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    计算两个向量的曼哈顿距离
    """
    if len(vector1) != len(vector2):
        raise ValueError("向量长度不一致")
    
    return sum(abs(a - b) for a, b in zip(vector1, vector2))

def get_score_vector(row: pd.Series) -> List[float]:
    """
    从数据行中提取评分向量
    """
    score_columns = [
        'Fundamental_Impact',
        'Impact_Cycle_Length', 
        'Timeliness_Weight',
        'Information_Certainty',
        'Information_Relevance'
    ]
    
    return [row[col] for col in score_columns]

def calculate_score_distances(
    file1_path: str, 
    file2_path: str,
    output_path: Union[str, None] = None
) -> Dict[str, Any]:
    """
    计算两个文件中相同ID的评分距离
    
    Args:
        file1_path: 第一个CSV文件路径
        file2_path: 第二个CSV文件路径
        output_path: 输出结果文件路径（可选）
    
    Returns:
        包含距离统计和详细结果的字典
    """
    
    # 加载数据
    print("正在加载数据...")
    df1 = load_and_clean_data(file1_path)
    df2 = load_and_clean_data(file2_path)
    
    print(f"文件1记录数: {len(df1)}")
    print(f"文件2记录数: {len(df2)}")
    
    # 找到共同的ID
    common_ids = set(df1['id']).intersection(set(df2['id']))
    print(f"共同ID数量: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("警告: 没有找到共同的ID")
        return {}
    
    # 筛选共同ID的数据
    df1_common = df1[df1['id'].isin(common_ids)].set_index('id')
    df2_common = df2[df2['id'].isin(common_ids)].set_index('id')
    
    # 确保顺序一致
    common_ids_list = list(common_ids)
    df1_common = df1_common.loc[common_ids_list]
    df2_common = df2_common.loc[common_ids_list]
    
    # 计算距离
    results: List[Dict[str, Any]] = []
    distances: List[float] = []
    
    for id_val in common_ids:
        row1: pd.Series = df1_common.loc[id_val]
        row2: pd.Series = df2_common.loc[id_val]
        
        vector1 = get_score_vector(row1)
        vector2 = get_score_vector(row2)
        
        distance = calculate_manhattan_distance(vector1, vector2)
        distances.append(distance)
        
        results.append({
            'id': id_val,
            'stock_code': row1['stock_code'],
            'pub_time': row1['pub_time'],
            'distance': distance,
            'file1_Fundamental_Impact': row1['Fundamental_Impact'],
            'file2_Fundamental_Impact': row2['Fundamental_Impact'],
            'file1_Impact_Cycle_Length': row1['Impact_Cycle_Length'],
            'file2_Impact_Cycle_Length': row2['Impact_Cycle_Length'],
            'file1_Timeliness_Weight': row1['Timeliness_Weight'],
            'file2_Timeliness_Weight': row2['Timeliness_Weight'],
            'file1_Information_Certainty': row1['Information_Certainty'],
            'file2_Information_Certainty': row2['Information_Certainty'],
            'file1_Information_Relevance': row1['Information_Relevance'],
            'file2_Information_Relevance': row2['Information_Relevance']
        })
    
    # 统计信息
    distances_array: np.ndarray = np.array(distances)
    stats: Dict[str, float] = {
        'total_pairs': float(len(common_ids)),
        'mean_distance': float(np.mean(distances_array)),
        'std_distance': float(np.std(distances_array)),
        'min_distance': float(np.min(distances_array)),
        'max_distance': float(np.max(distances_array)),
        'median_distance': float(np.median(distances_array)),
        'q25_distance': float(np.percentile(distances_array, 25)),
        'q75_distance': float(np.percentile(distances_array, 75))
    }
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 按距离排序
    results_df = results_df.sort_values('distance', ascending=False)
    
    # 输出结果
    print("\n=== 距离统计 ===")
    print(f"总对数: {stats['total_pairs']:.0f}")
    print(f"平均距离: {stats['mean_distance']:.4f}")
    print(f"距离标准差: {stats['std_distance']:.4f}")
    print(f"最小距离: {stats['min_distance']:.4f}")
    print(f"最大距离: {stats['max_distance']:.4f}")
    print(f"中位数距离: {stats['median_distance']:.4f}")
    print(f"25%分位数: {stats['q25_distance']:.4f}")
    print(f"75%分位数: {stats['q75_distance']:.4f}")
    
    print("\n=== 距离最大的前10个记录 ===")
    print(results_df.head(10)[['id', 'stock_code', 'pub_time', 'distance']].to_string(index=False))
    
    # 保存结果
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n详细结果已保存到: {output_path}")
    
    return {
        'statistics': stats,
        'detailed_results': results_df,
        'raw_data': {
            'file1': df1_common,
            'file2': df2_common
        }
    }

def main():
    """主函数"""
    # 文件路径
    # 默认输出目录与脚本同级的 output 文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    file1_path = os.path.join(output_dir, 'ai_score_with_thinking.csv')
    file2_path = os.path.join(output_dir, 'ai_score_without_thinking.csv')
    output_path = os.path.join(output_dir, 'score_distance_results.csv')
    
    # 检查文件是否存在
    for path in [file1_path, file2_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            return
    
    print("开始计算曼哈顿距离...")
    print(f"文件1: {file1_path}")
    print(f"文件2: {file2_path}")
    
    try:
        results = calculate_score_distances(file1_path, file2_path, output_path)
        
        if results:
            print("\n=== 计算完成 ===")
            
            # 显示距离分布
            distances = results['detailed_results']['distance'].values
            print(f"\n距离分布:")
            print(f"0-0.1: {len([d for d in distances if d <= 0.1])} 条")
            print(f"0.1-0.2: {len([d for d in distances if 0.1 < d <= 0.2])} 条")
            print(f"0.2-0.3: {len([d for d in distances if 0.2 < d <= 0.3])} 条")
            print(f"0.3-0.4: {len([d for d in distances if 0.3 < d <= 0.4])} 条")
            print(f"0.4-0.5: {len([d for d in distances if 0.4 < d <= 0.5])} 条")
            print(f">0.5: {len([d for d in distances if d > 0.5])} 条")
            
        else:
            print("计算失败")
            
    except Exception as e:
        print(f"计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()