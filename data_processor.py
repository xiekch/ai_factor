#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理模块：处理文件加载和保存的功能
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Set

from logger import logger

class DataProcessor:
    """处理文件加载和保存的功能"""
    
    @staticmethod
    def load_data_from_json_file(file_path: str) -> Optional[List[Dict[str, Any]]]:
        """从JSON文件加载数据"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # 检查文件是否为空
            if not raw_content.strip():
                logger.warning(f"文件 {file_path} 为空，跳过。")
                return None

            # 加载数据
            data = json.loads(raw_content)[10]  # 注意这里的索引[10]是根据原代码逻辑保留的
            logger.info(f"成功从 {file_path} 加载了 {len(data)} 条记录。")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"文件 {file_path} 不是有效的 JSON 格式: {e}")
            return None
        except FileNotFoundError:
            logger.error(f"找不到文件 {file_path}")
            return None
        except Exception as e:
            logger.error(f"加载文件时发生未知错误: {e}")
            return None
    
    @staticmethod
    def find_stock_json_files(directory: str) -> List[str]:
        """查找目录中的股票JSON文件"""
        found_files: List[str] = []
        try:
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    stock_code = filename.split(".")[0]
                    if stock_code.isdigit():
                        found_files.append(os.path.join(directory, filename))
        except FileNotFoundError:
            logger.error(f"找不到目录 {directory}。请检查配置")
        return found_files
    
    @staticmethod
    def load_processed_codes(output_csv_path: str) -> Set[str]:
        """加载已处理的股票代码"""
        processed_codes: Set[str] = set()
        if os.path.exists(output_csv_path):
            try:
                df_existing = pd.read_csv(output_csv_path)
                processed_codes = set(df_existing["stock_code"].astype(str).unique())
                logger.info(f"检测到已存在的CSV，已加载 {len(processed_codes)} 个处理过的股票代码")
            except Exception as e:
                logger.error(f"无法读取已存在的CSV: {e}。将作为新文件处理")
        return processed_codes
    
    @staticmethod
    def save_results(results: List[Dict[str, Any]], output_csv_path: str):
        """保存结果到CSV文件"""
        if not results:
            return
        
        df_new = pd.DataFrame(results)
        file_exists = os.path.exists(output_csv_path)
        
        df_new.to_csv(
            output_csv_path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )
        
        logger.info(f"{len(results)} 条结果已追加至 {output_csv_path}")
    
    @staticmethod
    def save_failed_tasks(failed_tasks: List[Dict[str, Any]], failed_json_path: str):
        """保存失败的任务到JSON文件"""
        if not failed_tasks:
            return
        
        with open(failed_json_path, "w", encoding="utf-8") as f:
            json.dump(failed_tasks, f, ensure_ascii=False, indent=4)
        
        logger.warning(f"总计有 {len(failed_tasks)} 条记录处理失败，详情请查看 {failed_json_path}")