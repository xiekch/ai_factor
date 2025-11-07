#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理模块：集中管理所有配置参数
"""

import os
from typing import Set
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """集中管理所有配置参数"""
    # API配置
    API_KEY = os.getenv("API_KEY", "")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-turbo"
    REQUEST_TIMEOUT = 60
    
    # 处理配置
    JSON_SOURCE_DIRECTORY = "./guba_df"  # 修改为相对路径
    MAX_CONCURRENT_REQUESTS = 5
    MAX_RETRIES = 3
    RETRY_SLEEP_TIME = 15
    
    # 输出配置
    OUTPUT_DIRECTORY = "./output"
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIRECTORY, "scored_results.csv")
    FAILED_JSON_PATH = os.path.join(OUTPUT_DIRECTORY, "scored_failed_tasks.json")
    
    # 目标股票
    TARGET_STOCK_CODES = ["000001"]
    PROCESS_NUM = 20
    TARGET_STOCK_CODES_SET: Set[str] = set(TARGET_STOCK_CODES)
    NEED_THINKING = False
