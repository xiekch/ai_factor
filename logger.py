#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志配置模块：设置统一的日志系统
"""

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_scorer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("优化版AI因子评分脚本启动")

# 屏蔽第三方库的INFO日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)