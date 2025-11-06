#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
程序入口模块：启动评分流程的主入口
"""

from logger import logger
from stock_scorer import StockScorer

if __name__ == "__main__":
    try:
        scorer = StockScorer()
        scorer.run()
    except KeyboardInterrupt:
        logger.info("\n用户中断程序...")
    except Exception as e:
        logger.error(f"程序运行过程中发生异常: {e}", exc_info=True)