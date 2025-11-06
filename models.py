#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据模型定义模块：定义评分结果的数据结构
"""

from pydantic import BaseModel, Field

class StockScore(BaseModel):
    """评分结果的数据模型"""
    Fundamental_Positive: float = Field(..., ge=0.0, le=1.0, description="基础面积极因素评分")
    Impact_Cycle_Length: float = Field(..., ge=0.0, le=1.0, description="影响周期长度评分")
    Timeliness_Weight: float = Field(..., ge=0.0, le=1.0, description="时效性权重评分")
    Information_Certainty: float = Field(..., ge=0.0, le=1.0, description="信息确定性评分")
    Information_Relevance: float = Field(..., ge=0.0, le=1.0, description="信息相关性评分")