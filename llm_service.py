#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM服务模块：封装与语言模型交互的功能
"""

import json
import time
from typing import List, Dict, Any, Optional
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import BaseMessage, SystemMessage

from config import Config
from logger import logger

class LLMService:
    """封装LLM调用相关的功能"""
    
    def __init__(self):
        """初始化LLM客户端和解析器"""
        try:
            self.llm = ChatTongyi(model=Config.MODEL_NAME, api_key=Config.API_KEY)
            logger.info(f"成功初始化LangChain LLM客户端，模型: {Config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise
    
    def create_prompt(self, task_item: Dict[str, Any]) -> List[BaseMessage]:
        """创建LangChain提示词"""
        title = task_item.get("title", "") 
        content =  task_item.get("content", "")
        source = task_item.get("source", "unknown")
        publish_time = task_item.get("pub_time", "unknown")
        stock_code = task_item.get("stock_code", "N/A")
        stock_name = task_item.get("stock_name", "N/A")
        
        # 系统提示词
        system_prompt = f"""你是一名顶尖的中国A股市场金融分析师，擅长从海量文本信息中挖掘对股价有影响的信号。
任务：分析以下股票新闻内容，针对每条新闻输出5个AI因子的取值（保留1位小数，范围0-1），每个因子基于新闻内容独立评估。

因子定义及取值说明：

1. 基本面正向变动强度（Fundamental_Positive）  
   - 定义：新闻对公司核心基本面（营收、利润、成本、技术壁垒、资产质量等）的正向影响程度。  
   - 取值1：极端利好，如“公司签订100亿元长期订单，预计年增营收50%”“研发出颠覆性技术，专利保护期20年”。  
   - 取值0：极端利空，如“公司暴雷30亿元财务造假，核心资产被冻结”“主力产品被监管认定为不合格，全面下架”。  
   - 中间值示例：0.3（“季度营收微增2%，但利润率下降1个百分点”）；0.7（“成本端原材料价格下降，预计年度利润提升10%”）。

2. 影响周期长度（Impact_Cycle_Length）
   - 定义：新闻事件对公司的影响持续时间。
   - 取值1：长期影响（如“签订5年独家供应协议”“发布未来3年战略规划，投入50亿研发”）。
   - 取值0：短期影响（如“公司获得1000万短期政府补贴”“单日促销活动，预计增销1000万”）。
   - 中间值：0.7（“发布新款产品，预计贡献未来1年营收”）、0.4（“季度性原材料降价，预计影响2个季度利润”）。


3. 事件时效性权重（Timeliness_Weight）  
   - 定义：新闻事件的即时性与紧迫性（突发/近期事件权重高，过时信息权重低），通过对比发布时间与新闻内容中的时间描述。  
   - 取值1：突发重大即时事件，如“今日盘中突发：公司获得国家级专项补贴5亿元”“半小时前公告：重大合同签署生效”。  
   - 取值0：过时或滞后信息，如“回顾半年前公司的战略调整（已被市场消化）”“转载3个月前的行业分析报告（无新信息）”。  
   - 中间值示例：0.6（“昨日晚间公告的季度业绩，尚未被充分交易”）；0.3（“一周前的产品发布会，市场已部分反应”）。

4. 信息确定性程度（Information_Certainty）  
   - 定义：新闻内容的明确性与可信度（官方公告/数据vs猜测/传闻）。  
   - 取值1：完全确定可验证，如“公司官网发布经审计的年度财报，净利润15.2亿元”“监管层正式发文批准公司新项目”。  
   - 取值0：高度模糊或存疑，如“匿名消息称公司可能有并购计划（未证实）”“分析师猜测业绩可能不及预期（无数据支撑）”。  
   - 中间值示例：0.5（“公司预告年度利润区间10-12亿元，尚未审计”）；0.8（“行业协会发布统计数据，公司市场份额提升至25%”）。

5. 信息相关度（Information_Relevance）  
   - 定义：新闻内容与目标公司的直接关联程度，无关行业整体或其他公司，仅聚焦目标公司自身。  
   - 取值1：高度相关且聚焦核心业务，如“公司官宣核心产品提价20%，预计影响全年利润”“目标公司与某巨头签订独家合作协议，涉及主营板块”。  
   - 取值0：完全无关，如“仅讨论新能源行业趋势，未提及任何具体公司”“新闻主体为同行业其他公司，仅附带提及目标公司名称”。  
   - 中间值示例：0.6（“新闻提及目标公司子公司的小额合作项目，不涉及母公司核心业务”）；0.3（“行业报告中列举目标公司为行业参与者之一，无具体信息”）。

输出格式（请严格按照JSON格式输出，键为因子名，值为浮点数），不需要输出其他内容：  
{{"Fundamental_Positive": 0.XX, "Impact_Cycle_Length": 0.XX, "Timeliness_Weight": 0.XX, "Information_Certainty": 0.XX, "Information_Relevance": 0.XX}}
---
股票名称: {stock_name}
股票代码: {stock_code}
{title}
来源：{source}
发布时间：{publish_time}
{content}
"""
        
        return [SystemMessage(system_prompt)]
    
    def get_score_with_retry(self, task_item: Dict[str, Any]) -> Dict[str, Any]:
        """调用LLM获取评分，带重试机制"""
        # 检查内容是否为空
        full_content = task_item.get("content", "")
        if not full_content.strip():
            logger.warning(f"ID {task_item.get('_id', 'N/A')} 的文本内容为空，跳过处理。")
            return {"status": "skipped", "error": "Empty content"}
        
        # 准备提示词
        messages = self.create_prompt(task_item)
        
        # 重试逻辑
        for attempt in range(Config.MAX_RETRIES):
            try:
                # 调用LLM
                response = self.llm.invoke(messages)
                print(response)
                # 解析结果
                try:
                    # 确保response.content是字符串类型，处理不同可能的返回类型
                    if isinstance(response.content, str):
                        content_str = response.content.strip()
                    elif isinstance(response.content, list):
                        # 如果是列表，尝试将每个元素转换为字符串并连接
                        content_str = " ".join(str(item) for item in response.content).strip()
                    else:
                        content_str = str(response.content).strip()
                    
                    score_data = json.loads(content_str)
                    return {
                        "status": "success", 
                        "data": score_data
                    }
                except Exception as parse_error:
                    # 安全处理日志输出
                    response_content_str = str(response.content)[:200] + "..." if len(str(response.content)) > 200 else str(response.content)
                    logger.error(
                        f"ID {task_item.get('_id', 'N/A')} 的响应解析失败。错误: {parse_error}. 内容: {response_content_str}"
                    )
                    # 尝试直接JSON解析作为备选方案
                    try:
                        # 安全处理content
                        if isinstance(response.content, str):
                            content_str = response.content
                        elif isinstance(response.content, list):
                            content_str = " ".join(str(item) for item in response.content)
                        else:
                            content_str = str(response.content)
                        score_data = json.loads(content_str)
                        # 验证字段
                        if all(field in score_data for field in ["Fundamental_Positive", "Impact_Cycle_Length", "Timeliness_Weight", "Information_Certainty", "Information_Relevance"]):
                            return {"status": "success", "data": score_data}
                    except Exception as json_error:
                        logger.error(f"JSON解析失败: {json_error}")
                    return {
                        "status": "failed",
                        "error": "ParsingError",
                        "content": response.content
                    }
            
            except Exception as e:
                logger.error(
                    f"ID {task_item.get('_id', 'N/A')} 调用LLM失败。错误: {e}. 第 {attempt + 1}/{Config.MAX_RETRIES} 次尝试。 {Config.RETRY_SLEEP_TIME} 秒后重试...",
                    exc_info=True
                )
                # 等待后重试
                time.sleep(Config.RETRY_SLEEP_TIME)
        
        logger.error(f"ID {task_item.get('_id', 'N/A')} 超过最大重试次数。")
        return {"status": "failed", "error": "Max retries exceeded"}