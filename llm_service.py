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
from pydantic.types import SecretStr


class LLMService:
    """封装LLM调用相关的功能"""

    def __init__(self):
        """初始化LLM客户端和解析器"""
        try:
            self.llm = ChatTongyi(
                model=Config.MODEL_NAME, api_key=SecretStr(Config.API_KEY), verbose=True
            )
            logger.info(f"成功初始化LangChain LLM客户端，模型: {Config.MODEL_NAME}")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
            raise

    def create_prompt(self, task_item: Dict[str, Any]) -> List[BaseMessage]:
        """创建LangChain提示词"""
        title = task_item.get("title", "")
        content = task_item.get("content", "")
        source = task_item.get("source", "unknown")
        publish_time = task_item.get("pub_time", "unknown")
        stock_code = task_item.get("stock_code", "N/A")
        stock_name = task_item.get("stock_name", "N/A")
        thingking_prompt = "输出格式（输出简短的思考，最后一行严格按照JSON格式输出，键为因子名，值为浮点数，不需要代码块格式包裹）："
        no_thinking_prompt = "输出格式(只输出JSON格式，键为因子名，值为浮点数，不需要代码块格式包裹）："
        # 系统提示词
        system_prompt = f"""你是一名顶尖的中国A股市场金融分析师，擅长从海量文本信息中挖掘对股价有影响的信号。
任务：分析以下股票新闻内容，针对每条新闻输出5个AI因子的取值（保留1位小数，范围0-1），每个因子基于新闻内容独立评估。

因子定义及取值说明：
1. 基本面影响强度（Fundamental_Impact）  
   - 定义：新闻对公司核心基本面（营收、利润、成本、技术壁垒、资产质量等）的影响程度。 大于 0.5 表示利好，小于 0.5 表示利空，等于 0.5 表示中性或不相干影响。
   - 取值1：极端利好，如“公司签订100亿元长期订单，预计年增营收50%”“研发出颠覆性技术，专利保护期20年”。  
   - 取值0.7：中等利好，如“季度营收增长15%”“获得地方政府补贴”。
   - 取值0.5：中性或不相干影响，如“未提及具体财务影响”“市场对公司产品的评价存在争议”。
   - 取值0.3：中等利空，如“季度营收下降10%”“面临行业竞争压力”。
   - 取值0.1：显著利空，如“主要产品价格下降”“面临监管调查”。
   - 取值0：极端利空，如“公司暴雷30亿元财务造假，核心资产被冻结”“主力产品被监管认定为不合格，全面下架”。

2. 影响周期长度（Impact_Cycle_Length）
   - 定义：新闻事件对公司的影响持续时间。
   - 取值1：长期影响（3年以上），如“签订5年独家供应协议”“发布未来3年战略规划”。
   - 取值0.6：中短期影响（3-6个月），如“季节性销售旺季”“半年度业绩预告”。
   - 取值0.5：短期影响（1-3个月），如“月度销售数据”“短期促销活动”。
   - 取值0.4：超短期影响（2-4周），如“周度经营数据”“短期市场波动”。
   - 取值0.2：即时影响（1-7天），如“单日交易异常”。
   - 取值0：无影响或影响已消化，如“历史数据回顾”“已被市场充分预期的信息”。

3. 事件时效性（Timeliness_Weight）  
   - 定义：新闻事件的即时性，通过对比**事件的发生时间**相对于**新闻发布时间**的时间差。  
   - 取值1：即时事件（当天），如“今日盘中突发”“9月10日发布公布（新闻发布时间也是9月10日）”。  
   - 取值0.9：近期事件（1天），如“昨日晚间公告的季度业绩”。
   - 取值0.5：近期事件（1周），如“一周前的产品发布会”。
   - 取值0.4：近期事件（2周），如“两周前的行业展会”“10天前的调研活动”。
   - 取值0.2：历史事件（3个月）如“3个月前的财报”“季度初的规划”。
   - 取值0：过时或滞后信息，如“回顾半年前公司的战略调整”。

4. 信息确定性程度（Information_Certainty）  
   - 定义：新闻内容的明确性与可信度（官方公告/数据vs猜测/传闻）。  
   - 取值1：完全确定可验证，如“公司官网发布经审计的年度财报，净利润15.2亿元”“监管层正式发文批准公司新项目”。  
   - 取值0.8：较为确定，如“公司官方预告年度利润区间10-12亿元，尚未审计”“主流媒体报道的签约仪式”。
   - 取值0.7：中等确定，如“公司高管在公开场合透露的经营计划”“分析师基于公开数据的预测”。
   - 取值0.5：中性不确定，如“市场传闻有待证实”“业内人士的匿名爆料”。
   - 取值0.3：较不确定，如“网络论坛的讨论”“个人投资者的猜测”。
   - 取值0.1：高度不确定，如“小道消息”“未经核实的传言”。
   - 取值0：完全模糊或存疑，如“未经任何验证”。

5. 信息相关度（Information_Relevance）  
   - 定义：新闻内容与目标公司的直接关联程度，无关行业整体或其他公司，仅聚焦目标公司自身。  
   - 取值1：高度相关且聚焦核心业务，如“公司官宣核心产品提价20%，预计影响全年利润”“目标公司与某巨头签订独家合作协议，涉及主营板块”。     - 取值0.8：较强相关，如“公司主要产品的市场表现”“重要客户的合作动态”。
   - 取值0.6：有一定相关，如“新闻提及目标公司子公司的小额合作项目，不涉及母公司核心业务”。
   - 取值0.5：中性相关，如“公司作为行业代表被提及”“行业政策对公司的影响分析”。
   - 取值0.4：较弱相关，如“行业报告中列举目标公司为行业参与者之一，无具体信息”“同行业对比中的提及”。
   - 取值0.2：相关性很弱，如“附带提及公司名称”“行业数据的组成部分”。
   - 取值0：完全无关，如“新闻主体为同行业其他公司，仅附带提及目标公司名称”。

{thingking_prompt if Config.NEED_THINKING else no_thinking_prompt}
{{"Fundamental_Impact": 0.XX, "Impact_Cycle_Length": 0.XX, "Timeliness_Weight": 0.XX, "Information_Certainty": 0.XX, "Information_Relevance": 0.XX}}
---
股票名称: {stock_name}
股票代码: {stock_code}
{title}
来源：{source}
新闻发布时间：{publish_time}
{content}
"""

        return [SystemMessage(system_prompt)]

    def get_score_with_retry(self, task_item: Dict[str, Any]) -> Dict[str, Any]:
        """调用LLM获取评分，带重试机制"""
        # 检查内容是否为空
        full_content = task_item.get("content", "")
        if not full_content.strip():
            logger.warning(
                f"ID {task_item.get('_id', 'N/A')} 的文本内容为空，跳过处理。"
            )
            return {"status": "skipped", "error": "Empty content"}

        # 准备提示词
        messages = self.create_prompt(task_item)

        # 重试逻辑
        for attempt in range(Config.MAX_RETRIES):
            try:
                # 调用LLM
                response = self.llm.invoke(messages)
                logger.info(f"ID {task_item.get('_id', 'N/A')} 的LLM响应内容: {response.content}")
                # 解析结果

                # 确保response.content是字符串类型，处理不同可能的返回类型
                if isinstance(response.content, str):
                    content_str = response.content.split("\n")[-1].strip()
                elif isinstance(response.content, list):
                    # 如果是列表，尝试将每个元素转换为字符串并连接
                    content_str = " ".join(
                        str(item) for item in response.content
                    ).strip()
                else:
                    content_str = str(response.content).strip()

                score_data = json.loads(content_str)
                return {"status": "success", "data": score_data}

            except Exception as e:
                logger.error(
                    f"ID {task_item.get('_id', 'N/A')} 调用LLM失败。错误: {e}. 第 {attempt + 1}/{Config.MAX_RETRIES} 次尝试。 {Config.RETRY_SLEEP_TIME} 秒后重试...",
                    exc_info=True,
                )
                # 等待后重试
                time.sleep(Config.RETRY_SLEEP_TIME)

        logger.error(f"ID {task_item.get('_id', 'N/A')} 超过最大重试次数。")
        return {"status": "failed", "error": "Max retries exceeded"}
