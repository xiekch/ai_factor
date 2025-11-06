#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI 因子评分脚本：
- 批量读取爬虫抓取的 JSON 文本数据。
- 通过并发 API 请求（通义千问）进行多维度评分。
- 实现多线程、API重试、断点续传和手动暂停功能。
- 将评分结果保存到 CSV 文件。
"""

import os
import json
import pandas as pd
import time
import logging
import re
from openai import OpenAI, RateLimitError  # 导入 RateLimitError 以处理API限流
from concurrent.futures import ThreadPoolExecutor  # 导入多线程执行器

# --- 1. 全局配置 ---
API_KEY = "sk-49ee3ae503864131a44223e48983f224"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo"
REQUEST_TIMEOUT = 60

# --- 2. 批量处理与并发配置 ---
JSON_SOURCE_DIRECTORY = "/guba_df"
MAX_CONCURRENT_REQUESTS = 40
MAX_RETRIES = 3
RETRY_SLEEP_TIME = 15
OUTPUT_CSV_PATH = "ALL_STOCKS_scored_results.csv"
FAILED_JSON_PATH = "ALL_STOCKS_failed_tasks.json"

# --- 3. 目标股票代码列表 ---
# 仅处理以下列表中的股票代码
TARGET_STOCK_CODES = ['600519', '601211', '000333', '601328', '601138', '000063', '600000', '600030', '601336', '601166', '600050', '600941', '600111', '601319', '000651', '000568', '600048', '600115', '600196', '600150', '600926', '600690', '600919', '600438', '600938', '600809', '600031', '600028', '000100', '600015', '600016', '601229', '000001', '000002', '601360', '600415', '301236', '600029', '000408', '600887', '302132']
TARGET_STOCK_CODES_SET = set(TARGET_STOCK_CODES)

# --- 4. 日志配置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 屏蔽 httpx 库的 INFO 日志，避免刷屏 "HTTP Request"
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 5. 初始化API客户端 ---
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    logging.info(f"成功初始化API客户端，目标模型: {MODEL_NAME}")
    logging.info(f"并发数设置为: {MAX_CONCURRENT_REQUESTS}")
except Exception as e:
    logging.error(f"API客户端初始化失败: {e}")
    exit()


# --- 6. JSON 数据加载 ---
def load_data_from_json_file(file_path):
    """
    从指定的本地 JSON 文件路径加载数据。
    """
    raw_content = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # 检查文件是否为空
        if not raw_content.strip():
            logging.warning(f"  文件 {file_path} 为空，跳过。")
            return None

        # 直接加载
        data = json.loads(raw_content)
        logging.info(f"  (标准加载) 成功从 {file_path} 加载了 {len(data)} 条记录。")
        return data

    except json.JSONDecodeError as e:
        # 如果 JSON 格式仍有问题，则直接报错
        logging.error(f"  错误：文件 {file_path} 不是有效的 JSON 格式。{e}")
        return None
    except FileNotFoundError:
        logging.error(f"  错误：找不到文件 {file_path}。")
        return None
    except Exception as e:
        logging.error(f"  加载文件时发生未知错误: {e}")
        return None


# --- 7. 核心API调用功能 (已集成优化后的Prompt) ---
def get_llm_score_with_retry(task_item):
    """
    (同步版本) 调用LLM API，带重试逻辑。
    此函数被多线程调用。
    """
    full_content = task_item.get("title", "") + " " + task_item.get("content", "")

    if not full_content.strip():
        logging.warning(
            f"  ID {task_item.get('_id', 'N/A')} 的文本内容为空，跳过处理。"
        )
        return {"status": "skipped", "error": "Empty content"}

    system_prompt = "你是一名顶尖的中国A股市场金融分析师，擅长从海量文本信息中挖掘对股价有影响的信号。"

    user_prompt = f"""
# Context
- 股票代码: {task_item.get('stock_code', 'N/A')}

# Task
请仔细阅读并分析以下文本。根据你的专业知识和下方 **# Scoring Definitions** 中的定义，对文本内容进行多维度评估。

# Text to Analyze
"{full_content}"

# Scoring Definitions
请严格按照以下定义和锚点进行评分：

1.  **sentiment_score (情绪倾向)**:
    - **定义**: 评估文本对该股票**未来股价**的整体情绪倾向。
    - **范围**: -1.0 (极度负面/看空) 到 1.0 (极度正面/看多)。
    - **锚点**: 
        - 0.0 代表中性、客观陈述或情绪不明确。
        - 示例: 利好财报/重大合同 ≈ 1.0; 监管调查/业绩巨亏 ≈ -1.0。

2.  **event_driven_score (事件驱动性)**:
    - **定义**: 评估文本是否描述了一个**具体的、可能导致股价发生变动的独立事件**。
    - **范围**: 0.0 (无具体事件) 到 1.0 (强事件驱动)。
    - **锚点**:
        - 0.0: 宏观分析、市场评论、无具体指向的常规讨论、旧闻总结。
        - 1.0: 明确的**业绩预告**、**并购重组**、**重大合同**、**高管重大变动**、**监管处罚**、**新产品重大突破**等。

3.  **time_horizon (影响周期)**:
    - **定义**: 评估该信息或事件对股价**潜在影响的持续时间**。
    - **选项**: ["短期", "中期", "长期"]
    - **锚点**:
        - "短期": 影响主要在未来几天到几周内 (如: 财报发布、技术反弹)。
        - "中期": 影响可能持续几周到几个月 (如: 季度趋势、新产品销售周期)。
        - "长期": 影响涉及公司基本面/战略，可能持续数月至数年 (如: 行业变革、战略转型、重大研发)。

4.  **novelty_score (信息新颖度)**:
    - **定义**: 评估该信息**对市场而言的新颖程度**，即信息是否**“出乎意料”**或**首次披露**。
    - **范围**: 0.0 (旧闻/预期内) 到 1.0 (全新/预期外)。
    - **锚点**:
        - 0.0: 重复报道、市场已充分知晓的旧信息、对已知事件的常规分析、完全在预期内的事件。
        - 1.0: **首次披露**的重大信息、**预期差极大**的数据(如业绩远超/低于预期)、可能导致市场重新定价的突发新闻。

# Output Instruction
你的分析结果必须以一个严格的JSON格式输出，不要包含任何额外的解释或代码块标记。JSON结构如下：
{{
  "sentiment_score": <从-1.0到1.0的浮点数>,
  "event_driven_score": <从0.0到1.0的浮点数>,
  "time_horizon": <从["短期", "中期", "长期"]中选择一个>,
  "novelty_score": <从0.0到1.0的浮点数>
}}
"""

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=REQUEST_TIMEOUT,
            )
            content_str = completion.choices[0].message.content

            try:
                score_data = json.loads(content_str)
                # logging.info(f"  [成功] ID: {task_item.get('_id', 'N/A')}") # 已注释，需要时可打开
                return {"status": "success", "data": score_data}
            except json.JSONDecodeError:
                logging.error(
                    f"  JSON解析失败 for ID {task_item.get('_id')}. Content: {content_str}"
                )
                return {
                    "status": "failed",
                    "error": "JSONDecodeError",
                    "content": content_str,
                }

        except RateLimitError as e:
            logging.warning(
                f"  [触发限流] ID {task_item.get('_id')}. 第 {attempt + 1}/{MAX_RETRIES} 次尝试。 {RETRY_SLEEP_TIME} 秒后重试..."
            )
            time.sleep(RETRY_SLEEP_TIME)
        except Exception as e:
            if "timeout" in str(e).lower():
                logging.warning(
                    f"  [请求超时] ID {task_item.get('_id')}. 第 {attempt + 1}/{MAX_RETRIES} 次尝试。 {RETRY_SLEEP_TIME} 秒后重试..."
                )
                time.sleep(RETRY_SLEEP_TIME)
            else:
                logging.error(f"  [API未知错误] ID {task_item.get('_id')}. Error: {e}")
                return {"status": "failed", "error": str(e)}

    logging.error(f"  [重试失败] ID {task_item.get('_id')} 超过最大重试次数。")
    return {"status": "failed", "error": "Max retries exceeded"}


# --- 8. 辅助函数：扫描JSON文件 ---
def find_stock_json_files(directory):
    """
    扫描指定目录，找出所有以数字股号命名的 .json 文件。
    """
    found_files = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                stock_code = filename.split(".")[0]
                if stock_code.isdigit():
                    found_files.append(os.path.join(directory, filename))
    except FileNotFoundError:
        logging.error(f"错误：找不到目录 {directory}。请检查 JSON_SOURCE_DIRECTORY")
        return []
    return found_files


# --- 9. 主执行流程 ---
def main():
    """
    主函数：实现断点续传、多线程、即时保存、手动控制。
    """

    processed_codes = set()
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV_PATH)
            processed_codes = set(df_existing["stock_code"].astype(str).unique())
            logging.info(
                f"检测到已存在的CSV，已加载 {len(processed_codes)} 个处理过的股票代码。"
            )
        except Exception as e:
            logging.error(f"无法读取已存在的CSV: {e}。将作为新文件处理。")

    all_json_files_in_dir = find_stock_json_files(JSON_SOURCE_DIRECTORY)

    # --- 文件过滤逻辑 ---
    target_json_files = []
    for f_path in all_json_files_in_dir:
        code = os.path.basename(f_path).split(".")[0]
        if code in TARGET_STOCK_CODES_SET:
            target_json_files.append(f_path)

    logging.info(
        f"在目录中找到了 {len(target_json_files)} 个与目标列表 {len(TARGET_STOCK_CODES_SET)} 匹配的文件。"
    )

    files_to_process = []
    for f_path in target_json_files:
        code = os.path.basename(f_path).split(".")[0]
        if code not in processed_codes:
            files_to_process.append(f_path)
        else:
            logging.info(f"  (跳过已处理) {code}.json")

    if not files_to_process:
        logging.info("所有 *目标* 股票文件均已处理。程序退出。")
        return

    logging.info(
        f"总计 {len(target_json_files)} 个目标文件。已处理 {len(processed_codes)} 个。剩余 {len(files_to_process)} 个待处理。"
    )

    all_failed_tasks_this_run = []

    for file_index, file_path in enumerate(files_to_process):
        logging.info(f"\n--- --------------------------------- ---")
        logging.info(
            f"--- 正在处理目标文件 {file_index + 1}/{len(files_to_process)}: {file_path} ---"
        )
        logging.info(f"--- --------------------------------- ---")

        all_data_to_process = load_data_from_json_file(file_path)

        if not all_data_to_process:
            logging.warning(f"文件 {file_path} 加载失败或为空，跳过。")
            continue

        successful_results_for_this_file = []
        failed_tasks_for_this_file = []

        logging.info(
            f"  开始为 {len(all_data_to_process)} 条记录启动 {MAX_CONCURRENT_REQUESTS} 个并发线程..."
        )

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            results = list(executor.map(get_llm_score_with_retry, all_data_to_process))

        logging.info(f"  文件 {file_path} 的API调用全部完成，开始整理结果...")

        for i, result in enumerate(results):
            item = all_data_to_process[i]

            if result["status"] == "success":
                flat_result = {
                    "id": item.get("_id"),
                    "stock_code": item.get("stock_code"),
                    "pub_time": item.get("pub_time"),
                    **result["data"],
                }
                successful_results_for_this_file.append(flat_result)
            elif result["status"] == "failed":
                failed_tasks_for_this_file.append(
                    {"input": item, "error_details": result}
                )

        # 即时保存
        if successful_results_for_this_file:
            df_new = pd.DataFrame(successful_results_for_this_file)
            file_exists = os.path.exists(OUTPUT_CSV_PATH)

            df_new.to_csv(
                OUTPUT_CSV_PATH,
                mode="a",
                header=not file_exists,
                index=False,
                encoding="utf-8-sig",
            )

            logging.info(
                f"--- [保存成功] {len(successful_results_for_this_file)} 条结果已追加至 {OUTPUT_CSV_PATH} ---"
            )
        else:
            logging.info(f"--- 文件 {file_path} 没有成功的结果 ---")

        if failed_tasks_for_this_file:
            all_failed_tasks_this_run.extend(failed_tasks_for_this_file)
            logging.warning(
                f"  文件 {file_path} 产生 {len(failed_tasks_for_this_file)} 条失败记录。"
            )

        # --- [修改] 手动暂停逻辑 (每处理 5 个文件暂停) ---
        is_last_file = file_index == len(files_to_process) - 1

        # 检查是否达到了 5 个文件的倍数 (file_index 0-4 -> 5个)，并且不是最后一个文件
        should_pause = ((file_index + 1) % 5 == 0) and (not is_last_file)

        if should_pause:
            try:
                logging.info(f"\n--- --------------------------------- ---")
                logging.info(
                    f"--- [暂停] 已累计处理 5 个文件。最近一个是: {os.path.basename(file_path)} ---"
                )
                user_input = input(
                    "[操作] 按 Enter 继续处理下 5 个文件, 或输入 'q' (然后按Enter) 退出: "
                )
                if user_input.lower() == "q":
                    logging.info("用户请求退出... 程序终止。")
                    break
            except KeyboardInterrupt:
                logging.info("\n检测到 Ctrl+C... 程序终止。")
                break

    # --- 10. 最终结果存储 ---
    logging.info(f"\n--- 批量处理全部完成！---")

    if all_failed_tasks_this_run:
        with open(FAILED_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_failed_tasks_this_run, f, ensure_ascii=False, indent=4)
        logging.warning(
            f"总计有 {len(all_failed_tasks_this_run)} 条记录处理失败，详情请查看 {FAILED_JSON_PATH}"
        )
    else:
        logging.info("本次运行没有记录失败。")


if __name__ == "__main__":
    main()
