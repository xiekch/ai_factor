#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主处理流程模块：实现评分流程的协调和管理
"""

import os
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from logger import logger
from llm_service import LLMService
from data_processor import DataProcessor
from stock_code_to_name_map import stock_code_to_name

class StockScorer:
    """主处理流程类"""
    
    def __init__(self):
        """初始化各个组件"""
        self.llm_service = LLMService()
        self.data_processor = DataProcessor()
    
    def process_stock_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """处理单个股票文件"""
        successful_results: List[Dict[str, Any]] = []
        failed_tasks: List[Dict[str, Any]] = []
        
        # 加载数据
        all_data_to_process = self.data_processor.load_data_from_json_file(file_path)
        if not all_data_to_process:
            logger.warning(f"文件 {file_path} 加载失败或为空，跳过")
            return successful_results, failed_tasks
        # 为每个item添加stock_name字段
        for item in all_data_to_process:
            stock_code = item.get('stock_code', '')
            # 从映射字典中获取股票名称，如果不存在则使用'Unknown'
            stock_name = stock_code_to_name.get(stock_code, '')
            item['stock_name'] = stock_name
        
        logger.info(f"开始为 {len(all_data_to_process)} 条记录启动 {Config.MAX_CONCURRENT_REQUESTS} 个并发线程...")
        
        # 并发处理
        with ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_REQUESTS) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(self.llm_service.get_score_with_retry, item): item 
                for item in all_data_to_process
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    
                    if result["status"] == "success":
                        flat_result: Dict[str, Any] = {
                            "id": item.get("_id"),
                            "stock_code": item.get("stock_code"),
                            "pub_time": item.get("pub_time"),
                            **result["data"],
                        }
                        successful_results.append(flat_result)
                    elif result["status"] == "failed":
                        failed_tasks.append({
                            "id": item.get("_id"), 
                            "error_details": result
                        })
                except Exception as e:
                    logger.error(f"处理ID {item.get('_id')} 时发生异常: {e}",exc_info=True)
                    failed_tasks.append({
                        "id": item.get("_id"), 
                        "error_details": {"status": "exception", "error": str(e)}
                    })
        
        logger.info(f"文件 {file_path} 的API调用全部完成，成功 {len(successful_results)} 条，失败 {len(failed_tasks)} 条")
        return successful_results, failed_tasks
    
    def run(self):
        """主运行方法"""
        # 加载已处理的股票代码
        processed_codes: Set[str] = self.data_processor.load_processed_codes(Config.OUTPUT_CSV_PATH)
        
        # 查找目标文件
        all_json_files: List[str] = self.data_processor.find_stock_json_files(Config.JSON_SOURCE_DIRECTORY)
        
        # 过滤目标文件
        target_json_files: List[str] = []
        for f_path in all_json_files:
            code = os.path.basename(f_path).split(".")[0]
            if code in Config.TARGET_STOCK_CODES_SET:
                target_json_files.append(f_path)
        
        logger.info(f"在目录中找到了 {len(target_json_files)} 个与目标列表 {len(Config.TARGET_STOCK_CODES_SET)} 匹配的文件")
        
        # 过滤未处理的文件
        files_to_process: List[str] = []
        for f_path in target_json_files:
            code = os.path.basename(f_path).split(".")[0]
            if code not in processed_codes:
                files_to_process.append(f_path)
            else:
                logger.info(f"(跳过已处理) {code}.json")
        
        if not files_to_process:
            logger.info("所有目标股票文件均已处理。程序退出")
            return
        
        logger.info(f"总计 {len(target_json_files)} 个目标文件。已处理 {len(processed_codes)} 个。剩余 {len(files_to_process)} 个待处理")
        
        all_failed_tasks: List[Dict[str, Any]] = []
        
        try:
            # 处理每个文件
            for file_index, file_path in enumerate(files_to_process):
                logger.info(f"\n--- --------------------------------- ---")
                logger.info(f"--- 正在处理目标文件 {file_index + 1}/{len(files_to_process)}: {file_path} ---")
                logger.info(f"--- --------------------------------- ---")
                
                # 处理文件
                successful_results, failed_tasks = self.process_stock_file(file_path)
                
                # 保存成功结果
                if successful_results:
                    self.data_processor.save_results(successful_results, Config.OUTPUT_CSV_PATH)
                
                # 收集失败任务
                if failed_tasks:
                    all_failed_tasks.extend(failed_tasks)
                    logger.warning(f"文件 {file_path} 产生 {len(failed_tasks)} 条失败记录")
                
                # 手动暂停逻辑
                is_last_file = file_index == len(files_to_process) - 1
                should_pause = ((file_index + 1) % 5 == 0) and (not is_last_file)
                
                if should_pause:
                    try:
                        logger.info(f"\n--- --------------------------------- ---")
                        logger.info(f"--- [暂停] 已累计处理 5 个文件。最近一个是: {os.path.basename(file_path)} ---")
                        user_input = input("[操作] 按 Enter 继续处理下 5 个文件, 或输入 'q' (然后按Enter) 退出: ")
                        if user_input.lower() == "q":
                            logger.info("用户请求退出... 程序终止")
                            break
                    except KeyboardInterrupt:
                        logger.info("\n检测到 Ctrl+C... 程序终止")
                        break
        
        finally:
            # 保存失败任务
            self.data_processor.save_failed_tasks(all_failed_tasks, Config.FAILED_JSON_PATH)
            logger.info(f"\n--- 批量处理全部完成！---")
            if not all_failed_tasks:
                logger.info("本次运行没有记录失败")
