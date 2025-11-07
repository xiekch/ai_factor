## 如何运行

### 配置.env文件

```
API_KEY = "sk-"
```

API_KEY 的内容换成你的阿里云api key。

### 配置 config.py

```python
# 目标股票
TARGET_STOCK_CODES = ["000001"]
# 处理的新闻个数，换成10000或更大
PROCESS_NUM = 20
```



### 运行

```
python3 main.py
```

