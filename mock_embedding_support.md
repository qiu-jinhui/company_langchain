# 在模拟API服务器中添加嵌入模型支持

此文档提供了如何在现有的模拟API服务器中添加对嵌入模型API的支持，以便本地测试CompanyEmbeddings适配器。

## 修改流程

1. 打开 `mock_api.py` 文件
2. 添加嵌入模型API路由
3. 实现嵌入生成逻辑
4. 更新服务器配置（如有必要）

## 具体步骤

### 1. 导入所需库

确保在文件顶部导入所有必要的库：

```python
import numpy as np  # 用于生成随机嵌入向量
```

### 2. 添加嵌入API路由

在 `create_app` 函数中，加入嵌入API的路由处理：

```python
@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """模拟嵌入模型API"""
    # 检查认证头
    if not verify_auth_headers():
        return jsonify({
            "error": {
                "message": "认证失败，请提供有效的认证信息",
                "type": "auth_error",
                "code": 401
            }
        }), 401
    
    # 随机生成错误
    if random.random() < llm_error_rate:
        status_code = random.choice([429, 500, 502, 503, 504])
        error_messages = {
            429: "达到速率限制，请稍后再试",
            500: "服务器内部错误",
            502: "错误网关",
            503: "服务不可用",
            504: "网关超时"
        }
        return jsonify({
            "error": {
                "message": error_messages[status_code],
                "type": "api_error",
                "code": status_code
            }
        }), status_code
    
    # 解析请求
    request_json = request.get_json()
    
    if not request_json:
        return jsonify({
            "error": {
                "message": "无效的请求体",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400
    
    # 检查必要字段
    if "input" not in request_json:
        return jsonify({
            "error": {
                "message": "缺少必要字段: input",
                "type": "invalid_request_error",
                "code": 400
            }
        }), 400
    
    # 获取输入文本
    input_texts = request_json["input"]
    
    # 确保输入是列表
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    
    # 生成模拟嵌入（使用随机值）
    embedding_dimension = 1536  # OpenAI模型的标准维度
    result = {
        "object": "list",
        "data": [],
        "model": request_json.get("model", "text-embedding-ada-002"),
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in input_texts),
            "total_tokens": sum(len(text.split()) for text in input_texts)
        }
    }
    
    # 为每个输入文本生成一个嵌入向量
    for i, text in enumerate(input_texts):
        # 创建确定性但看似随机的嵌入
        # 使用文本的哈希值作为随机种子，使相同文本总是产生相同的嵌入
        text_hash = abs(hash(text)) % (10 ** 8)
        np.random.seed(text_hash)
        
        # 生成单位长度的嵌入向量（L2范数=1）
        embedding = np.random.normal(0, 1, embedding_dimension)
        embedding = embedding / np.linalg.norm(embedding)
        
        # 添加到结果中
        result["data"].append({
            "object": "embedding",
            "embedding": embedding.tolist(),
            "index": i
        })
    
    # 重置随机种子
    np.random.seed(None)
    
    # 延迟响应以模拟实际API
    time.sleep(random.uniform(0.1, 0.5))
    
    return jsonify(result)
```

### 3. 更新全局变量和参数

如果需要，可以将嵌入模型的错误率也作为参数：

```python
def create_app(llm_error_rate=0.3, azure_error_rate=0.3, token_error_rate=0.1, embedding_error_rate=0.3):
    """创建Flask应用"""
    app = Flask(__name__)
    # ...现有代码...
```

然后在命令行参数中添加：

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行模拟API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器监听端口")
    parser.add_argument("--llm-error-rate", type=float, default=0.3, help="公司LLM API错误率（0-1）")
    parser.add_argument("--azure-error-rate", type=float, default=0.3, help="Azure API错误率（0-1）")
    parser.add_argument("--token-error-rate", type=float, default=0.1, help="令牌API错误率（0-1）")
    parser.add_argument("--embedding-error-rate", type=float, default=0.3, help="嵌入模型API错误率（0-1）")
    
    args = parser.parse_args()
    
    app = create_app(
        llm_error_rate=args.llm_error_rate,
        azure_error_rate=args.azure_error_rate,
        token_error_rate=args.token_error_rate,
        embedding_error_rate=args.embedding_error_rate
    )
    app.run(host=args.host, port=args.port, debug=True)
```

## 使用方法

更新模拟服务器后，可以使用以下命令启动它：

```bash
python run_mock_api.py --port 5001 --embedding-error-rate 0.2
```

然后，可以使用CompanyEmbeddings适配器连接到模拟服务器：

```python
from company_embedding import CompanyEmbeddings
from pydantic import SecretStr

# 初始化嵌入模型（指向模拟服务器）
embeddings = CompanyEmbeddings(
    api_url="http://localhost:5001/v1/embeddings",
    application_id="test-app-id",
    trust_token=SecretStr("test-token"),
)

# 测试嵌入生成
text = "这是一个测试文本"
embedding = embeddings.embed_query(text)
print(f"生成的嵌入向量维度: {len(embedding)}")

# 确认相同的文本总是产生相同的嵌入
embedding2 = embeddings.embed_query(text)
import numpy as np
similarity = np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))
print(f"相同文本的嵌入相似度: {similarity}")  # 应该非常接近1.0
```

## 注意事项

1. 这个模拟API生成的嵌入向量是随机的，但对于相同的输入文本会产生一致的嵌入（通过使用文本的哈希值作为随机种子）。
2. 嵌入向量被归一化为单位长度（L2范数=1），符合大多数嵌入模型的标准。
3. 嵌入维度设置为1536，这是OpenAI的text-embedding-ada-002模型的标准维度。
4. 您可以根据需要调整嵌入维度和错误率。 