# 欢迎使用公司LangChain适配器

这个项目提供了一组适配器，用于将公司内部的API与LangChain框架无缝集成。通过这些适配器，您可以在LangChain应用中轻松使用公司的LLM和嵌入模型，同时保持与框架其他组件的兼容性。

## 主要组件

### 1. 公司LLM聊天模型适配器 (CompanyChatModel)

连接公司内部的LLM API，提供与OpenAI兼容的接口。支持同步和异步调用、流式响应和令牌自动刷新。

```python
from company_llm import CompanyChatModel
model = CompanyChatModel(api_url="...", application_id="...", trust_token="...")
```

### 2. 公司嵌入模型适配器 (CompanyEmbeddings)

连接公司内部的嵌入模型API，用于生成文本嵌入向量。支持批量处理、异步调用和与向量数据库集成。

```python
from company_embedding import CompanyEmbeddings
embeddings = CompanyEmbeddings(api_url="...", application_id="...", trust_token="...")
```

### 3. Azure OpenAI适配器 (CompanyAzureChatModel)

连接Azure OpenAI服务，提供与公司认证集成的访问方式。

```python
from company_azure_llm import CompanyAzureChatModel
model = CompanyAzureChatModel(deployment_name="...", application_id="...", trust_token="...")
```

## 快速开始

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

2. 配置认证信息:
   ```python
   application_id = "您的应用ID"
   trust_token = "您的信任令牌"
   ```

3. 创建模型实例:
   ```python
   from company_llm import CompanyChatModel
   from company_embedding import CompanyEmbeddings
   
   llm = CompanyChatModel(
       api_url="https://your-llm-api.example.com/v1/chat/completions",
       application_id=application_id,
       trust_token=trust_token
   )
   
   embeddings = CompanyEmbeddings(
       api_url="https://your-embedding-api.example.com/v1/embeddings",
       application_id=application_id,
       trust_token=trust_token
   )
   ```

4. 使用LangChain组件:
   ```python
   from langchain.chains import LLMChain
   from langchain.prompts import PromptTemplate
   from langchain_community.vectorstores import FAISS
   
   # 使用LLM
   chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("回答以下问题: {question}"))
   result = chain.invoke({"question": "什么是LangChain?"})
   
   # 使用嵌入模型
   texts = ["文档1", "文档2", "文档3"]
   vector_store = FAISS.from_texts(texts, embeddings)
   relevant_docs = vector_store.similarity_search("查询文本")
   ```

## 更多信息

详细文档请查看以下文件:
- `README.md`: 完整的项目说明和参数配置
- `example_usage.py`: LLM适配器的使用示例
- `embedding_example.py`: 嵌入模型适配器的使用示例
- `test_company_llm.py` 和 `test_company_embedding.py`: 单元测试，展示了API的预期行为

## 本地开发

本项目包含一个模拟API服务器，可用于本地开发和测试:

```bash
python run_mock_api.py --port 5001
```

请参阅 `mock_embedding_support.md` 了解如何为模拟服务器添加嵌入模型支持。 