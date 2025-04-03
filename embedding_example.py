"""
示例：如何使用公司嵌入模型适配器与LangChain
"""

import os
import asyncio
from company_embedding import CompanyEmbeddings
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def basic_usage_example():
    """基本使用示例：直接调用公司嵌入模型"""
    
    # 初始化公司嵌入模型
    # 注意：这些是示例值，请替换为您的实际配置
    embeddings = CompanyEmbeddings(
        api_url="https://your-company-embedding-api.example.com/v1/embeddings",
        application_id="your-application-id",
        trust_token="your-trust-token",
        model="text-embedding-ada-002",  # 或您公司支持的模型名称
    )
    
    # 生成单个文本的嵌入
    single_text = "这是一个测试文本用于演示嵌入功能。"
    single_embedding = embeddings.embed_query(single_text)
    print("单个文本嵌入向量（截取前5个值）:")
    print(single_embedding[:5])  # 只打印前5个值，嵌入向量可能很长
    print("\n" + "-"*50 + "\n")
    
    # 生成多个文本的嵌入
    texts = [
        "LangChain是一个用于构建LLM应用的框架。",
        "嵌入向量可以用于文本相似度计算和语义搜索。",
        "向量数据库存储嵌入向量并支持快速检索。"
    ]
    embeddings_list = embeddings.embed_documents(texts)
    print(f"多个文本嵌入结果，共{len(embeddings_list)}个向量:")
    for i, emb in enumerate(embeddings_list):
        print(f"文本{i+1}嵌入向量（截取前5个值）: {emb[:5]}")
    print("\n" + "-"*50 + "\n")


def vector_store_example():
    """向量存储示例：将嵌入与FAISS向量数据库结合使用"""
    
    # 初始化公司嵌入模型
    embeddings = CompanyEmbeddings(
        api_url="https://your-company-embedding-api.example.com/v1/embeddings",
        application_id="your-application-id",
        trust_token="your-trust-token",
    )
    
    # 准备文档数据
    documents = [
        Document(page_content="LangChain是一个强大的框架，用于开发由语言模型驱动的应用程序。", metadata={"source": "介绍文档", "page": 1}),
        Document(page_content="向量数据库是存储向量嵌入的专用数据库，支持高效的相似性搜索。", metadata={"source": "技术文档", "page": 5}),
        Document(page_content="嵌入模型将文本转换为数值向量，捕获语义信息。", metadata={"source": "教程", "page": 3}),
        Document(page_content="语义搜索使用嵌入向量来查找与查询语义相关的文档。", metadata={"source": "搜索指南", "page": 7}),
        Document(page_content="RAG（检索增强生成）结合了检索系统和生成模型的优势。", metadata={"source": "高级指南", "page": 12}),
    ]
    
    # 创建向量存储
    print("正在创建FAISS向量存储...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("向量存储创建完成\n")
    
    # 执行相似性搜索
    query = "什么是语义搜索？"
    print(f"查询: '{query}'")
    results = vector_store.similarity_search(query, k=2)
    
    print("搜索结果:")
    for i, doc in enumerate(results):
        print(f"结果 {i+1}:")
        print(f"  内容: {doc.page_content}")
        print(f"  元数据: {doc.metadata}")
    print("\n" + "-"*50 + "\n")


def token_refresh_example():
    """令牌自动刷新示例"""
    # 初始化公司嵌入模型（启用令牌自动刷新）
    embeddings = CompanyEmbeddings(
        api_url="https://your-company-embedding-api.example.com/v1/embeddings",
        application_id="your-application-id",
        trust_token="your-initial-token",  # 初始令牌
        
        # 令牌刷新配置
        token_refresh_enabled=True,  # 启用令牌自动刷新
        token_url="https://your-company-auth-api.example.com/token",  # 令牌刷新API
        username=SecretStr("your-username"),  # 用户名
        password=SecretStr("your-password"),  # 密码
        token_refresh_interval=3600,  # 令牌刷新间隔（秒）
    )
    
    # 测试查询
    query = "测试令牌刷新功能的查询文本"
    
    try:
        print("\n令牌自动刷新示例:")
        print("正在生成嵌入（如果需要，会自动刷新令牌）...")
        embedding = embeddings.embed_query(query)
        print(f"嵌入向量生成成功，维度: {len(embedding)}")
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def async_example():
    """异步调用示例"""
    # 初始化公司嵌入模型
    embeddings = CompanyEmbeddings(
        api_url="https://your-company-embedding-api.example.com/v1/embeddings",
        application_id="your-application-id",
        trust_token="your-trust-token",
    )
    
    # 准备测试文本
    texts = [
        "异步编程允许程序在等待I/O操作完成时执行其他任务。",
        "Python的asyncio库提供了编写异步代码的工具。",
        "异步API调用可以提高处理大量请求时的性能。"
    ]
    
    try:
        print("\n异步调用示例:")
        
        # 异步生成多个文本的嵌入
        print("正在异步生成多个文本的嵌入...")
        embeddings_list = await embeddings.aembed_documents(texts)
        print(f"生成了{len(embeddings_list)}个嵌入向量")
        
        # 异步生成单个查询的嵌入
        query = "什么是异步编程？"
        print(f"\n正在异步生成查询文本的嵌入: '{query}'")
        query_embedding = await embeddings.aembed_query(query)
        print(f"查询嵌入向量维度: {len(query_embedding)}")
        
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def async_token_refresh_example():
    """异步令牌刷新示例"""
    # 初始化公司嵌入模型（启用令牌自动刷新）
    embeddings = CompanyEmbeddings(
        api_url="https://your-company-embedding-api.example.com/v1/embeddings",
        application_id="your-application-id",
        trust_token="your-initial-token",  # 初始令牌
        
        # 令牌刷新配置
        token_refresh_enabled=True,  # 启用令牌自动刷新
        token_url="https://your-company-auth-api.example.com/token",  # 令牌刷新API
        username=SecretStr("your-username"),  # 用户名
        password=SecretStr("your-password"),  # 密码
        token_refresh_interval=3600,  # 令牌刷新间隔（秒）
    )
    
    # 测试查询
    query = "测试异步令牌刷新功能的查询文本"
    
    try:
        print("\n异步令牌刷新示例:")
        print("正在异步生成嵌入（如果需要，会自动刷新令牌）...")
        embedding = await embeddings.aembed_query(query)
        print(f"嵌入向量生成成功，维度: {len(embedding)}")
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def run_async_examples():
    """运行所有异步示例"""
    await async_example()
    await async_token_refresh_example()


if __name__ == "__main__":
    print("公司嵌入模型适配器示例")
    print("="*50)
    print("注意：这些示例需要有效的API凭据才能运行\n")
    
    # 运行示例
    print("这些示例展示了如何使用公司嵌入模型适配器。")
    print("请替换API凭据后再运行。")
    
    print("\n可用的示例功能:")
    print("1. 基本嵌入 (basic_usage_example)")
    print("2. 向量存储集成 (vector_store_example)")
    print("3. 令牌自动刷新 (token_refresh_example)")
    print("4. 异步嵌入生成 (asyncio.run(async_example()))")
    print("5. 异步令牌刷新 (asyncio.run(async_token_refresh_example()))")
    print("6. 运行所有异步示例 (asyncio.run(run_async_examples()))")
    
    # 取消注释以下行来运行示例（需要提供有效的API凭据）
    # basic_usage_example()
    # vector_store_example()
    # token_refresh_example()
    # asyncio.run(async_example())
    # asyncio.run(async_token_refresh_example())
    # asyncio.run(run_async_examples()) 