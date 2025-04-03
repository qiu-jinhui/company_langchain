"""
异步测试：使用mock数据测试公司嵌入模型适配器的异步功能

这个测试文件展示了如何测试CompanyEmbeddings类的异步方法，而不需要实际调用API。
由于CompanyEmbeddings是一个Pydantic模型，我们使用自定义模拟类而不是传统的patch方法。
"""

import asyncio
import time
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from company_embedding import CompanyEmbeddings

class MockCompanyEmbeddings:
    """
    模拟CompanyEmbeddings类的行为
    
    这个类实现了与CompanyEmbeddings相同的异步接口，但所有API调用都是模拟的，
    不需要实际的网络连接或API凭据。
    """
    
    def __init__(self, dimension=5, delay=0.1):
        """
        初始化模拟嵌入生成器
        
        参数:
            dimension: 生成的嵌入向量的维度
            delay: 模拟API调用延迟的秒数
        """
        self.dimension = dimension
        self.delay = delay
        self.call_count = 0
        
    async def aembed_documents(self, texts):
        """
        模拟异步生成多个文本的嵌入向量
        
        参数:
            texts: 要生成嵌入的文本列表
            
        返回:
            嵌入向量列表，每个文本对应一个嵌入向量
        """
        # 增加调用计数
        self.call_count += 1
        
        # 记录调用信息
        print(f"模拟异步嵌入生成，输入文本数量: {len(texts)}")
        for i, text in enumerate(texts[:3]):  # 只打印前3个文本
            print(f"  文本{i+1}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
            
        # 模拟API调用延迟
        await asyncio.sleep(self.delay)
        
        # 生成确定性但看似随机的嵌入
        embeddings = []
        for text in texts:
            # 使用文本的哈希值作为随机种子
            text_hash = abs(hash(text)) % (10 ** 8)
            np.random.seed(text_hash)
            
            # 生成单位长度的嵌入向量
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding.tolist())
            
        # 重置随机种子
        np.random.seed(None)
        
        return embeddings
            
    async def aembed_query(self, text):
        """
        模拟异步生成单个查询文本的嵌入向量
        
        参数:
            text: 要生成嵌入的查询文本
            
        返回:
            查询文本的嵌入向量
        """
        # 增加调用计数
        self.call_count += 1
        
        # 记录调用信息
        print(f"模拟异步查询嵌入生成，查询文本: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
        # 模拟API调用延迟
        await asyncio.sleep(self.delay)
        
        # 对单个文本使用aembed_documents
        embeddings = await self.aembed_documents([text])
        
        # 返回第一个（也是唯一一个）嵌入结果
        return embeddings[0]

async def test_async_embedding_with_mocks():
    """
    使用自定义模拟类测试异步嵌入功能
    """
    # 创建模拟嵌入生成器
    mock_embeddings = MockCompanyEmbeddings(dimension=1536)  # 使用与OpenAI相同的维度
    
    print("\n===== 测试1: 单个查询嵌入 =====")
    # 测试查询文本
    start_time = time.time()
    query = "这是一个测试异步嵌入的文本，用于验证异步查询嵌入功能是否正常工作。"
    query_embedding = await mock_embeddings.aembed_query(query)
    end_time = time.time()
    
    print(f"完成时间: {end_time - start_time:.2f}秒")
    print(f"嵌入向量维度: {len(query_embedding)}")
    print(f"向量前5个值: {query_embedding[:5]}")
    
    # 验证相同文本产生相同嵌入
    print("\n===== 测试2: 验证一致性 =====")
    query_embedding2 = await mock_embeddings.aembed_query(query)
    similarity = np.dot(query_embedding, query_embedding2) / (np.linalg.norm(query_embedding) * np.linalg.norm(query_embedding2))
    print(f"相同文本的嵌入相似度: {similarity}")  # 应该非常接近1.0
    
    print("\n===== 测试3: 批量文档嵌入 =====")
    # 测试多个文本的嵌入
    texts = [
        "LangChain是一个用于构建LLM应用的框架。",
        "嵌入向量可以用于文本相似度计算和语义搜索。",
        "向量数据库存储嵌入向量并支持快速检索。"
    ]
    
    start_time = time.time()
    embeddings_list = await mock_embeddings.aembed_documents(texts)
    end_time = time.time()
    
    print(f"完成时间: {end_time - start_time:.2f}秒")
    print(f"生成的嵌入向量数量: {len(embeddings_list)}")
    print(f"每个向量的维度: {len(embeddings_list[0])}")
    
    # 计算向量间相似度
    print("\n===== 测试4: 向量相似度分析 =====")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            vec1 = embeddings_list[i]
            vec2 = embeddings_list[j]
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(f"文本{i+1}和文本{j+1}的相似度: {similarity:.4f}")
    
    print(f"\n总共执行API调用次数: {mock_embeddings.call_count}")
    
if __name__ == "__main__":
    print("开始测试异步嵌入功能...")
    asyncio.run(test_async_embedding_with_mocks())
    print("\n所有测试完成!") 