"""
LangChain嵌入模型集成简化示例

本文件展示如何将CompanyEmbeddings适配器与LangChain框架集成使用，
仅演示基本的向量存储和检索功能。
"""

import numpy as np
from pydantic import SecretStr, Field

# 导入公司嵌入模型适配器
from company_embedding import CompanyEmbeddings

# 导入LangChain组件
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 创建模拟嵌入类用于本地测试（避免需要实际API凭据）
class MockEmbeddings(CompanyEmbeddings):
    """本地测试用的模拟嵌入类"""
    
    # 将自定义属性添加为Pydantic字段
    dimension: int = Field(1536, description="嵌入向量的维度")
    call_counter: int = Field(0, description="API调用计数")
    
    def __init__(self, **kwargs):
        """初始化模拟嵌入生成器"""
        # 设置默认的API参数
        api_params = {
            "api_url": "https://mock-api.example.com/v1/embeddings",
            "application_id": "mock-app-id",
            "trust_token": "mock-token"
        }
        # 更新传入的参数
        api_params.update(kwargs)
        # 调用父类初始化方法
        super().__init__(**api_params)
    
    def embed_documents(self, texts):
        """模拟生成多个文本的嵌入向量"""
        self.call_counter += 1
        print(f"生成{len(texts)}个文本的嵌入向量")
        
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
    
    def embed_query(self, text):
        """模拟生成单个查询文本的嵌入向量"""
        self.call_counter += 1
        print(f"生成查询的嵌入向量: '{text[:50]}...'")
        
        # 对单个文本使用embed_documents
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
    @property
    def call_count(self):
        """获取API调用计数"""
        return self.call_counter


def basic_vector_store_example():
    """基本向量存储与相似度搜索示例"""
    print("\n===== 基本向量存储与相似度搜索示例 =====")
    
    # 初始化模拟嵌入模型
    embeddings = MockEmbeddings(dimension=1536)
    
    # 准备示例文档
    documents = [
        Document(page_content="LangChain是一个强大的框架，用于开发由语言模型驱动的应用程序。", 
                 metadata={"source": "intro_doc", "category": "framework"}),
        Document(page_content="向量数据库是存储向量嵌入的专用数据库，支持高效的相似性搜索。", 
                 metadata={"source": "tech_doc", "category": "database"}),
        Document(page_content="嵌入模型将文本转换为数值向量，捕获语义信息。", 
                 metadata={"source": "tutorial", "category": "embedding"}),
        Document(page_content="语义搜索使用嵌入向量来查找与查询语义相关的文档。", 
                 metadata={"source": "search_guide", "category": "search"}),
        Document(page_content="RAG（检索增强生成）结合了检索系统和生成模型的优势。", 
                 metadata={"source": "advanced_guide", "category": "rag"}),
    ]
    
    # 创建向量存储
    print("创建FAISS向量存储...")
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"向量存储创建完成，包含{len(documents)}个文档\n")
    
    # 执行相似性搜索
    queries = [
        "什么是语义搜索？",
        "向量数据库如何工作？",
        "LangChain有什么用途？"
    ]
    
    for query in queries:
        print(f"查询: '{query}'")
        results = vector_store.similarity_search_with_score(query, k=2)
        
        print("搜索结果:")
        for i, (doc, score) in enumerate(results):
            # FAISS中，分数越小表示越相似
            similarity = 1.0 / (1.0 + score)
            print(f"  结果 {i+1} (相似度: {similarity:.4f}):")
            print(f"    内容: {doc.page_content}")
            print(f"    元数据: {doc.metadata}")
        print()
    
    # 添加新文档演示
    new_docs = [
        Document(page_content="文本分割器可以将长文档分解为更小的块，便于处理和检索。", 
                 metadata={"source": "processing_doc", "category": "preprocessing"})
    ]
    
    print("向现有向量存储添加新文档...")
    vector_store.add_documents(new_docs)
    print(f"向量存储现在包含{len(documents) + len(new_docs)}个文档")
    
    # 使用元数据过滤执行搜索
    print("\n使用元数据过滤执行搜索...")
    filtered_docs = vector_store.similarity_search(
        "什么是语义搜索？",
        k=1,
        filter={"category": "search"}
    )
    
    print("按类别'search'过滤的结果:")
    for doc in filtered_docs:
        print(f"  内容: {doc.page_content}")
        print(f"  元数据: {doc.metadata}")
    
    # 最终的统计
    print(f"\n总共执行API调用次数: {embeddings.call_count}")


if __name__ == "__main__":
    print("LangChain嵌入模型集成简化示例")
    print("=" * 50)
    
    # 运行示例
    basic_vector_store_example()
    
    print("\n示例运行完成!") 