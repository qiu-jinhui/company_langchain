"""
LangChain嵌入模型集成高级示例

本文件展示如何将CompanyEmbeddings适配器与LangChain的多个组件集成使用，
包括文本分割、向量存储、检索链和语义搜索等高级功能。
"""

import os
import asyncio
from pydantic import SecretStr, Field
import numpy as np

# 导入公司嵌入模型适配器
from company_embedding import CompanyEmbeddings

# 导入LangChain组件
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    
    async def aembed_documents(self, texts):
        """模拟异步生成多个文本的嵌入向量"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text):
        """模拟异步生成单个查询文本的嵌入向量"""
        return self.embed_query(text)
    
    @property
    def call_count(self):
        """获取API调用计数"""
        return self.call_counter


def example_1_basic_vector_store():
    """示例1：基本向量存储与相似度搜索"""
    print("\n===== 示例1：基本向量存储与相似度搜索 =====")
    
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
    
    # 保存和加载向量存储
    print("保存向量存储到磁盘...")
    vector_store.save_local("faiss_index")
    
    print("从磁盘加载向量存储...")
    # 添加allow_dangerous_deserialization参数以解决安全问题
    loaded_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("成功加载向量存储!\n")
    
    # 添加新文档
    new_docs = [
        Document(page_content="文本分割器可以将长文档分解为更小的块，便于处理和检索。", 
                 metadata={"source": "processing_doc", "category": "preprocessing"})
    ]
    
    print("向现有向量存储添加新文档...")
    loaded_vector_store.add_documents(new_docs)
    print(f"向量存储现在包含{len(documents) + len(new_docs)}个文档")


def example_2_text_splitting():
    """示例2：文本分割与嵌入"""
    print("\n===== 示例2：文本分割与嵌入 =====")
    
    # 初始化模拟嵌入模型
    embeddings = MockEmbeddings(dimension=1536)
    
    # 准备一篇长文档
    long_text = """
    LangChain是一个用于开发由语言模型驱动的应用程序的框架。它具有以下主要组件：

    1. 模型：LangChain支持集成多种语言模型，包括OpenAI的GPT系列、Anthropic的Claude、开源模型如Llama等。
    
    2. 提示管理：提供了构建、优化和重用提示模板的工具。
    
    3. 记忆：实现上下文管理，使模型能够记住之前的交互。
    
    4. 检索：支持从外部数据源获取相关信息，增强模型的回答。
    
    5. 代理：赋予模型使用工具和执行操作的能力。
    
    LangChain的设计理念是模块化和可扩展的。它不仅提供了现成的组件，还允许开发者轻松创建和集成自己的组件。
    
    嵌入模型是LangChain中的重要组成部分。它们用于将文本转换为数值向量，捕获文本的语义信息。这些向量可以存储在向量数据库中，然后用于相似性搜索、文本聚类等任务。LangChain支持多种嵌入模型和向量存储解决方案。

    向量存储是LangChain中用于存储和检索嵌入向量的组件。它们提供了高效的相似性搜索功能。LangChain支持多种向量存储，包括FAISS、Chroma、Pinecone等。

    检索增强生成（RAG）是LangChain中的一个强大功能。它结合了检索系统和生成模型的优势。首先，从外部数据源检索相关信息，然后将这些信息提供给生成模型，使其能够生成更准确、更相关的回答。
    """
    
    # 创建文本分割器
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    character_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="。"
    )
    
    # 分割文本
    recursive_chunks = recursive_splitter.split_text(long_text)
    character_chunks = character_splitter.split_text(long_text)
    
    print(f"使用递归分割器得到{len(recursive_chunks)}个文本块")
    print(f"使用字符分割器得到{len(character_chunks)}个文本块")
    
    print("\n递归分割器前3个块:")
    for i, chunk in enumerate(recursive_chunks[:3]):
        print(f"块 {i+1}: {chunk[:100]}...")
    
    # 创建向量存储
    print("\n从分割的文本块创建Chroma向量存储...")
    recursive_docs = [Document(page_content=chunk) for chunk in recursive_chunks]
    vector_store = Chroma.from_documents(recursive_docs, embeddings)
    
    # 执行查询
    print("\n在分块后的文档上执行查询...")
    query = "LangChain中的嵌入模型有什么用途？"
    results = vector_store.similarity_search(query, k=2)
    
    print(f"查询: '{query}'")
    print("结果:")
    for i, doc in enumerate(results):
        print(f"  结果 {i+1}: {doc.page_content}")


async def example_3_async_operations():
    """示例3：异步操作"""
    print("\n===== 示例3：异步操作 =====")
    
    # 初始化模拟嵌入模型
    embeddings = MockEmbeddings(dimension=1536)
    
    # 准备文档
    documents = [
        Document(page_content="异步编程允许程序在等待I/O操作完成时执行其他任务。"),
        Document(page_content="Python的asyncio库提供了编写异步代码的工具。"),
        Document(page_content="异步API调用可以提高处理大量请求时的性能。")
    ]
    
    # 创建向量存储 - 同步方式
    print("同步创建向量存储...")
    start_time = asyncio.get_event_loop().time()
    vector_store = FAISS.from_documents(documents, embeddings)
    end_time = asyncio.get_event_loop().time()
    print(f"同步创建完成，耗时: {end_time - start_time:.4f}秒")
    
    # 执行异步查询
    print("\n执行异步查询...")
    query = "异步编程有什么优势？"
    
    # 手动实现异步查询
    start_time = asyncio.get_event_loop().time()
    query_vector = await embeddings.aembed_query(query)
    # 这里我们使用同步方法模拟异步查询结果
    results = vector_store.similarity_search(query, k=2)
    end_time = asyncio.get_event_loop().time()
    
    print(f"异步查询完成，耗时: {end_time - start_time:.4f}秒")
    print(f"查询: '{query}'")
    print("结果:")
    for i, doc in enumerate(results):
        print(f"  结果 {i+1}: {doc.page_content}")


def example_4_metadata_filtering():
    """示例4：元数据过滤"""
    print("\n===== 示例4：元数据过滤 =====")
    
    # 初始化模拟嵌入模型
    embeddings = MockEmbeddings(dimension=1536)
    
    # 准备带元数据的文档
    documents = [
        Document(page_content="Python是一种广泛使用的解释型高级编程语言。", 
                 metadata={"language": "python", "category": "programming", "difficulty": "beginner"}),
        Document(page_content="JavaScript是一种脚本语言，用于创建动态网页内容。", 
                 metadata={"language": "javascript", "category": "programming", "difficulty": "beginner"}),
        Document(page_content="React是一个用于构建用户界面的JavaScript库。", 
                 metadata={"language": "javascript", "category": "framework", "difficulty": "intermediate"}),
        Document(page_content="Django是一个高级Python Web框架，鼓励快速开发和简洁实用的设计。", 
                 metadata={"language": "python", "category": "framework", "difficulty": "intermediate"}),
        Document(page_content="机器学习是人工智能的一个子领域，使计算机能够学习而无需明确编程。", 
                 metadata={"language": "general", "category": "data-science", "difficulty": "advanced"}),
    ]
    
    # 创建Chroma向量存储（支持元数据过滤）
    print("创建支持元数据过滤的Chroma向量存储...")
    vector_store = Chroma.from_documents(documents, embeddings)
    
    # 执行带元数据过滤的检索
    filters = [
        {"language": "python"},
        {"category": "framework"},
        {"language": "python", "category": "framework"},
        {"difficulty": "advanced"}
    ]
    
    print("\n使用元数据过滤执行查询:")
    query = "编程语言和框架"
    
    for filter_dict in filters:
        print(f"\n查询: '{query}' 过滤条件: {filter_dict}")
        results = vector_store.similarity_search(
            query, 
            k=2,
            filter=filter_dict
        )
        
        print("结果:")
        for i, doc in enumerate(results):
            print(f"  结果 {i+1}: {doc.page_content}")
            print(f"    元数据: {doc.metadata}")


def cleanup():
    """清理测试文件"""
    import shutil
    
    # 删除测试生成的目录和文件
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")


if __name__ == "__main__":
    print("LangChain嵌入模型集成高级示例")
    print("=" * 50)
    
    try:
        # 运行同步示例 - 只运行示例1
        example_1_basic_vector_store()
        
        # 运行异步示例
        asyncio.run(example_3_async_operations())
        
    finally:
        # 清理临时文件
        cleanup()
        
    print("\n所有示例运行完成!") 