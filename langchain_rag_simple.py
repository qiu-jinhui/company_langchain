"""
简单的RAG（检索增强生成）示例

本示例展示如何使用CompanyEmbeddings创建一个基本的RAG应用，
包括文档的嵌入、检索和模拟生成响应的流程。
"""

import numpy as np
from pydantic import Field

# 导入公司嵌入模型适配器
from company_embedding import CompanyEmbeddings

# 导入LangChain组件
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 创建模拟嵌入类
class MockEmbeddings(CompanyEmbeddings):
    """本地测试用的模拟嵌入类"""
    
    dimension: int = Field(1536, description="嵌入向量的维度")
    call_counter: int = Field(0, description="API调用计数")
    
    def __init__(self, **kwargs):
        """初始化模拟嵌入生成器"""
        api_params = {
            "api_url": "https://mock-api.example.com/v1/embeddings",
            "application_id": "mock-app-id",
            "trust_token": "mock-token"
        }
        api_params.update(kwargs)
        super().__init__(**api_params)
    
    def embed_documents(self, texts):
        """模拟生成文档的嵌入向量"""
        self.call_counter += 1
        print(f"生成{len(texts)}个文档的嵌入向量")
        
        embeddings = []
        for text in texts:
            text_hash = abs(hash(text)) % (10 ** 8)
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
            
        np.random.seed(None)
        return embeddings
    
    def embed_query(self, text):
        """模拟生成查询的嵌入向量"""
        self.call_counter += 1
        print(f"生成查询的嵌入向量: '{text[:50]}...'")
        
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
    @property
    def call_count(self):
        """获取API调用计数"""
        return self.call_counter


# 创建模拟LLM类
class MockLLM:
    """模拟大语言模型，用于RAG演示"""
    
    def __init__(self):
        self.call_count = 0
    
    def invoke(self, prompt):
        """模拟LLM生成回答"""
        self.call_count += 1
        
        # 处理StringPromptValue对象
        if hasattr(prompt, "to_string"):
            prompt_text = prompt.to_string()
        else:
            prompt_text = str(prompt)
            
        print(f"LLM收到的Prompt长度: {len(prompt_text)}")
        
        # 非常简单的模拟回答生成
        if "LangChain" in prompt_text:
            return "LangChain是一个用于构建LLM应用的框架，它提供了各种工具和组件，使开发者能够轻松创建复杂的AI应用。"
        elif "嵌入" in prompt_text or "向量" in prompt_text:
            return "嵌入向量是将文本转换为数值表示的方法，它能够捕获文本的语义信息，使计算机能够理解文本之间的相似性。"
        elif "RAG" in prompt_text:
            return "RAG（检索增强生成）是一种结合检索系统和生成模型的方法，它能够使模型生成基于事实、更加准确的回答。"
        else:
            return "您的问题很有趣。基于我检索到的信息，我可以说这是一个与AI和语言模型相关的话题。您是否想了解更多关于LangChain、嵌入向量或RAG的信息？"


def create_knowledge_base():
    """创建知识库"""
    print("创建知识库...")
    
    # 准备文档
    documents = [
        Document(
            page_content="LangChain是一个用于开发由大语言模型驱动的应用程序的框架。它支持文档加载、提示管理、模型接口等功能。",
            metadata={"source": "langchain_intro.txt", "topic": "framework"}
        ),
        Document(
            page_content="嵌入模型将文本转换为向量表示，这些向量可以用于相似性搜索和语义匹配。",
            metadata={"source": "embedding_guide.txt", "topic": "embedding"}
        ),
        Document(
            page_content="向量数据库是专门设计用于存储和检索向量数据的数据库系统。FAISS和Chroma是常用的向量数据库。",
            metadata={"source": "vector_db_intro.txt", "topic": "database"}
        ),
        Document(
            page_content="RAG（检索增强生成）通过从外部知识库检索相关信息，增强语言模型的回答能力。它结合了检索和生成的优势。",
            metadata={"source": "rag_overview.txt", "topic": "technique"}
        ),
        Document(
            page_content="提示工程是设计和优化提示的过程，以引导语言模型生成所需的输出。它包括提示模板、少样本学习等技术。",
            metadata={"source": "prompt_engineering.txt", "topic": "technique"}
        )
    ]
    
    # 创建嵌入模型
    embeddings = MockEmbeddings(dimension=1536)
    
    # 创建向量存储
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"知识库创建完成，包含{len(documents)}个文档\n")
    
    return vector_store, embeddings


def setup_rag_chain(vector_store):
    """设置RAG检索链"""
    
    # 创建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 创建模拟LLM
    llm = MockLLM()
    
    # 创建提示模板
    template = """基于以下上下文回答用户的问题：

上下文:
{context}

用户问题: {question}

请提供详细、准确的回答，并仅基于上下文中的信息:
"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 创建RAG链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm.invoke
        | StrOutputParser()
    )
    
    return rag_chain


def run_rag_demo():
    """运行RAG演示"""
    print("=" * 50)
    print("简单RAG演示")
    print("=" * 50)
    
    # 创建知识库
    vector_store, embeddings = create_knowledge_base()
    
    # 设置RAG链
    rag_chain = setup_rag_chain(vector_store)
    
    # 准备问题
    questions = [
        "什么是LangChain框架？",
        "嵌入向量有什么用途？",
        "RAG技术的工作原理是什么？",
        "向量数据库与传统数据库有何不同？"
    ]
    
    # 运行问答
    for i, question in enumerate(questions):
        print(f"\n问题 {i+1}: {question}")
        
        print("检索相关文档...")
        answer = rag_chain.invoke(question)
        
        print(f"回答: {answer}\n")
        print("-" * 40)
    
    # 打印统计信息
    print(f"\n总共执行了{embeddings.call_count}次嵌入API调用")


if __name__ == "__main__":
    run_rag_demo() 