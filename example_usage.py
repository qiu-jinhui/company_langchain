"""
示例：如何使用公司LLM适配器与LangChain
"""

import os
import asyncio
from company_llm import CompanyChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

def basic_usage_example():
    """基本使用示例：直接调用公司LLM"""
    
    # 初始化公司LLM模型
    # 注意：这些是示例值，请替换为您的实际配置
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        model="gpt-3.5-turbo",  # 或您公司支持的模型名称
        temperature=0.7,
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个有用的AI助手。"),
        HumanMessage(content="请简要介绍一下LangChain框架。")
    ]
    
    # 调用LLM
    response = llm.invoke(messages)
    print("基本调用示例响应:")
    print(response.content)
    print("\n" + "-"*50 + "\n")


def langchain_chain_example():
    """LangChain链示例：将公司LLM与Chain结合使用"""
    
    # 初始化公司LLM模型
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
    )
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}。请用简洁的语言回答用户的问题。"),
        ("human", "{query}")
    ])
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 运行链
    response = chain.invoke({
        "role": "技术文档作者",
        "query": "解释API网关的主要功能和优势"
    })
    
    print("LangChain链示例响应:")
    print(response["text"])


def token_refresh_example():
    """令牌自动刷新示例"""
    # 初始化公司LLM模型（启用令牌自动刷新）
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-initial-token",  # 初始令牌
        
        # 令牌刷新配置
        token_refresh_enabled=True,  # 启用令牌自动刷新
        token_url="https://your-company-auth-api.example.com/token",  # 令牌刷新API
        username="your-username",  # 用户名
        password="your-password",  # 密码
        token_refresh_interval=3600,  # 令牌刷新间隔（秒）
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的安全顾问。"),
        HumanMessage(content="请解释JWT令牌的工作原理和安全性。")
    ]
    
    try:
        # 调用LLM（会自动刷新令牌）
        print("\n令牌自动刷新示例:")
        print("正在调用LLM（如果需要，会自动刷新令牌）...")
        response = llm.invoke(messages)
        print("响应内容:")
        print(response.content)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


def streaming_example():
    """流式传输示例"""
    # 初始化公司LLM模型（启用流式传输）
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        temperature=0.7,
        streaming=True,  # 启用流式传输
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的技术讲师。"),
        HumanMessage(content="请详细解释LangChain框架的核心组件。")
    ]
    
    try:
        print("\n流式传输示例响应:")
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def async_example():
    """异步调用示例"""
    # 初始化公司LLM模型
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的软件工程师。"),
        HumanMessage(content="请解释WebSocket与HTTP的区别。")
    ]
    
    try:
        print("\n异步调用示例响应:")
        response = await llm.ainvoke(messages)
        print(response.content)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def async_token_refresh_example():
    """异步令牌刷新示例"""
    # 初始化公司LLM模型（启用令牌自动刷新）
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-initial-token",  # 初始令牌
        
        # 令牌刷新配置
        token_refresh_enabled=True,  # 启用令牌自动刷新
        token_url="https://your-company-auth-api.example.com/token",  # 令牌刷新API
        username="your-username",  # 用户名
        password="your-password",  # 密码
        token_refresh_interval=3600,  # 令牌刷新间隔（秒）
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的网络安全专家。"),
        HumanMessage(content="请解释OAuth 2.0的授权流程。")
    ]
    
    try:
        print("\n异步令牌刷新示例:")
        print("正在异步调用LLM（如果需要，会自动刷新令牌）...")
        response = await llm.ainvoke(messages)
        print("响应内容:")
        print(response.content)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def async_streaming_example():
    """异步流式传输示例"""
    # 初始化公司LLM模型（启用流式传输）
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        streaming=True,
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的AI研究员。"),
        HumanMessage(content="请解释大型语言模型的工作原理。")
    ]
    
    try:
        print("\n异步流式传输示例响应:")
        async for chunk in llm.astream(messages):
            print(chunk.content, end="", flush=True)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")


async def run_async_examples():
    """运行所有异步示例"""
    await async_example()
    await async_token_refresh_example()
    await async_streaming_example()


if __name__ == "__main__":
    print("公司LLM适配器示例")
    print("="*50)
    print("注意：这些示例需要有效的API凭据才能运行\n")
    
    # 运行示例
    print("这些示例展示了如何使用公司LLM适配器。")
    print("请替换API凭据后再运行。")
    
    print("\n可用的示例功能:")
    print("1. 基本调用 (basic_usage_example)")
    print("2. LangChain链集成 (langchain_chain_example)")
    print("3. 令牌自动刷新 (token_refresh_example)")
    print("4. 流式传输 (streaming_example)")
    print("5. 异步调用 (asyncio.run(async_example()))")
    print("6. 异步令牌刷新 (asyncio.run(async_token_refresh_example()))")
    print("7. 异步流式传输 (asyncio.run(async_streaming_example()))")
    print("8. 运行所有异步示例 (asyncio.run(run_async_examples()))")
    
    # 取消注释以下行来运行示例（需要提供有效的API凭据）
    # basic_usage_example()
    # langchain_chain_example()
    # token_refresh_example()
    # streaming_example()
    # asyncio.run(async_example())
    # asyncio.run(async_token_refresh_example())
    # asyncio.run(async_streaming_example())
    # asyncio.run(run_async_examples()) 