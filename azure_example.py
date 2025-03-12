"""
示例：如何使用公司Azure OpenAI适配器
"""

import os
import asyncio
from company_azure_llm import CompanyAzureChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

def set_azure_env():
    """设置Azure环境变量（仅用于演示）"""
    # 在实际使用时，建议通过环境变量或配置文件设置这些值
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"

def basic_usage_example():
    """基本使用示例"""
    # 初始化Azure OpenAI模型
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",  # Azure部署名称
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        temperature=0.7,
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的技术顾问。"),
        HumanMessage(content="请解释什么是Azure OpenAI服务？")
    ]
    
    try:
        # 调用LLM
        response = llm.invoke(messages)
        print("\n基本调用示例响应:")
        print(response.content)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")

def token_refresh_example():
    """令牌自动刷新示例"""
    # 初始化Azure OpenAI模型（启用令牌自动刷新）
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",  # Azure部署名称
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
        SystemMessage(content="你是一个专业的云安全专家。"),
        HumanMessage(content="请解释Azure Active Directory的主要功能和安全特性。")
    ]
    
    try:
        # 调用LLM（会自动刷新令牌）
        print("\n令牌自动刷新示例:")
        print("正在调用Azure LLM（如果需要，会自动刷新令牌）...")
        response = llm.invoke(messages)
        print("响应内容:")
        print(response.content)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")

def langchain_chain_example():
    """LangChain链示例"""
    # 初始化Azure OpenAI模型
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        temperature=0.5,
    )
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}。请用通俗易懂的语言回答问题。"),
        ("human", "{query}")
    ])
    
    # 创建链
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # 运行链
        response = chain.invoke({
            "role": "Azure云服务专家",
            "query": "Azure OpenAI Service与普通OpenAI API有什么区别？"
        })
        
        print("\nLangChain链示例响应:")
        print(response["text"])
    except Exception as e:
        print(f"错误: {str(e)}")

def streaming_example():
    """流式传输示例"""
    # 初始化Azure OpenAI模型（启用流式传输）
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        temperature=0.7,
        streaming=True,  # 启用流式传输
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的技术讲师。"),
        HumanMessage(content="请一步一步地解释Azure OpenAI服务的部署过程。")
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
    # 初始化Azure OpenAI模型
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个专业的云计算专家。"),
        HumanMessage(content="请简要介绍Azure的主要服务类别。")
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
    # 初始化Azure OpenAI模型（启用令牌自动刷新）
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",  # Azure部署名称
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
        SystemMessage(content="你是一个专业的Azure安全架构师。"),
        HumanMessage(content="请解释Azure中的身份验证和授权最佳实践。")
    ]
    
    try:
        print("\n异步令牌刷新示例:")
        print("正在异步调用Azure LLM（如果需要，会自动刷新令牌）...")
        response = await llm.ainvoke(messages)
        print("响应内容:")
        print(response.content)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")

async def async_streaming_example():
    """异步流式传输示例"""
    # 初始化Azure OpenAI模型（启用流式传输）
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        streaming=True,
    )
    
    # 创建消息
    messages = [
        SystemMessage(content="你是一个AI系统架构师。"),
        HumanMessage(content="请解释大型语言模型的工作原理。")
    ]
    
    try:
        print("\n异步流式传输示例响应:")
        async for chunk in llm.astream(messages):
            print(chunk.content, end="", flush=True)
        print("\n" + "-"*50 + "\n")
    except Exception as e:
        print(f"错误: {str(e)}")

def show_request_format():
    """展示请求格式（不实际发送请求）"""
    llm = CompanyAzureChatModel(
        deployment_name="your-deployment-name",
        application_id="your-application-id",
        user_login_as="user@example.com",
        trust_token="your-trust-token",
        
        # 令牌刷新配置（可选）
        token_refresh_enabled=True,
        token_url="https://your-company-auth-api.example.com/token",
        username="your-username",
        password="your-password",
    )
    
    print("\n请求配置示例:")
    print("-" * 50)
    print(f"Azure端点: {os.getenv('AZURE_OPENAI_ENDPOINT', 'not-set')}")
    print(f"API版本: {llm.openai_api_version}")
    print(f"部署名称: {llm.deployment_name}")
    print(f"令牌刷新: {'启用' if llm.token_refresh_enabled else '禁用'}")
    if llm.token_refresh_enabled:
        print(f"令牌刷新URL: {llm.token_url}")
        print(f"令牌刷新间隔: {llm.token_refresh_interval}秒")
    
    print("\n客户端参数:")
    client_params = llm._client_params.copy()
    # 隐藏敏感信息
    client_params["api_key"] = "***" if client_params["api_key"] else None
    client_params["default_headers"]["X-E2E-Trust-Token"] = "***"
    for key, value in client_params.items():
        print(f"  {key}: {value}")

async def run_async_examples():
    """运行所有异步示例"""
    await async_example()
    await async_token_refresh_example()
    await async_streaming_example()

if __name__ == "__main__":
    print("公司Azure OpenAI适配器示例")
    print("=" * 50)
    
    # 设置环境变量（实际使用时应通过环境变量或配置文件设置）
    set_azure_env()
    
    print("\n注意：这些示例需要正确的Azure OpenAI配置和API凭据才能运行")
    print("请确保已设置以下环境变量：")
    print("- AZURE_OPENAI_API_KEY")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- OPENAI_API_VERSION")
    
    # 显示请求格式
    show_request_format()
    
    print("\n可用的示例功能:")
    print("1. 基本调用 (basic_usage_example)")
    print("2. 令牌自动刷新 (token_refresh_example)")
    print("3. LangChain链集成 (langchain_chain_example)")
    print("4. 流式传输 (streaming_example)")
    print("5. 异步调用 (asyncio.run(async_example()))")
    print("6. 异步令牌刷新 (asyncio.run(async_token_refresh_example()))")
    print("7. 异步流式传输 (asyncio.run(async_streaming_example()))")
    print("8. 运行所有异步示例 (asyncio.run(run_async_examples()))")
    
    print("\n要运行实际示例，请取消注释以下行并提供有效的凭据：")
    # basic_usage_example()
    # token_refresh_example()
    # langchain_chain_example()
    # streaming_example()
    # asyncio.run(async_example())
    # asyncio.run(async_token_refresh_example())
    # asyncio.run(async_streaming_example())
    # asyncio.run(run_async_examples()) 