# 公司LLM适配器

这个项目提供了自定义的LangChain模型实现，用于连接公司内部的LLM API以及Azure OpenAI API，并与LangChain生态系统进行集成。

## 功能特点

- 提供两种适配器实现：
  - `CompanyChatModel`: 连接公司内部LLM API
  - `CompanyAzureChatModel`: 连接Azure OpenAI API
- 支持公司特定的认证头部（AGI-Platform-Application-ID、X-DSP-User-Login-As、X-E2E-Trust-Token）
- 内置令牌刷新机制，可自动刷新过期令牌
- 强大的重试机制，能够自动处理临时性错误
- 支持同步和异步调用方式
- 支持流式响应（streaming）
- 与OpenAI API和Azure OpenAI API兼容的请求和响应格式
- 与LangChain框架无缝集成
- 支持各种LLM参数配置（温度、最大token数等）
- 包含模拟API服务器，用于开发和测试

## 安装

1. 克隆此仓库：

```bash
git clone https://github.com/yourusername/companylangchain.git
cd companylangchain
```

2. 创建并激活虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 公司内部LLM API适配器

```python
from company_llm import CompanyChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化模型
llm = CompanyChatModel(
    api_url="您的公司LLM API URL",
    application_id="您的应用ID",
    user_login_as="您的用户登录名",
    trust_token=SecretStr("您的信任令牌"),
    model="您使用的模型名称",
    temperature=0.7,
    # 启用令牌刷新
    token_refresh_enabled=True,
    token_url="https://api.company.com/token",
    username=SecretStr("your_username"),
    password=SecretStr("your_password"),
    # 配置重试机制
    max_retries=6,
    retry_min_delay=1.0,
    retry_max_delay=60.0,
    retry_backoff_factor=2.0,
)

# 创建消息
messages = [
    SystemMessage(content="你是一个有用的AI助手。"),
    HumanMessage(content="请介绍一下LangChain框架。")
]

# 调用LLM
response = llm.invoke(messages)
print(response.content)

# 使用流式传输
llm.streaming = True
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Azure OpenAI API适配器

```python
from company_azure_llm import CompanyAzureChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

# 初始化模型
llm = CompanyAzureChatModel(
    deployment_name="您的Azure部署名称",
    application_id="您的应用ID",
    user_login_as="您的用户登录名",
    trust_token=SecretStr("您的信任令牌"),
    # 可选的Azure配置
    openai_api_key=SecretStr("您的Azure API密钥"),  # 也可从环境变量读取
    azure_endpoint="https://your-resource-name.openai.azure.com",  # 也可从环境变量读取
    # 启用令牌刷新
    token_refresh_enabled=True,
    token_url="https://api.company.com/token",
    username=SecretStr("your_username"),
    password=SecretStr("your_password"),
    # 配置重试机制
    max_retries=6,
    retry_min_delay=1.0,
    retry_max_delay=60.0,
    retry_backoff_factor=2.0,
)

# 创建消息
messages = [
    SystemMessage(content="你是一个有用的AI助手。"),
    HumanMessage(content="请介绍一下LangChain框架。")
]

# 调用LLM
response = llm.invoke(messages)
print(response.content)

# 使用流式传输
llm.streaming = True
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### 异步调用示例

```python
import asyncio
from company_llm import CompanyChatModel
from langchain_core.messages import HumanMessage

async def main():
    llm = CompanyChatModel(
        api_url="您的公司LLM API URL",
        application_id="您的应用ID",
        user_login_as="您的用户登录名",
        trust_token=SecretStr("您的信任令牌"),
    )
    
    # 异步调用
    response = await llm.ainvoke([HumanMessage(content="你好，请简单介绍一下你自己")])
    print(response.content)
    
    # 异步流式调用
    llm.streaming = True
    async for chunk in llm.astream([HumanMessage(content="异步流式响应测试")]):
        print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## 模拟API服务器

项目包含一个模拟API服务器，用于开发和测试LLM适配器，无需连接真实的API。

### 启动模拟服务器

```bash
python run_mock_api.py --port 5001 --llm-error-rate 0.2 --azure-error-rate 0.2
```

参数说明：
- `--host`: 服务器主机地址，默认为0.0.0.0
- `--port`: 服务器监听端口，默认为5000
- `--llm-error-rate`: 公司LLM API错误率（0-1），默认为0.3
- `--azure-error-rate`: Azure API错误率（0-1），默认为0.3
- `--token-error-rate`: 令牌API错误率（0-1），默认为0.1

### 使用模拟服务器进行测试

```python
from company_llm import CompanyChatModel
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

# 连接到模拟服务器
llm = CompanyChatModel(
    api_url="http://localhost:5001/v1/chat/completions",
    application_id="test-app-id",
    user_login_as="test-user@example.com",
    trust_token=SecretStr("test-token"),
    # 启用令牌刷新
    token_refresh_enabled=True,
    token_url="http://localhost:5001/api/token",
    username=SecretStr("test_user"),
    password=SecretStr("test_password"),
)

# 调用模拟API
response = llm.invoke([HumanMessage(content="你好，这是一个测试")])
print(response.content)
```

## 配置参数

### CompanyChatModel 参数

| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| api_url | 字符串 | 公司LLM API的URL | 必填 |
| application_id | 字符串 | AGI-Platform-Application-ID头部 | 必填 |
| user_login_as | 字符串 | X-DSP-User-Login-As头部 | 必填 |
| trust_token | SecretStr | X-E2E-Trust-Token头部 | 必填 |
| model | 字符串 | 使用的模型名称 | "gpt-3.5-turbo" |
| temperature | 浮点数 | 随机性程度（0-1） | 0.7 |
| max_tokens | 整数 | 生成的最大token数 | None |
| top_p | 浮点数 | 核心采样的概率质量 | 1.0 |
| frequency_penalty | 浮点数 | 频率惩罚系数 | 0.0 |
| presence_penalty | 浮点数 | 存在惩罚系数 | 0.0 |
| timeout | 浮点数 | API请求超时时间（秒） | None |
| request_timeout | 浮点数 | 请求超时时间（秒） | None |
| token_refresh_enabled | 布尔值 | 是否启用令牌自动刷新 | False |
| token_url | 字符串 | 令牌刷新API的URL | None |
| username | SecretStr | 用于令牌刷新的用户名 | None |
| password | SecretStr | 用于令牌刷新的密码 | None |
| token_refresh_interval | 整数 | 令牌最小刷新间隔（秒） | 0 |
| max_retries | 整数 | 最大重试次数 | 6 |
| retry_min_delay | 浮点数 | 重试的最小延迟时间（秒） | 1.0 |
| retry_max_delay | 浮点数 | 重试的最大延迟时间（秒） | 60.0 |
| retry_backoff_factor | 浮点数 | 重试延迟的退避系数 | 2.0 |
| retry_jitter | 布尔值 | 是否在重试延迟时间上添加随机抖动 | True |
| retry_on_status_codes | 整数列表 | 需要重试的HTTP状态码 | [429, 500, 502, 503, 504] |
| streaming | 布尔值 | 是否使用流式传输 | False |
| n | 整数 | 生成的回复数量 | 1 |

### CompanyAzureChatModel 参数

| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| deployment_name | 字符串 | Azure OpenAI API部署名称 | 必填 |
| application_id | 字符串 | AGI-Platform-Application-ID头部 | 必填 |
| user_login_as | 字符串 | X-DSP-User-Login-As头部 | 必填 |
| trust_token | SecretStr | X-E2E-Trust-Token头部 | 必填 |
| openai_api_version | 字符串 | Azure OpenAI API版本 | "2023-05-15" |
| openai_api_key | SecretStr | Azure OpenAI API密钥 | 环境变量 |
| azure_endpoint | 字符串 | Azure OpenAI API端点 | 环境变量 |
| azure_deployment | 字符串 | Azure部署名称（别名） | deployment_name |
| temperature | 浮点数 | 随机性程度（0-1） | 0.7 |
| max_tokens | 整数 | 生成的最大token数 | None |
| top_p | 浮点数 | 核心采样的概率质量 | 1.0 |
| frequency_penalty | 浮点数 | 频率惩罚系数 | 0.0 |
| presence_penalty | 浮点数 | 存在惩罚系数 | 0.0 |
| timeout | 浮点数 | API请求超时时间（秒） | None |
| request_timeout | 浮点数 | 请求超时时间（秒） | None |
| token_refresh_enabled | 布尔值 | 是否启用令牌自动刷新 | False |
| token_url | 字符串 | 令牌刷新API的URL | None |
| username | SecretStr | 用于令牌刷新的用户名 | None |
| password | SecretStr | 用于令牌刷新的密码 | None |
| token_refresh_interval | 整数 | 令牌最小刷新间隔（秒） | 0 |
| max_retries | 整数 | 最大重试次数 | 6 |
| retry_min_delay | 浮点数 | 重试的最小延迟时间（秒） | 1.0 |
| retry_max_delay | 浮点数 | 重试的最大延迟时间（秒） | 60.0 |
| retry_backoff_factor | 浮点数 | 重试延迟的退避系数 | 2.0 |
| retry_jitter | 布尔值 | 是否在重试延迟时间上添加随机抖动 | True |
| retry_on_status_codes | 整数列表 | 需要重试的HTTP状态码 | [429, 500, 502, 503, 504] |
| streaming | 布尔值 | 是否使用流式传输 | False |
| n | 整数 | 生成的回复数量 | 1 |

## 运行测试

1. 创建测试用的虚拟环境：

```bash
python -m venv venv_test
source venv_test/bin/activate  # 在Windows上使用: venv_test\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. 运行单元测试：

```bash
python -m unittest
```

3. 运行集成测试：

```bash
python test_mock_api_integration.py
```

## 测试健壮性

本项目的测试框架设计具有高度的健壮性，特别是在处理随机错误方面：

1. **随机错误处理**：集成测试使用模拟API服务器，该服务器会按照设定的概率随机生成各种错误（如429、500等），用于测试适配器的重试机制和错误处理能力。

2. **智能重试**：测试用例内置了多次尝试机制，当遇到随机错误时会自动重试，最多重试5次，并在重试之间添加适当的延迟。

3. **优雅跳过**：如果所有重试尝试都失败（这在高错误率设置下可能发生），测试会优雅地跳过而不是失败，同时记录详细的错误信息，确保CI/CD流程的稳定性。

4. **错误率配置**：通过调整`mock_api.py`中的`ERROR_PROBABILITY`字典或在启动模拟服务器时使用命令行参数，可以控制不同API端点的错误率，便于测试各种错误场景。

这种设计使得测试既能够验证适配器在错误情况下的行为，又不会因为随机错误而导致测试不稳定，特别适合在自动化测试环境中使用。

## 项目文件结构

- `company_llm.py`: 公司LLM API的适配器
- `company_azure_llm.py`: Azure OpenAI API的适配器
- `mock_api.py`: 模拟API服务器的实现
- `run_mock_api.py`: 启动模拟API服务器的脚本
- `test_mock_api_integration.py`: 使用模拟API服务器的集成测试
- `test_company_llm.py`: 公司LLM适配器的单元测试
- `test_company_azure_llm.py`: Azure适配器的单元测试

## 使用说明

1. **重试机制**: 适配器内置了指数退避重试机制，可以自动处理API的临时性错误（如速率限制、服务暂时不可用等）。重试策略包括：
   - 指数退避：每次重试的等待时间按指数增长
   - 可配置的抖动：随机化重试间隔，避免同时重试导致的"惊群效应"
   - 状态码筛选：只对特定的错误状态码（如429、500、502等）进行重试
   - 最大重试次数限制：防止无限重试消耗资源

2. **令牌刷新**: 如果启用了令牌刷新功能，适配器会在令牌过期或无效时自动刷新令牌。适配器会：
   - 在每次API调用前检查令牌状态
   - 根据设定的最小刷新间隔决定是否刷新令牌
   - 处理刷新失败的情况并提供明确的错误信息

3. **流式响应**: 通过设置`streaming=True`并使用`stream()`或`astream()`方法，可以获取流式响应，适用于实时显示LLM生成结果的场景。

4. **环境变量**: Azure适配器支持从环境变量读取配置：
   - `AZURE_OPENAI_API_KEY`: Azure OpenAI API密钥
   - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI API端点
   - `OPENAI_API_VERSION`: OpenAI API版本

5. **模拟服务器**: 使用模拟API服务器可以在不消耗真实API配额的情况下进行开发和测试。模拟服务器支持设置错误率，可用于测试适配器的重试机制。
   - 可通过命令行参数控制不同API的错误率
   - 支持模拟各种HTTP错误状态码（429、500、502、503、504）
   - 模拟令牌刷新流程
   - 提供健康检查端点便于监控 