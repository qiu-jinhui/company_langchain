"""
简单示例：如何使用公司LLM适配器
"""

from company_llm import CompanyChatModel
from langchain_core.messages import HumanMessage

def main():
    """
    简单示例
    
    请替换以下参数为您的实际API信息
    """
    
    # 创建模型实例
    llm = CompanyChatModel(
        api_url="https://your-company-llm-api.example.com/v1/chat/completions",
        application_id="your-application-id",
        user_login_as="your-user-login",
        trust_token="your-trust-token",
        temperature=0.5,
    )
    
    # 创建用户消息
    message = HumanMessage(content="请用简单的语言解释什么是API？")
    
    # 打印配置信息
    print("模型配置:")
    print(f"  API URL: {llm.api_url}")
    print(f"  模型: {llm.model}")
    print(f"  温度: {llm.temperature}")
    print()
    
    # 调用LLM (注意：这将实际调用API，请确保您的API凭据正确)
    print("发送请求...")
    print(f"消息: \"{message.content}\"")
    print()
    
    try:
        # 这里实际调用API
        # response = llm.invoke([message])
        # print("回复:")
        # print(response.content)
        
        # 由于这是示例，我们不实际调用API，而是打印请求将如何构建
        print("请求示例 (实际未发送):")
        print(f"POST {llm.api_url}")
        print("请求头:")
        for key, value in llm._headers.items():
            # 敏感信息用星号替代
            if key in ["X-E2E-Trust-Token"]:
                value = "*" * 8
            print(f"  {key}: {value}")
        
        # 获取OpenAI格式消息
        openai_messages = llm._convert_messages_to_openai_format([message])
        print("\n请求体:")
        print("  {")
        print(f"    \"model\": \"{llm.model}\",")
        print(f"    \"messages\": {openai_messages},")
        print(f"    \"temperature\": {llm.temperature},")
        print("    ...")
        print("  }")
        
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main() 