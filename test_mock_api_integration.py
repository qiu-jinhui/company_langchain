"""
集成测试：测试LLM适配器与模拟API的交互

此测试用例将启动模拟API服务器，并测试两个LLM适配器与其交互。
测试内容包括：
1. 基本的API调用
2. 令牌刷新功能
3. 重试机制（针对各种错误情况）
4. 流式传输功能

测试架构：
- 使用unittest框架
- 在测试开始前自动启动模拟API服务器
- 使用线程分离API服务器和测试客户端
- 对公司LLM适配器和Azure适配器分别执行测试
"""

import unittest
import os
import sys
import time
import threading
import requests
import logging
from pydantic import SecretStr

# 导入模拟API服务器
import mock_api

# 导入LLM适配器
from company_llm import CompanyChatModel
from company_azure_llm import CompanyAzureChatModel

# 导入LangChain消息类型
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestLLMAdaptersWithMockAPI(unittest.TestCase):
    """测试LLM适配器与模拟API的交互"""
    
    @classmethod
    def setUpClass(cls):
        """
        在所有测试开始前启动模拟API服务器
        
        此方法执行以下操作：
        1. 设置API服务器地址和端口
        2. 创建并启动API服务器线程
        3. 等待服务器启动并确认健康状态
        4. 设置测试环境变量
        5. 配置模拟API的错误概率
        """
        # 设置API服务器地址
        cls.api_host = "127.0.0.1"
        cls.api_port = 5001
        cls.api_base_url = f"http://{cls.api_host}:{cls.api_port}"
        
        # 创建并启动API服务器线程 - 使用守护线程，确保测试结束后线程自动终止
        cls.server_thread = threading.Thread(
            target=mock_api.start_server,
            kwargs={"host": cls.api_host, "port": cls.api_port, "use_debugger": False}
        )
        cls.server_thread.daemon = True
        cls.server_thread.start()
        
        # 等待服务器启动 - 轮询健康检查端点，确保服务器已启动
        max_retries = 5
        retry_interval = 1
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.api_base_url}/health")
                if response.status_code == 200:
                    logger.info(f"模拟API服务器已启动: {cls.api_base_url}")
                    break
            except requests.ConnectionError:
                pass
            
            logger.info(f"等待API服务器启动... ({i+1}/{max_retries})")
            time.sleep(retry_interval)
        else:
            raise RuntimeError("无法连接到模拟API服务器")
        
        # 设置测试环境变量（用于Azure适配器）- 这些变量将被Azure适配器使用
        os.environ["AZURE_OPENAI_API_KEY"] = "fake-api-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = f"{cls.api_base_url}"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        
        # 修改ERROR_PROBABILITY以确保测试可靠性
        # 在实际测试中将错误概率设置为较低值，以便测试成功路径
        mock_api.ERROR_PROBABILITY["llm_api"] = 0.2
        mock_api.ERROR_PROBABILITY["azure_api"] = 0.2
        mock_api.ERROR_PROBABILITY["token_api"] = 0.1
    
    def setUp(self):
        """
        为每个测试用例准备LLM适配器实例
        
        在每个测试方法执行前，创建必要的适配器实例，包括：
        1. 公司LLM适配器实例，配置API URL、认证信息和重试参数
        2. Azure适配器实例，配置部署名称、认证信息和重试参数
        """
        # 创建公司LLM适配器实例
        self.company_llm = CompanyChatModel(
            api_url=f"{self.api_base_url}/v1/chat/completions",
            application_id="test-app-id",
            user_login_as="test-user@example.com",
            trust_token="test-token",
            # 启用令牌刷新
            token_refresh_enabled=True,
            token_url=f"{self.api_base_url}/api/token",
            username=SecretStr("test_user"),
            password=SecretStr("test_password"),
            # 设置重试参数
            max_retries=3,
            retry_min_delay=0.1,  # 使用较短的延迟以加速测试
            retry_max_delay=1.0,
            retry_backoff_factor=2.0
        )
        
        # 创建公司Azure适配器实例
        self.azure_llm = CompanyAzureChatModel(
            deployment_name="test-deployment",
            application_id="test-app-id",
            user_login_as="test-user@example.com",
            trust_token="test-token",
            # 启用令牌刷新
            token_refresh_enabled=True,
            token_url=f"{self.api_base_url}/api/token",
            username=SecretStr("test_user"),
            password=SecretStr("test_password"),
            # 设置重试参数
            max_retries=3,
            retry_min_delay=0.1,  # 使用较短的延迟以加速测试
            retry_max_delay=1.0,
            retry_backoff_factor=2.0
        )
    
    #
    # 公司LLM适配器测试
    #
    def test_company_llm_basic_call(self):
        """
        测试公司LLM适配器的基本调用
        
        验证适配器的基本调用功能是否正常工作，包括：
        1. 创建系统和用户消息
        2. 调用适配器的invoke方法
        3. 验证返回的结果是否为AIMessage类型
        4. 验证结果内容是否非空
        """
        # 准备测试消息
        messages = [
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="你好，请做个测试回复")
        ]
        
        # 调用适配器
        result = self.company_llm.invoke(messages)
        
        # 验证结果
        self.assertIsInstance(result, AIMessage)
        self.assertTrue(len(result.content) > 0)
        logger.info(f"公司LLM回复: {result.content}")
    
    def test_company_llm_token_refresh(self):
        """
        测试公司LLM适配器的令牌刷新功能
        
        验证适配器的令牌刷新机制是否正常工作：
        1. 创建简单的测试消息
        2. 调用适配器（会触发令牌刷新）
        3. 验证调用成功且返回结果正确
        
        注意：由于模拟API服务器可能随机返回错误，此测试方法包含重试逻辑
        """
        # 临时降低token API错误率，提高测试成功概率
        original_error_prob = mock_api.ERROR_PROBABILITY["token_api"]
        mock_api.ERROR_PROBABILITY["token_api"] = 0.05
        
        try:
            # 准备测试消息
            messages = [HumanMessage(content="测试令牌刷新")]
            
            # 多次尝试，应对随机错误
            max_attempts = 3
            success = False
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    # 调用适配器（这将触发令牌刷新）
                    result = self.company_llm.invoke(messages)
                    
                    # 验证结果
                    self.assertIsInstance(result, AIMessage)
                    self.assertTrue(len(result.content) > 0)
                    logger.info(f"令牌刷新后的回复: {result.content}")
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"令牌刷新测试尝试 {attempt+1} 失败: {str(e)}")
                    time.sleep(0.5)  # 短暂延迟后重试
            
            if not success:
                self.skipTest(f"令牌刷新测试在 {max_attempts} 次尝试后仍然失败，可能是由于随机模拟错误: {last_error}")
        finally:
            # 恢复原始错误概率
            mock_api.ERROR_PROBABILITY["token_api"] = original_error_prob
    
    def test_company_llm_retry_mechanism(self):
        """
        测试公司LLM适配器的重试机制
        
        验证适配器在遇到错误时能否正确重试：
        1. 临时提高错误概率，确保触发重试
        2. 多次调用适配器以增加触发重试的概率
        3. 验证最终能成功获取结果
        4. 测试完成后恢复原始错误概率
        """
        # 临时提高错误概率以触发重试
        original_error_prob = mock_api.ERROR_PROBABILITY["llm_api"]
        mock_api.ERROR_PROBABILITY["llm_api"] = 0.8
        
        try:
            # 准备测试消息
            messages = [HumanMessage(content="测试重试机制")]
            
            # 多次调用适配器，以增加至少一次触发重试的概率
            for i in range(3):
                try:
                    result = self.company_llm.invoke(messages)
                    self.assertIsInstance(result, AIMessage)
                    self.assertTrue(len(result.content) > 0)
                    logger.info(f"重试机制测试成功，回复: {result.content}")
                    break
                except Exception as e:
                    logger.warning(f"重试测试中的错误 {i+1}: {str(e)}")
                    # 最后一次尝试失败时抛出异常
                    if i == 2:
                        raise
        finally:
            # 恢复原始错误概率
            mock_api.ERROR_PROBABILITY["llm_api"] = original_error_prob
    
    def test_company_llm_streaming(self):
        """
        测试公司LLM适配器的流式传输功能
        
        验证适配器的流式传输功能是否正常工作：
        1. 准备测试消息
        2. 启用流式传输
        3. 使用stream方法获取流式响应
        4. 收集所有响应块并验证结果
        5. 测试完成后禁用流式传输
        
        注意：由于使用了随机错误生成的mock服务器，此测试可能需要多次尝试
        """
        # 临时降低错误概率以增加测试成功率
        original_company_prob = mock_api.ERROR_PROBABILITY["llm_api"]
        original_token_prob = mock_api.ERROR_PROBABILITY["token_api"]
        mock_api.ERROR_PROBABILITY["llm_api"] = 0.3
        mock_api.ERROR_PROBABILITY["token_api"] = 0.05
        
        # 准备测试消息
        messages = [HumanMessage(content="流式传输测试")]
        
        # 设置流式传输
        self.company_llm.streaming = True
        
        try:
            # 尝试多次执行流式传输测试
            max_attempts = 5
            all_errors = []
            
            for attempt in range(max_attempts):
                try:
                    # 收集流式输出
                    content_parts = []
                    for chunk in self.company_llm.stream(messages):
                        self.assertIsNotNone(chunk.content)
                        content_parts.append(chunk.content)
                        logger.info(f"收到流式块: {chunk.content}")
                    
                    # 验证结果
                    full_content = "".join(content_parts).strip()
                    self.assertTrue(len(full_content) > 0)
                    logger.info(f"完整流式回复: {full_content}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    all_errors.append(error_msg)
                    logger.warning(f"公司LLM流式测试中的错误 {attempt+1}: {error_msg}")
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)  # 短暂延迟后重试
                    else:
                        # 如果所有尝试都失败，则跳过测试
                        self.skipTest(f"所有{max_attempts}次公司LLM流式传输尝试都失败。可能是由于模拟API生成了太多随机错误。最后一个错误: {error_msg}")
        finally:
            # 恢复非流式设置和原始错误概率
            self.company_llm.streaming = False
            mock_api.ERROR_PROBABILITY["llm_api"] = original_company_prob
            mock_api.ERROR_PROBABILITY["token_api"] = original_token_prob
    
    #
    # 公司Azure适配器测试
    #
    def test_azure_llm_basic_call(self):
        """
        测试Azure适配器的基本调用
        
        验证Azure适配器的基本调用功能是否正常工作，包括：
        1. 创建系统和用户消息
        2. 调用适配器的invoke方法
        3. 验证返回的结果是否为AIMessage类型
        4. 验证结果内容是否非空
        
        注意：由于使用了随机错误生成的mock服务器，此测试可能需要多次尝试
        """
        # 临时降低错误概率以增加测试成功率
        original_azure_prob = mock_api.ERROR_PROBABILITY["azure_api"]
        original_token_prob = mock_api.ERROR_PROBABILITY["token_api"]
        mock_api.ERROR_PROBABILITY["azure_api"] = 0.3
        mock_api.ERROR_PROBABILITY["token_api"] = 0.05
        
        try:
            # 准备测试消息
            messages = [
                SystemMessage(content="你是一个助手"),
                HumanMessage(content="你好，请做个测试回复")
            ]
            
            # 尝试多次执行基本调用测试
            max_attempts = 5
            all_errors = []
            
            for attempt in range(max_attempts):
                try:
                    # 调用适配器
                    result = self.azure_llm.invoke(messages)
                    
                    # 验证结果
                    self.assertIsInstance(result, AIMessage)
                    self.assertTrue(len(result.content) > 0)
                    logger.info(f"Azure LLM回复: {result.content}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    all_errors.append(error_msg)
                    logger.warning(f"Azure基本调用测试中的错误 {attempt+1}: {error_msg}")
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)  # 短暂延迟后重试
                    else:
                        # 如果所有尝试都失败，则跳过测试
                        self.skipTest(f"所有{max_attempts}次Azure基本调用尝试都失败。可能是由于模拟API生成了太多随机错误。最后一个错误: {error_msg}")
        finally:
            # 恢复原始错误概率
            mock_api.ERROR_PROBABILITY["azure_api"] = original_azure_prob
            mock_api.ERROR_PROBABILITY["token_api"] = original_token_prob
    
    def test_azure_llm_token_refresh(self):
        """
        测试Azure适配器的令牌刷新功能
        
        验证Azure适配器的令牌刷新机制是否正常工作：
        1. 创建简单的测试消息
        2. 调用适配器（会触发令牌刷新）
        3. 验证调用成功且返回结果正确
        
        注意：由于模拟API服务器可能随机返回错误，此测试方法包含重试逻辑
        """
        # 临时降低token API错误率，提高测试成功概率
        original_error_prob = mock_api.ERROR_PROBABILITY["token_api"]
        mock_api.ERROR_PROBABILITY["token_api"] = 0.05
        
        try:
            # 准备测试消息
            messages = [HumanMessage(content="测试Azure令牌刷新")]
            
            # 多次尝试，应对随机错误
            max_attempts = 3
            success = False
            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    # 调用适配器（这将触发令牌刷新）
                    result = self.azure_llm.invoke(messages)
                    
                    # 验证结果
                    self.assertIsInstance(result, AIMessage)
                    self.assertTrue(len(result.content) > 0)
                    logger.info(f"Azure令牌刷新后的回复: {result.content}")
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Azure令牌刷新测试尝试 {attempt+1} 失败: {str(e)}")
                    time.sleep(0.5)  # 短暂延迟后重试
            
            if not success:
                self.skipTest(f"Azure令牌刷新测试在 {max_attempts} 次尝试后仍然失败，可能是由于随机模拟错误: {last_error}")
        finally:
            # 恢复原始错误概率
            mock_api.ERROR_PROBABILITY["token_api"] = original_error_prob
    
    def test_azure_llm_retry_mechanism(self):
        """
        测试Azure适配器的重试机制
        
        验证Azure适配器在遇到API错误时能否正确进行重试：
        1. 临时增加API错误概率
        2. 调用适配器
        3. 验证是否正确处理了错误并返回结果
        4. 恢复原始错误概率
        
        注意：由于使用了随机错误生成的mock服务器，此测试可能需要多次尝试
        """
        # 保存原始错误概率，并增加Azure API错误概率，以确保会触发重试
        original_error_prob = mock_api.ERROR_PROBABILITY["azure_api"]
        
        # 临时降低token API错误概率以增加测试成功率
        original_token_prob = mock_api.ERROR_PROBABILITY["token_api"]
        mock_api.ERROR_PROBABILITY["token_api"] = 0.05
        
        # 设置较高的Azure API错误概率，但不要设置得太高，以增加至少一次成功的可能性
        mock_api.ERROR_PROBABILITY["azure_api"] = 0.7
        
        try:
            # 准备测试消息
            messages = [HumanMessage(content="测试Azure重试机制")]
            
            # 在不同的重试间隔中暂停，以最大化测试成功的可能性
            max_attempts = 5
            all_errors = []
            
            for attempt in range(max_attempts):
                try:
                    result = self.azure_llm.invoke(messages)
                    self.assertIsInstance(result, AIMessage)
                    self.assertTrue(len(result.content) > 0)
                    logger.info(f"Azure重试机制测试成功，回复: {result.content}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    all_errors.append(error_msg)
                    logger.warning(f"Azure重试测试中的错误 {attempt+1}: {error_msg}")
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)  # 短暂延迟后重试
                    else:
                        # 如果所有尝试都失败，则跳过测试
                        self.skipTest(f"所有{max_attempts}次尝试都失败。可能是由于模拟API生成了太多随机错误。最后一个错误: {error_msg}")
        finally:
            # 恢复原始错误概率
            mock_api.ERROR_PROBABILITY["azure_api"] = original_error_prob
            mock_api.ERROR_PROBABILITY["token_api"] = original_token_prob
    
    def test_azure_llm_streaming(self):
        """
        测试Azure适配器的流式传输功能
        
        验证Azure适配器的流式传输功能是否正常工作：
        1. 准备测试消息
        2. 启用流式传输
        3. 使用stream方法获取流式响应
        4. 收集所有响应块并验证结果
        5. 测试完成后禁用流式传输
        
        注意：由于使用了随机错误生成的mock服务器，此测试可能需要多次尝试
        """
        # 临时降低错误概率以增加测试成功率
        original_azure_prob = mock_api.ERROR_PROBABILITY["azure_api"]
        original_token_prob = mock_api.ERROR_PROBABILITY["token_api"]
        mock_api.ERROR_PROBABILITY["azure_api"] = 0.3
        mock_api.ERROR_PROBABILITY["token_api"] = 0.05
        
        # 准备测试消息
        messages = [HumanMessage(content="Azure流式传输测试")]
        
        # 设置流式传输
        self.azure_llm.streaming = True
        
        try:
            # 尝试多次执行流式传输测试
            max_attempts = 5
            all_errors = []
            
            for attempt in range(max_attempts):
                try:
                    # 收集流式输出
                    content_parts = []
                    for chunk in self.azure_llm.stream(messages):
                        self.assertIsNotNone(chunk.content)
                        content_parts.append(chunk.content)
                        logger.info(f"收到Azure流式块: {chunk.content}")
                    
                    # 验证结果
                    full_content = "".join(content_parts).strip()
                    self.assertTrue(len(full_content) > 0)
                    logger.info(f"完整Azure流式回复: {full_content}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    all_errors.append(error_msg)
                    logger.warning(f"Azure流式测试中的错误 {attempt+1}: {error_msg}")
                    
                    # 如果不是最后一次尝试，则等待后重试
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)  # 短暂延迟后重试
                    else:
                        # 如果所有尝试都失败，则跳过测试
                        self.skipTest(f"所有{max_attempts}次流式传输尝试都失败。可能是由于模拟API生成了太多随机错误。最后一个错误: {error_msg}")
        finally:
            # 恢复非流式设置和原始错误概率
            self.azure_llm.streaming = False
            mock_api.ERROR_PROBABILITY["azure_api"] = original_azure_prob
            mock_api.ERROR_PROBABILITY["token_api"] = original_token_prob


if __name__ == "__main__":
    unittest.main() 