"""
测试公司Azure OpenAI适配器
"""

import unittest
from unittest.mock import patch, MagicMock, call
import json
import time
import os

from company_azure_llm import CompanyAzureChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class TestCompanyAzureChatModel(unittest.TestCase):
    """测试CompanyAzureChatModel的功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 设置测试环境变量
        os.environ["AZURE_OPENAI_API_KEY"] = "fake-api-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fake-azure.openai.azure.com"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        
        self.model = CompanyAzureChatModel(
            deployment_name="test-deployment",
            application_id="test-app-id",
            user_login_as="test-user@example.com",
            trust_token="test-token",
        )
    
    def test_calculate_retry_delay(self):
        """测试重试延迟计算功能"""
        # 设置固定的重试参数进行测试
        self.model.retry_min_delay = 1.0
        self.model.retry_max_delay = 60.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False  # 关闭抖动以便于测试
        
        # 测试不同尝试次数下的延迟
        self.assertEqual(self.model._calculate_retry_delay(1), 1.0)  # 第1次 -> 1.0
        self.assertEqual(self.model._calculate_retry_delay(2), 2.0)  # 第2次 -> 1.0 * 2^1
        self.assertEqual(self.model._calculate_retry_delay(3), 4.0)  # 第3次 -> 1.0 * 2^2
        self.assertEqual(self.model._calculate_retry_delay(4), 8.0)  # 第4次 -> 1.0 * 2^3
        
        # 测试最大延迟限制
        self.model.retry_max_delay = 5.0
        self.assertEqual(self.model._calculate_retry_delay(4), 5.0)  # 应该被限制在最大值5.0
        
    def test_should_retry(self):
        """测试是否应该重试的判断功能"""
        # 默认情况下应该重试的状态码
        self.assertTrue(self.model._should_retry(429))  # 速率限制
        self.assertTrue(self.model._should_retry(500))  # 服务器错误
        self.assertTrue(self.model._should_retry(502))  # 网关错误
        self.assertTrue(self.model._should_retry(503))  # 服务不可用
        self.assertTrue(self.model._should_retry(504))  # 网关超时
        
        # 不应该重试的状态码
        self.assertFalse(self.model._should_retry(400))  # 客户端错误
        self.assertFalse(self.model._should_retry(401))  # 未授权
        self.assertFalse(self.model._should_retry(404))  # 未找到
        
        # 自定义重试状态码
        self.model.retry_on_status_codes = [429, 408]  # 只在速率限制和请求超时时重试
        self.assertTrue(self.model._should_retry(429))
        self.assertTrue(self.model._should_retry(408))
        self.assertFalse(self.model._should_retry(500))  # 现在不应该重试500
    
    @patch("time.sleep")  # 模拟sleep以加速测试
    @patch("company_azure_llm.CompanyAzureChatModel._get_client")
    def test_retry_success(self, mock_get_client, mock_sleep):
        """测试重试成功的情况"""
        # 设置模型的重试参数
        self.model.max_retries = 3
        self.model.retry_min_delay = 1.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟OpenAI客户端和响应
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # 模拟第一次和第二次调用失败，第三次成功
        error_msg = 'Error communicating with OpenAI: status_code=500'
        mock_client.create.side_effect = [
            Exception(error_msg),
            Exception(error_msg),
            MagicMock(model_dump=lambda: {
                "id": "mock-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "这是重试成功的回复"
                        },
                        "finish_reason": "stop"
                    }
                ]
            })
        ]
        
        # 测试消息
        messages = [HumanMessage(content="测试重试")]
        
        # 调用生成方法
        result = self.model._generate(messages)
        
        # 验证客户端被调用了3次
        self.assertEqual(mock_client.create.call_count, 3)
        
        # 验证sleep被调用了2次（每次失败后）
        self.assertEqual(mock_sleep.call_count, 2)
        # 验证sleep的参数（第一次1.0秒，第二次2.0秒）
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
        
        # 验证最终结果是成功的
        self.assertEqual(result.generations[0].message.content, "这是重试成功的回复")
        
    @patch("time.sleep")  # 模拟sleep以加速测试
    @patch("company_azure_llm.CompanyAzureChatModel._get_client")
    def test_retry_max_attempts(self, mock_get_client, mock_sleep):
        """测试达到最大重试次数的情况"""
        # 设置模型的重试参数
        self.model.max_retries = 2  # 最多尝试3次（初始尝试 + 2次重试）
        self.model.retry_min_delay = 1.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟OpenAI客户端
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # 模拟所有请求都失败
        error_msg = 'Error communicating with OpenAI: status_code=500'
        mock_client.create.side_effect = Exception(error_msg)
        
        # 测试消息
        messages = [HumanMessage(content="测试重试失败")]
        
        # 调用生成方法应该抛出异常
        with self.assertRaises(ValueError) as context:
            self.model._generate(messages)
            
        # 验证异常消息
        error_text = str(context.exception)
        self.assertTrue(
            "Azure OpenAI服务器错误" in error_text or
            "在 2 次尝试后" in error_text
        )
        
        # 验证客户端被调用了3次（初始 + 2次重试）
        self.assertEqual(mock_client.create.call_count, 3)
        
        # 验证sleep被调用了2次（每次失败后）
        self.assertEqual(mock_sleep.call_count, 2)
        
        # 验证sleep的参数（第一次1.0秒，第二次2.0秒）
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
    
    @patch("time.sleep")
    @patch("company_azure_llm.CompanyAzureChatModel._get_client")
    def test_retry_429_error(self, mock_get_client, mock_sleep):
        """测试速率限制错误的重试"""
        # 设置模型的重试参数
        self.model.max_retries = 2
        self.model.retry_min_delay = 1.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False
        
        # 模拟OpenAI客户端
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # 模拟前两次请求返回速率限制错误，第三次成功
        error_msg_429 = 'Error communicating with OpenAI: status_code=429'
        mock_client.create.side_effect = [
            Exception(error_msg_429),
            Exception(error_msg_429),
            MagicMock(model_dump=lambda: {
                "id": "mock-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-35-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "速率限制重试成功的回复"
                        },
                        "finish_reason": "stop"
                    }
                ]
            })
        ]
        
        # 测试消息
        messages = [HumanMessage(content="测试速率限制重试")]
        
        # 调用生成方法
        result = self.model._generate(messages)
        
        # 验证客户端被调用了3次
        self.assertEqual(mock_client.create.call_count, 3)
        
        # 验证sleep被调用了2次
        self.assertEqual(mock_sleep.call_count, 2)
        
        # 验证最终结果是成功的
        self.assertEqual(result.generations[0].message.content, "速率限制重试成功的回复")


if __name__ == "__main__":
    unittest.main() 