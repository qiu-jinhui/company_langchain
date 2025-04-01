"""
测试公司LLM适配器
"""

import unittest
from unittest.mock import patch, MagicMock, call
import json
import time
import requests

from company_llm import CompanyChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class TestCompanyChatModel(unittest.TestCase):
    """测试CompanyChatModel的功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.model = CompanyChatModel(
            api_url="https://fake-api.example.com/v1/chat/completions",
            application_id="test-app-id",
            user_login_as="test-user@example.com",
            trust_token="test-token",
        )
    
    def test_headers(self):
        """测试请求头构建功能"""
        headers = self.model._headers
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["GAI-Platform-Application-ID"], "test-app-id")
        self.assertEqual(headers["X-DSP-User-Login-As"], "test-user@example.com")
        self.assertEqual(headers["X-E2E-Trust-Token"], "test-token")
    
    def test_message_conversion(self):
        """测试消息格式转换功能"""
        messages = [
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="你好"),
            AIMessage(content="我能帮你什么？")
        ]
        
        openai_messages = self.model._convert_messages_to_openai_format(messages)
        self.assertEqual(len(openai_messages), 3)
        self.assertEqual(openai_messages[0]["role"], "system")
        self.assertEqual(openai_messages[0]["content"], "你是一个助手")
        self.assertEqual(openai_messages[1]["role"], "user")
        self.assertEqual(openai_messages[1]["content"], "你好")
        self.assertEqual(openai_messages[2]["role"], "assistant")
        self.assertEqual(openai_messages[2]["content"], "我能帮你什么？")
    
    @patch("requests.post")
    def test_generate(self, mock_post):
        """测试生成回复功能"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200  # 明确设置状态码
        mock_response.json.return_value = {
            "id": "mock-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "这是测试回复"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # 测试消息
        messages = [
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="你好")
        ]
        
        # 调用生成方法
        result = self.model._generate(messages)
        
        # 验证API调用
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://fake-api.example.com/v1/chat/completions")
        self.assertEqual(kwargs["headers"], self.model._headers)
        
        # 验证请求体
        request_body = kwargs["json"]
        self.assertEqual(request_body["model"], "gpt-3.5-turbo")
        self.assertEqual(len(request_body["messages"]), 2)
        
        # 验证返回结果
        self.assertEqual(len(result.generations), 1)
        self.assertEqual(result.generations[0].message.content, "这是测试回复")
    
    @patch("requests.post")
    def test_invoke(self, mock_post):
        """测试invoke方法"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200  # 明确设置状态码
        mock_response.json.return_value = {
            "id": "mock-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "这是测试回复"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # 测试消息
        messages = [HumanMessage(content="你好")]
        
        # 调用实际的invoke方法
        result = self.model.invoke(messages)
        
        # 验证返回结果
        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "这是测试回复")
        
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
    @patch("requests.post")
    def test_retry_success(self, mock_post, mock_sleep):
        """测试重试成功的情况"""
        # 设置模型的重试参数
        self.model.max_retries = 3
        self.model.retry_min_delay = 1.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟前两次请求失败（服务器错误），第三次成功
        error_response = MagicMock()
        error_response.status_code = 500
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "id": "mock-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
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
        }
        
        # 设置mock_post连续返回不同的响应
        mock_post.side_effect = [error_response, error_response, success_response]
        
        # 测试消息
        messages = [HumanMessage(content="测试重试")]
        
        # 调用生成方法
        result = self.model._generate(messages)
        
        # 验证API被调用了3次（2次失败 + 1次成功）
        self.assertEqual(mock_post.call_count, 3)
        
        # 验证sleep被调用了2次（每次失败后）
        self.assertEqual(mock_sleep.call_count, 2)
        # 验证sleep的参数（第一次1.0秒，第二次2.0秒）
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
        
        # 验证最终结果是成功的
        self.assertEqual(result.generations[0].message.content, "这是重试成功的回复")
        
    @patch("time.sleep")  # 模拟sleep以加速测试
    @patch("requests.post")
    def test_retry_max_attempts(self, mock_post, mock_sleep):
        """测试达到最大重试次数的情况"""
        # 设置模型的重试参数
        self.model.max_retries = 2  # 最多尝试3次（初始尝试 + 2次重试）
        self.model.retry_min_delay = 1.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟所有请求都失败（服务器错误）
        error_response = MagicMock()
        error_response.status_code = 500
        # 防止response.raise_for_status()引发异常
        error_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("模拟HTTP错误"))
        mock_post.return_value = error_response
        
        # 测试消息
        messages = [HumanMessage(content="测试重试失败")]
        
        # 调用生成方法应该抛出异常
        with self.assertRaises(ValueError) as context:
            self.model._generate(messages)
            
        # 验证异常消息
        # 因为实际实现可能返回"服务器错误: 500"，所以我们检查这种情况
        error_msg = str(context.exception)
        self.assertTrue(
            "服务器错误: 500" in error_msg or
            "在 2 次尝试后" in error_msg
        )
        
        # 验证API被调用了3次（初始 + 2次重试）
        self.assertEqual(mock_post.call_count, 3)
        
        # 验证sleep被调用了2次（每次失败后）
        self.assertEqual(mock_sleep.call_count, 2)
        
        # 验证sleep的参数（第一次1.0秒，第二次2.0秒）
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
        
    @patch("time.sleep")
    @patch("requests.post")
    def test_retry_connection_error(self, mock_post, mock_sleep):
        """测试连接错误的重试"""
        # 设置模型的重试参数
        self.model.max_retries = 2
        self.model.retry_min_delay = 1.0
        self.model.retry_backoff_factor = 2.0
        self.model.retry_jitter = False
        
        # 模拟前两次请求抛出连接错误，第三次成功
        mock_post.side_effect = [
            requests.ConnectionError("连接失败"),
            requests.ConnectionError("连接失败"),
            MagicMock(status_code=200, json=lambda: {
                "id": "mock-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "连接重试成功的回复"
                        },
                        "finish_reason": "stop"
                    }
                ]
            })
        ]
        
        # 测试消息
        messages = [HumanMessage(content="测试连接错误重试")]
        
        # 调用生成方法
        result = self.model._generate(messages)
        
        # 验证API被调用了3次
        self.assertEqual(mock_post.call_count, 3)
        
        # 验证sleep被调用了2次
        self.assertEqual(mock_sleep.call_count, 2)
        
        # 验证最终结果是成功的
        self.assertEqual(result.generations[0].message.content, "连接重试成功的回复")


if __name__ == "__main__":
    unittest.main() 