"""
测试公司嵌入模型适配器
"""

import unittest
from unittest.mock import patch, MagicMock, call
import json
import time
import requests
import asyncio

from company_embedding import CompanyEmbeddings
from pydantic import SecretStr


class TestCompanyEmbeddings(unittest.TestCase):
    """测试CompanyEmbeddings的功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.embeddings = CompanyEmbeddings(
            api_url="https://fake-api.example.com/v1/embeddings",
            application_id="test-app-id",
            trust_token="test-token",
        )
    
    def test_headers(self):
        """测试请求头构建功能"""
        headers = self.embeddings._headers
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["GAI-Platform-Application-ID"], "test-app-id")
        self.assertEqual(headers["X-E2E-Trust-Token"], "test-token")
    
    @patch("requests.post")
    def test_embed_documents(self, mock_post):
        """测试生成文档嵌入功能"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
        mock_post.return_value = mock_response
        
        # 测试文本
        texts = ["这是第一个测试文本", "这是第二个测试文本"]
        
        # 调用嵌入方法
        result = self.embeddings.embed_documents(texts)
        
        # 验证API调用
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://fake-api.example.com/v1/embeddings")
        self.assertEqual(kwargs["headers"], self.embeddings._headers)
        
        # 验证请求体
        request_body = kwargs["json"]
        self.assertEqual(request_body["model"], "text-embedding-ada-002")
        self.assertEqual(request_body["input"], texts)
        
        # 验证返回结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])
    
    @patch("requests.post")
    def test_embed_query(self, mock_post):
        """测试生成查询嵌入功能"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.7, 0.8, 0.9], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_post.return_value = mock_response
        
        # 测试查询文本
        query = "这是一个查询文本"
        
        # 调用嵌入查询方法
        result = self.embeddings.embed_query(query)
        
        # 验证API调用
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://fake-api.example.com/v1/embeddings")
        self.assertEqual(kwargs["headers"], self.embeddings._headers)
        
        # 验证请求体
        request_body = kwargs["json"]
        self.assertEqual(request_body["model"], "text-embedding-ada-002")
        self.assertEqual(request_body["input"], [query])  # 应该将单个查询转为列表
        
        # 验证返回结果
        self.assertEqual(result, [0.7, 0.8, 0.9])
    
    def test_calculate_retry_delay(self):
        """测试重试延迟计算功能"""
        # 设置固定的重试参数进行测试
        self.embeddings.retry_min_delay = 1.0
        self.embeddings.retry_max_delay = 60.0
        self.embeddings.retry_backoff_factor = 2.0
        self.embeddings.retry_jitter = False  # 关闭抖动以便于测试
        
        # 测试不同尝试次数下的延迟
        self.assertEqual(self.embeddings._calculate_retry_delay(1), 1.0)  # 第1次 -> 1.0
        self.assertEqual(self.embeddings._calculate_retry_delay(2), 2.0)  # 第2次 -> 1.0 * 2^1
        self.assertEqual(self.embeddings._calculate_retry_delay(3), 4.0)  # 第3次 -> 1.0 * 2^2
        self.assertEqual(self.embeddings._calculate_retry_delay(4), 8.0)  # 第4次 -> 1.0 * 2^3
        
        # 测试最大延迟限制
        self.embeddings.retry_max_delay = 5.0
        self.assertEqual(self.embeddings._calculate_retry_delay(4), 5.0)  # 应该被限制在最大值5.0
        
    def test_should_retry(self):
        """测试是否应该重试的判断功能"""
        # 默认情况下应该重试的状态码
        self.assertTrue(self.embeddings._should_retry(429))  # 速率限制
        self.assertTrue(self.embeddings._should_retry(500))  # 服务器错误
        self.assertTrue(self.embeddings._should_retry(502))  # 网关错误
        self.assertTrue(self.embeddings._should_retry(503))  # 服务不可用
        self.assertTrue(self.embeddings._should_retry(504))  # 网关超时
        
        # 不应该重试的状态码
        self.assertFalse(self.embeddings._should_retry(400))  # 客户端错误
        self.assertFalse(self.embeddings._should_retry(401))  # 未授权
        self.assertFalse(self.embeddings._should_retry(404))  # 未找到
        
        # 自定义重试状态码
        self.embeddings.retry_on_status_codes = [429, 408]  # 只在速率限制和请求超时时重试
        self.assertTrue(self.embeddings._should_retry(429))
        self.assertTrue(self.embeddings._should_retry(408))
        self.assertFalse(self.embeddings._should_retry(500))  # 现在不应该重试500
        
    @patch("time.sleep")  # 模拟sleep以加速测试
    @patch("requests.post")
    def test_retry_success(self, mock_post, mock_sleep):
        """测试重试成功的情况"""
        # 设置模型的重试参数
        self.embeddings.max_retries = 3
        self.embeddings.retry_min_delay = 1.0
        self.embeddings.retry_backoff_factor = 2.0
        self.embeddings.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟前两次请求失败（服务器错误），第三次成功
        error_response = MagicMock()
        error_response.status_code = 500
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        # 设置mock_post连续返回不同的响应
        mock_post.side_effect = [error_response, error_response, success_response]
        
        # 测试文本
        text = "测试重试"
        
        # 调用嵌入方法
        result = self.embeddings.embed_query(text)
        
        # 验证API被调用了3次（2次失败 + 1次成功）
        self.assertEqual(mock_post.call_count, 3)
        
        # 验证sleep被调用了2次（每次失败后）
        self.assertEqual(mock_sleep.call_count, 2)
        # 验证sleep的参数（第一次1.0秒，第二次2.0秒）
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
        
        # 验证最终结果是成功的
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
    @patch("time.sleep")  # 模拟sleep以加速测试
    @patch("requests.post")
    def test_retry_max_attempts(self, mock_post, mock_sleep):
        """测试达到最大重试次数的情况"""
        # 设置模型的重试参数
        self.embeddings.max_retries = 2  # 最多尝试3次（初始尝试 + 2次重试）
        self.embeddings.retry_min_delay = 1.0
        self.embeddings.retry_backoff_factor = 2.0
        self.embeddings.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟所有请求都失败（服务器错误）
        error_response = MagicMock()
        error_response.status_code = 500
        # 防止response.raise_for_status()引发异常
        error_response.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError("模拟HTTP错误"))
        mock_post.return_value = error_response
        
        # 测试文本
        text = "测试重试失败"
        
        # 调用嵌入方法应该抛出异常
        with self.assertRaises(ValueError) as context:
            self.embeddings.embed_query(text)
            
        # 验证API被调用了3次（初始 + 2次重试）
        self.assertEqual(mock_post.call_count, 3)
        
        # 验证sleep被调用了2次
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch("time.sleep")
    @patch("requests.post")
    def test_retry_connection_error(self, mock_post, mock_sleep):
        """测试连接错误的重试情况"""
        # 设置模型的重试参数
        self.embeddings.max_retries = 3
        self.embeddings.retry_min_delay = 1.0
        self.embeddings.retry_backoff_factor = 2.0
        self.embeddings.retry_jitter = False  # 关闭抖动以便于测试
        
        # 模拟连接错误，然后成功
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("连接错误1"),
            requests.exceptions.ConnectionError("连接错误2"),
            MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "data": [
                        {"embedding": [0.1, 0.2, 0.3], "index": 0}
                    ],
                    "model": "text-embedding-ada-002",
                    "usage": {
                        "prompt_tokens": 5,
                        "total_tokens": 5
                    }
                })
            )
        ]
        
        # 测试文本
        text = "测试连接错误重试"
        
        # 调用嵌入方法
        result = self.embeddings.embed_query(text)
        
        # 验证最终结果是成功的
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # 验证mock_post被调用了3次
        self.assertEqual(mock_post.call_count, 3)
        
        # 验证sleep被调用了2次
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])
        
    @patch("requests.post")
    def test_token_refresh(self, mock_post):
        """测试令牌刷新功能"""
        # 创建一个启用令牌刷新的模型
        embeddings = CompanyEmbeddings(
            api_url="https://fake-api.example.com/v1/embeddings",
            application_id="test-app-id",
            trust_token="initial-token",
            token_refresh_enabled=True,
            token_url="https://fake-api.example.com/api/token",
            username=SecretStr("test-user"),
            password=SecretStr("test-password"),
        )
        
        # 模拟令牌刷新响应
        token_response = MagicMock()
        token_response.status_code = 200
        token_response.json.return_value = {"token": "new-token"}
        
        # 模拟嵌入API响应
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        # 设置模拟响应序列
        mock_post.side_effect = [token_response, embed_response]
        
        # 调用嵌入方法
        result = embeddings.embed_query("测试令牌刷新")
        
        # 验证令牌刷新和嵌入API都被调用
        self.assertEqual(mock_post.call_count, 2)
        
        # 验证令牌已更新
        self.assertEqual(embeddings.trust_token.get_secret_value(), "new-token")
        
        # 验证结果正确
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # 验证第一次调用是令牌刷新
        token_call = mock_post.call_args_list[0]
        self.assertEqual(token_call[0][0], "https://fake-api.example.com/api/token")
        
        # 验证第二次调用是嵌入API
        embed_call = mock_post.call_args_list[1]
        self.assertEqual(embed_call[0][0], "https://fake-api.example.com/v1/embeddings")
        self.assertEqual(embed_call[1]["headers"]["X-E2E-Trust-Token"], "new-token")


class TestAsyncCompanyEmbeddings(unittest.IsolatedAsyncioTestCase):
    """测试CompanyEmbeddings的异步功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.embeddings = CompanyEmbeddings(
            api_url="https://fake-api.example.com/v1/embeddings",
            application_id="test-app-id",
            trust_token="test-token",
        )
    
    @patch("aiohttp.ClientSession.post")
    async def test_aembed_documents(self, mock_post):
        """测试异步生成文档嵌入功能"""
        # 模拟异步响应
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json = MagicMock(
            return_value=asyncio.Future()
        )
        mock_response.json.return_value.set_result({
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        })
        mock_post.return_value = mock_response
        
        # 测试文本
        texts = ["这是第一个测试文本", "这是第二个测试文本"]
        
        # 调用异步嵌入方法
        result = await self.embeddings.aembed_documents(texts)
        
        # 验证API调用
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["url"], "https://fake-api.example.com/v1/embeddings")
        self.assertEqual(kwargs["headers"], self.embeddings._headers)
        
        # 验证请求体
        request_body = kwargs["json"]
        self.assertEqual(request_body["model"], "text-embedding-ada-002")
        self.assertEqual(request_body["input"], texts)
        
        # 验证返回结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])
    
    @patch("aiohttp.ClientSession.post")
    async def test_aembed_query(self, mock_post):
        """测试异步生成查询嵌入功能"""
        # 模拟异步响应
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.json = MagicMock(
            return_value=asyncio.Future()
        )
        mock_response.json.return_value.set_result({
            "data": [
                {"embedding": [0.7, 0.8, 0.9], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        })
        mock_post.return_value = mock_response
        
        # 测试查询文本
        query = "这是一个查询文本"
        
        # 调用异步嵌入查询方法
        result = await self.embeddings.aembed_query(query)
        
        # 验证API调用
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["url"], "https://fake-api.example.com/v1/embeddings")
        self.assertEqual(kwargs["headers"], self.embeddings._headers)
        
        # 验证请求体
        request_body = kwargs["json"]
        self.assertEqual(request_body["model"], "text-embedding-ada-002")
        self.assertEqual(request_body["input"], [query])  # 应该将单个查询转为列表
        
        # 验证返回结果
        self.assertEqual(result, [0.7, 0.8, 0.9])


if __name__ == "__main__":
    unittest.main() 