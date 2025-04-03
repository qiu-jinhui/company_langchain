"""
公司嵌入模型适配器：用于连接公司自有嵌入模型API与LangChain的适配器
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast
import logging
import json
import time
import requests
import aiohttp
from aiohttp import ClientTimeout
import asyncio
import random

from langchain_core.embeddings import Embeddings
from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
from pydantic import Field, SecretStr, model_validator, BaseModel

logger = logging.getLogger(__name__)

class CompanyEmbeddings(Embeddings, BaseModel):
    """
    公司自有嵌入模型API的适配器，与OpenAI格式兼容
    """
    
    # API配置
    api_url: str = Field(..., description="公司嵌入模型API的URL")
    application_id: str = Field(..., description="GAI-Platform-Application-ID")
    trust_token: SecretStr = Field(..., description="X-E2E-Trust-Token")
    
    # 令牌刷新配置
    token_url: Optional[str] = Field(None, description="令牌刷新API的URL")
    username: Optional[SecretStr] = Field(None, description="用于令牌刷新的用户名")
    password: Optional[SecretStr] = Field(None, description="用于令牌刷新的密码")
    token_refresh_enabled: bool = Field(False, description="是否启用令牌自动刷新")
    token_refresh_interval: int = Field(0, description="令牌最小刷新间隔（秒），0表示每次都刷新")
    
    # 模型配置（类似OpenAI）
    model: str = "text-embedding-ada-002"
    timeout: Optional[float] = None
    
    # 重试配置
    max_retries: int = 6  # 最大重试次数（包括初始请求）
    retry_min_delay: float = Field(1.0, description="重试的最小延迟时间（秒）")
    retry_max_delay: float = Field(60.0, description="重试的最大延迟时间（秒）")
    retry_backoff_factor: float = Field(2.0, description="重试延迟的退避系数")
    retry_jitter: bool = Field(True, description="是否在重试延迟时间上添加随机抖动")
    retry_on_status_codes: List[int] = Field([429, 500, 502, 503, 504], description="需要重试的HTTP状态码")
    
    # 内部状态
    _last_token_refresh: float = 0.0  # 上次刷新令牌的时间戳
    
    class Config:
        arbitrary_types_allowed = True
    
    @model_validator(mode='after')
    def validate_params(self) -> 'CompanyEmbeddings':
        """验证参数合法性"""
        # 验证令牌刷新配置
        if self.token_refresh_enabled:
            if not self.token_url:
                raise ValueError("启用令牌刷新时token_url是必需的")
            if not self.username:
                raise ValueError("启用令牌刷新时username是必需的")
            if not self.password:
                raise ValueError("启用令牌刷新时password是必需的")
                
        # 验证重试配置
        if self.retry_min_delay < 0:
            raise ValueError("retry_min_delay必须大于或等于0")
        if self.retry_max_delay < self.retry_min_delay:
            raise ValueError("retry_max_delay必须大于或等于retry_min_delay")
        if self.retry_backoff_factor < 1:
            raise ValueError("retry_backoff_factor必须大于或等于1")
        if self.max_retries < 0:
            raise ValueError("max_retries必须大于或等于0")
                
        return self
    
    @property
    def _headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Content-Type": "application/json",
            "GAI-Platform-Application-ID": self.application_id,
            "X-E2E-Trust-Token": self.trust_token.get_secret_value(),
        }
    
    def _refresh_token(self) -> None:
        """刷新令牌（同步方法）"""
        # 检查是否需要刷新令牌
        if not self.token_refresh_enabled:
            return
            
        # 检查令牌刷新间隔
        current_time = time.time()
        if current_time - self._last_token_refresh < self.token_refresh_interval:
            logger.debug("令牌刷新间隔未到，跳过刷新")
            return
            
        # 构建请求头和请求体
        headers = {"Content-Type": "application/json"}
        payload = {
            "input_token_state": {
                "token_type": "CREDENTIAL",
                "username": self.username.get_secret_value(),
                "password": self.password.get_secret_value()
            },
            "output_token_state": {
                "token_type": "JWT"
            }
        }
        
        try:
            # 发送令牌刷新请求
            response = requests.post(
                self.token_url,
                headers=headers,
                json=payload,
                timeout=self.timeout or 30
            )
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"令牌刷新失败: 状态码 {response.status_code}")
                raise ValueError(f"令牌刷新失败: 状态码 {response.status_code}")
                
            # 解析响应
            token_data = response.json()
            if not token_data.get("token"):
                logger.error("令牌刷新失败: 响应中没有token字段")
                raise ValueError("令牌刷新失败: 响应中没有token字段")
                
            # 更新令牌
            new_token = token_data.get("token")
            self.trust_token = SecretStr(new_token)
            self._last_token_refresh = current_time
            
            logger.debug("令牌刷新成功")
            
        except requests.RequestException as e:
            logger.error(f"令牌刷新请求失败: {str(e)}")
            raise ValueError(f"令牌刷新请求失败: {str(e)}")
        except json.JSONDecodeError:
            logger.error("令牌刷新响应解析失败: 无效的JSON")
            raise ValueError("令牌刷新响应解析失败: 无效的JSON")
    
    async def _arefresh_token(self) -> None:
        """刷新令牌（异步方法）"""
        # 检查是否需要刷新令牌
        if not self.token_refresh_enabled:
            return
            
        # 检查令牌刷新间隔
        current_time = time.time()
        if current_time - self._last_token_refresh < self.token_refresh_interval:
            logger.debug("令牌刷新间隔未到，跳过刷新")
            return
            
        # 构建请求头和请求体
        headers = {"Content-Type": "application/json"}
        payload = {
            "input_token_state": {
                "token_type": "CREDENTIAL",
                "username": self.username.get_secret_value(),
                "password": self.password.get_secret_value()
            },
            "output_token_state": {
                "token_type": "JWT"
            }
        }
        
        # 设置超时
        timeout = ClientTimeout(total=self.timeout or 30)
        
        try:
            # 发送异步令牌刷新请求
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.token_url,
                    headers=headers,
                    json=payload
                ) as response:
                    # 检查响应状态
                    if response.status != 200:
                        logger.error(f"令牌刷新失败: 状态码 {response.status}")
                        raise ValueError(f"令牌刷新失败: 状态码 {response.status}")
                    
                    # 解析响应
                    token_data = await response.json()
                    if not token_data.get("token"):
                        logger.error("令牌刷新失败: 响应中没有token字段")
                        raise ValueError("令牌刷新失败: 响应中没有token字段")
                    
                    # 更新令牌
                    new_token = token_data.get("token")
                    self.trust_token = SecretStr(new_token)
                    self._last_token_refresh = current_time
                    
                    logger.debug("令牌刷新成功")
                    
        except aiohttp.ClientError as e:
            logger.error(f"异步令牌刷新请求失败: {str(e)}")
            raise ValueError(f"异步令牌刷新请求失败: {str(e)}")
        except json.JSONDecodeError:
            logger.error("异步令牌刷新响应解析失败: 无效的JSON")
            raise ValueError("异步令牌刷新响应解析失败: 无效的JSON")
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        计算重试延迟时间
        
        根据重试次数计算指数退避延迟时间。公式为：
        delay = min(max_delay, min_delay * (backoff_factor ^ (attempt - 1)))
        
        如果启用了抖动，会添加±15%的随机变化，以避免多个客户端同时重试导致的"惊群效应"。
        
        参数:
            attempt: 当前尝试次数（从1开始）
            
        返回:
            float: 计算出的延迟时间（秒）
        """
        # 计算基础延迟时间（指数退避）
        delay = min(
            self.retry_max_delay,
            self.retry_min_delay * (self.retry_backoff_factor ** (attempt - 1))
        )
        
        # 如果启用抖动，添加±15%的随机变化
        if self.retry_jitter:
            jitter = random.uniform(0.85, 1.15)
            delay *= jitter
            
        return delay
    
    def _should_retry(self, status_code: int) -> bool:
        """
        判断是否应该重试请求
        
        根据HTTP状态码判断是否应该重试请求。默认情况下，以下状态码会触发重试：
        - 429: 请求过多（速率限制）
        - 500: 服务器内部错误
        - 502: 错误网关
        - 503: 服务不可用
        - 504: 网关超时
        
        参数:
            status_code: HTTP状态码
            
        返回:
            bool: 如果应该重试则返回True，否则返回False
        """
        return status_code in self.retry_on_status_codes
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成多个文本的嵌入向量
        
        参数:
            texts: 要生成嵌入的文本列表
            
        返回:
            List[List[float]]: 嵌入向量列表，每个文本对应一个嵌入向量
        """
        # 先刷新令牌
        if self.token_refresh_enabled:
            self._refresh_token()
            
        # 构建请求体（保持与OpenAI兼容）
        request_body = {
            "model": self.model,
            "input": texts
        }
        
        # 实现重试逻辑
        attempt = 0  # 当前尝试次数
        last_error = None  # 记录最后一个错误
        
        # 重试循环，最多尝试max_retries次（包括初始请求）
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # 发送API请求
                response = requests.post(
                    self.api_url,
                    headers=self._headers,
                    json=request_body,
                    timeout=self.timeout or 60
                )
                
                if response.status_code == 200:
                    # 请求成功，解析响应并返回结果
                    result = response.json()
                    # 提取嵌入向量（保持与OpenAI兼容的响应格式）
                    embeddings = [item["embedding"] for item in result["data"]]
                    return embeddings
                elif self._should_retry(response.status_code):
                    # 需要重试的状态码，且尚未达到最大重试次数
                    if attempt <= self.max_retries:
                        # 计算重试延迟时间
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(
                            f"API请求返回状态码 {response.status_code}，将在 {delay:.2f} 秒后进行第 {attempt} 次重试..."
                        )
                        # 等待一段时间后重试
                        time.sleep(delay)
                        continue
                
                # 处理其他错误状态码（不重试的情况）
                if response.status_code == 401:
                    raise ValueError("认证失败，请检查您的认证凭据")
                elif response.status_code == 403:
                    raise ValueError("无权限访问此资源，请检查您的权限设置")
                elif response.status_code == 404:
                    raise ValueError(f"API端点不存在: {self.api_url}")
                elif response.status_code >= 500:
                    raise ValueError(f"服务器错误: {response.status_code}")
                
                # 其他未处理的状态码
                response.raise_for_status()
                
            except (requests.ConnectionError, requests.Timeout) as e:
                # 连接错误或超时，可以重试
                last_error = e
                if attempt <= self.max_retries:
                    # 计算重试延迟时间
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"连接错误或超时，将在 {delay:.2f} 秒后进行第 {attempt} 次重试: {str(e)}"
                    )
                    # 等待一段时间后重试
                    time.sleep(delay)
                    continue
                # 达到最大重试次数，抛出最后一个错误
                if isinstance(e, requests.ConnectionError):
                    logger.error(f"连接到API服务器失败: {self.api_url}")
                    raise ValueError(f"连接到API服务器失败: {self.api_url}")
                elif isinstance(e, requests.Timeout):
                    logger.error(f"API请求超时: {self.timeout}秒")
                    raise ValueError(f"API请求超时: {self.timeout}秒")
            except requests.RequestException as e:
                # 其他请求错误，可以重试
                last_error = e
                if attempt <= self.max_retries:
                    # 计算重试延迟时间
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"请求错误，将在 {delay:.2f} 秒后进行第 {attempt} 次重试: {str(e)}"
                    )
                    # 等待一段时间后重试
                    time.sleep(delay)
                    continue
                logger.error(f"调用公司嵌入模型API时出错: {str(e)}")
                raise ValueError(f"调用公司嵌入模型API时出错: {str(e)}")
            except json.JSONDecodeError as e:
                # JSON解析错误，一般是服务器返回的不是有效JSON
                last_error = e
                if attempt <= self.max_retries:
                    # 计算重试延迟时间
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"JSON解析错误，将在 {delay:.2f} 秒后进行第 {attempt} 次重试: {str(e)}"
                    )
                    # 等待一段时间后重试
                    time.sleep(delay)
                    continue
                logger.error("API返回了无效的JSON响应")
                raise ValueError("API返回了无效的JSON响应")
        
        # 如果到达这里，说明所有重试都失败了
        if last_error:
            error_type = type(last_error).__name__
            error_msg = str(last_error)
            logger.error(f"在 {self.max_retries} 次尝试后调用API失败: {error_type}: {error_msg}")
            raise ValueError(f"在 {self.max_retries} 次尝试后调用API失败: {error_type}: {error_msg}")
        else:
            logger.error(f"在 {self.max_retries} 次尝试后调用API失败，未知错误")
            raise ValueError(f"在 {self.max_retries} 次尝试后调用API失败，未知错误")
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成单个查询文本的嵌入向量
        
        参数:
            text: 要生成嵌入的文本
            
        返回:
            List[float]: 查询文本的嵌入向量
        """
        # 对单个文本使用embed_documents
        embeddings = self.embed_documents([text])
        # 返回第一个（也是唯一一个）嵌入结果
        return embeddings[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步生成多个文本的嵌入向量
        
        参数:
            texts: 要生成嵌入的文本列表
            
        返回:
            List[List[float]]: 嵌入向量列表，每个文本对应一个嵌入向量
        """
        # 先异步刷新令牌
        if self.token_refresh_enabled:
            await self._arefresh_token()
            
        # 构建请求体（保持与OpenAI兼容）
        request_body = {
            "model": self.model,
            "input": texts
        }
        
        # 设置超时
        request_timeout = ClientTimeout(total=self.timeout or 60)
        
        # 实现异步重试逻辑
        attempt = 0  # 当前尝试次数
        last_error = None  # 记录最后一个错误
        
        # 重试循环，最多尝试max_retries次（包括初始请求）
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # 发送异步API请求
                async with aiohttp.ClientSession(timeout=request_timeout) as session:
                    async with session.post(
                        url=self.api_url,
                        headers=self._headers,
                        json=request_body
                    ) as response:
                        if response.status == 200:
                            # 请求成功，解析响应并返回结果
                            result = await response.json()
                            # 提取嵌入向量（保持与OpenAI兼容的响应格式）
                            embeddings = [item["embedding"] for item in result["data"]]
                            return embeddings
                        elif self._should_retry(response.status):
                            # 需要重试的状态码，且尚未达到最大重试次数
                            if attempt <= self.max_retries:
                                # 计算重试延迟时间
                                delay = self._calculate_retry_delay(attempt)
                                logger.warning(
                                    f"异步API请求返回状态码 {response.status}，将在 {delay:.2f} 秒后进行第 {attempt} 次重试..."
                                )
                                # 异步等待一段时间后重试
                                await asyncio.sleep(delay)
                                continue
                                
                        # 处理其他错误状态码（不重试的情况）
                        if response.status == 401:
                            raise ValueError("认证失败，请检查您的认证凭据")
                        elif response.status == 403:
                            raise ValueError("无权限访问此资源，请检查您的权限设置")
                        elif response.status == 404:
                            raise ValueError(f"API端点不存在: {self.api_url}")
                        elif response.status >= 500:
                            raise ValueError(f"服务器错误: {response.status}")
                            
                        # 其他未处理的状态码
                        response.raise_for_status()
                        
            except aiohttp.ClientConnectorError as e:
                # 连接错误，可以重试
                last_error = e
                if attempt <= self.max_retries:
                    # 计算重试延迟时间
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"连接错误，将在 {delay:.2f} 秒后进行第 {attempt} 次重试: {str(e)}"
                    )
                    # 异步等待一段时间后重试
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"连接到API服务器失败: {self.api_url}")
                raise ValueError(f"连接到API服务器失败: {self.api_url}")
            except asyncio.TimeoutError as e:
                # 超时错误，可以重试
                last_error = e
                if attempt <= self.max_retries:
                    # 计算重试延迟时间
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"异步请求超时，将在 {delay:.2f} 秒后进行第 {attempt} 次重试"
                    )
                    # 异步等待一段时间后重试
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"异步API请求超时: {self.timeout}秒")
                raise ValueError(f"异步API请求超时: {self.timeout}秒")
            except aiohttp.ClientError as e:
                # 其他客户端错误，可以重试
                last_error = e
                if attempt <= self.max_retries:
                    # 计算重试延迟时间
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"客户端错误，将在 {delay:.2f} 秒后进行第 {attempt} 次重试: {str(e)}"
                    )
                    # 异步等待一段时间后重试
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"异步调用公司嵌入模型API时出错: {str(e)}")
                raise ValueError(f"异步调用公司嵌入模型API时出错: {str(e)}")
        
        # 如果到达这里，说明所有重试都失败了
        if last_error:
            error_type = type(last_error).__name__
            error_msg = str(last_error)
            logger.error(f"在 {self.max_retries} 次尝试后异步调用API失败: {error_type}: {error_msg}")
            raise ValueError(f"在 {self.max_retries} 次尝试后异步调用API失败: {error_type}: {error_msg}")
        else:
            logger.error(f"在 {self.max_retries} 次尝试后异步调用API失败，未知错误")
            raise ValueError(f"在 {self.max_retries} 次尝试后异步调用API失败，未知错误")
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        异步生成单个查询文本的嵌入向量
        
        参数:
            text: 要生成嵌入的文本
            
        返回:
            List[float]: 查询文本的嵌入向量
        """
        # 对单个文本使用异步embed_documents
        embeddings = await self.aembed_documents([text])
        # 返回第一个（也是唯一一个）嵌入结果
        return embeddings[0] 