"""
公司LLM适配器：用于连接公司自有LLM API与LangChain的适配器
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, Iterator, AsyncIterator, cast
import logging
import json
import time
import requests
import aiohttp
from aiohttp import ClientTimeout
import asyncio  # 确保导入asyncio

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr, model_validator

logger = logging.getLogger(__name__)

class CompanyChatModel(BaseChatModel):
    """
    公司自有LLM API的适配器，与OpenAI格式兼容
    """
    
    # API配置
    api_url: str = Field(..., description="公司LLM API的URL")
    application_id: str = Field(..., description="GAI-Platform-Application-ID")
    user_login_as: str = Field(..., description="X-DSP-User-Login-As")
    trust_token: SecretStr = Field(..., description="X-E2E-Trust-Token")
    
    # 令牌刷新配置
    token_url: Optional[str] = Field(None, description="令牌刷新API的URL")
    username: Optional[SecretStr] = Field(None, description="用于令牌刷新的用户名")
    password: Optional[SecretStr] = Field(None, description="用于令牌刷新的密码")
    token_refresh_enabled: bool = Field(False, description="是否启用令牌自动刷新")
    token_refresh_interval: int = Field(0, description="令牌最小刷新间隔（秒），0表示每次都刷新")
    
    # 模型配置（类似OpenAI）
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    timeout: Optional[float] = None
    
    # 其他配置
    streaming: bool = False
    n: int = 1
    max_retries: int = 6  # 最大重试次数（包括初始请求）
    
    # 重试配置
    retry_min_delay: float = Field(1.0, description="重试的最小延迟时间（秒）")
    retry_max_delay: float = Field(60.0, description="重试的最大延迟时间（秒）")
    retry_backoff_factor: float = Field(2.0, description="重试延迟的退避系数")
    retry_jitter: bool = Field(True, description="是否在重试延迟时间上添加随机抖动")
    retry_on_status_codes: List[int] = Field([429, 500, 502, 503, 504], description="需要重试的HTTP状态码")
    
    # 内部状态
    _last_token_refresh: float = 0.0  # 上次刷新令牌的时间戳
    
    @model_validator(mode='after')
    def validate_params(self) -> 'CompanyChatModel':
        """验证参数合法性"""
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("temperature必须在0和1之间")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p必须在0和1之间")
        if self.frequency_penalty < -2 or self.frequency_penalty > 2:
            raise ValueError("frequency_penalty必须在-2和2之间")
        if self.presence_penalty < -2 or self.presence_penalty > 2:
            raise ValueError("presence_penalty必须在-2和2之间")
        if self.n < 1:
            raise ValueError("n必须至少为1")
        if self.n > 1 and self.streaming:
            raise ValueError("streaming模式下n必须为1")
            
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
    def _default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "stream": self.streaming,
        }
    
    @property
    def _headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Content-Type": "application/json",
            "GAI-Platform-Application-ID": self.application_id,
            "X-DSP-User-Login-As": self.user_login_as,
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
                    self.token_url,
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
            import random
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
    
    def _convert_messages_to_openai_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """将LangChain消息格式转换为OpenAI格式"""
        openai_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                openai_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, ChatMessage):
                openai_messages.append({"role": message.role, "content": message.content})
            else:
                raise ValueError(f"不支持的消息类型: {type(message)}")
        return openai_messages
    
    def _create_chat_result(self, response: Dict[str, Any]) -> ChatResult:
        """解析API响应并创建ChatResult"""
        if not response.get("choices"):
            raise ValueError(f"API返回了无效的响应: {response}")
        
        generations = []
        for choice in response["choices"]:
            message = choice.get("message", {})
            if not message:
                continue
            
            role = message.get("role", "assistant")
            content = message.get("content", "")
            
            if role == "assistant":
                generation = ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info=dict(
                        finish_reason=choice.get("finish_reason"),
                        logprobs=choice.get("logprobs"),
                    ),
                )
                generations.append(generation)
        
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": response.get("model", "")}
        
        return ChatResult(generations=generations, llm_output=llm_output)
    
    def _process_chunk(self, chunk_data: Dict[str, Any]) -> Optional[ChatGenerationChunk]:
        """处理流式传输的数据块"""
        if not chunk_data.get("choices") or not chunk_data["choices"]:
            return None
            
        choice = chunk_data["choices"][0]
        delta = choice.get("delta", {})
        if not delta or not delta.get("content"):
            return None
            
        content = delta.get("content", "")
        return ChatGenerationChunk(
            message=AIMessageChunk(content=content),
            generation_info={"finish_reason": choice.get("finish_reason")},
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成LLM回复的核心方法
        
        此方法包含完整的重试逻辑，可以处理临时性故障，如网络错误、服务器错误等。
        重试使用指数退避策略，并支持自定义重试参数。
        
        参数:
            messages: 用于生成回复的消息列表
            stop: 停止标记列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        返回:
            ChatResult: 生成的聊天结果
            
        抛出:
            ValueError: 如果API调用失败且达到最大重试次数
        """
        # 先刷新令牌
        if self.token_refresh_enabled:
            self._refresh_token()
            
        # 准备请求参数
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop
            
        # 如果开启了流式传输但调用了_generate，则关闭流式传输
        if params.get("stream"):
            params["stream"] = False
        
        # 转换消息格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求体
        request_body = {
            "model": params["model"],
            "messages": openai_messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "frequency_penalty": params["frequency_penalty"],
            "presence_penalty": params["presence_penalty"],
            "n": params["n"],
        }
        
        if params.get("max_tokens"):
            request_body["max_tokens"] = params["max_tokens"]
        
        # 实现重试逻辑
        attempt = 0  # 当前尝试次数
        last_error = None  # 记录最后一个错误，用于在所有重试都失败时提供详细信息
        
        # 重试循环，最多尝试max_retries次（包括初始请求）
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # 发送API请求
                response = requests.post(
                    self.api_url,
                    headers=self._headers,  # 使用带有刷新后令牌的请求头
                    json=request_body,
                    timeout=self.timeout or 60,
                )
                
                if response.status_code == 200:
                    # 请求成功，解析响应并返回结果
                    raw_response = response.json()
                    chat_result = self._create_chat_result(raw_response)
                    return chat_result
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
                logger.error(f"调用公司LLM API时出错: {str(e)}")
                raise ValueError(f"调用公司LLM API时出错: {str(e)}")
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
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式生成LLM回复
        
        此方法实现了流式响应的重试逻辑，可以处理临时性故障。
        当流式请求失败时，会尝试重新建立连接并继续请求。
        
        参数:
            messages: 用于生成回复的消息列表
            stop: 停止标记列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        返回:
            Iterator[ChatGenerationChunk]: 生成的聊天片段迭代器
            
        抛出:
            ValueError: 如果API调用失败且达到最大重试次数
        """
        # 先刷新令牌
        if self.token_refresh_enabled:
            self._refresh_token()
            
        # 准备请求参数
        params = {**self._default_params, **kwargs, "stream": True}
        if stop:
            params["stop"] = stop
        
        # 转换消息格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求体
        request_body = {
            "model": params["model"],
            "messages": openai_messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "frequency_penalty": params["frequency_penalty"],
            "presence_penalty": params["presence_penalty"],
            "stream": True,
            "n": 1,  # 流式传输只支持1个生成结果
        }
        
        if params.get("max_tokens"):
            request_body["max_tokens"] = params["max_tokens"]
        
        # 实现重试逻辑
        attempt = 0  # 当前尝试次数
        last_error = None  # 记录最后一个错误
        
        # 重试循环，最多尝试max_retries次（包括初始请求）
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # 发送流式API请求
                response = requests.post(
                    self.api_url,
                    headers=self._headers,  # 使用带有刷新后令牌的请求头
                    json=request_body,
                    stream=True,  # 启用流式传输
                    timeout=self.timeout or 60,
                )
                
                if response.status_code == 200:
                    # 成功建立流式连接，处理返回的SSE数据
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        if line.startswith(b"data: "):
                            line = line[6:]  # 移除 "data: " 前缀
                            
                            if line.strip() == b"[DONE]":
                                break
                            
                            try:
                                # 解析并处理每个数据块
                                chunk_data = json.loads(line)
                                chunk = self._process_chunk(chunk_data)
                                if chunk is not None:
                                    yield chunk
                                    
                                    # 回调处理
                                    if run_manager and chunk.message.content:
                                        run_manager.on_llm_new_token(chunk.message.content)
                            except json.JSONDecodeError:
                                pass
                    
                    # 如果成功完成流式传输，则返回
                    return
                elif self._should_retry(response.status_code):
                    # 需要重试的状态码，且尚未达到最大重试次数
                    if attempt <= self.max_retries:
                        # 计算重试延迟时间
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(
                            f"流式API请求返回状态码 {response.status_code}，将在 {delay:.2f} 秒后进行第 {attempt} 次重试..."
                        )
                        # 等待一段时间后重试
                        time.sleep(delay)
                        continue
                
                # 处理错误状态码（不重试的情况）
                if response.status_code != 200:
                    response_text = response.text
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
                else:
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
                logger.error(f"调用流式API时出错: {str(e)}")
                raise ValueError(f"调用流式API时出错: {str(e)}")
        
        # 如果到达这里，说明所有重试都失败了
        if last_error:
            error_type = type(last_error).__name__
            error_msg = str(last_error)
            logger.error(f"在 {self.max_retries} 次尝试后调用流式API失败: {error_type}: {error_msg}")
            raise ValueError(f"在 {self.max_retries} 次尝试后调用流式API失败: {error_type}: {error_msg}")
        else:
            logger.error(f"在 {self.max_retries} 次尝试后调用流式API失败，未知错误")
            raise ValueError(f"在 {self.max_retries} 次尝试后调用流式API失败，未知错误")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        异步生成LLM回复
        
        此方法实现了异步请求的重试逻辑，可以处理临时性故障。
        使用asyncio实现异步重试，避免阻塞事件循环。
        
        参数:
            messages: 用于生成回复的消息列表
            stop: 停止标记列表
            run_manager: 异步回调管理器
            **kwargs: 其他参数
            
        返回:
            ChatResult: 生成的聊天结果
            
        抛出:
            ValueError: 如果API调用失败且达到最大重试次数
        """
        # 先刷新令牌
        if self.token_refresh_enabled:
            await self._arefresh_token()
            
        # 准备请求参数
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop
            
        # 如果开启了流式传输但调用了_agenerate，则关闭流式传输
        if params.get("stream"):
            params["stream"] = False
        
        # 转换消息格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求体
        request_body = {
            "model": params["model"],
            "messages": openai_messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "frequency_penalty": params["frequency_penalty"],
            "presence_penalty": params["presence_penalty"],
            "n": params["n"],
        }
        
        if params.get("max_tokens"):
            request_body["max_tokens"] = params["max_tokens"]
            
        request_timeout = aiohttp.ClientTimeout(total=self.timeout or 60)
        
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
                        self.api_url,
                        headers=self._headers,  # 使用带有刷新后令牌的请求头
                        json=request_body
                    ) as response:
                        if response.status == 200:
                            # 请求成功，解析响应并返回结果
                            raw_response = await response.json()
                            return self._create_chat_result(raw_response)
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
                logger.error(f"异步调用公司LLM API时出错: {str(e)}")
                raise ValueError(f"异步调用公司LLM API时出错: {str(e)}")
        
        # 如果到达这里，说明所有重试都失败了
        if last_error:
            error_type = type(last_error).__name__
            error_msg = str(last_error)
            logger.error(f"在 {self.max_retries} 次尝试后异步调用API失败: {error_type}: {error_msg}")
            raise ValueError(f"在 {self.max_retries} 次尝试后异步调用API失败: {error_type}: {error_msg}")
        else:
            logger.error(f"在 {self.max_retries} 次尝试后异步调用API失败，未知错误")
            raise ValueError(f"在 {self.max_retries} 次尝试后异步调用API失败，未知错误")
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        异步流式生成LLM回复
        
        此方法实现了异步流式请求的重试逻辑，可以处理临时性故障。
        当流式连接中断时，会尝试重新建立连接并继续请求。
        
        参数:
            messages: 用于生成回复的消息列表
            stop: 停止标记列表
            run_manager: 异步回调管理器
            **kwargs: 其他参数
            
        返回:
            AsyncIterator[ChatGenerationChunk]: 异步生成的聊天片段迭代器
            
        抛出:
            ValueError: 如果API调用失败且达到最大重试次数
        """
        # 先刷新令牌
        if self.token_refresh_enabled:
            await self._arefresh_token()
            
        # 准备请求参数
        params = {**self._default_params, **kwargs, "stream": True}
        if stop:
            params["stop"] = stop
        
        # 转换消息格式
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # 构建请求体
        request_body = {
            "model": params["model"],
            "messages": openai_messages,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "frequency_penalty": params["frequency_penalty"],
            "presence_penalty": params["presence_penalty"],
            "stream": True,
            "n": 1,  # 流式传输只支持1个生成结果
        }
        
        if params.get("max_tokens"):
            request_body["max_tokens"] = params["max_tokens"]
            
        request_timeout = aiohttp.ClientTimeout(total=self.timeout or 60)
        
        # 实现异步重试逻辑
        attempt = 0  # 当前尝试次数
        last_error = None  # 记录最后一个错误
        
        # 重试循环，最多尝试max_retries次（包括初始请求）
        while attempt <= self.max_retries:
            attempt += 1
            try:
                # 发送异步流式API请求
                async with aiohttp.ClientSession(timeout=request_timeout) as session:
                    async with session.post(
                        self.api_url,
                        headers=self._headers,  # 使用带有刷新后令牌的请求头
                        json=request_body
                    ) as response:
                        if response.status == 200:
                            # 成功建立异步流式连接，处理SSE流
                            async for line in response.content:
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                line_text = line.decode("utf-8")
                                if not line_text.startswith("data: "):
                                    continue
                                    
                                line_data = line_text[6:]  # 移除 "data: " 前缀
                                if line_data == "[DONE]":
                                    break
                                    
                                try:
                                    # 解析并处理每个数据块
                                    chunk_data = json.loads(line_data)
                                    chunk = self._process_chunk(chunk_data)
                                    if chunk is not None:
                                        yield chunk
                                        
                                        # 回调处理
                                        if run_manager and chunk.message.content:
                                            await run_manager.on_llm_new_token(chunk.message.content)
                                except json.JSONDecodeError:
                                    pass
                            
                            # 如果成功完成流式传输，则返回
                            return
                        elif self._should_retry(response.status):
                            # 需要重试的状态码，且尚未达到最大重试次数
                            if attempt <= self.max_retries:
                                # 计算重试延迟时间
                                delay = self._calculate_retry_delay(attempt)
                                logger.warning(
                                    f"异步流式API请求返回状态码 {response.status}，将在 {delay:.2f} 秒后进行第 {attempt} 次重试..."
                                )
                                # 异步等待一段时间后重试
                                await asyncio.sleep(delay)
                                continue
                            
                        # 处理错误状态码（不重试的情况）
                        if response.status != 200:
                            response_text = await response.text()
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
                        f"异步流式请求超时，将在 {delay:.2f} 秒后进行第 {attempt} 次重试"
                    )
                    # 异步等待一段时间后重试
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"异步流式API请求超时: {self.timeout}秒")
                raise ValueError(f"异步流式API请求超时: {self.timeout}秒")
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
                logger.error(f"异步调用流式API时出错: {str(e)}")
                raise ValueError(f"异步调用流式API时出错: {str(e)}")
        
        # 如果到达这里，说明所有重试都失败了
        if last_error:
            error_type = type(last_error).__name__
            error_msg = str(last_error)
            logger.error(f"在 {self.max_retries} 次尝试后异步调用流式API失败: {error_type}: {error_msg}")
            raise ValueError(f"在 {self.max_retries} 次尝试后异步调用流式API失败: {error_type}: {error_msg}")
        else:
            logger.error(f"在 {self.max_retries} 次尝试后异步调用流式API失败，未知错误")
            raise ValueError(f"在 {self.max_retries} 次尝试后异步调用流式API失败，未知错误")
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识符"""
        return "company-chat-model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型的标识参数"""
        return {
            "model": self.model,
            "api_url": self.api_url,
            "application_id": self.application_id,
            "token_refresh_enabled": self.token_refresh_enabled,
            # 不返回敏感信息
        } 