"""
模拟API服务器：提供模拟的公司LLM API和Token获取API

此模块使用Flask创建模拟的API端点，用于测试LLM适配器。
主要提供以下功能：
1. 模拟公司LLM API（与OpenAI格式兼容）
2. 模拟公司Azure OpenAI API 
3. 模拟令牌获取API

模拟服务器的主要特点：
- 可配置的错误率，用于测试适配器的重试机制
- 支持流式和非流式响应
- 模拟认证和授权流程
- 提供健康检查端点
"""

from flask import Flask, request, jsonify, Response, stream_with_context
import time
import json
import random
import uuid
import threading
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 模拟的用户凭证 - 用于令牌获取API的用户名和密码验证
VALID_CREDENTIALS = {
    "test_user": "test_password",
    "admin": "admin123"
}

# 模拟的有效令牌 - 格式: {令牌字符串: {过期时间戳, 用户名}}
VALID_TOKENS = {
    "test-token": {"expires_at": time.time() + 3600, "user": "test_user"},
    "admin-token": {"expires_at": time.time() + 3600, "user": "admin"}
}

# 模拟故障概率 - 控制不同API端点产生错误的概率
ERROR_PROBABILITY = {
    "llm_api": 0.3,  # 公司LLM API故障概率
    "azure_api": 0.3,  # Azure API故障概率
    "token_api": 0.1  # Token获取API故障概率
}

# 模拟的错误类型及其概率 - 当产生错误时，各种HTTP状态码的概率分布
ERROR_TYPES = {
    "429": 0.5,  # Too Many Requests
    "500": 0.3,  # Internal Server Error
    "502": 0.1,  # Bad Gateway
    "503": 0.05,  # Service Unavailable
    "504": 0.05   # Gateway Timeout
}

# 模拟的应用ID白名单 - 用于认证检查
VALID_APP_IDS = ["test-app-id", "production-app-id", "demo-app-id"]

# 模拟响应延迟范围（秒）- 模拟真实API的响应时间
RESPONSE_DELAY_RANGE = (0.1, 0.5)

def should_generate_error(api_type):
    """
    决定是否生成模拟错误
    
    根据预设的错误概率，确定当前请求是否应该返回错误响应。
    这用于测试适配器的重试机制和错误处理能力。
    
    参数:
        api_type: API类型，可以是"llm_api"、"azure_api"或"token_api"
        
    返回:
        bool: 如果应该生成错误则返回True，否则返回False
    """
    return random.random() < ERROR_PROBABILITY.get(api_type, 0.1)

def get_error_status_code():
    """
    随机选择一个错误状态码
    
    根据预设的错误状态码概率分布，随机选择一个HTTP错误状态码。
    这用于模拟不同类型的API错误，如速率限制、服务器错误等。
    
    返回:
        int: 选择的HTTP错误状态码
    """
    # 使用累积概率选择错误类型
    r = random.random()
    cumulative_prob = 0
    for code, prob in ERROR_TYPES.items():
        cumulative_prob += prob
        if r <= cumulative_prob:
            return int(code)
    return 500  # 默认返回500

def simulate_response_delay():
    """
    模拟API响应延迟
    
    引入随机延迟，模拟真实API的响应时间，使测试更接近真实环境。
    延迟范围由RESPONSE_DELAY_RANGE常量定义。
    """
    delay = random.uniform(*RESPONSE_DELAY_RANGE)
    time.sleep(delay)

def validate_token(token):
    """
    验证令牌是否有效
    
    检查提供的令牌是否存在于VALID_TOKENS中，并且是否已过期。
    这是API认证流程的一部分。
    
    参数:
        token: 要验证的令牌字符串
        
    返回:
        bool: 如果令牌有效且未过期则返回True，否则返回False
    """
    token_info = VALID_TOKENS.get(token)
    if not token_info:
        return False
    if token_info["expires_at"] < time.time():
        # 令牌已过期
        return False
    return True

def validate_app_id(app_id):
    """
    验证应用ID是否有效
    
    检查提供的应用ID是否在VALID_APP_IDS白名单中。
    这是API认证流程的另一部分。
    
    参数:
        app_id: 要验证的应用ID
        
    返回:
        bool: 如果应用ID有效则返回True，否则返回False
    """
    return app_id in VALID_APP_IDS

def build_chat_completion_response(messages, model="gpt-3.5-turbo", stream=False):
    """
    构建模拟的聊天完成响应
    
    根据输入的消息生成一个模拟的API响应，格式与OpenAI兼容。
    可以返回完整响应或流式响应，取决于stream参数。
    
    参数:
        messages: 用户输入的消息列表，格式与OpenAI兼容
        model: 使用的模型名称
        stream: 是否生成流式响应
        
    返回:
        dict或list: 如果stream为False，返回完整响应字典；
                  如果stream为True，返回响应块列表
    """
    # 提取用户消息
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        user_message = "你好！"
    else:
        user_message = user_messages[-1].get("content", "你好！")
    
    # 根据用户消息生成模拟回复 - 简单的关键词匹配
    if "错误" in user_message:
        response_text = "抱歉，我不知道如何处理这个错误。"
    elif "问题" in user_message:
        response_text = "这是一个很好的问题！让我来回答..."
    elif "测试" in user_message:
        response_text = "这是一个测试回复，用于验证API功能是否正常。"
    else:
        # 默认回复 - 随机选择一个预设回复
        responses = [
            "我是模拟的LLM API，很高兴为您服务！",
            "您好，我能帮您解决什么问题？",
            "感谢您使用我们的服务，这是一条自动生成的回复。",
            "模拟API正常工作中，这是预设的回复消息。"
        ]
        response_text = random.choice(responses)
    
    # 如果是流式响应，返回一个生成器
    if stream:
        return generate_stream_response(response_text)
    
    # 否则返回完整的响应 - 格式与OpenAI兼容
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(response_text.split()),
            "total_tokens": 10 + len(response_text.split())
        }
    }
    return response

def generate_stream_response(text):
    """
    生成流式响应的内容
    
    将输入文本分割成多个块，模拟流式传输响应。
    格式与OpenAI的流式响应兼容。
    
    参数:
        text: 要流式传输的文本内容
        
    返回:
        list: 响应块列表，每个块是一个格式化的字典
    """
    # 将文本分成多个部分模拟流式传输
    words = text.split()
    chunks = []
    
    # 构建开始部分 - 流式响应的第一个块，只包含角色信息
    chunks.append({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None
            }
        ]
    })
    
    # 构建内容部分，每1-3个词作为一个块
    current_position = 0
    while current_position < len(words):
        chunk_size = random.randint(1, min(3, len(words) - current_position))
        chunk_text = " ".join(words[current_position:current_position + chunk_size])
        current_position += chunk_size
        
        chunks.append({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk_text + " "
                    },
                    "finish_reason": None
                }
            ]
        })
    
    # 构建结束部分 - 流式响应的最后一个块，包含finish_reason
    chunks.append({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    })
    
    return chunks

@app.route('/v1/chat/completions', methods=['POST'])
def company_llm_api():
    """
    模拟公司LLM API（与OpenAI格式兼容）
    
    处理POST请求，模拟公司自有的LLM API。
    包括认证检查、错误生成和响应构建。
    支持流式和非流式响应。
    
    返回:
        Response: 包含响应内容的Flask响应对象
    """
    # 模拟API响应延迟
    simulate_response_delay()
    
    # 验证请求头 - 检查认证信息
    headers = request.headers
    app_id = headers.get('AGI-Platform-Application-ID')
    user_login = headers.get('X-DSP-User-Login-As')
    token = headers.get('X-E2E-Trust-Token')
    
    # 基本验证 - 如果认证失败，返回401错误
    if not app_id or not validate_app_id(app_id):
        return jsonify({"error": "无效的应用ID"}), 401
    if not user_login:
        return jsonify({"error": "缺少用户登录信息"}), 401
    if not token or not validate_token(token):
        return jsonify({"error": "无效或过期的令牌"}), 401
    
    # 随机产生错误 - 用于测试适配器的重试机制
    if should_generate_error("llm_api"):
        error_code = get_error_status_code()
        error_messages = {
            429: "请求过多，请稍后再试",
            500: "服务器内部错误",
            502: "错误网关",
            503: "服务暂时不可用",
            504: "网关超时"
        }
        return jsonify({"error": error_messages.get(error_code, "未知错误")}), error_code
    
    # 解析请求
    try:
        data = request.json
        stream = data.get('stream', False)
        model = data.get('model', 'gpt-3.5-turbo')
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "消息列表不能为空"}), 400
        
        # 构建响应
        response = build_chat_completion_response(messages, model, stream)
        
        # 处理流式响应 - 使用Flask的stream_with_context
        if stream:
            def generate():
                for chunk in response:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

@app.route('/openai/deployments/<deployment_id>/chat/completions', methods=['POST'])
def azure_openai_api(deployment_id):
    """
    模拟Azure OpenAI API
    
    处理POST请求，模拟Azure OpenAI API。
    包括认证检查、错误生成和响应构建。
    支持流式和非流式响应。
    
    参数:
        deployment_id: Azure部署ID，从URL路径中提取
        
    返回:
        Response: 包含响应内容的Flask响应对象
    """
    # 模拟API响应延迟
    simulate_response_delay()
    
    # 验证请求头
    headers = request.headers
    app_id = headers.get('AGI-Platform-Application-ID')
    user_login = headers.get('X-DSP-User-Login-As')
    token = headers.get('X-E2E-Trust-Token')
    api_key = headers.get('api-key')
    
    # 基本验证 - Azure API需要额外的api-key验证
    if not app_id or not validate_app_id(app_id):
        return jsonify({"error": "无效的应用ID"}), 401
    if not user_login:
        return jsonify({"error": "缺少用户登录信息"}), 401
    if not token or not validate_token(token):
        return jsonify({"error": "无效或过期的令牌"}), 401
    if not api_key or api_key != "fake-api-key":
        return jsonify({"error": "无效的API密钥"}), 401
    
    # 随机产生错误
    if should_generate_error("azure_api"):
        error_code = get_error_status_code()
        error_messages = {
            429: "请求过多，请稍后再试",
            500: "服务器内部错误",
            502: "错误网关",
            503: "服务暂时不可用",
            504: "网关超时"
        }
        return jsonify({"error": error_messages.get(error_code, "未知错误")}), error_code
    
    # 验证部署ID - 检查是否为有效的部署
    valid_deployments = ["test-deployment", "gpt-35-turbo", "gpt-4"]
    if deployment_id not in valid_deployments:
        return jsonify({"error": f"找不到部署: {deployment_id}"}), 404
    
    # 解析请求
    try:
        data = request.json
        stream = data.get('stream', False)
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "消息列表不能为空"}), 400
        
        # 构建响应
        response = build_chat_completion_response(messages, deployment_id, stream)
        
        # 处理流式响应
        if stream:
            def generate():
                for chunk in response:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"处理Azure API请求时出错: {str(e)}")
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

@app.route('/api/token', methods=['POST'])
def token_api():
    """
    模拟令牌获取API
    
    处理POST请求，模拟令牌获取/刷新API。
    验证用户凭证并生成新的令牌。
    
    返回:
        Response: 包含新令牌和过期时间的Flask响应对象
    """
    # 模拟API响应延迟
    simulate_response_delay()
    
    # 随机产生错误
    if should_generate_error("token_api"):
        error_code = get_error_status_code()
        error_messages = {
            429: "请求过多，请稍后再试",
            500: "服务器内部错误",
            502: "错误网关",
            503: "服务暂时不可用",
            504: "网关超时"
        }
        return jsonify({"error": error_messages.get(error_code, "未知错误")}), error_code
    
    # 解析请求
    try:
        data = request.json
        input_token_state = data.get('input_token_state', {})
        
        if input_token_state.get('token_type') != "CREDENTIAL":
            return jsonify({"error": "无效的令牌类型"}), 400
        
        username = input_token_state.get('username')
        password = input_token_state.get('password')
        
        if not username or not password:
            return jsonify({"error": "缺少用户名或密码"}), 400
        
        # 验证凭证
        if username not in VALID_CREDENTIALS or VALID_CREDENTIALS[username] != password:
            return jsonify({"error": "无效的用户名或密码"}), 401
        
        # 生成新令牌
        new_token = f"{username}-{uuid.uuid4().hex[:8]}"
        expires_at = time.time() + 3600  # 令牌有效期1小时
        
        # 存储令牌
        VALID_TOKENS[new_token] = {"expires_at": expires_at, "user": username}
        
        # 返回响应
        return jsonify({
            "token": new_token,
            "expires_at": int(expires_at)
        })
        
    except Exception as e:
        logger.error(f"处理令牌请求时出错: {str(e)}")
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查端点
    
    处理GET请求，提供API服务器的健康状态。
    用于确认服务器是否正常运行。
    
    返回:
        Response: 包含健康状态和时间戳的Flask响应对象
    """
    return jsonify({"status": "healthy", "timestamp": time.time()})

def start_server(host='0.0.0.0', port=5000, use_debugger=True):
    """
    启动API服务器
    
    使用Flask运行模拟API服务器，监听指定的主机和端口。
    
    参数:
        host: 监听的主机地址，默认为0.0.0.0（所有网络接口）
        port: 监听的端口号，默认为5000
        use_debugger: 是否使用Flask的调试器，在线程中应设为False
    """
    import threading
    
    # 在线程中运行时，禁用调试器，因为信号处理只能在主线程中工作
    is_in_thread = threading.current_thread() is not threading.main_thread()
    use_debug = use_debugger and not is_in_thread
    
    if is_in_thread:
        logger.info("在线程中运行，已禁用调试器")
        
    app.run(host=host, port=port, debug=use_debug, threaded=True)

if __name__ == '__main__':
    start_server() 