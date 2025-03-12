"""
启动模拟API服务器

此脚本用于启动模拟的公司LLM API和Token获取API服务器。
默认监听在 http://0.0.0.0:5000

命令行参数:
--host: 指定服务器监听的主机地址，默认为0.0.0.0（所有网络接口）
--port: 指定服务器监听的端口号，默认为5000
--llm-error-rate: 指定LLM API的错误率，范围0-1，默认为0.3
--azure-error-rate: 指定Azure API的错误率，范围0-1，默认为0.3
--token-error-rate: 指定Token API的错误率，范围0-1，默认为0.1
--no-debug: 不使用调试模式启动服务器

使用示例:
python run_mock_api.py --port 5001 --llm-error-rate 0.2
"""

import argparse
import logging
from mock_api import start_server, ERROR_PROBABILITY

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="启动模拟API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址，默认为0.0.0.0")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口，默认为5000")
    
    # 配置错误率 - 用于测试适配器的重试机制
    parser.add_argument("--llm-error-rate", type=float, default=0.3, help="LLM API错误率，默认为0.3")
    parser.add_argument("--azure-error-rate", type=float, default=0.3, help="Azure API错误率，默认为0.3")
    parser.add_argument("--token-error-rate", type=float, default=0.1, help="Token API错误率，默认为0.1")
    
    # 调试模式选项
    parser.add_argument("--no-debug", action="store_true", help="不使用调试模式启动服务器")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置模拟API的错误率 - 通过修改mock_api模块中的常量
    ERROR_PROBABILITY["llm_api"] = args.llm_error_rate
    ERROR_PROBABILITY["azure_api"] = args.azure_error_rate
    ERROR_PROBABILITY["token_api"] = args.token_error_rate
    
    # 打印服务器启动信息和端点说明
    logger.info(f"启动模拟API服务器在 http://{args.host}:{args.port}")
    logger.info(f"错误率设置: LLM API={args.llm_error_rate}, Azure API={args.azure_error_rate}, Token API={args.token_error_rate}")
    logger.info("提供以下API端点:")
    logger.info(f"- 公司LLM API: http://{args.host}:{args.port}/v1/chat/completions")
    logger.info(f"- Azure OpenAI API: http://{args.host}:{args.port}/openai/deployments/<deployment_id>/chat/completions")
    logger.info(f"- 令牌刷新API: http://{args.host}:{args.port}/api/token")
    logger.info(f"- 健康检查: http://{args.host}:{args.port}/health")
    
    # 启动服务器 - 调用mock_api模块中的start_server函数
    start_server(host=args.host, port=args.port, use_debugger=not args.no_debug) 