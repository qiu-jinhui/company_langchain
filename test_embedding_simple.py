"""
简单测试：使用mock数据测试公司嵌入模型适配器
"""

import unittest
from unittest.mock import patch, MagicMock
from company_embedding import CompanyEmbeddings
from pydantic import SecretStr

def test_embedding_with_mocks():
    """
    使用mock模拟API调用，测试嵌入功能
    """
    # 初始化公司嵌入模型
    embeddings = CompanyEmbeddings(
        api_url="https://fake-api.example.com/v1/embeddings",
        application_id="test-app-id",
        trust_token="test-token",
    )
    
    # 模拟API响应
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0},
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }
    
    # 打补丁替换实际API调用
    with patch("requests.post", return_value=mock_response):
        # 调用嵌入方法
        text = "这是一个测试文本"
        result = embeddings.embed_query(text)
        
        print("模拟测试成功！")
        print(f"输入文本: '{text}'")
        print(f"嵌入结果: {result}")

if __name__ == "__main__":
    test_embedding_with_mocks() 