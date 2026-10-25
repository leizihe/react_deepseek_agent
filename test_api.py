#!/usr/bin/env python
"""
快速测试DeepSeek API连通性，不用启动完整Agent
用法: python test_api.py
"""
import os
import sys

# 先加载vendor依赖
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import requests


def test_api():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 请先设置 DEEPSEEK_API_KEY 环境变量")
        print("   export DEEPSEEK_API_KEY=your-key-here")
        sys.exit(1)

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    print(f"🔍 测试API连接...")
    print(f"   地址: {base_url}")
    print(f"   模型: {model}")
    print()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "你好，请回复一句话：API连接成功"}
        ],
        "temperature": 0.1,
        "max_tokens": 50,
        "stream": False,
    }

    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            print(f"✅ API连接成功！模型回复: {answer}")
            print()
            print("🎉 可以正常启动Agent: python run.py")
        else:
            print(f"❌ 调用失败，HTTP {resp.status_code}")
            try:
                err = resp.json()
                print(f"   错误信息: {err.get('error', {}).get('message', resp.text)}")
            except:
                print(f"   响应内容: {resp.text}")
            print()
            print("常见原因:")
            print("  1. API Key 填错了（检查前后有没有空格/换行）")
            print("  2. 账户余额不足，需要充值")
            print("  3. base_url不对（国内代理地址填错了）")
            print("  4. 网络无法访问 api.deepseek.com")
    except Exception as e:
        print(f"❌ 网络请求失败: {e}")
        print()
        print("常见原因:")
        print("  1. 网络连接不通，检查代理设置")
        print("  2. 防火墙拦截了出站请求")


if __name__ == "__main__":
    test_api()
