import json
import requests
from typing import List, Dict, Any, Generator, Optional


class DeepSeekModel:
    """DeepSeek 模型原生HTTP调用封装，支持流式输出"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        调用DeepSeek Chat API，支持流式返回
        返回生成器，每次yield一个增量事件：
            {"type": "content", "delta": "..."}  # 文本增量
            {"type": "tool_call", "index": 0, "id": "...", "name": "...", "arguments": "..."}  # 工具调用增量
            {"type": "done", "response": full_response}  # 结束，返回完整响应对象
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            # 只保留type/function字段，去掉handler等Python对象
            clean_tools = []
            for t in tools:
                clean_tools.append({
                    "type": t["type"],
                    "function": {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "parameters": t["function"]["parameters"],
                    }
                })
            payload["tools"] = clean_tools
            payload["tool_choice"] = "auto"

        if not stream:
            # 非流式，直接返回完整结果
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                # 打印详细错误信息
                error_detail = ""
                try:
                    err = resp.json()
                    error_detail = err.get("error", {}).get("message", resp.text)
                except:
                    error_detail = resp.text
                raise Exception(f"HTTP {resp.status_code}: {error_detail}\n请求payload preview: {json.dumps(payload, ensure_ascii=False)[:500]}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            if choice.get("message", {}).get("tool_calls"):
                yield {
                    "type": "done",
                    "response": {
                        "content": choice["message"].get("content", ""),
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            }
                            for tc in choice["message"]["tool_calls"]
                        ],
                    }
                }
            else:
                yield {
                    "type": "done",
                    "response": {
                        "content": choice["message"].get("content", ""),
                        "tool_calls": [],
                    }
                }
            return

        # 流式SSE解析
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=300)
        if resp.status_code != 200:
            error_detail = ""
            try:
                err = resp.json()
                error_detail = err.get("error", {}).get("message", resp.text)
            except:
                error_detail = resp.text
            raise Exception(f"HTTP {resp.status_code}: {error_detail}\n请求payload preview: {json.dumps(payload, ensure_ascii=False)[:1000]}")
        resp.raise_for_status()

        # 收集完整结果
        full_content = ""
        full_tool_calls: Dict[int, Dict[str, Any]] = {}

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if data.get("object") == "chat.completion.chunk":
                delta = data["choices"][0].get("delta", {})
                # 处理文本增量
                if delta.get("content"):
                    content_delta = delta["content"]
                    full_content += content_delta
                    yield {"type": "content", "delta": content_delta}
                # 处理工具调用增量
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta["index"]
                        if idx not in full_tool_calls:
                            full_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.get("id"):
                            full_tool_calls[idx]["id"] += tc_delta["id"]
                        if tc_delta.get("function", {}).get("name"):
                            full_tool_calls[idx]["name"] += tc_delta["function"]["name"]
                        if tc_delta.get("function", {}).get("arguments"):
                            full_tool_calls[idx]["arguments"] += tc_delta["function"]["arguments"]
                        yield {
                            "type": "tool_call_part",
                            "index": idx,
                            "delta_name": tc_delta.get("function", {}).get("name", ""),
                            "delta_args": tc_delta.get("function", {}).get("arguments", ""),
                        }

        # 结束，返回完整结果
        sorted_tool_calls = []
        for i in sorted(full_tool_calls.keys()):
            tc = full_tool_calls[i]
            sorted_tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
            })

        yield {
            "type": "done",
            "response": {
                "content": full_content,
                "tool_calls": sorted_tool_calls,
            }
        }
