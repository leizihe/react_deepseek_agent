# -*- coding: utf-8 -*-
"""
离线验证脚本：不访问真实网络，用 mock HTTP 响应验证 webfetch 工具。
验证点：
  1. HTML 转纯文本：提取标题、去脚本/样式、实体解码、块级换行
  2. JSON 格式化、GBK 页面解码
  3. 长度截断、重定向提示
  4. 非法URL、HTTP错误、超时、二进制内容
  5. 工具注册与 schema 合法性
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import tool as tool_mod
from tool import ToolManager, WebFetchTool


class FakeResponse:
    def __init__(self, body, status=200, headers=None, url="https://example.com/page",
                 reason="OK"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body
        self.status_code = status
        self.headers = headers or {}
        self.url = url
        self.reason = reason
        self.encoding = None

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FetchMock:
    """按URL返回预设响应，记录调用参数"""
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.original = tool_mod.requests.get

    def __enter__(self):
        def fake_get(url, **kwargs):
            self.calls.append({"url": url, "kwargs": kwargs})
            result = self.responses[url]
            if isinstance(result, Exception):
                raise result
            return result
        tool_mod.requests.get = fake_get
        return self

    def __exit__(self, *args):
        tool_mod.requests.get = self.original
        return False


def run_fetch(url, responses, **kwargs):
    with FetchMock(responses):
        return WebFetchTool().execute(url=url, **kwargs)


def test_1_html_to_text():
    print("\n[测试1] HTML → 纯文本（标题/去脚本/去样式/实体/换行）")
    html = """<!DOCTYPE html>
<html><head>
<title> 示例 &amp; 页面 </title>
<style>.a { color: red; }</style>
<script>alert('should not appear')</script>
</head>
<body>
<h1>主标题</h1>
<p>第一段&nbsp;内容 &lt;test&gt;</p>
<ul><li>条目一</li><li>条目二</li></ul>
<br/>
<pre>  pre  内容  </pre>
<div hidden>可见文本</div>
</body></html>"""
    resp = FakeResponse(html, headers={"Content-Type": "text/html; charset=utf-8"})
    out = run_fetch("https://example.com/page", {"https://example.com/page": resp})

    assert "🏷️ 标题: 示例 & 页面" in out, out
    assert "主标题" in out and "第一段 内容 <test>" in out
    assert "条目一" in out and "条目二" in out
    assert "可见文本" in out
    assert "alert(" not in out, "脚本内容未被移除"
    assert "color: red" not in out, "样式内容未被移除"
    assert "<p>" not in out and "<h1>" not in out, "HTML标签未被移除"
    assert "Content-Type" not in out
    print("  输出片段:\n{}".format("\n".join(out.splitlines()[:8])))
    print("  ✅ 通过")


def test_2_json_and_gbk():
    print("\n[测试2] JSON 格式化 + GBK 编码页面")
    payload = json.dumps({"name": "测试", "ok": True}, ensure_ascii=False)
    jresp = FakeResponse(payload, headers={"Content-Type": "application/json; charset=utf-8"})
    out = run_fetch("https://example.com/api", {"https://example.com/api": jresp})
    assert '"name": "测试"' in out and '"ok": true' in out, out

    html = '<html><head><meta charset="gbk"><title>中文页面</title></head><body><p>这是GBK编码的内容</p></body></html>'
    gbk_resp = FakeResponse(html.encode("gbk"), headers={"Content-Type": "text/html"})
    out = run_fetch("https://example.com/gbk", {"https://example.com/gbk": gbk_resp})
    assert "中文页面" in out and "这是GBK编码的内容" in out, "GBK解码失败:\n{}".format(out)
    print("  ✅ 通过")


def test_3_truncation_and_redirect():
    print("\n[测试3] 长度截断 + 重定向提示")
    body = "<html><body><p>" + ("很长的内容" * 300) + "</p></body></html>"
    resp = FakeResponse(body, headers={"Content-Type": "text/html"})
    out = run_fetch(
        "https://example.com/short",
        {"https://example.com/short": FakeResponse(
            body, headers={"Content-Type": "text/html"}, url="https://example.com/final")},
        max_chars=600,
    )
    assert "原始URL: https://example.com/short" in out, "缺少重定向提示"
    assert "URL: https://example.com/final" in out
    assert "已截断到 600 字符" in out, "截断提示缺失"
    assert len(out) < 2000, "截断未生效，输出 {} 字符".format(len(out))
    print("  ✅ 通过")


def test_4_errors():
    print("\n[测试4] 非法URL / HTTP错误 / 超时 / 二进制内容")
    assert "仅支持 http/https" in run_fetch("file:///etc/passwd", {}), "非法协议未被拦截"
    assert "仅支持 http/https" in run_fetch("ftp://example.com/a", {}), "非法协议未被拦截"

    not_found = FakeResponse("not found", status=404, reason="Not Found")
    out = run_fetch("https://example.com/404", {"https://example.com/404": not_found})
    assert "HTTP 404" in out, out

    timeout = tool_mod.requests.exceptions.Timeout("timed out")
    out = run_fetch("https://example.com/slow", {"https://example.com/slow": timeout})
    assert "抓取超时" in out, out

    png = FakeResponse(b"\x89PNG\r\n\x1a\n\x00\x00", headers={"Content-Type": "image/png"})
    out = run_fetch("https://example.com/a.png", {"https://example.com/a.png": png})
    assert "不支持的内容类型: image/png" in out, out
    print("  ✅ 通过")


def test_5_schema_and_registration():
    print("\n[测试5] 工具注册 + schema 合法性")
    tm = ToolManager("./workspace", skills_dir="./skills")
    names = list(tm.tools.keys())
    print("  工具: {}".format(names))
    assert "webfetch" in names, "webfetch 未注册"

    schema = tm.tools["webfetch"].get_schema()
    params = schema["function"]["parameters"]
    assert params["required"] == ["url"], params["required"]
    assert "required" not in params["properties"]["url"], "schema 泄漏布尔 required"
    assert params["properties"]["max_chars"]["type"] == "integer"
    json.dumps(schema)
    print("  ✅ 通过")


if __name__ == "__main__":
    print("=" * 60)
    print("WebFetch 工具离线验证")
    print("=" * 60)
    test_1_html_to_text()
    test_2_json_and_gbk()
    test_3_truncation_and_redirect()
    test_4_errors()
    test_5_schema_and_registration()
    print("\n" + "=" * 60)
    print("🎉 全部测试通过！WebFetch 功能工作正常")
    print("=" * 60)
