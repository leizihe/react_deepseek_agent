# -*- coding: utf-8 -*-
"""
离线验证脚本：消息格式校验逻辑（不调用真实 API）。
重点回归：
  assistant 带多个 tool_calls 时，校验后必须紧跟对应数量的 tool 消息，
  中间不能插入任何消息，否则 DeepSeek 会报 400
  "An assistant message with 'tool_calls' must be followed by tool messages..."。

验证点：
  1. 并行工具调用的连续 tool 消息原样保留
  2. 缺失的 tool 响应自动补占位消息
  3. 孤立的 tool 消息被丢弃
  4. 连续 user 消息合并，不插入假 assistant
  5. 内部 tool_calls 格式归一化
  6. 端到端：模型一轮返回两个工具调用，下一轮请求结构合法
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import agent as agent_mod


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass


def make_agent():
    original = agent_mod.DeepSeekModel
    agent_mod.DeepSeekModel = DummyModel
    try:
        with tempfile.TemporaryDirectory() as base:
            a = agent_mod.ReActAgent(
                api_key="fake",
                workspace_root=os.path.join(base, "workspace"),
                enable_subagent=False,
                enable_skills=False,
            )
        return a
    finally:
        agent_mod.DeepSeekModel = original


def tc(call_id, name="ls"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def roles(messages):
    return [m["role"] for m in messages]


def test_1_parallel_tools_preserved():
    print("\n[测试1] 并行工具调用：连续 tool 消息必须原样保留")
    a = make_agent()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [tc("call_a"), tc("call_b")]},
        {"role": "tool", "tool_call_id": "call_a", "content": "结果A"},
        {"role": "tool", "tool_call_id": "call_b", "content": "结果B"},
    ]
    out = a._validate_messages(messages)
    print("  角色序列: {}".format(roles(out)))
    assert roles(out) == ["system", "user", "assistant", "tool", "tool"], "插入了额外消息！"
    assert out[2]["tool_calls"][0]["id"] == "call_a"
    assert out[3]["tool_call_id"] == "call_a"
    assert out[4]["tool_call_id"] == "call_b"
    print("  ✅ 通过")


def test_2_missing_tool_reply_filled():
    print("\n[测试2] 缺失 tool 响应：补占位消息，保持配对数一致")
    a = make_agent()
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [tc("call_a"), tc("call_b")]},
        {"role": "tool", "tool_call_id": "call_a", "content": "结果A"},
    ]
    out = a._validate_messages(messages)
    print("  角色序列: {}".format(roles(out)))
    assert roles(out) == ["user", "assistant", "tool", "tool"]
    assert out[2]["tool_call_id"] == "call_a"
    assert out[3]["tool_call_id"] == "call_b"
    assert "未返回结果" in out[3]["content"]
    print("  ✅ 通过")


def test_3_orphan_tool_dropped():
    print("\n[测试3] 孤立 tool 消息：没有对应 assistant tool_calls，应丢弃")
    a = make_agent()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "tool_call_id": "ghost", "content": "孤立结果"},
    ]
    out = a._validate_messages(messages)
    print("  角色序列: {}".format(roles(out)))
    assert roles(out) == ["system", "user"]
    print("  ✅ 通过")


def test_4_consecutive_user_merged():
    print("\n[测试4] 连续 user 消息：合并，而不是插入假 assistant")
    a = make_agent()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "第一句"},
        {"role": "user", "content": "第二句"},
    ]
    out = a._validate_messages(messages)
    print("  角色序列: {}".format(roles(out)))
    assert roles(out) == ["system", "user"]
    assert out[1]["content"] == "第一句\n第二句"
    print("  ✅ 通过")


def test_5_internal_tool_call_format():
    print("\n[测试5] 内部 tool_calls 格式归一化为API格式")
    a = make_agent()
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "name": "ls", "arguments": {"path": "."}}
        ]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"},
    ]
    out = a._validate_messages(messages)
    call = out[1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "ls"
    assert call["function"]["arguments"] == '{"path": "."}'
    assert out[1]["content"], "带tool_calls的assistant content不应为空"
    print("  归一化结果: {}".format(call))
    print("  ✅ 通过")


class ParallelToolMockModel:
    """第一轮返回两个工具调用，第二轮检查请求结构是否合法并收尾"""
    struct_valid = None
    seen_roles = None

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [
                    {"id": "call_ls", "name": "ls", "arguments": '{"path": "."}'},
                    {"id": "call_bash", "name": "bash", "arguments": '{"command": "echo hi"}'},
                ],
            }}
            return

        # 检查所有 assistant(tool_calls) 后面是否紧跟对应数量的 tool 消息
        valid = True
        for idx, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                ids = [c["id"] for c in msg["tool_calls"]]
                following = messages[idx + 1: idx + 1 + len(ids)]
                if (len(following) != len(ids)
                        or any(m.get("role") != "tool" for m in following)
                        or [m.get("tool_call_id") for m in following] != ids):
                    valid = False
        ParallelToolMockModel.struct_valid = valid
        ParallelToolMockModel.seen_roles = [m.get("role") for m in messages]

        answer = "PARALLEL_OK" if len(tool_messages) == 2 and valid else "PARALLEL_BAD"
        yield {"type": "done", "response": {"content": answer, "tool_calls": []}}


def test_6_parallel_end_to_end():
    print("\n[测试6] 端到端：一轮两个工具调用，下一轮请求结构合法")
    original = agent_mod.DeepSeekModel
    agent_mod.DeepSeekModel = ParallelToolMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            a = agent_mod.ReActAgent(
                api_key="fake",
                workspace_root=os.path.join(base, "workspace"),
                enable_subagent=False,
                enable_skills=False,
                max_iterations=5,
            )
            answer = a.chat("并行调用两个工具", stream=False)
            print("  Agent 回答: {}".format(answer))
            print("  第二轮角色序列: {}".format(ParallelToolMockModel.seen_roles))
            assert answer == "PARALLEL_OK", "并行工具调用的历史结构不合法"
            assert ParallelToolMockModel.struct_valid is True

            # 落盘的历史本身也应是配对的
            assistant_idx = next(
                i for i, m in enumerate(a.messages)
                if m.get("role") == "assistant" and m.get("tool_calls")
            )
            calls = a.messages[assistant_idx]["tool_calls"]
            following = a.messages[assistant_idx + 1: assistant_idx + 1 + len(calls)]
            assert len(following) == len(calls)
            assert [m["tool_call_id"] for m in following] == [c["id"] for c in calls]
        print("  ✅ 通过")
    finally:
        agent_mod.DeepSeekModel = original


def check_pairing(messages):
    """检查每条 assistant(tool_calls) 后是否紧跟对应数量的 tool 消息"""
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = [c["id"] for c in msg["tool_calls"]]
            following = messages[idx + 1: idx + 1 + len(ids)]
            if (len(following) != len(ids)
                    or any(m.get("role") != "tool" for m in following)
                    or [m.get("tool_call_id") for m in following] != ids):
                return False
    return True


class SubParallelMockModel:
    """子智能体mock：第一轮并行两个工具调用，第二轮检查结构并收尾"""

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [
                    {"id": "sub_ls", "name": "ls", "arguments": '{"path": "."}'},
                    {"id": "sub_bash", "name": "bash", "arguments": '{"command": "echo hi"}'},
                ],
            }}
            return

        ok = len(tool_messages) == 2 and check_pairing(messages)
        yield {"type": "done", "response": {
            "content": "SUB_PARALLEL_OK" if ok else "SUB_PARALLEL_BAD",
            "tool_calls": [],
        }}


def test_7_subagent_parallel_tools():
    print("\n[测试7] 子智能体复用校验逻辑：并行工具调用不报错")
    import model as model_mod
    from subagent import SubAgent

    original = model_mod.DeepSeekModel
    model_mod.DeepSeekModel = SubParallelMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            sub = SubAgent(
                api_key="fake",
                workspace_root=os.path.join(base, "workspace"),
                depth=1,
                max_depth=1,
            )
            answer = sub.run("并行调用两个工具")
            print("  子智能体回答: {}".format(answer))
            assert answer == "SUB_PARALLEL_OK", \
                "子智能体并行工具调用失败: {}".format(answer)
        print("  ✅ 通过")
    finally:
        model_mod.DeepSeekModel = original


if __name__ == "__main__":
    print("=" * 60)
    print("消息格式校验离线验证")
    print("=" * 60)
    test_1_parallel_tools_preserved()
    test_2_missing_tool_reply_filled()
    test_3_orphan_tool_dropped()
    test_4_consecutive_user_merged()
    test_5_internal_tool_call_format()
    test_6_parallel_end_to_end()
    test_7_subagent_parallel_tools()
    print("\n" + "=" * 60)
    print("🎉 全部测试通过！消息格式校验工作正常")
    print("=" * 60)
