# -*- coding: utf-8 -*-
"""
离线验证脚本：不调用真实 DeepSeek API，用 mock 模型验证 SubAgent 机制。
验证点：
  1. SubAgent 工具被正确注册进主 Agent 工具集
  2. 主 Agent 调用 task 时能成功创建子智能体并拿回结论
  3. 子智能体拥有独立上下文（消息历史与主 Agent 隔离）
  4. 嵌套深度达到上限后，子智能体不再拥有 SubAgent 工具
  5. allowed_tools 白名单真实生效
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import tool as tool_mod
from subagent import SubAgent, TaskTool


# ------------------------------------------------------------------ #
# Mock 模型：模拟 DeepSeek 的行为
# ------------------------------------------------------------------ #
class MockModel:
    """根据系统提示判断角色，模拟不同的响应序列"""
    call_log = []

    def __init__(self, api_key, base_url, model):
        self.model = model
        self.key = api_key

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        tool_names = [t["function"]["name"] for t in (tools or [])]
        system = messages[0]["content"]
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m["content"]

        # 用子智能体系统提示里的独有标记来区分主/子（主Agent系统提示也含"子智能体"字样）
        is_sub = "嵌套层级: 第" in system and "请直接开始执行任务" in system
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        MockModel.call_log.append({
            "is_subagent": is_sub,
            "tools": tool_names,
            "user": user_msg[:60],
            "has_tool_result": has_tool_result,
        })

        # 子智能体：直接产出结论，报告自己看到的工具集
        if is_sub:
            yield {"type": "done", "response": {
                "content": "SUBAGENT_RESULT 结论：已使用工具({})完成【{}】".format(
                    ",".join(tool_names), user_msg[:30]),
                "tool_calls": [],
            }}
            return

        # 主 Agent 第一轮：发起 SubAgent 调用
        if "派遣" in user_msg and not has_tool_result:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "name": "SubAgent",
                    "arguments": '{"description": "调研一下 DeepSeek API 的定价", "role": "researcher"}',
                }],
            }}
            return

        # 主 Agent 收到 SubAgent 结果后收尾
        if has_tool_result:
            last_tool = [m for m in messages if m.get("role") == "tool"][-1]["content"]
            yield {"type": "done", "response": {
                "content": "MAIN_SUMMARY 汇总：{}".format(last_tool[:80]),
                "tool_calls": [],
            }}
            return

        yield {"type": "done", "response": {"content": "普通回答", "tool_calls": []}}
        return


def patch_model():
    import model as model_mod
    model_mod.DeepSeekModel = MockModel
    import subagent as sa
    sa_sub_globals = SubAgent.__init__.__globals__
    sa_sub_globals["DeepSeekModel"] = MockModel


def test_1_task_tool_registered():
    print("\n[测试1] SubAgent 工具是否注册进主 Agent 工具集")
    import agent as agent_mod
    agent_mod.DeepSeekModel = MockModel
    a = agent_mod.ReActAgent(api_key="fake", workspace_root="./workspace", enable_subagent=True)
    names = list(a.tool_manager.tools.keys())
    print("  主Agent工具: {}".format(names))
    assert "SubAgent" in names, "SubAgent 工具未注册！"
    print("  ✅ 通过")


def test_2_subagent_dispatch_and_isolated_context():
    print("\n[测试2] 派遣子智能体 + 上下文隔离 + 结果回传")
    import agent as agent_mod
    agent_mod.DeepSeekModel = MockModel
    patch_model()
    MockModel.call_log = []

    a = agent_mod.ReActAgent(api_key="fake", workspace_root="./workspace",
                             max_iterations=5, enable_subagent=True)
    main_msg_count_before = len(a.messages)
    answer = a.chat("帮我派遣一个子智能体去调研", stream=False)

    print("  主Agent最终回答: {}".format(answer))
    print("  调用日志:")
    for c in MockModel.call_log:
        print("    is_subagent={} tools={}".format(c["is_subagent"], c["tools"]))

    sub_calls = [c for c in MockModel.call_log if c["is_subagent"]]
    assert len(sub_calls) >= 1, "子智能体没有被调用！"
    assert "SUBAGENT_RESULT" in answer, "子智能体结论未回传给主Agent，实际: {}".format(answer)
    assert "MAIN_SUMMARY" in answer, "主Agent未汇总子智能体结论"
    # 子智能体的消息历史独立：主 Agent 历史里不应出现子智能体的 system 提示
    for m in a.messages:
        assert "子智能体，由主智能体派遣" not in (m.get("content") or ""), "上下文未隔离！"
    print("  ✅ 通过（主Agent历史 {} 条，未混入子智能体内部消息）".format(len(a.messages)))


def test_3_depth_limit():
    print("\n[测试3] 嵌套深度限制：达到上限后子智能体不再有 SubAgent 工具")
    t_task = TaskTool(api_key="fake", parent_depth=0, max_depth=1, workspace_root="./workspace")
    # depth=1 的子智能体，因 1 < max_depth(1) 不成立，不应注册 SubAgent
    sub = SubAgent(api_key="fake", workspace_root="./workspace", depth=1, max_depth=1)
    names = list(sub.tool_manager.tools.keys())
    print("  depth=1,max_depth=1 子智能体工具: {}".format(names))
    assert "SubAgent" not in names, "深度限制失效，子智能体不应再拥有 SubAgent！"

    # depth=0 派生的 depth=1 子智能体，max_depth=2 时应有 SubAgent
    sub2 = SubAgent(api_key="fake", workspace_root="./workspace", depth=1, max_depth=2)
    names2 = list(sub2.tool_manager.tools.keys())
    print("  depth=1,max_depth=2 子智能体工具: {}".format(names2))
    assert "SubAgent" in names2, "未达深度上限时子智能体应能继续派遣！"
    print("  ✅ 通过")


def test_4_allowed_tools():
    print("\n[测试4] 子智能体工具白名单是否生效")
    sub = SubAgent(api_key="fake", workspace_root="./workspace",
                   allowed_tools=["ls", "grep"], depth=1, max_depth=1)
    names = sorted(sub.tool_manager.tools.keys())
    print("  白名单[ls,grep] → 实际工具: {}".format(names))
    assert set(names) == {"ls", "grep"}, "白名单过滤失效！"
    print("  ✅ 通过")


def test_5_task_exceeds_depth():
    print("\n[测试5] 超深度时 SubAgent 工具应拒绝派遣")
    t = TaskTool(api_key="fake", parent_depth=1, max_depth=1, workspace_root="./workspace")
    r = t.execute(description="再来一个")
    print("  返回: {}".format(r))
    assert "最大子智能体嵌套深度" in r, "深度超限未被拒绝！"
    print("  ✅ 通过")


if __name__ == "__main__":
    print("=" * 60)
    print("SubAgent 机制离线验证")
    print("=" * 60)
    test_1_task_tool_registered()
    test_2_subagent_dispatch_and_isolated_context()
    test_3_depth_limit()
    test_4_allowed_tools()
    test_5_task_exceeds_depth()
    print("\n" + "=" * 60)
    print("🎉 全部测试通过！SubAgent 功能工作正常")
    print("=" * 60)
