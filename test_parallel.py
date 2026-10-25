# -*- coding: utf-8 -*-
"""
离线验证脚本：工具与子智能体的并行执行。
验证点：
  1. run_parallel 真并发执行，且结果按输入顺序返回
  2. 主 Agent 一轮多个工具调用并行执行（Barrier 证明同时开始）
  3. 主 Agent 一轮多个 SubAgent 调用时，子智能体并行工作（时间区间重叠）
"""
import os
import sys
import json
import time
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import tool as tool_mod
import model as model_mod
import agent as agent_mod
from tool import BaseTool, run_parallel


class DummyModel:
    def __init__(self, *args, **kwargs):
        pass


def test_1_run_parallel():
    print("\n[测试1] run_parallel：真并发 + 保持输入顺序")
    parties = 3
    barrier = threading.Barrier(parties, timeout=2)
    delays = {0: 0.12, 1: 0.06, 2: 0.0}

    def task(i):
        # 串行执行时第一个任务会等满2秒并抛BrokenBarrierError
        barrier.wait()
        time.sleep(delays[i])
        return i

    start = time.time()
    results = run_parallel([lambda i=i: task(i) for i in range(parties)])
    elapsed = time.time() - start

    print("  结果: {}，耗时 {:.3f}s".format(results, elapsed))
    assert results == [0, 1, 2], "结果顺序未按输入保持"
    assert elapsed < 1.0, "执行过慢，可能没有并发"

    assert run_parallel([]) == []
    assert run_parallel([lambda: "single"]) == ["single"]
    print("  ✅ 通过")


class SlowTool(BaseTool):
    """等待所有调用到齐后才返回，用来证明并发执行"""
    name = "slow"
    description = "测试用慢工具"
    parameters = {
        "label": {"type": "string", "description": "标签", "required": True},
        "delay": {"type": "number", "description": "返回前额外等待秒数", "required": False},
    }

    def __init__(self, parties=2):
        self.barrier = threading.Barrier(parties, timeout=2)
        self.lock = threading.Lock()
        self.starts = []

    def execute(self, label, delay=0, **kwargs):
        with self.lock:
            self.starts.append((label, time.time()))
        self.barrier.wait()
        time.sleep(float(delay))
        return "done:" + label


class TwoToolMockModel:
    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "slow",
                     "arguments": json.dumps({"label": "first", "delay": 0.2})},
                    {"id": "c2", "name": "slow",
                     "arguments": json.dumps({"label": "second", "delay": 0})},
                ],
            }}
            return

        contents = [m["content"] for m in tool_messages]
        ok = contents == ["done:first", "done:second"]
        yield {"type": "done", "response": {
            "content": "MAIN_PARALLEL_OK" if ok else "MAIN_PARALLEL_BAD: {}".format(contents),
            "tool_calls": [],
        }}


def test_2_agent_parallel_tools():
    print("\n[测试2] 主 Agent：一轮多个工具调用并行执行")
    original = agent_mod.DeepSeekModel
    agent_mod.DeepSeekModel = TwoToolMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            a = agent_mod.ReActAgent(
                api_key="fake",
                workspace_root=os.path.join(base, "workspace"),
                enable_subagent=False,
                enable_skills=False,
                max_iterations=5,
            )
            slow = SlowTool(parties=2)
            a.tool_manager.register_tool(slow)
            answer = a.chat("并行调用两个慢工具", stream=False)

            print("  Agent 回答: {}".format(answer))
            print("  工具开始时间: {}".format([(label, round(t, 3)) for label, t in slow.starts]))
            assert answer == "MAIN_PARALLEL_OK", answer
            assert len(slow.starts) == 2, "两个工具未都执行"
            # 两个工具都在 barrier 等待中被释放，说明确实是同时执行的
            assert abs(slow.starts[0][1] - slow.starts[1][1]) < 1.0
        print("  ✅ 通过")
    finally:
        agent_mod.DeepSeekModel = original


class ParallelTaskMockModel:
    """主Agent一轮派两个SubAgent；子智能体各睡眠0.3秒，记录起止时间"""
    log = []
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        system = messages[0].get("content", "")
        is_sub = "嵌套层级: 第" in system and "请直接开始执行任务" in system
        user_msg = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")

        if is_sub:
            start = time.time()
            time.sleep(0.3)
            end = time.time()
            with ParallelTaskMockModel.lock:
                ParallelTaskMockModel.log.append((user_msg, start, end))
            yield {"type": "done", "response": {
                "content": "SUB_RESULT:" + user_msg,
                "tool_calls": [],
            }}
            return

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [
                    {"id": "task_a", "name": "SubAgent",
                     "arguments": json.dumps({"description": "任务A", "role": "general"},
                                             ensure_ascii=False)},
                    {"id": "task_b", "name": "SubAgent",
                     "arguments": json.dumps({"description": "任务B", "role": "general"},
                                             ensure_ascii=False)},
                ],
            }}
            return

        combined = "\n".join(m["content"] for m in tool_messages)
        ok = "SUB_RESULT:任务A" in combined and "SUB_RESULT:任务B" in combined
        yield {"type": "done", "response": {
            "content": "MAIN_TASKS_OK" if ok else "MAIN_TASKS_BAD",
            "tool_calls": [],
        }}


def test_3_parallel_subagents():
    print("\n[测试3] 主 Agent：一轮多个 SubAgent 调用，子智能体并行工作")
    original_agent_model = agent_mod.DeepSeekModel
    original_model_model = model_mod.DeepSeekModel
    agent_mod.DeepSeekModel = ParallelTaskMockModel
    model_mod.DeepSeekModel = ParallelTaskMockModel
    ParallelTaskMockModel.log = []
    try:
        with tempfile.TemporaryDirectory() as base:
            a = agent_mod.ReActAgent(
                api_key="fake",
                workspace_root=os.path.join(base, "workspace"),
                enable_skills=False,
                max_iterations=5,
            )
            start = time.time()
            answer = a.chat("并行派两个子智能体", stream=False)
            elapsed = time.time() - start

            print("  Agent 回答: {}".format(answer))
            for desc, s, e in ParallelTaskMockModel.log:
                print("  子任务 {!r}: {:.3f} → {:.3f}".format(desc, s - start, e - start))
            print("  总耗时: {:.3f}s".format(elapsed))

            assert answer == "MAIN_TASKS_OK", answer
            assert len(ParallelTaskMockModel.log) == 2, "两个子智能体未都执行"
            descriptions = {entry[0] for entry in ParallelTaskMockModel.log}
            assert descriptions == {"任务A", "任务B"}

            # 时间区间重叠 => 并行；串行时后一个会等前一个结束才开始
            (_, a_start, a_end), (_, b_start, b_end) = sorted(
                ParallelTaskMockModel.log, key=lambda x: x[1])
            overlap = max(a_start, b_start) < min(a_end, b_end)
            assert overlap, "子智能体没有并行执行（时间区间不重叠）"
            assert elapsed < 1.0, "总耗时异常: {:.3f}s".format(elapsed)
        print("  ✅ 通过")
    finally:
        agent_mod.DeepSeekModel = original_agent_model
        model_mod.DeepSeekModel = original_model_model


class SubTwoToolMockModel:
    """子智能体mock：一轮返回两个慢工具调用"""

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "slow",
                     "arguments": json.dumps({"label": "first", "delay": 0.2})},
                    {"id": "c2", "name": "slow",
                     "arguments": json.dumps({"label": "second", "delay": 0})},
                ],
            }}
            return

        contents = [m["content"] for m in tool_messages]
        ok = contents == ["done:first", "done:second"]
        yield {"type": "done", "response": {
            "content": "SUB_TOOLS_PARALLEL_OK" if ok else "SUB_TOOLS_BAD: {}".format(contents),
            "tool_calls": [],
        }}


def test_4_subagent_internal_parallel_tools():
    print("\n[测试4] 子智能体内部：一轮多个工具调用并行执行")
    from subagent import SubAgent

    original = model_mod.DeepSeekModel
    model_mod.DeepSeekModel = SubTwoToolMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            sub = SubAgent(
                api_key="fake",
                workspace_root=os.path.join(base, "workspace"),
                depth=1,
                max_depth=1,
                verbose=True,
            )
            slow = SlowTool(parties=2)
            sub.tool_manager.register_tool(slow)
            answer = sub.run("并行调用两个慢工具")

            print("  子智能体回答: {}".format(answer))
            print("  工具开始时间: {}".format([(label, round(t, 3)) for label, t in slow.starts]))
            assert answer == "SUB_TOOLS_PARALLEL_OK", answer
            assert len(slow.starts) == 2, "两个工具未都执行"
            assert abs(slow.starts[0][1] - slow.starts[1][1]) < 1.0
        print("  ✅ 通过")
    finally:
        model_mod.DeepSeekModel = original


if __name__ == "__main__":
    print("=" * 60)
    print("并行执行机制离线验证")
    print("=" * 60)
    test_1_run_parallel()
    test_2_agent_parallel_tools()
    test_3_parallel_subagents()
    test_4_subagent_internal_parallel_tools()
    print("\n" + "=" * 60)
    print("🎉 全部测试通过！并行执行工作正常")
    print("=" * 60)
