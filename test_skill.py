# -*- coding: utf-8 -*-
"""
离线验证脚本：不调用真实 DeepSeek API，验证 Skill 技能机制。
验证点：
  1. 技能目录扫描、frontmatter 解析、prompt 渲染
  2. read_file 工具可读取技能目录，但拒绝越界路径
  3. 主 Agent 系统提示注入技能清单，模型能按路径加载技能并得到正文
  4. 没有技能时不注入 skill-system 提示
  5. 子智能体继承技能；白名单禁用 read_file 时不注入技能清单
"""
import os
import sys
import json
import re
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(VENDOR):
    sys.path.insert(0, VENDOR)

import model as model_mod
from skill import SkillManager, SKILL_SYSTEM_PROMPT
from tool import ReadFileTool, ToolManager, build_parameters


def make_skills(base):
    """构造一个临时技能目录，返回路径"""
    skills_dir = os.path.join(base, "skills")
    os.makedirs(skills_dir)

    # 标准技能：frontmatter + 引用资源
    demo = os.path.join(skills_dir, "demo")
    os.makedirs(os.path.join(demo, "references"))
    with open(os.path.join(demo, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(
            "---\n"
            "name: demo-skill\n"
            "description: 演示技能，用于离线测试。\n"
            "---\n\n"
            "# Python 调试工作流\n\n"
            "1. 读取 references/checklist.md\n"
        )
    with open(os.path.join(demo, "references", "checklist.md"), "w", encoding="utf-8") as f:
        f.write("# 检查清单\n- 已复现\n")

    # 无 frontmatter：名称回退为目录名，描述回退为正文首行
    plain = os.path.join(skills_dir, "plain")
    os.makedirs(plain)
    with open(os.path.join(plain, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("# 标题\n\n这是一个没有 frontmatter 的技能说明。\n")

    # 折叠块描述
    folded = os.path.join(skills_dir, "folded")
    os.makedirs(folded)
    with open(os.path.join(folded, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(
            "---\n"
            "name: folded-skill\n"
            "description: >\n"
            "  第一行描述，\n"
            "  第二行描述。\n"
            "---\n\n"
            "# 正文\n"
        )

    # 没有 SKILL.md 的目录应被忽略
    os.makedirs(os.path.join(skills_dir, "not-a-skill"))
    with open(os.path.join(skills_dir, "not-a-skill", "README.md"), "w", encoding="utf-8") as f:
        f.write("普通文件\n")

    return skills_dir


def test_1_skill_discovery_and_prompt():
    print("\n[测试1] 技能扫描 + frontmatter 解析 + prompt 渲染")
    with tempfile.TemporaryDirectory() as base:
        skills_dir = make_skills(base)
        sm = SkillManager(skills_dir)
        names = [s.name for s in sm.skills]
        print("  发现技能: {}".format(names))
        assert names == ["demo-skill", "folded-skill", "plain"], "技能扫描结果不正确: {}".format(names)

        demo = sm.get_skill("demo-skill")
        assert demo.description == "演示技能，用于离线测试。"
        assert os.path.isabs(demo.path) and demo.path.endswith("SKILL.md")
        assert sm.get_skill("folded-skill").description == "第一行描述， 第二行描述。"
        assert sm.get_skill("plain").description == "这是一个没有 frontmatter 的技能说明。"

        prompt = sm.get_prompt()
        assert SKILL_SYSTEM_PROMPT in prompt, "缺少 skill-system 提示"
        assert "<available_skills>" in prompt and "</available_skills>" in prompt
        assert '<skill name="demo-skill" path="{}">'.format(demo.path) in prompt, "技能标签格式不正确"
        print("  prompt 长度: {} 字符".format(len(prompt)))
        print("  ✅ 通过")


def test_2_read_file_tool():
    print("\n[测试2] read_file：读取技能目录 + 分页 + 越界拦截")
    with tempfile.TemporaryDirectory() as base:
        skills_dir = make_skills(base)
        workspace = os.path.join(base, "workspace")
        os.makedirs(workspace)
        with open(os.path.join(workspace, "a.txt"), "w", encoding="utf-8") as f:
            f.write("l1\nl2\nl3\nl4\nl5\n")

        rt = ReadFileTool(workspace, extra_roots=[skills_dir])
        skill_path = os.path.join(skills_dir, "demo", "SKILL.md")

        result = rt.execute(path=skill_path)
        assert "Python 调试工作流" in result, "无法读取技能主文件"
        assert "共 8 行" in result, "行数统计不正确: {}".format(result)

        result = rt.execute(path=skill_path, offset=5, limit=2)
        print("  分页结果片段: {}".format(result.splitlines()[0]))
        assert "显示 5-6 行" in result, "分页不正确: {}".format(result)

        result = rt.execute(path="a.txt", offset=2, limit=2)
        assert "2| l2" in result and "3| l3" in result, "相对路径分页读取失败"
        assert "还有 2 行" in result

        assert "路径越界" in rt.execute(path="/etc/passwd"), "越界路径未被拦截"
        assert "这是一个目录" in rt.execute(path=os.path.join(skills_dir, "demo")), "目录未被识别"
        assert "文件不存在" in rt.execute(path=os.path.join(workspace, "nope.md"))

        schema = rt.get_schema()["function"]["parameters"]
        assert schema["required"] == ["path"], "read_file schema required 不正确"
        assert "required" not in schema["properties"]["path"], "schema 泄漏了非标准 required 标记"
        json.dumps(rt.get_schema())
        print("  ✅ 通过")


class SkillMockModel:
    """模拟：从系统提示里找到技能路径 → 调用 read_file → 根据结果收尾"""

    def __init__(self, api_key, base_url, model):
        self.model = model

    def chat(self, messages, tools=None, temperature=0.7, stream=False):
        system = messages[0]["content"]
        tool_results = [m for m in messages if m.get("role") == "tool"]

        if tool_results:
            content = tool_results[-1]["content"]
            answer = "SKILL_LOADED" if "Python 调试工作流" in content else "NO_SKILL_CONTENT"
            yield {"type": "done", "response": {"content": answer, "tool_calls": []}}
            return

        match = re.search(r'<skill name="demo-skill" path="([^"]+)"', system)
        if match:
            yield {"type": "done", "response": {
                "content": "",
                "tool_calls": [{
                    "id": "call_skill",
                    "name": "read_file",
                    "arguments": json.dumps({"path": match.group(1)}, ensure_ascii=False),
                }],
            }}
            return

        yield {"type": "done", "response": {"content": "NO_SKILL_IN_PROMPT", "tool_calls": []}}


def test_3_agent_loads_skill_end_to_end():
    print("\n[测试3] 主 Agent 注入技能 → 模型 read_file 加载 → 拿到正文")
    import agent as agent_mod

    original_model = agent_mod.DeepSeekModel
    agent_mod.DeepSeekModel = SkillMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            skills_dir = make_skills(base)
            workspace = os.path.join(base, "workspace")
            a = agent_mod.ReActAgent(
                api_key="fake", workspace_root=workspace,
                enable_subagent=False, skills_dir=skills_dir, max_iterations=5,
            )
            system = a.messages[0]["content"]
            assert "<skill-system>" in system and 'name="demo-skill"' in system, "技能未注入系统提示"

            answer = a.chat("帮我用演示技能处理一下", stream=False)
            print("  Agent 回答: {}".format(answer))
            assert answer == "SKILL_LOADED", "技能正文未成功加载，实际: {}".format(answer)
        print("  ✅ 通过")
    finally:
        agent_mod.DeepSeekModel = original_model


def test_4_no_skills_no_prompt():
    print("\n[测试4] 空技能目录：不注入 skill-system，read_file 仍可用")
    import agent as agent_mod

    original_model = agent_mod.DeepSeekModel
    agent_mod.DeepSeekModel = SkillMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            empty_skills = os.path.join(base, "skills")
            os.makedirs(empty_skills)
            a = agent_mod.ReActAgent(
                api_key="fake", workspace_root=os.path.join(base, "workspace"),
                enable_subagent=False, skills_dir=empty_skills,
            )
            system = a.messages[0]["content"]
            assert "<skill-system>" not in system, "没有技能时不应注入 skill-system"
            assert "read_file" in a.tool_manager.tools, "read_file 工具应始终可用"
        print("  ✅ 通过")
    finally:
        agent_mod.DeepSeekModel = original_model


def test_5_subagent_inherits_skills():
    print("\n[测试5] 子智能体继承技能；白名单禁用 read_file 时不注入")
    from subagent import SubAgent

    original_model = model_mod.DeepSeekModel
    model_mod.DeepSeekModel = SkillMockModel
    try:
        with tempfile.TemporaryDirectory() as base:
            skills_dir = make_skills(base)
            workspace = os.path.join(base, "workspace")

            sub = SubAgent(api_key="fake", workspace_root=workspace, skills_dir=skills_dir,
                           depth=1, max_depth=1)
            assert "<skill-system>" in sub.messages[0]["content"], "子智能体未注入技能"

            limited = SubAgent(api_key="fake", workspace_root=workspace, skills_dir=skills_dir,
                               allowed_tools=["ls"], depth=1, max_depth=1)
            assert "read_file" not in limited.tool_manager.tools
            assert "<skill-system>" not in limited.messages[0]["content"], \
                "无 read_file 时不应注入技能清单"
        print("  ✅ 通过")
    finally:
        model_mod.DeepSeekModel = original_model


def test_6_tool_schemas_valid():
    print("\n[测试6] 所有工具 schema 合法（无布尔 required）")
    tm = ToolManager("./workspace", skills_dir="./skills")
    bad = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "required" and not isinstance(v, list):
                    bad.append("{}.{}".format(path, k))
                walk(v, "{}.{}".format(path, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "{}[{}]".format(path, i))

    names = []
    for schema in tm.get_tool_schemas():
        names.append(schema["function"]["name"])
        json.dumps(schema)
        walk(schema, schema["function"]["name"])

    print("  工具: {}".format(names))
    assert "read_file" in names
    assert not bad, "非法 required 字段: {}".format(bad)
    print("  ✅ 通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Skill 技能机制离线验证")
    print("=" * 60)
    test_1_skill_discovery_and_prompt()
    test_2_read_file_tool()
    test_3_agent_loads_skill_end_to_end()
    test_4_no_skills_no_prompt()
    test_5_subagent_inherits_skills()
    test_6_tool_schemas_valid()
    print("\n" + "=" * 60)
    print("🎉 全部测试通过！Skill 功能工作正常")
    print("=" * 60)
