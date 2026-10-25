# -*- coding: utf-8 -*-
"""
SubAgent 子智能体模块

在主 Agent 之外，提供可派遣的独立子智能体（Sub Agent），特性：
- 独立上下文：子智能体拥有自己独立的消息历史，不污染主会话
- 独立工具集：可限制子智能体只用部分工具（例如只允许只读工具）
- 递归控制：通过 depth / max_depth 限制嵌套层数，避免无限递归
- 静默执行：子智能体的中间思考过程默认不打印，只把最终结论回传给主 Agent
- 系统提示隔离：每个子智能体可有专属 system prompt / 角色定义
- 结果回传：将子智能体最终回答作为 SubAgent 工具的 observation 返回给主 Agent
"""

import threading
from typing import List, Dict, Any, Optional

from tool import ToolManager, Colors, cprint, build_parameters, run_parallel
from skill import SkillManager


# 子智能体可用的预设角色（内置模板），也可由调用方自定义
PRESET_ROLES = {
    "general": (
        "你是一个子智能体，由主智能体派遣来完成一项具体任务。\n"
        "你拥有独立的工作上下文，请专注完成交给你的这项任务。\n"
        "完成后，请给出清晰、准确、可直接使用的最终结论。"
    ),
    "researcher": (
        "你是一个调研型子智能体，专注信息检索与汇总。\n"
        "请充分使用搜索、文件读取等工具收集信息，交叉验证后给出结论。\n"
        "输出时请标注信息来源，不要编造。"
    ),
    "coder": (
        "你是一个编码型子智能体，专注代码编写与调试。\n"
        "请先阅读相关代码理解上下文，再动手修改，修改后尽量验证。\n"
        "输出时说明你改了哪些文件、关键改动点是什么。"
    ),
    "explorer": (
        "你是一个探索型子智能体，专注快速摸清代码库或目录结构。\n"
        "请用 ls、grep、bash 等工具高效定位关键文件与实现。\n"
        "输出时给出结构化的发现摘要，注明关键文件路径。"
    ),
}


class SubAgent:
    """子智能体：拥有独立上下文与工具集的迷你 ReAct Agent"""

    # 全局计数器，用于生成子智能体 ID（并行派遣时需加锁）
    _counter = 0
    _counter_lock = threading.Lock()

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat",
        workspace_root: str = ".",
        tavily_api_key: Optional[str] = None,
        role: str = "general",
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        max_iterations: int = 12,
        depth: int = 1,
        max_depth: int = 2,
        verbose: bool = False,
        name: Optional[str] = None,
        skills_dir: Optional[str] = None,
    ):
        # 延迟导入，避免与 agent.py 形成循环依赖
        from model import DeepSeekModel

        with SubAgent._counter_lock:
            SubAgent._counter += 1
            sub_number = SubAgent._counter
        self.id = "sub_{}".format(sub_number)
        self.name = name or self.id
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.workspace_root = workspace_root
        self.tavily_api_key = tavily_api_key
        self.skills_dir = skills_dir

        self.role = role if role in PRESET_ROLES else "general"
        self.custom_system_prompt = system_prompt
        self.allowed_tools = allowed_tools  # None 表示全部工具
        self.max_iterations = max_iterations
        self.depth = depth
        self.max_depth = max_depth
        self.verbose = verbose

        # 子智能体使用独立的模型实例与工具管理器
        self.model = DeepSeekModel(api_key, base_url, model_name)
        self.tool_manager = self._build_tool_manager()
        self.skill_manager = SkillManager(self.skills_dir) if self.skills_dir else None

        # 独立的消息历史
        self.messages: List[Dict[str, Any]] = []
        self._init_messages()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def _build_tool_manager(self) -> ToolManager:
        """构建子智能体的工具管理器，并按 allowed_tools 过滤"""
        tm = ToolManager(self.workspace_root, self.tavily_api_key, skills_dir=self.skills_dir)

        if self.allowed_tools is not None:
            allowed = set(self.allowed_tools)
            for tname in list(tm.tools.keys()):
                if tname not in allowed:
                    tm.tools.pop(tname, None)

        # 只有当还有嵌套余量时，才给子智能体注册 SubAgent 工具（实现递归 subagent）
        if self.depth < self.max_depth:
            subagent_tool = TaskTool(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                workspace_root=self.workspace_root,
                tavily_api_key=self.tavily_api_key,
                parent_depth=self.depth,
                max_depth=self.max_depth,
                allowed_tools=self.allowed_tools,
                skills_dir=self.skills_dir,
            )
            tm.register_tool(subagent_tool)
        else:
            # 达到最大嵌套深度，移除 SubAgent 工具，防止无限递归
            tm.tools.pop("SubAgent", None)

        return tm

    def _init_messages(self) -> None:
        """初始化子智能体的系统提示"""
        base_prompt = self.custom_system_prompt or PRESET_ROLES.get(self.role, PRESET_ROLES["general"])
        tool_names = ", ".join(self.tool_manager.tools.keys())
        suffix = (
            "\n\n可用工具: {tools}\n"
            "工作目录: {ws}\n"
            "嵌套层级: 第 {depth} 层（最大 {max_depth} 层）\n"
            "请直接开始执行任务，完成后输出最终结论，不要输出多余寒暄。"
        ).format(tools=tool_names, ws=self.workspace_root, depth=self.depth, max_depth=self.max_depth)

        content = base_prompt + suffix
        # 只有子智能体拥有 read_file 工具时才注入技能清单（白名单可能限制掉）
        if self.skill_manager is not None and "read_file" in self.tool_manager.tools:
            skill_prompt = self.skill_manager.get_prompt()
            if skill_prompt:
                content = content + "\n\n" + skill_prompt

        self.messages = [{"role": "system", "content": content}]

    # ------------------------------------------------------------------ #
    # ReAct 循环
    # ------------------------------------------------------------------ #
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """复用主 Agent 的消息校验逻辑（保证符合 DeepSeek API 要求）"""
        from agent import ReActAgent
        return ReActAgent._validate_messages(self, messages)  # type: ignore

    def _run_single_step(self) -> Dict[str, Any]:
        """执行单轮推理（子智能体默认非流式，保持输出干净）"""
        messages = self._validate_messages([dict(m) for m in self.messages])
        tools = self.tool_manager.get_tool_schemas()

        full_response: Dict[str, Any] = {"content": "", "tool_calls": []}
        for event in self.model.chat(messages, tools=tools, stream=False):
            if event["type"] == "done":
                full_response = event["response"]
        return full_response

    def _execute_tool_call(self, tc: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个工具调用，返回对应的tool消息"""
        if self.verbose:
            cprint("   ↳ 子智能体 [{}] 执行工具: {}".format(self.name, tc["name"]), Colors.MAGENTA)

        try:
            tool_result = self.tool_manager.execute_tool(tc["name"], tc["arguments"])
        except Exception as e:
            tool_result = "工具执行异常: {}".format(str(e))

        return {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(tool_result),
        }

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行一轮内的所有工具调用；多于一个时并行执行，返回顺序与tool_calls一致"""
        if not tool_calls:
            return []
        if len(tool_calls) == 1:
            return [self._execute_tool_call(tool_calls[0])]

        if self.verbose:
            names = ", ".join(tc["name"] for tc in tool_calls)
            cprint("   ↳ 子智能体 [{}] 并行执行 {} 个工具: {}".format(
                self.name, len(tool_calls), names), Colors.MAGENTA)
        funcs = [(lambda tc=tc: self._execute_tool_call(tc)) for tc in tool_calls]
        return run_parallel(funcs)

    def run(self, task_description: str) -> str:
        """
        执行一个任务，返回最终结论字符串。
        这是子智能体对外的唯一入口，可被 SubAgent 工具调用。
        """
        self.messages.append({"role": "user", "content": task_description})

        final_answer = ""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            try:
                response = self._run_single_step()
            except Exception as e:
                return "❌ 子智能体 [{}] 调用模型失败: {}".format(self.name, str(e))

            tool_calls = response.get("tool_calls", [])
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response.get("content", "")}

            if not tool_calls:
                final_answer = response.get("content", "")
                self.messages.append(assistant_msg)
                break

            # 处理工具调用
            assistant_msg["content"] = response.get("content") or "我需要调用工具来完成这个任务"
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
            tool_messages = self._execute_tool_calls(tool_calls)

            self.messages.append(assistant_msg)
            for tm in tool_messages:
                self.messages.append(tm)
        else:
            final_answer = final_answer or "⚠️ 子智能体 [{}] 达到最大迭代次数（{}）仍未完成".format(
                self.name, self.max_iterations
            )

        self.tool_manager.reset_todo()
        return final_answer


class TaskTool:
    """
    SubAgent 工具：供主 Agent（或上层子 Agent）派遣子智能体。

    模型通过该工具声明：要做什么(description)、用哪个角色(role)、
    以及可选的专属提示与工具白名单。工具负责创建子智能体、执行、
    并把最终结论作为 observation 返回。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat",
        workspace_root: str = ".",
        tavily_api_key: Optional[str] = None,
        parent_depth: int = 0,
        max_depth: int = 2,
        allowed_tools: Optional[List[str]] = None,
        verbose: bool = False,
        skills_dir: Optional[str] = None,
    ):
        self.name = "SubAgent"
        self.description = (
            "派遣一个子智能体（SubAgent）去独立完成一项子任务。子智能体拥有独立上下文，"
            "可自行调用工具，最终只把结论回传给你。适合：需要大量检索/探索的调研、"
            "需要独立完成的编码子任务、可并行拆分的复杂任务。"
            "注意：子智能体看不到你和用户的对话历史，description 必须自包含、说清背景与验收标准。"
        )
        self.parameters = {
            "description": {
                "type": "string",
                "description": "交给子智能体的任务说明，必须自包含（背景+目标+要求+验收标准）",
                "required": True,
            },
            "role": {
                "type": "string",
                "enum": ["general", "researcher", "coder", "explorer"],
                "description": "子智能体角色：general通用/researcher调研/coder编码/explorer探索，默认general",
                "required": False,
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "限制子智能体可用的工具名列表，如['ls','grep','bash']；不传则用全部工具",
                "required": False,
            },
            "system_prompt": {
                "type": "string",
                "description": "可选的专属系统提示，用于覆盖默认角色设定",
                "required": False,
            },
        }

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.workspace_root = workspace_root
        self.tavily_api_key = tavily_api_key
        self.parent_depth = parent_depth
        self.max_depth = max_depth
        self.allowed_tools = allowed_tools
        self.verbose = verbose
        self.skills_dir = skills_dir

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": build_parameters(self.parameters),
            },
        }

    def execute(
        self,
        description: str,
        role: str = "general",
        allowed_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """创建并运行子智能体，返回其最终结论"""
        child_depth = self.parent_depth + 1

        if child_depth > self.max_depth:
            return "❌ 已达最大子智能体嵌套深度（{}），无法继续派遣".format(self.max_depth)

        # 工具白名单：取「调用方指定」与「上层限制」的交集
        effective_tools = allowed_tools
        if self.allowed_tools is not None:
            if effective_tools is None:
                effective_tools = list(self.allowed_tools)
            else:
                effective_tools = [t for t in effective_tools if t in self.allowed_tools]

        cprint(
            "\n🚀 派遣子智能体 [{}] 层级={} 角色={}".format(self.name, child_depth, role),
            Colors.CYAN,
        )
        cprint("   任务: {}".format(description[:200] + ("..." if len(description) > 200 else "")), Colors.CYAN)

        sub = SubAgent(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            workspace_root=self.workspace_root,
            tavily_api_key=self.tavily_api_key,
            role=role,
            system_prompt=system_prompt,
            allowed_tools=effective_tools,
            max_iterations=30,
            depth=child_depth,
            max_depth=self.max_depth,
            verbose=self.verbose,
            skills_dir=self.skills_dir,
        )

        try:
            result = sub.run(description)
        except Exception as e:
            return "❌ 子智能体执行失败: {}".format(str(e))

        cprint("✅ 子智能体 [{}] 完成，返回 {} 字结论".format(sub.name, len(result)), Colors.GREEN)
        return "【子智能体 {} 的结论】\n{}".format(sub.name, result)
