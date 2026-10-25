import os
import sys
import json
import argparse
from typing import List, Dict, Any
from datetime import datetime

from model import DeepSeekModel
from tool import ToolManager, Colors, cprint, run_parallel
from skill import SkillManager, default_skills_dir


SYSTEM_PROMPT = """你是一个有用的AI助手，运行在用户的本地电脑上，可以通过工具帮助用户完成各种任务。

你可以使用以下工具：
- todo: 创建和管理任务计划，开始复杂任务前必须先制定计划
- read_file: 读取文件内容（支持分页），读取技能文件时使用
- bash: 执行终端命令
- edit: 精确编辑文件
- ls: 列出目录内容
- grep: 在文件中搜索内容
- websearch: 搜索互联网上的信息
- webfetch: 抓取指定URL的网页并转换为可读文本
- SubAgent: 派遣子智能体（SubAgent）独立完成一项子任务，返回其结论

工作原则：
1. 处理复杂任务时，先调用todo制定详细计划，再逐步执行
2. 每完成一个步骤，调用todo更新任务状态，将完成的任务标记为completed，下一个任务标记为in_progress
3. 所有文件操作都在当前工作目录下进行，不要访问工作目录外的文件
4. 编辑文件时，old_text必须精确匹配，包含正确的缩进和换行
5. 如果命令执行出错，仔细分析错误原因，调整后重试，不要反复执行同样的错误命令
6. 回答要简洁准确，优先用工具结果说话
7. 所有任务完成后，给用户一个清晰的总结

关于子智能体（SubAgent工具）：
- 当某项子任务需要大量检索/探索、或可以独立完成时，用 SubAgent 工具派遣子智能体，避免占用你的主上下文
- 子智能体看不到你和用户的对话历史，任务的 description 必须自包含：写清背景、目标、要求与验收标准
- 可通过 role 指定角色（general/researcher/coder/explorer），通过 allowed_tools 限制其可用工具
- 子智能体只回传最终结论，你负责验收并把结果汇总给用户
- 任务可以拆分成多个独立子任务时 → 每个子任务派一个，并行执行

当前工作目录：{workspace_root}
当前时间：{current_time}

"""


def _normalize_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把内部工具调用格式统一成API要求的tool_calls格式（模块级，SubAgent也可复用）"""
    normalized = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        function = tc.get("function") or {}
        name = function.get("name") or tc.get("name") or ""
        arguments = function.get("arguments")
        if arguments is None:
            arguments = tc.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append({
            "id": tc.get("id") or "",
            "type": "function",
            "function": {"name": name, "arguments": arguments}
        })
    return normalized


class ReActAgent:
    """ReAct Agent主类，实现思考-行动-观察循环"""

    def __init__(
        self,
        api_key: str,
        workspace_root: str = ".",
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat",
        tavily_api_key: str = None,
        max_iterations: int = 30,
        enable_subagent: bool = True,
        max_subagent_depth: int = 2,
        subagent_allowed_tools: List[str] = None,
        subagent_verbose: bool = False,
        skills_dir: str = None,
        enable_skills: bool = True,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        os.makedirs(self.workspace_root, exist_ok=True)

        # 初始化技能目录
        self.enable_skills = enable_skills
        self.skills_dir = None
        self.skill_manager = None
        if enable_skills:
            self.skills_dir = os.path.abspath(skills_dir) if skills_dir else default_skills_dir()
            self.skill_manager = SkillManager(self.skills_dir)

        # 初始化模型和工具
        self.model = DeepSeekModel(api_key, base_url, model_name)
        self.tool_manager = ToolManager(self.workspace_root, tavily_api_key, skills_dir=self.skills_dir)

        # 子智能体配置
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.tavily_api_key = tavily_api_key
        self.enable_subagent = enable_subagent
        self.max_subagent_depth = max_subagent_depth
        self.subagent_allowed_tools = subagent_allowed_tools
        self.subagent_verbose = subagent_verbose

        # 注册 SubAgent 工具（派遣子智能体）
        self.task_tool = None
        if self.enable_subagent:
            self._register_task_tool()

        self.max_iterations = max_iterations
        self.messages: List[Dict[str, Any]] = []
        self._init_messages()

    def _register_task_tool(self):
        """注册 SubAgent 工具，用于派遣子智能体"""
        from subagent import TaskTool

        self.task_tool = TaskTool(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            workspace_root=self.workspace_root,
            tavily_api_key=self.tavily_api_key,
            parent_depth=0,
            max_depth=self.max_subagent_depth,
            allowed_tools=self.subagent_allowed_tools,
            verbose=self.subagent_verbose,
            skills_dir=self.skills_dir,
        )
        self.tool_manager.register_tool(self.task_tool)

    def _init_messages(self):
        """初始化系统提示"""
        content = SYSTEM_PROMPT.format(
            workspace_root=self.workspace_root,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        skill_prompt = self.skill_manager.get_prompt() if self.skill_manager else ""
        if skill_prompt:
            content = content + "\n\n" + skill_prompt
        self.messages = [{"role": "system", "content": content}]

    def _get_todo_prompt(self) -> str:
        """获取当前Todo状态提示"""
        return self.tool_manager.todo_tool.get_status_prompt()

    def reset(self):
        """重置对话历史和Todo状态"""
        self._init_messages()
        self.tool_manager.reset_todo()
        cprint("\n🔄 对话已重置", Colors.CYAN)

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        校验并修复消息格式，确保符合DeepSeek API要求。

        核心约束：assistant 消息带 tool_calls 时，后面必须紧跟着为每个
        tool_call_id 各提供一条 tool 消息。因此：
        - 绝不能在这类 assistant 与其 tool 响应之间插入任何消息；
        - 连续多条 tool 消息是合法的（并行工具调用），必须原样保留；
        - 缺失的 tool 响应要补一条占位消息，孤立的 tool 消息要丢弃。
        """
        # 第一步：清洗字段，统一格式
        cleaned = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content")
            content = "" if content is None else str(content)

            if role == "tool":
                cleaned.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", "") or "",
                    "content": content
                })
            elif role in ("user", "system"):
                cleaned.append({"role": role, "content": content})
            elif role == "assistant":
                cleaned_msg = {"role": "assistant", "content": content}
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    normalized = _normalize_tool_calls(tool_calls)
                    if normalized:
                        cleaned_msg["tool_calls"] = normalized
                        if not content:
                            cleaned_msg["content"] = "我需要调用工具来完成这个任务"
                cleaned.append(cleaned_msg)

        # 第二步：修复 assistant(tool_calls) 与 tool 响应的配对关系
        fixed = []
        i = 0
        while i < len(cleaned):
            msg = cleaned[i]
            role = msg["role"]

            if role == "assistant" and msg.get("tool_calls"):
                # 收集紧随其后的所有 tool 消息
                j = i + 1
                replies = {}
                while j < len(cleaned) and cleaned[j]["role"] == "tool":
                    reply = cleaned[j]
                    replies.setdefault(reply.get("tool_call_id", ""), reply)
                    j += 1

                fixed.append(msg)
                for tc in msg["tool_calls"]:
                    reply = replies.get(tc.get("id", ""))
                    if reply is None:
                        reply = {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": "（该工具调用未返回结果，已跳过）"
                        }
                    fixed.append(reply)
                i = j
                continue

            if role == "tool":
                # 前面没有 assistant tool_calls，属于孤立消息，丢弃
                i += 1
                continue

            if role == "assistant" and fixed and fixed[-1]["role"] == "assistant" \
                    and not fixed[-1].get("tool_calls"):
                fixed[-1] = {
                    "role": "assistant",
                    "content": fixed[-1]["content"] + "\n" + msg["content"]
                }
                i += 1
                continue

            if role == "user" and fixed and fixed[-1]["role"] == "user":
                fixed[-1] = {
                    "role": "user",
                    "content": fixed[-1]["content"] + "\n" + msg["content"]
                }
                i += 1
                continue

            fixed.append(msg)
            i += 1

        return fixed

    def _run_single_step(self, stream: bool = True) -> Dict[str, Any]:
        """运行单轮推理，返回模型响应"""
        # 每次调用前注入Todo状态到系统提示
        todo_prompt = self._get_todo_prompt()
        messages = self.messages.copy()
        if todo_prompt:
            # 在系统提示末尾追加Todo状态
            messages[0] = {
                "role": "system",
                "content": self.messages[0]["content"] + "\n" + todo_prompt
            }

        # 自动校验修复消息格式
        messages = self._validate_messages(messages)

        tools = self.tool_manager.get_tool_schemas()

        full_response = {"content": "", "tool_calls": []}
        current_content = ""
        current_tool_calls: Dict[int, Dict[str, Any]] = {}

        # 流式调用模型
        if stream:
            for event in self.model.chat(messages, tools=tools, stream=True):
                if event["type"] == "content":
                    if not current_content:
                        cprint("\n💭 思考: ", Colors.BLUE, end="")
                    cprint(event["delta"], Colors.BLUE, end="")
                    current_content += event["delta"]
                elif event["type"] == "tool_call_part":
                    idx = event["index"]
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if event.get("delta_name"):
                        current_tool_calls[idx]["name"] += event["delta_name"]
                        cprint(f"\n🔧 调用工具: {Colors.YELLOW}{current_tool_calls[idx]['name']}{Colors.RESET}", end="")
                    if event.get("delta_args"):
                        current_tool_calls[idx]["arguments"] += event["delta_args"]
                elif event["type"] == "done":
                    full_response = event["response"]
        else:
            # 非流式
            for event in self.model.chat(messages, tools=tools, stream=False):
                full_response = event["response"]
            if full_response["content"]:
                cprint("\n💭 思考: " + full_response["content"], Colors.BLUE)

        return full_response

    def _execute_tool_call(self, tc: Dict[str, Any], stream: bool = True,
                           show_name: bool = False) -> Dict[str, Any]:
        """执行单个工具调用，返回对应的tool消息"""
        tc_id = tc["id"]
        tc_name = tc["name"]
        tc_args = tc["arguments"]

        args_display = tc_args
        if len(args_display) > 200:
            args_display = args_display[:200] + "..."
        start_lines = [f"\n⚙️  执行 {tc_name}..."]
        if stream:
            start_lines.append(f"   参数: {args_display}")
        cprint("\n".join(start_lines), Colors.MAGENTA)

        try:
            tool_result = self.tool_manager.execute_tool(tc_name, tc_args)
        except Exception as e:
            tool_result = f"工具执行异常: {str(e)}"

        result_display = str(tool_result)
        if len(result_display) > 500:
            result_display = result_display[:500] + "..."
        header = f"📝 结果 [{tc_name}]:" if show_name else "📝 结果:"
        cprint(f"{header}\n{result_display}", Colors.GREEN)

        # DeepSeek tool消息格式：只需要role/tool_call_id/content，不要带name
        return {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": str(tool_result)
        }

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]],
                            stream: bool = True) -> List[Dict[str, Any]]:
        """执行一轮内的所有工具调用；多于一个时并行执行，返回顺序与tool_calls一致"""
        if not tool_calls:
            return []
        if len(tool_calls) == 1:
            return [self._execute_tool_call(tool_calls[0], stream=stream)]

        names = ", ".join(tc["name"] for tc in tool_calls)
        cprint(f"\n⚡ 并行执行 {len(tool_calls)} 个工具: {names}", Colors.CYAN)
        funcs = [
            (lambda tc=tc: self._execute_tool_call(tc, stream=stream, show_name=True))
            for tc in tool_calls
        ]
        return run_parallel(funcs)

    def chat(self, user_input: str, stream: bool = True) -> str:
        """处理用户输入，执行多轮ReAct循环直到返回最终答案"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        iteration = 0
        final_answer = ""

        while iteration < self.max_iterations:
            iteration += 1

            try:
                response = self._run_single_step(stream=stream)
            except Exception as e:
                error_msg = f"\n❌ 模型调用失败: {str(e)}"
                cprint(error_msg, Colors.RED)
                return f"模型调用失败: {str(e)}"

            # 构建assistant消息
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": response["content"]}
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # 没有工具调用，就是最终回答
                final_answer = response["content"]
                self.messages.append(assistant_msg)
                cprint("\n", end="")
                break

            # 有工具调用，执行工具
            assistant_msg["tool_calls"] = []
            # DeepSeek要求content必须是字符串，不能是null
            assistant_msg["content"] = response["content"] if response.get("content") else "我需要调用工具来完成这个任务"
            if stream and response.get("content"):
                cprint("\n", end="")

            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                }
                for tc in tool_calls
            ]

            tool_messages = self._execute_tool_calls(tool_calls, stream=stream)

            # 添加到消息历史
            self.messages.append(assistant_msg)
            for tm in tool_messages:
                self.messages.append(tm)

            # 检查Todo是否全部完成，如果是的话并且没有更多待办，自动结束循环让模型总结
            if self.tool_manager.todo_tool.all_completed() and len(self.tool_manager.todo_tool.todos) > 0:
                # 不插入额外user消息，直接让下一轮模型自己收尾，避免破坏消息格式
                pass

        else:
            # 达到最大迭代次数
            cprint(f"\n⚠️  达到最大迭代次数({self.max_iterations})，自动结束", Colors.YELLOW)

        return final_answer

    def run_interactive(self, stream: bool = True):
        """启动交互式命令行界面"""
        cprint("=" * 60, Colors.CYAN)
        cprint("🤖 DeepSeek ReAct Agent", Colors.CYAN)
        cprint("=" * 60, Colors.CYAN)
        cprint(f"📂 工作目录: {self.workspace_root}", Colors.CYAN)
        cprint(f"🧠 模型: {self.model.model}", Colors.CYAN)
        sub_status = "已启用" if self.enable_subagent else "已禁用"
        cprint(f"🚀 子智能体: {sub_status}（最大嵌套 {self.max_subagent_depth} 层）", Colors.CYAN)
        if self.skill_manager is not None:
            skill_names = ", ".join(s.name for s in self.skill_manager.skills) or "无"
            cprint(f"🧩 技能: {len(self.skill_manager.skills)} 个（{skill_names}）", Colors.CYAN)
            cprint(f"📂 技能目录: {self.skills_dir}", Colors.CYAN)
        else:
            cprint("🧩 技能: 已禁用", Colors.CYAN)
        cprint("💡 输入 /exit 退出, /clear 清空对话, /reset 重置, /todo 查看任务计划", Colors.CYAN)
        cprint("=" * 60, Colors.CYAN)

        while True:
            try:
                cprint("\n👤 你: ", Colors.BOLD, end="")
                user_input = input().strip()
            except (KeyboardInterrupt, EOFError):
                cprint("\n👋 再见！", Colors.CYAN)
                break

            if not user_input:
                continue

            # 处理斜杠命令
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().split()[0] if user_input[1:] else ""
                if cmd == "exit" or cmd == "quit":
                    cprint("👋 再见！", Colors.CYAN)
                    break
                elif cmd == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                elif cmd == "reset":
                    self.reset()
                    continue
                elif cmd == "todo":
                    todo_status = self._get_todo_prompt()
                    if todo_status:
                        cprint(todo_status, Colors.YELLOW)
                    else:
                        cprint("📋 当前没有任务计划", Colors.YELLOW)
                    continue
                elif cmd == "help":
                    cprint("命令列表:", Colors.CYAN)
                    cprint("  /exit    退出程序", Colors.CYAN)
                    cprint("  /clear   清屏", Colors.CYAN)
                    cprint("  /reset   重置对话和任务", Colors.CYAN)
                    cprint("  /todo    查看当前任务计划", Colors.CYAN)
                    cprint("  /help    显示帮助", Colors.CYAN)
                    continue
                else:
                    cprint(f"❌ 未知命令: {cmd}，输入/help查看帮助", Colors.RED)
                    continue

            # 处理正常对话
            try:
                self.chat(user_input, stream=stream)
            except KeyboardInterrupt:
                cprint("\n⚠️  已中断当前任务", Colors.YELLOW)
                continue
            except Exception as e:
                cprint(f"\n❌ 发生错误: {str(e)}", Colors.RED)
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="DeepSeek ReAct Agent - 支持工具调用的命令行Agent")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"), help="DeepSeek API Key")
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="API基础URL")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"), help="模型名称")
    parser.add_argument("--tavily-key", default=os.environ.get("TAVILY_API_KEY"), help="Tavily搜索API Key")
    parser.add_argument("--workspace", default="./workspace", help="工作目录")
    parser.add_argument("--max-iterations", type=int, default=15, help="单轮对话最大ReAct迭代次数")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument("--no-subagent", action="store_true", help="禁用子智能体（SubAgent工具）")
    parser.add_argument("--max-subagent-depth", type=int, default=2, help="子智能体最大嵌套深度，默认2")
    parser.add_argument("--subagent-tools", default=None,
                        help="限制子智能体可用工具，逗号分隔，例如 ls,grep,bash；不传则用全部工具")
    parser.add_argument("--subagent-verbose", action="store_true", help="显示子智能体内部工具调用过程")
    parser.add_argument("--skills-dir", default=os.environ.get("DEEPSEEK_SKILLS_DIR"),
                        help="技能目录，默认使用项目下的skills/目录")
    parser.add_argument("--no-skills", action="store_true", help="禁用技能功能")

    args = parser.parse_args()

    if not args.api_key:
        cprint("❌ 请设置DEEPSEEK_API_KEY环境变量，或通过--api-key参数传入", Colors.RED)
        cprint("示例: export DEEPSEEK_API_KEY=your-key-here", Colors.YELLOW)
        sys.exit(1)

    subagent_tools = None
    if args.subagent_tools:
        subagent_tools = [t.strip() for t in args.subagent_tools.split(",") if t.strip()]

    agent = ReActAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        workspace_root=args.workspace,
        tavily_api_key=args.tavily_key,
        max_iterations=args.max_iterations,
        enable_subagent=not args.no_subagent,
        max_subagent_depth=args.max_subagent_depth,
        subagent_allowed_tools=subagent_tools,
        subagent_verbose=args.subagent_verbose,
        skills_dir=None if args.no_skills else args.skills_dir,
        enable_skills=not args.no_skills
    )

    agent.run_interactive(stream=not args.no_stream)


if __name__ == "__main__":
    main()
