import subprocess
import os
import re
import json
import shlex
import threading
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # 直接运行 agent.py 且未加载 vendor 目录时
    requests = None


# 单轮内最多并行执行的工具数
MAX_PARALLEL_TOOLS = 8

# 多个工具并发输出时，保证每次打印不被其它线程打断
_PRINT_LOCK = threading.Lock()


# 终端颜色定义
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def cprint(text: str, color: str = Colors.RESET, end: str = "\n"):
    """彩色打印（线程安全）"""
    with _PRINT_LOCK:
        print(f"{color}{text}{Colors.RESET}", end=end, flush=True)


def run_parallel(funcs: List[Callable[[], Any]]) -> List[Any]:
    """
    并行执行一组零参函数，返回值按输入顺序排列（与完成先后无关）。
    单个任务直接同步执行，避免无意义的线程开销。
    """
    if not funcs:
        return []
    if len(funcs) == 1:
        return [funcs[0]()]

    results: List[Any] = [None] * len(funcs)
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_TOOLS, len(funcs))) as executor:
        future_to_index = {
            executor.submit(func): index for index, func in enumerate(funcs)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return results


def is_safe_path(path: str, workspace_root: str) -> bool:
    """检查路径是否在指定目录内，防止路径穿越"""
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(workspace_root)
        return os.path.commonpath([abs_path, abs_root]) == abs_root
    except ValueError:
        # Windows下不同盘符会抛ValueError
        return False


def build_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    把内部参数定义转换为标准 JSON Schema。
    parameters 中每个属性里的 "required": True/False 只是内部标记，
    必须从属性schema中剔除，并汇总为顶层的 required 数组；
    否则会作为非法JSON Schema（required必须是数组）发给API导致400。
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, spec in parameters.items():
        properties[name] = {k: v for k, v in spec.items() if k != "required"}
        if spec.get("required", False):
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


class BaseTool:
    """工具基类"""
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    def get_schema(self) -> Dict[str, Any]:
        """返回OpenAI格式的工具schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": build_parameters(self.parameters)
            }
        }

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class TodoTool(BaseTool):
    """Todo规划工具，类似LangGraph的Todo中间件"""
    name = "todo"
    description = "创建和管理任务计划，在开始复杂任务时必须先调用此工具制定计划，然后逐步执行。每完成一个步骤请更新任务状态。"
    parameters = {
        "todos": {
            "type": "array",
            "description": "任务列表，每个任务包含content和status字段",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "任务内容描述"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "任务状态"
                    }
                },
                "required": ["content", "status"]
            }
        }
    }

    def __init__(self):
        self.todos: List[Dict[str, str]] = []
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.todos = []

    def execute(self, todos: List[Dict[str, str]], **kwargs) -> str:
        # 并行工具调用时可能同时更新计划，加锁避免读到半更新的状态
        with self._lock:
            self.todos = list(todos)
        result = ["📋 当前任务计划:"]
        for i, todo in enumerate(todos, 1):
            status_icon = "⏳" if todo["status"] == "pending" else "🔄" if todo["status"] == "in_progress" else "✅"
            result.append(f"  {i}. {status_icon} [{todo['status']}] {todo['content']}")
        return "\n".join(result)

    def get_status_prompt(self) -> str:
        """返回当前Todo状态，注入到系统提示末尾"""
        if not self.todos:
            return ""
        lines = ["\n--- 当前任务计划 ---"]
        completed = 0
        in_progress = 0
        pending = 0
        for i, todo in enumerate(self.todos, 1):
            status_icon = "⏳" if todo["status"] == "pending" else "🔄" if todo["status"] == "in_progress" else "✅"
            lines.append(f"{i}. {status_icon} [{todo['status']}] {todo['content']}")
            if todo["status"] == "completed":
                completed += 1
            elif todo["status"] == "in_progress":
                in_progress += 1
            else:
                pending += 1
        lines.append(f"--- 进度: {completed}完成/{in_progress}进行中/{pending}待办 ---")
        if in_progress == 0 and pending > 0:
            lines.append("提示: 请将第一个pending任务标记为in_progress，然后开始执行。")
        return "\n".join(lines)

    def all_completed(self) -> bool:
        if not self.todos:
            return True
        return all(t["status"] == "completed" for t in self.todos)


class BashTool(BaseTool):
    """Bash命令执行工具"""
    name = "bash"
    description = "在终端执行bash命令，返回执行结果。可以用来运行程序、安装包、查看文件内容、搜索文件等。"
    parameters = {
        "command": {
            "type": "string",
            "description": "要执行的bash命令",
            "required": True
        },
        "timeout": {
            "type": "integer",
            "description": "超时时间（秒），默认60秒",
            "required": False
        }
    }

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def execute(self, command: str, timeout: int = 60, **kwargs) -> str:
        # 危险命令黑名单
        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:", "chmod -R 777 /", "wget -O- | sh", "curl | sh"]
        for d in dangerous:
            if d in command:
                return f"❌ 禁止执行危险命令: {d}"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if not output:
                output = "(命令执行成功，无输出)"
            if result.returncode != 0:
                output += f"\n[退出码: {result.returncode}]"
            return output[:8000]  # 限制输出长度
        except subprocess.TimeoutExpired:
            return f"⏰ 命令执行超时（{timeout}秒），已自动终止"
        except Exception as e:
            return f"❌ 执行失败: {str(e)}"


class EditTool(BaseTool):
    """文件编辑工具"""
    name = "edit"
    description = "精确编辑文件，将文件中的old_text替换为new_text。必须提供精确匹配的old_text，包含正确的缩进。"
    parameters = {
        "path": {
            "type": "string",
            "description": "要编辑的文件路径（相对或绝对路径，必须在工作目录内）",
            "required": True
        },
        "old_text": {
            "type": "string",
            "description": "要替换的原文片段，必须精确匹配",
            "required": True
        },
        "new_text": {
            "type": "string",
            "description": "替换后的新文本",
            "required": True
        }
    }

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def execute(self, path: str, old_text: str, new_text: str, **kwargs) -> str:
        # 处理相对路径
        if not os.path.isabs(path):
            path = os.path.join(self.workspace_root, path)
        if not is_safe_path(path, self.workspace_root):
            return "❌ 路径越界，只能编辑工作目录内的文件"
        if not os.path.exists(path):
            return f"❌ 文件不存在: {path}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_text not in content:
                # 尝试找相似匹配
                matches = []
                for i in range(min(5, len(content.split("\n")))):
                    line = content.split("\n")[i]
                    if old_text[:20] in line:
                        matches.append(f"行{i+1}: {line[:80]}")
                hint = "\n可能的匹配位置:\n" + "\n".join(matches) if matches else ""
                return f"❌ 未在文件中找到精确匹配的old_text{hint}"
            count = content.count(old_text)
            if count > 1:
                return f"❌ 找到{count}处匹配，请提供更长的old_text确保唯一匹配"
            new_content = content.replace(old_text, new_text, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"✅ 编辑成功，已替换并保存文件 {path}"
        except Exception as e:
            return f"❌ 编辑失败: {str(e)}"


class ReadFileTool(BaseTool):
    """文件读取工具，支持分页，并可额外读取技能目录"""
    name = "read_file"
    description = "读取文本文件内容，返回带行号的结果。支持offset/limit分页。读取技能(Skill)主文件及其引用资源时使用此工具。"
    parameters = {
        "path": {
            "type": "string",
            "description": "要读取的文件路径（相对工作目录或绝对路径，必须在允许的目录内）",
            "required": True
        },
        "offset": {
            "type": "integer",
            "description": "起始行号（从1开始），默认1",
            "required": False
        },
        "limit": {
            "type": "integer",
            "description": "最多读取的行数，默认400",
            "required": False
        }
    }

    MAX_OUTPUT_CHARS = 20000
    MAX_LINE_CHARS = 1000

    def __init__(self, workspace_root: str, extra_roots: Optional[List[str]] = None):
        self.workspace_root = workspace_root
        self.allowed_roots = [workspace_root] + [r for r in (extra_roots or []) if r]

    def execute(self, path: str, offset: int = 1, limit: int = 400, **kwargs) -> str:
        try:
            offset = max(1, int(offset or 1))
            limit = max(1, int(limit or 400))
        except (TypeError, ValueError):
            return "❌ offset/limit 必须是整数"

        if not os.path.isabs(path):
            path = os.path.join(self.workspace_root, path)
        path = os.path.abspath(path)

        if not any(is_safe_path(path, root) for root in self.allowed_roots):
            return "❌ 路径越界，只能读取工作目录或技能目录内的文件"
        if not os.path.exists(path):
            return f"❌ 文件不存在: {path}"
        if os.path.isdir(path):
            return f"❌ 这是一个目录，请用 ls 工具查看: {path}"

        try:
            with open(path, "rb") as f:
                raw = f.read()
            if b"\x00" in raw[:4096]:
                return f"❌ 这是二进制文件，无法按文本读取: {path}"
            text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"❌ 读取失败: {str(e)}"

        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return f"📄 {path} 是空文件"

        if offset > total:
            return f"❌ offset({offset}) 超出文件总行数({total})"

        selected = lines[offset - 1: offset - 1 + limit]
        width = len(str(offset + len(selected) - 1))
        output = [f"📄 {path}（共 {total} 行，显示 {offset}-{offset + len(selected) - 1} 行）:"]
        for lineno, line in enumerate(selected, offset):
            output.append(f"{lineno:>{width}}| {line[:self.MAX_LINE_CHARS]}")

        result = "\n".join(output)
        remaining = total - (offset - 1 + len(selected))
        if len(result) > self.MAX_OUTPUT_CHARS:
            result = result[:self.MAX_OUTPUT_CHARS] + "\n...（输出过长已截断，请用offset/limit分页读取）"
        elif remaining > 0:
            result += f"\n...（还有 {remaining} 行，可使用 offset={offset + len(selected)} 继续读取）"
        return result


class LsTool(BaseTool):
    """目录列表工具"""
    name = "ls"
    description = "列出指定目录下的文件和子目录"
    parameters = {
        "path": {
            "type": "string",
            "description": "目录路径，默认为当前工作目录",
            "required": False
        }
    }

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def execute(self, path: str = ".", **kwargs) -> str:
        if not os.path.isabs(path):
            path = os.path.join(self.workspace_root, path)
        if not is_safe_path(path, self.workspace_root):
            return "❌ 路径越界"
        if not os.path.exists(path):
            return f"❌ 路径不存在: {path}"
        if not os.path.isdir(path):
            return f"❌ 不是目录: {path}"
        try:
            items = sorted(os.listdir(path))
            result = [f"📂 目录内容 {path}:"]
            for item in items:
                if item.startswith("."):
                    continue  # 跳过隐藏文件
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    result.append(f"  📁 {item}/")
                else:
                    size = os.path.getsize(full_path)
                    size_str = f"{size}B" if size < 1024 else f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
                    result.append(f"  📄 {item}  ({size_str})")
            return "\n".join(result) if len(result) > 1 else "目录为空"
        except Exception as e:
            return f"❌ 列出目录失败: {str(e)}"


class GrepTool(BaseTool):
    """Grep内容搜索工具"""
    name = "grep"
    description = "在文件中搜索指定的文本模式，返回匹配的行"
    parameters = {
        "pattern": {
            "type": "string",
            "description": "要搜索的文本或正则表达式",
            "required": True
        },
        "path": {
            "type": "string",
            "description": "要搜索的文件或目录路径，默认为当前目录",
            "required": False
        },
        "include": {
            "type": "string",
            "description": "文件过滤模式，例如 '*.py'",
            "required": False
        }
    }

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root

    def execute(self, pattern: str, path: str = ".", include: str = "*", **kwargs) -> str:
        if not os.path.isabs(path):
            path = os.path.join(self.workspace_root, path)
        if not is_safe_path(path, self.workspace_root):
            return "❌ 路径越界"
        if not os.path.exists(path):
            return f"❌ 路径不存在: {path}"
        matches = []
        try:
            regex = re.compile(pattern)
            if os.path.isfile(path):
                files = [path]
            else:
                files = []
                for root, dirs, filenames in os.walk(path):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                    for f in filenames:
                        if f.startswith("."):
                            continue
                        import fnmatch
                        if include != "*" and not fnmatch.fnmatch(f, include):
                            continue
                        files.append(os.path.join(root, f))
            for filepath in files:
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(filepath, self.workspace_root)
                                matches.append(f"{rel_path}:{i}: {line.rstrip()[:200]}")
                                if len(matches) >= 100:
                                    matches.append("... 结果过多，截断显示前100条")
                                    return "\n".join(matches)
                except Exception:
                    continue
            if not matches:
                return f"未找到匹配 '{pattern}' 的内容"
            return "\n".join(matches)
        except re.error as e:
            return f"❌ 正则表达式错误: {str(e)}"
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"


class WebSearchTool(BaseTool):
    """Tavily联网搜索工具"""
    name = "websearch"
    description = "使用Tavily搜索引擎搜索互联网上的实时信息，适合查找最新资讯、文档、事实。"
    parameters = {
        "query": {
            "type": "string",
            "description": "搜索关键词",
            "required": True
        },
        "max_results": {
            "type": "integer",
            "description": "返回结果数量，默认5条",
            "required": False
        }
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.enabled = bool(self.api_key)

    def execute(self, query: str, max_results: int = 5, **kwargs) -> str:
        if not self.enabled:
            return "❌ Web搜索未配置，请设置TAVILY_API_KEY环境变量"
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic"
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return f"未找到关于 '{query}' 的搜索结果"
            output = [f"🔍 搜索结果: {query}\n"]
            for i, r in enumerate(results, 1):
                output.append(f"{i}. {r['title']}")
                output.append(f"   {r['url']}")
                output.append(f"   {r['content'][:300]}...")
                output.append("")
            return "\n".join(output)
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


def _clean_text(text: str) -> str:
    """压缩空白：行内多空格合一，连续空行最多保留一行"""
    lines = []
    blank = False
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\f\v\xa0]+", " ", raw_line).strip()
        if line:
            lines.append(line)
            blank = False
        elif not blank:
            lines.append("")
            blank = True
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


_META_CHARSET_RE = re.compile(br'<meta[^>]+charset\s*=\s*["\']?\s*([a-zA-Z0-9_\-]+)', re.I)


def _decode_bytes(raw: bytes, header_charset: Optional[str]) -> Tuple[str, str]:
    """
    解码响应字节。优先 header 里的 charset，其次 HTML meta charset，
    然后依次尝试 utf-8、gb18030，最后用 utf-8 容错解码。
    """
    candidates: List[str] = []
    meta_match = _META_CHARSET_RE.search(raw[:4096])
    if meta_match:
        candidates.append(meta_match.group(1).decode("ascii", "ignore"))
    if header_charset:
        candidates.append(header_charset)
    candidates.extend(["utf-8", "gb18030"])

    seen = set()
    for encoding in candidates:
        encoding = encoding.strip().lower()
        if not encoding or encoding in seen:
            continue
        seen.add(encoding)
        try:
            return raw.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue

    return raw.decode("utf-8", errors="replace"), "utf-8"


class _HTMLTextExtractor(HTMLParser):
    """把 HTML 转换为纯文本：去掉脚本/样式，块级标签换行，提取标题"""

    SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "object"}
    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
        "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul"
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "pre":
            self._pre_depth += 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # 自闭合标签（如 <br/>）只按开始标签处理，避免重复换行
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag == "pre" and self._pre_depth:
            self._pre_depth -= 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._pre_depth:
            self.parts.append(data)
        else:
            self.parts.append(re.sub(r"[ \t\r\f\v\xa0]+", " ", data))

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()

    def get_text(self) -> str:
        return _clean_text("".join(self.parts))


class WebFetchTool(BaseTool):
    """网页抓取工具：下载URL并转换为可读文本"""
    name = "webfetch"
    description = (
        "抓取指定URL的网页内容并转换为可读文本，适合阅读在线文档、博客、公告等。"
        "会自动去除HTML标签、脚本和样式，并限制返回长度。"
    )
    parameters = {
        "url": {
            "type": "string",
            "description": "要抓取的完整URL，必须以 http:// 或 https:// 开头",
            "required": True
        },
        "max_chars": {
            "type": "integer",
            "description": "返回文本的最大字符数，默认20000，最大50000",
            "required": False
        }
    }

    TIMEOUT = 30
    MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
    DEFAULT_MAX_CHARS = 20000
    MAX_MAX_CHARS = 50000
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
            "DeepSeekReActAgent/1.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    def execute(self, url: str, max_chars: int = 20000, **kwargs) -> str:
        if requests is None:
            return "❌ requests库未加载，请使用 run.py 启动程序"
        if not isinstance(url, str):
            return "❌ url 必须是字符串"
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "❌ 仅支持 http/https URL，例如 https://example.com"

        try:
            limit = int(max_chars or self.DEFAULT_MAX_CHARS)
        except (TypeError, ValueError):
            limit = self.DEFAULT_MAX_CHARS
        limit = max(500, min(limit, self.MAX_MAX_CHARS))

        try:
            resp = requests.get(
                url,
                headers=self.HEADERS,
                timeout=self.TIMEOUT,
                stream=True,
                allow_redirects=True
            )
        except requests.exceptions.Timeout:
            return f"⏰ 抓取超时（{self.TIMEOUT}秒）: {url}"
        except requests.exceptions.RequestException as e:
            return f"❌ 抓取失败: {str(e)}"

        with resp:
            status = resp.status_code
            if status >= 400:
                return f"❌ 抓取失败: HTTP {status} {getattr(resp, 'reason', '')}\nURL: {url}"

            chunks = []
            downloaded = 0
            download_truncated = False
            try:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    if downloaded >= self.MAX_DOWNLOAD_BYTES:
                        download_truncated = True
                        break
            except Exception as e:
                return f"❌ 读取响应失败: {str(e)}"

            raw = b"".join(chunks)[:self.MAX_DOWNLOAD_BYTES]
            headers = resp.headers or {}
            content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
            content_type_full = headers.get("Content-Type") or ""
            final_url = resp.url or url

        if not raw:
            return f"⚠️ 页面返回空内容（HTTP {status}）: {final_url}"

        binary_prefixes = (
            "image/", "audio/", "video/", "font/",
            "application/pdf", "application/zip", "application/octet-stream"
        )
        if content_type.startswith(binary_prefixes) or (
            b"\x00" in raw[:4096] and not content_type.startswith("text/")
        ):
            return (
                f"❌ 不支持的内容类型: {content_type or '未知'}\n"
                f"URL: {final_url}\n大小: {_format_size(len(raw))}"
            )

        charset_match = re.search(r"charset\s*=\s*([\w\-]+)", content_type_full, re.I)
        header_charset = charset_match.group(1) if charset_match else None
        text, _ = _decode_bytes(raw, header_charset)

        is_html = content_type in ("text/html", "application/xhtml+xml") or (
            not content_type and re.match(br"\s*<(?:!doctype\s+html|html)", raw[:512], re.I)
        )

        title = ""
        if is_html:
            parser = _HTMLTextExtractor()
            try:
                parser.feed(text)
                parser.close()
                body = parser.get_text()
                title = parser.title
            except Exception:
                body = _clean_text(text)
            if not body:
                body = _clean_text(text)
        elif content_type == "application/json" or content_type.endswith("+json"):
            try:
                body = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except Exception:
                body = text
        else:
            body = text.strip()

        total_chars = len(body)
        body_truncated = total_chars > limit
        if body_truncated:
            body = body[:limit]

        lines = [f"🌐 URL: {final_url}"]
        if final_url != url:
            lines.append(f"↪️ 原始URL: {url}")
        lines.append(
            f"📄 类型: {content_type or '未知'} | 状态: {status} | 下载: {_format_size(len(raw))}"
        )
        if title:
            lines.append(f"🏷️ 标题: {title}")
        lines.append("---")

        result = "\n".join(lines) + "\n" + body
        if body_truncated:
            result += (
                f"\n\n...（内容过长，已截断到 {limit} 字符；原文约 {total_chars} 字符，"
                f"可通过 max_chars 调整，最大 {self.MAX_MAX_CHARS}）"
            )
        if download_truncated:
            result += f"\n...（响应超过 {_format_size(self.MAX_DOWNLOAD_BYTES)}，仅下载了前一部分）"
        return result


class ToolManager:
    """工具管理器：注册、查找、执行工具"""

    def __init__(self, workspace_root: str, tavily_api_key: str = None, skills_dir: str = None):
        self.workspace_root = workspace_root
        self.skills_dir = os.path.abspath(skills_dir) if skills_dir else None
        self.tools: Dict[str, Any] = {}
        self.todo_tool = TodoTool()
        self._register_default_tools(tavily_api_key)

    def _register_default_tools(self, tavily_api_key: str = None):
        """注册默认工具"""
        self.register_tool(self.todo_tool)
        self.register_tool(ReadFileTool(self.workspace_root, extra_roots=[self.skills_dir]))
        self.register_tool(BashTool(self.workspace_root))
        self.register_tool(EditTool(self.workspace_root))
        self.register_tool(LsTool(self.workspace_root))
        self.register_tool(GrepTool(self.workspace_root))
        self.register_tool(WebSearchTool(tavily_api_key))
        self.register_tool(WebFetchTool())

    def register_tool(self, tool):
        self.tools[tool.name] = tool

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回所有工具的schema列表，给模型调用"""
        return [t.get_schema() for t in self.tools.values()]

    def get_tool_handler(self, name: str) -> Optional[Callable]:
        """获取工具执行函数"""
        if name in self.tools:
            return self.tools[name].execute
        return None

    def execute_tool(self, name: str, arguments: str) -> str:
        """执行工具，解析JSON参数"""
        handler = self.get_tool_handler(name)
        if not handler:
            return f"❌ 未知工具: {name}"
        try:
            args = json.loads(arguments) if arguments else {}
            return handler(**args)
        except json.JSONDecodeError as e:
            return f"❌ 工具参数JSON解析失败: {str(e)}\n参数内容: {arguments}"
        except Exception as e:
            return f"❌ 工具执行错误: {str(e)}"

    def reset_todo(self):
        self.todo_tool.reset()
