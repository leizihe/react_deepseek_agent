# DeepSeek ReAct Agent

一个纯Python实现、支持工具调用和流式输出的命令行Agent，不需要pip安装任何依赖，解压即用。

## ✨ 功能特性

- 🧠 **原生DeepSeek API调用**：不依赖OpenAI SDK / LangChain / LangGraph，纯HTTP原生实现
- 🔧 **内置8个工具**：bash、edit、ls、grep、todo、read_file、websearch（Tavily）、webfetch
- 🧩 **Skill技能系统**：扫描 `skills/` 目录，模型按需渐进式加载技能主文件与引用资源
- 🚀 **子智能体（SubAgent）**：可派遣独立子智能体并行/分层处理子任务，独立上下文、独立工具集、支持嵌套
- ⚡ **并行执行**：一轮内多个工具调用并发执行，多个子智能体可同时工作，结果仍按调用顺序回填
- 📋 **Todo规划中间件**：类似LangGraph的自动任务规划，复杂任务自动拆解步骤
- 🌊 **流式输出**：逐字实时打印思考过程，工具调用状态实时展示
- 💬 **多轮对话**：完整上下文记忆
- 🐍 **Python 3.7兼容**：requests包已随完整发行包内置（git 仓库使用 requirements.txt）
- 🛡️ **消息格式自动校验**：自动修复API请求格式，避免400错误
- 🔍 **内置连通性测试**：一键检查API配置是否正确

## 🚀 快速开始

### 1. 获取代码
```bash
# 方式一：使用完整发行包（含 vendor，解压即用）
unzip react-deepseek-agent.zip
cd react_deepseek_agent

# 方式二：从 git 仓库克隆（仓库不包含 vendor/skills/workspace）
git clone <仓库地址>
cd react_deepseek_agent
pip install -r requirements.txt
```

> git 仓库按需排除了 `vendor/`、`skills/`、`workspace/` 三个本地目录。
> 从仓库克隆后请先执行 `pip install -r requirements.txt`；技能目录可按
> [Skill 技能系统](#-skill-技能系统) 的格式自行创建。

### 2. 配置API Key
```bash
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
# 可选：Tavily搜索Key，用于联网搜索
# export TAVILY_API_KEY="你的Tavily Key"
```

### 3. 测试API连通性（推荐先做）
```bash
python test_api.py
```
如果显示✅ API连接成功，再启动Agent。如果失败，脚本会给出具体原因。

### 4. 启动Agent
```bash
python run.py
```
使用完整发行包（含 `vendor/`）时不需要任何 pip 安装；从 git 仓库克隆时请先安装 `requirements.txt`。

## 🚀 子智能体（SubAgent）

主 Agent 可以通过内置的 `SubAgent` 工具派遣子智能体（SubAgent）去独立完成子任务。

### 核心机制

| 特性 | 说明 |
|------|------|
| **独立上下文** | 每个子智能体拥有自己的消息历史，不污染主会话，主 Agent 只拿到最终结论 |
| **独立工具集** | 子智能体使用独立的工具管理器，可通过白名单限制其可用工具 |
| **角色模板** | 内置 general / researcher / coder / explorer 四种角色，也可自定义 system prompt |
| **递归嵌套** | 子智能体自己也能派遣下级子智能体，通过 max_depth 限制嵌套层数防止失控 |
| **结果回传** | 子智能体最终结论以 `【子智能体 xxx 的结论】` 形式作为工具结果回传给上层 |
| **并行派遣** | 同一轮内多个互不依赖的 `SubAgent` 调用会并发执行，总耗时接近最慢的那个子任务 |

除 `SubAgent` 外，普通工具调用同样支持并行：模型一轮返回多个工具调用（如同时 `read_file` + `grep`）时会并发执行，最多 `MAX_PARALLEL_TOOLS`（默认 8）个；工具结果按原始调用顺序回填，保证消息配对关系正确。

### 使用方式

直接对 Agent 说"派遣一个子智能体去调研 XXX"即可，主 Agent 会自行判断是否调用 `SubAgent` 工具。

`SubAgent` 工具的参数：

| 参数 | 说明 |
|------|------|
| `description` | **必填**，任务说明。子智能体看不到主对话历史，必须自包含（背景+目标+要求+验收标准） |
| `role` | 角色：`general` / `researcher` / `coder` / `explorer`，默认 general |
| `allowed_tools` | 限制子智能体可用工具，如 `["ls","grep","bash"]` |
| `system_prompt` | 可选，覆盖默认角色设定的专属提示 |

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--no-subagent` | 禁用子智能体功能（不注册 SubAgent 工具） |
| `--max-subagent-depth` | 子智能体最大嵌套深度，默认 2 |
| `--subagent-tools` | 限制子智能体可用工具，逗号分隔，如 `ls,grep,bash` |
| `--subagent-verbose` | 显示子智能体内部的工具调用过程 |

示例：
```bash
# 只允许子智能体使用只读工具，最多嵌套1层
python run.py --subagent-tools ls,grep --max-subagent-depth 1

# 完全禁用子智能体
python run.py --no-subagent
```

### 验证

附带的 `test_subagent.py` 为离线验证脚本（用 mock 模型，不消耗 API），可验证：
SubAgent 工具注册、派遣与回传、上下文隔离、嵌套深度限制、工具白名单。
```bash
python test_subagent.py
```

## 🧩 Skill 技能系统

Agent 会自动扫描默认的 `skills/` 目录，把「技能名 + 描述 + 主文件路径」放入系统提示。
模型判断任务匹配某个技能后，才用 `read_file` 读取该技能的 `SKILL.md`，并按需读取同目录下的引用资源，
即**渐进式加载**：平时只占几十个 token，不会把所有技能正文都塞进上下文。

### 目录格式

```
skills/
└── <skill-name>/
    ├── SKILL.md          # 必需，技能主文件
    └── references/...    # 可选，按需加载的引用资源
```

`SKILL.md` 使用 YAML frontmatter：

```markdown
---
name: python-debug
description: Python 报错排查与修复工作流。当程序运行报错、测试失败时使用。
---

# 工作流
1. 用 bash 复现问题，拿到完整 traceback
2. 用 read_file 读取相关代码
3. 对照 references/checklist.md 逐项检查
```

- `name` 缺省时使用目录名；`description` 缺省时取正文第一段。
- 只支持简单 frontmatter 子集（`key: value`、引号、`>` / `|` 折叠块），不依赖 PyYAML。
- 内置示例技能 `skills/python-debug/`，包含引用资源 `references/checklist.md`。

### 使用方式

把技能目录放进 `skills/` 即可，启动时自动加载：

```bash
# 使用默认 skills/ 目录
python run.py

# 自定义技能目录
python run.py --skills-dir /path/to/my-skills

# 禁用技能功能
python run.py --no-skills
```

主 Agent 和子智能体都会注入技能清单；如果子智能体被 `--subagent-tools` 白名单禁用了 `read_file`，
则不会向它注入技能清单，避免提示模型调用不可用的工具。

### 验证

```bash
python test_skill.py
```

离线验证技能扫描、frontmatter 解析、prompt 渲染、read_file 越界拦截、端到端加载流程。

## 📁 文件结构

| 文件 | 说明 |
|------|------|
| `run.py` | 启动入口，自动加载内置依赖 |
| `agent.py` | ReAct Agent主逻辑，思考-行动-观察循环 |
| `model.py` | DeepSeek模型封装，原生HTTP/SSE流式调用 |
| `tool.py` | 所有工具实现 + 工具管理器 |
| `skill.py` | 技能扫描、frontmatter 解析、技能提示渲染 |
| `subagent.py` | 子智能体（SubAgent）与 SubAgent 工具实现 |
| `skills/` | 技能目录，每个子目录一个技能（本地目录，不纳入 git） |
| `test_api.py` | API连通性测试脚本 |
| `test_subagent.py` | 子智能体机制离线验证脚本 |
| `test_skill.py` | 技能机制离线验证脚本 |
| `test_webfetch.py` | 网页抓取工具离线验证脚本 |
| `test_messages.py` | 消息格式校验（并行工具调用配对）离线验证脚本 |
| `test_parallel.py` | 工具与子智能体并行执行离线验证脚本 |
| `vendor/` | 内置requests及依赖（本地目录，不纳入 git） |
| `workspace/` | 默认工作目录，所有文件操作都在这里（本地目录，不纳入 git） |

## 🎮 使用命令

在对话中可以使用斜杠命令：

| 命令 | 作用 |
|------|------|
| `/exit` 或 `/quit` | 退出程序 |
| `/clear` | 清屏 |
| `/reset` | 重置对话和任务计划 |
| `/todo` | 查看当前任务计划 |
| `/help` | 显示帮助 |

## 🔧 内置工具说明

| 工具 | 功能 |
|------|------|
| `todo` | 任务规划，创建/更新/查看任务清单，复杂任务自动调用 |
| `read_file` | 读取文本文件，支持 offset/limit 分页；用于加载技能主文件与引用资源 |
| `bash` | 执行Shell命令，默认限制在workspace目录内 |
| `edit` | 精确替换文本编辑文件，old_text需要精确匹配 |
| `ls` | 列出目录内容和文件信息 |
| `grep` | 在文件中搜索关键词，支持正则 |
| `websearch` | 联网搜索（需要Tavily API Key） |
| `webfetch` | 抓取指定URL网页并转为可读文本：去HTML标签/脚本/样式、自动识别编码、支持长度限制 |
| `SubAgent` | **派遣子智能体**独立完成子任务，返回其结论（可用 `--no-subagent` 关闭） |

## ❓ 常见问题

**Q: 启动提示400 Bad Request？**
A: 先运行`python test_api.py`，脚本会告诉你具体错误原因，通常是API Key错误、余额不足或者base_url不对。

**Q: 不需要pip install吗？**
A: 不需要！所有依赖（requests及其依赖）都已经打包在vendor目录里，run.py会自动加载。

**Q: 怎么使用代理？**
A: 设置环境变量`HTTPS_PROXY=http://your-proxy:port`即可。

**Q: 可以用其他兼容OpenAI格式的模型吗？**
A: 可以，通过`--base-url`参数指定API地址，比如：
```bash
python run.py --base-url https://your-custom-api-endpoint/v1 --model your-model-name
```
