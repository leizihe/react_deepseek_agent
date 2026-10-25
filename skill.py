# -*- coding: utf-8 -*-
"""
Skill 技能模块

技能（Skill）是一份自带工作流说明的 Markdown 文件，模型按需渐进式加载：
系统提示里只放「技能名 + 描述 + 主文件路径」，当任务匹配某个技能时，
模型先调用 read_file 读取 SKILL.md，再读取同目录下的引用资源。

目录约定：
    skills/
        <skill-name>/
            SKILL.md            # 技能主文件，YAML frontmatter + Markdown 正文
            references/...      # 可选，按需加载的引用资源

SKILL.md 示例：
    ---
    name: code-review
    description: 代码审查工作流。当用户要求审查代码时使用。
    ---

    # 工作流
    1. ...
"""

import os
import re
import html
from typing import Dict, List, Optional, Tuple


# 与用户提供的 <skill-system> 提示保持一致
SKILL_SYSTEM_PROMPT = """<skill-system>
You have access to skills that provide optimized workflows for specific tasks. Each skill contains best practices, frameworks, and references to additional resources.

**Progressive Loading Pattern:**
1. When a user query matches a skill's use case, immediately call `read_file` on the skill's main file using the path attribute provided in the skill tag below
2. If an explicit requested skill is provided in the system context, load that skill first even if the user message is short
3. Read and understand the skill's workflow and instructions
4. The skill file contains references to external resources under the same folder
5. Load referenced resources only when needed during execution
6. Follow the skill's instructions precisely
</skill-system>"""


def default_skills_dir() -> str:
    """默认技能目录：项目根目录下的 skills/"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """
    解析 Markdown 的 YAML frontmatter。
    只支持本项目需要的简单子集（key: value、引号、> / | 折叠块），
    避免引入 PyYAML 依赖。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end is None:
        return {}, text

    meta: Dict[str, str] = {}
    i = 1
    while i < end:
        line = lines[i]
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
        if not match:
            i += 1
            continue

        key, value = match.group(1), match.group(2).strip()
        if value in (">", "|", ">-", "|-"):
            # 折叠块：收集后续缩进行
            block: List[str] = []
            j = i + 1
            while j < end:
                nxt = lines[j]
                if nxt.strip() and not nxt.startswith((" ", "\t")):
                    break
                block.append(nxt.strip())
                j += 1
            separator = " " if value.startswith(">") else "\n"
            value = separator.join(part for part in block if part).strip()
            i = j
        else:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            i += 1

        if value:
            meta[key] = value

    return meta, "\n".join(lines[end + 1:])


class Skill:
    """单个技能：名称、描述、主文件绝对路径"""

    def __init__(self, name: str, description: str, path: str):
        self.name = name
        self.description = description
        self.path = path
        self.base_dir = os.path.dirname(path)


class SkillManager:
    """扫描技能目录，并把技能清单渲染进系统提示"""

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = os.path.abspath(skills_dir) if skills_dir else None
        self.skills: List[Skill] = []
        self.reload()

    def reload(self) -> None:
        """重新扫描 skills/<name>/SKILL.md"""
        self.skills = []
        if not self.skills_dir or not os.path.isdir(self.skills_dir):
            return

        seen = set()
        for entry in sorted(os.listdir(self.skills_dir)):
            skill_dir = os.path.join(self.skills_dir, entry)
            skill_file = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isdir(skill_dir) or not os.path.isfile(skill_file):
                continue

            try:
                with open(skill_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

            meta, body = _parse_frontmatter(content)
            name = (meta.get("name") or entry).strip()
            description = (meta.get("description") or "").strip()
            if not description:
                description = self._fallback_description(body)
            if not description:
                continue
            if name in seen:
                continue

            seen.add(name)
            self.skills.append(Skill(name, description, skill_file))

    @staticmethod
    def _fallback_description(body: str) -> str:
        """没有 description 时，取正文第一段非标题文字"""
        for line in body.splitlines():
            line = line.strip()
            if not line or line in ("---", "...", "```"):
                continue
            if not line.startswith("#") and not line.startswith("```"):
                return line[:200]
        return ""

    def is_empty(self) -> bool:
        return not self.skills

    def get_prompt(self) -> str:
        """渲染技能提示块；没有技能时返回空字符串"""
        if not self.skills:
            return ""

        tags = []
        for skill in self.skills:
            description = " ".join(skill.description.split())
            tags.append(
                '<skill name="{name}" path="{path}">\n{description}\n</skill>'.format(
                    name=html.escape(skill.name, quote=True),
                    path=html.escape(skill.path, quote=True),
                    description=html.escape(description),
                )
            )

        return "{system}\n\n<available_skills>\n{skills}\n</available_skills>".format(
            system=SKILL_SYSTEM_PROMPT,
            skills="\n".join(tags),
        )

    def get_skill(self, name: str) -> Optional[Skill]:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def get_paths(self) -> List[str]:
        return [skill.path for skill in self.skills]
