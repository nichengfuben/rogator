#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目合规检查：目录子项数量、文件行数、函数长度、嵌套深度、禁用目录名、空目录、语法错误。
   强制使用 tree-sitter-language-pack；依赖缺失或解析失败时立即报错，无 legacy 回退。
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

from echotools.base.logger import configure, get_logger

# ============================================================
# 日志配置
# ============================================================

configure(
    level="DEBUG",
    color=True,
    show_time=True,
    show_level=True,
    show_name=True,
    time_format="%Y-%m-%d %H:%M:%S",
)
logger = get_logger("checker")

# ============================================================
# 全局常量配置
# ============================================================

# 扫描根目录：硬编码到项目根，无论从哪个 cwd 调用都扫这个目录。
# 注释本行、置 None、置空字符串、或路径不存在时，自动回退到"脚本所在目录"逻辑。
##SCAN_ROOT: Optional[str] = "X:/Project/Public/Provider-Evo"

# 目录子项约束
MAX_CHILDREN: int = 7
MIN_CHILDREN: int = 3

# 目录豁免配置
EXEMPT_SUBTREE_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("scripts",),
    ("provider-core", "plugins"),
    ("src", "persist"),
    ("provider-core", "plugins"),
    ("provider-core", "persist"),
    ("logs",),
    ("persist",),
    ("template",),
    ("provider-core", "config"),
    ("provider-core", "template"),
    ("tmp",),
    ("agent_log",),
)

EXEMPT_SHALLOW_REL_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("provider-docs",),
    ("provider-plugin",),
    ("provider-core",),
)

# 运行时目录/文件（自动忽略）
RUNTIME_DIR_NAMES: Set[str] = {
    "__pycache__",
    "config",
    "configs",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    "node_modules",
    "dist",
}

RUNTIME_FILE_SUFFIXES: Tuple[str, ...] = (".pyc", ".pyo")

# 文件行数约束
FILE_MIN_LINES: int = 0
FILE_MAX_LINES: int = 400
FILE_HARD_MAX_LINES: int = 800
FILE_EXEMPT_NAMES: Set[str] = {"__init__.py", "conftest.py", "init.js"}
# 数据/配置/样式文件不受最小行数限制（其长度由数据规模决定，与代码质量无关）
FILE_LOWER_EXEMPT_EXTS: Set[str] = {".json", ".toml", ".yaml", ".yml", ".css"}
# 数据/配置/样式文件同样不受最大行数限制（JSON locale、CSS 样式表等允许任意大）
FILE_UPPER_EXEMPT_EXTS: Set[str] = {".json", ".toml", ".yaml", ".yml", ".css"}

# 文件排除配置
EXEMPT_FILE_REL_PATHS: Tuple[str, ...] = ()  # 相对路径排除
EXEMPT_FILE_NAMES: Set[str] = {
    "file_merger.py",
    "achecker.py",
    "accounts.json",
    "plugin_details.json",
    "abnormis.txt",
    "astro.config.mjs",
    "Base.astro",
}
# 文件命名约束：stem 部分不得含 "-"、"."、emoji，"_" 最多出现 1 次
# 以 "." 开头的文件（dotfiles）豁免文件名规则
FILENAME_EXEMPT_EXTS_DOTFILE: bool = True

# 函数长度约束
FUNC_MAX_LINES: int = 50

# 嵌套深度约束
MAX_NESTING_DEPTH: int = 4

# 禁用目录名
FORBIDDEN_DIR_NAMES: Set[str] = {
    "infra", "domain","spec",
    "application","infrastructure", "interfaces"
}

# 豁免子树（不扫描文件行数等）
FILE_EXEMPT_SUBTREE_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("provider-core", "scripts"),
    ("provider-core", "plugins"),
    ("provider-core", "persist"),
    ("logs",),
    ("persist",),
    ("template",),
    ("provider-core", "config"),
    ("provider-core", "template"),
    ("tmp",),
    ("tests",),
    ("docs-src",),
    ("provider-docs",),
)

# 空目录豁免
EMPTY_DIR_EXEMPT_NAMES: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "logs",
    "tmp",
    "template",
    "config",
    "persist",
    "emoji_assets",
    "node_modules",
    "dist",
}

# 文件名约束（仅检查 stem 部分，忽略扩展名后缀）
# 禁止字符: "-", "."（stem 内不允许出现多段点号）, emoji
# 下划线 "_" 最多出现 1 次
FILENAME_FORBIDDEN_CHARS: Set[str] = {"-", "."}
FILENAME_MAX_UNDERSCORES: int = 1
FILENAME_EXEMPT_NAMES: Set[str] = {"__init__.py", "conftest.py", "__main__.py", "docker-compose.yml", "docker-compose.yaml", "_manifest.json"}

# README.md 仅允许出现在各 Git 仓库根目录
README_ALLOWED_REL_PATHS: Tuple[str, ...] = (
    "README.md",
    "provider-core/README.md",
    "provider-docs/README.md",
    "plugin-repo/README.md",
)

# 报告输出配置
REPORT_OUTPUT: str = "abnormis.txt"

# 语法检查配置
ENABLE_SYNTAX_CHECK: bool = True  # 是否启用语法错误检查

# ============================================================
# Tree-sitter 集成
# ============================================================

try:
    from tree_sitter_language_pack import (
        get_parser,
        get_language,
        available_languages,
        downloaded_languages,
    )
except ImportError as exc:
    raise SystemExit(
        "achecker requires tree-sitter-language-pack (no fallback). "
        "Install: py -m pip install tree-sitter tree-sitter-language-pack"
    ) from exc

_TS_GET_PARSER = get_parser
_TS_GET_LANGUAGE = get_language
# Seed from pack listings; membership also accepts on-demand get_parser success.
_TS_AVAILABLE_LANGUAGES: Set[str] = set(downloaded_languages()) or set(available_languages())
try:
    get_parser("python")
    _TS_AVAILABLE_LANGUAGES.add("python")
except Exception as exc:
    raise SystemExit(
        "tree-sitter-language-pack cannot load python parser (no legacy fallback). "
        "Pre-download parsers (proxy may be required), e.g.:\n"
        "  py -c \"from tree_sitter_language_pack import download; "
        "download(['python','javascript','typescript','tsx','json',"
        "'yaml','toml','html','css','vue','bash','markdown'])\""
    ) from exc
_TS_AVAILABLE = True
logger.info(
    "Tree-sitter-language-pack loaded (seed=%d languages; more via on-demand parse)",
    len(_TS_AVAILABLE_LANGUAGES),
)

# 文件扩展名到 tree-sitter 语言名的映射（支持 40+ 种常见语言）
EXT_TO_TS_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".lua": "lua",
    ".r": "r",
    ".sql": "sql",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".vue": "vue",
    ".svelte": "svelte",
    ".zig": "zig",
    ".dart": "dart",
    ".elm": "elm",
    ".hs": "haskell",
    ".haskell": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".erl": "erlang",
    ".ex": "elixir",
    ".exs": "elixir",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".proto": "proto",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".svg": "xml",
    ".vue": "vue",
    ".svelte": "svelte",
    ".dart": "dart",
    ".elm": "elm",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".erl": "erlang",
    ".ex": "elixir",
    ".exs": "elixir",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".proto": "proto",
    ".cmake": "cmake",
    ".make": "make",
    ".mk": "make",
    ".cmake": "cmake",
}

def get_ts_language_for_file(filepath: Path) -> Optional[str]:
    """根据文件扩展名获取对应的 tree-sitter 语言名。"""
    return EXT_TO_TS_LANG.get(filepath.suffix.lower())

def is_ts_language_supported(lang: str) -> bool:
    """检查 tree-sitter 是否支持该语言（含 on-demand 加载）。"""
    if not _TS_AVAILABLE:
        return False
    if lang in _TS_AVAILABLE_LANGUAGES:
        return True
    try:
        get_parser(lang)
        _TS_AVAILABLE_LANGUAGES.add(lang)
        return True
    except Exception:
        return False


# ============================================================
# 异常类
# ============================================================

class CheckerError(RuntimeError):
    """检查器基础异常。"""
    pass


class ScanError(CheckerError):
    """扫描过程异常。"""
    pass


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class DirViolation:
    """目录子项违规。"""
    path: Path
    child_count: int
    children: Tuple[str, ...]
    kind: str = "over_max"  # "over_max" | "under_min"


@dataclass(frozen=True)
class FileLineViolation:
    """文件行数违规。"""
    path: Path
    line_count: int
    kind: str  # "under_min" | "over_soft" | "over_hard"


@dataclass(frozen=True)
class FuncLengthViolation:
    """函数长度违规。"""
    path: Path
    func_name: str
    line_count: int
    start_line: int


@dataclass(frozen=True)
class NestingViolation:
    """嵌套深度违规。"""
    path: Path
    func_name: str
    depth: int
    line: int


@dataclass(frozen=True)
class ForbiddenDirViolation:
    """禁用目录名违规。"""
    path: Path
    name: str


@dataclass(frozen=True)
class EmptyDirViolation:
    """空目录违规。"""
    path: Path


@dataclass(frozen=True)
class SyntaxViolation:
    """语法错误违规。"""
    path: Path
    language: str
    line: Optional[int]
    message: str


@dataclass(frozen=True)
class FileNameViolation:
    """文件命名违规。"""
    path: Path
    reason: str


@dataclass(frozen=True)
class ReadmeViolation:
    """README.md 位置违规（仅允许仓库根目录）。"""
    path: Path


@dataclass
class ScanResult:
    """扫描结果汇总。"""
    dir_violations: List[DirViolation] = None
    file_violations: List[FileLineViolation] = None
    func_violations: List[FuncLengthViolation] = None
    nesting_violations: List[NestingViolation] = None
    forbidden_violations: List[ForbiddenDirViolation] = None
    empty_dir_violations: List[EmptyDirViolation] = None
    syntax_violations: List[SyntaxViolation] = None
    filename_violations: List[FileNameViolation] = None
    readme_violations: List[ReadmeViolation] = None

    def __post_init__(self):
        self.dir_violations = self.dir_violations or []
        self.file_violations = self.file_violations or []
        self.func_violations = self.func_violations or []
        self.nesting_violations = self.nesting_violations or []
        self.forbidden_violations = self.forbidden_violations or []
        self.filename_violations = self.filename_violations or []
        self.readme_violations = self.readme_violations or []
        self.empty_dir_violations = self.empty_dir_violations or []
        self.syntax_violations = self.syntax_violations or []

    @property
    def total(self) -> int:
        count = (
            len(self.dir_violations) +
            len(self.file_violations) +
            len(self.func_violations) +
            len(self.nesting_violations) +
            len(self.forbidden_violations) +
            len(self.empty_dir_violations) +
            len(self.filename_violations) +
            len(self.readme_violations)
        )
        if ENABLE_SYNTAX_CHECK:
            count += len(self.syntax_violations)
        return count

    @property
    def is_clean(self) -> bool:
        return self.total == 0


# ============================================================
# 工具函数
# ============================================================

def _resolve_scan_root() -> Path:
    """解析扫描根目录。未配置 SCAN_ROOT 时使用脚本所在目录；配置无效则立即退出。"""
    script_dir = Path(__file__).resolve().parent
    try:
        candidate = SCAN_ROOT
    except NameError:
        return script_dir
    if candidate is None:
        return script_dir
    if not isinstance(candidate, str) or not candidate.strip():
        return script_dir
    try:
        resolved = Path(candidate).resolve()
    except (OSError, TypeError, ValueError) as e:
        raise SystemExit(f"Invalid SCAN_ROOT {candidate!r}: {e}") from e
    if not resolved.is_dir():
        raise SystemExit(f"SCAN_ROOT is not a directory: {resolved}")
    return resolved


def _relative_parts(root: Path, path: Path) -> Optional[Tuple[str, ...]]:
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return None


# ============================================================
# 路径检查逻辑
# ============================================================

def is_shallow_exempt(root: Path, directory: Path) -> bool:
    parts = _relative_parts(root, directory)
    if parts is None:
        return False
    return parts in EXEMPT_SHALLOW_REL_PATHS


def is_exempt_subtree(root: Path, directory: Path) -> bool:
    root_resolved = root.resolve()
    directory_resolved = directory.resolve()
    if directory_resolved == root_resolved:
        return True
    parts = _relative_parts(root, directory)
    if parts is None:
        return False
    if any(part.startswith(".") for part in parts):
        return True
    # <--- 修改：添加 "emoji_assets"、"__pycache__" 到豁免目录名列表
    exempt_dir_names = {"logs", "docs-src", "tests", "template", "config", "tmp", "vendor", "emoji_assets", "__pycache__"}
    if directory.name in exempt_dir_names:
        return True
    for prefix in EXEMPT_SUBTREE_PREFIXES:
        if len(parts) >= len(prefix) and parts[:len(prefix)] == prefix:
            return True
    return False


def is_exempt_dir(root: Path, directory: Path) -> bool:
    return is_exempt_subtree(root, directory) or is_shallow_exempt(root, directory)


def is_readme_path_allowed(root: Path, filepath: Path) -> bool:
    if filepath.name != "README.md":
        return True
    parts = _relative_parts(root, filepath.parent)
    if parts is None:
        return False
    rel = str(Path(*parts) / filepath.name).replace("\\", "/")
    if rel in README_ALLOWED_REL_PATHS:
        return True
    path_parts = rel.split("/")
    if (
        len(path_parts) == 3
        and path_parts[0] == "provider-plugin"
        and path_parts[1].startswith("Provider-")
    ):
        return True
    return False


def _is_file_exempt_for_lines(root: Path, filepath: Path) -> bool:
    if filepath.name in EXEMPT_FILE_NAMES:
        logger.debug("File exempt by name: %s", filepath.name)
        return True
    parts = _relative_parts(root, filepath.parent)
    if parts is None:
        return True
    rel_path = str(Path(*parts) / filepath.name)
    if rel_path in EXEMPT_FILE_REL_PATHS:
        logger.debug("File exempt by relative path: %s", rel_path)
        return True
    # <--- 修改：检查父目录是否为 emoji_assets
    for parent in filepath.parents:
        if parent.name in {"logs", "docs-src", "tests", "template", "config", "tmp", "vendor", "emoji_assets"}:
            return True
    for prefix in FILE_EXEMPT_SUBTREE_PREFIXES:
        if len(parts) >= len(prefix) and parts[:len(prefix)] == prefix:
            return True
    return False


def _is_init_py(filepath: Path) -> bool:
    return filepath.name == "__init__.py"


def countable_children(directory: Path) -> List[Path]:
    children: List[Path] = []
    script_path = Path(__file__).resolve()
    for item in directory.iterdir():
        if item.name in RUNTIME_DIR_NAMES:
            continue
        if item.is_file() and item.suffix in RUNTIME_FILE_SUFFIXES:
            continue
        if item.resolve() == script_path:
            continue
        children.append(item)
    return sorted(children, key=lambda entry: entry.name.lower())


def is_standard_two_child_dir(children: List[Path]) -> bool:
    """Architectural layouts that legitimately have two direct children."""
    if len(children) != 2:
        return False
    names = {c.name for c in children}
    if names == {"__init__.py", "py.typed"}:
        return True
    if "__init__.py" in names:
        others = [c for c in children if c.name != "__init__.py"]
        if len(others) == 1 and (others[0].suffix == ".py" or others[0].is_dir()):
            return True
    prompts = [c for c in children if c.suffix == ".prompt"]
    meta = [
        c
        for c in children
        if c.name == ".meta.toml" or c.name.endswith(".meta.toml")
    ]
    if len(prompts) == 1 and len(meta) == 1:
        return True
    if names == {"config", "core"} and all(c.is_dir() for c in children):
        return True
    if names == {"core", "vendor"} and all(c.is_dir() for c in children):
        return True
    if names == {"layouts", "pages"} and all(c.is_dir() for c in children):
        return True
    return False


def satisfies_min_children(children: List[Path]) -> bool:
    count = len(children)
    if count > MAX_CHILDREN:
        return False
    if count >= MIN_CHILDREN:
        return True
    if count == 2 and is_standard_two_child_dir(children):
        return True
    return False


def _walk_dirs(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as e:
            logger.warning("Failed to read directory %s: %s", current, e)
            continue
        dirs = [item for item in entries if item.is_dir()]
        files = [item for item in entries if item.is_file()]
        yield str(current), [item.name for item in dirs], [item.name for item in files]
        for item in reversed(dirs):
            if item.name in RUNTIME_DIR_NAMES:
                continue
            if is_exempt_subtree(root, item):
                continue
            stack.append(item)


# ============================================================
# Tree-sitter 分析器（修复版）
# ============================================================

class TreeSitterAnalyzer:
    """使用 Tree-sitter 分析代码的函数长度和嵌套深度。"""

    def __init__(self):
        self._parser_cache: Dict[str, Any] = {}

    def _get_parser(self, lang: str):
        if lang not in self._parser_cache:
            try:
                self._parser_cache[lang] = _TS_GET_PARSER(lang)
            except Exception as e:
                raise RuntimeError(
                    f"tree-sitter get_parser({lang!r}) failed (no fallback): {e}"
                ) from e
        return self._parser_cache[lang]

    def analyze(
        self,
        source: str,
        filepath: Path,
        lang: str
    ) -> Tuple[List[FuncLengthViolation], List[NestingViolation]]:
        """使用 tree-sitter 分析文件，返回函数长度和嵌套深度违规。"""
        func_violations: List[FuncLengthViolation] = []
        nesting_violations: List[NestingViolation] = []

        parser = self._get_parser(lang)
        try:
            tree = parser.parse(source.encode('utf-8'))
            root = tree.root_node
        except Exception as e:
            raise RuntimeError(
                f"tree-sitter parse failed for {filepath} (no fallback): {e}"
            ) from e

        # 遍历所有节点，查找函数定义
        self._find_functions(
            root,
            filepath,
            lang,
            func_violations,
            nesting_violations
        )

        return func_violations, nesting_violations

    def _find_functions(
        self,
        node,
        filepath: Path,
        lang: str,
        func_violations: List[FuncLengthViolation],
        nesting_violations: List[NestingViolation]
    ):
        """遍历语法树，查找所有函数定义。"""
        if self._is_function_node(node, lang):
            self._analyze_function(
                node,
                filepath,
                lang,
                func_violations,
                nesting_violations
            )
        for child in node.children:
            self._find_functions(
                child,
                filepath,
                lang,
                func_violations,
                nesting_violations
            )

    def _analyze_function(
        self,
        func_node,
        filepath: Path,
        lang: str,
        func_violations: List[FuncLengthViolation],
        nesting_violations: List[NestingViolation]
    ):
        """分析单个函数。"""
        # 获取函数名
        name = self._get_function_name(func_node, lang) or "<anonymous>"

        # 计算函数行数
        start_line = func_node.start_point[0] + 1
        end_line = func_node.end_point[0] + 1
        func_lines = end_line - start_line + 1

        if func_lines > FUNC_MAX_LINES:
            func_violations.append(
                FuncLengthViolation(
                    path=filepath,
                    func_name=name,
                    line_count=func_lines,
                    start_line=start_line,
                )
            )

        # 计算嵌套深度
        max_depth = self._compute_max_nesting(func_node, lang)
        if max_depth > MAX_NESTING_DEPTH:
            nesting_violations.append(
                NestingViolation(
                    path=filepath,
                    func_name=name,
                    depth=max_depth,
                    line=start_line,
                )
            )

    def _compute_max_nesting(self, node, lang: str) -> int:
        """计算节点内最大嵌套深度（控制流嵌套）。"""
        max_depth = 0

        def walk(n, depth):
            nonlocal max_depth
            if self._is_control_node(n, lang):
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            for child in n.children:
                # 跳过嵌套的函数定义，避免将内部函数当作控制流
                if not self._is_function_node(child, lang):
                    walk(child, depth)

        walk(node, 0)
        return max_depth

    def _is_function_node(self, node, lang: str) -> bool:
        """判断节点是否为函数定义。"""
        func_types = {
            "python": {"function_definition", "async_function_definition"},
            "javascript": {"function_declaration", "function_expression",
                          "arrow_function", "method_definition"},
            "typescript": {"function_declaration", "function_expression",
                          "arrow_function", "method_definition"},
            "jsx": {"function_declaration", "function_expression",
                   "arrow_function", "method_definition"},
            "tsx": {"function_declaration", "function_expression",
                   "arrow_function", "method_definition"},
            "java": {"method_declaration", "constructor_declaration"},
            "go": {"function_declaration", "method_declaration"},
            "rust": {"function_item", "method_declaration"},
            "c": {"function_definition"},
            "cpp": {"function_definition", "method_definition"},
            "c_sharp": {"method_declaration", "constructor_declaration"},
            "ruby": {"method", "def", "defs"},
            "php": {"function_definition", "method_declaration"},
            "swift": {"function_declaration", "method_declaration"},
            "kotlin": {"function_declaration"},
            "scala": {"function_declaration"},
            "bash": {"function_definition"},
            "lua": {"function_declaration", "function_definition"},
            "r": {"function_definition"},
            "sql": {"create_function_statement"},
            "html": {"script_element"},
            "vue": {"script_element", "function_declaration"},
            "svelte": {"script_element", "function_declaration"},
            "dart": {"function_declaration", "method_declaration"},
            "elm": {"function_declaration"},
            "haskell": {"function_declaration"},
            "ocaml": {"function_declaration"},
            "fsharp": {"function_declaration"},
            "erlang": {"function_declaration"},
            "elixir": {"function_declaration"},
            "clojure": {"function_declaration"},
            "proto": {"rpc", "service"},
            "cmake": {"function"},
            "make": {"rule"},
        }
        types = func_types.get(lang, set())
        return node.type in types

    def _is_control_node(self, node, lang: str) -> bool:
        """判断节点是否为控制流节点（增加嵌套深度）。"""
        control_types = {
            "python": {"if_statement", "for_statement", "while_statement",
                      "with_statement", "try_statement"},
            "javascript": {"if_statement", "for_statement", "while_statement",
                          "switch_statement", "try_statement", "with_statement"},
            "typescript": {"if_statement", "for_statement", "while_statement",
                          "switch_statement", "try_statement"},
            "jsx": {"if_statement", "for_statement", "while_statement",
                   "switch_statement", "try_statement"},
            "tsx": {"if_statement", "for_statement", "while_statement",
                   "switch_statement", "try_statement"},
            "java": {"if_statement", "for_statement", "while_statement",
                    "switch_statement", "try_statement"},
            "go": {"if_statement", "for_statement", "switch_statement", "select_statement"},
            "rust": {"if_expression", "for_expression", "while_expression",
                    "match_expression", "loop_expression"},
            "c": {"if_statement", "for_statement", "while_statement", "switch_statement"},
            "cpp": {"if_statement", "for_statement", "while_statement",
                   "switch_statement", "try_statement"},
            "c_sharp": {"if_statement", "for_statement", "while_statement",
                       "switch_statement", "try_statement"},
            "ruby": {"if", "unless", "while", "until", "for", "case"},
            "php": {"if_statement", "for_statement", "while_statement",
                   "switch_statement", "try_statement"},
            "swift": {"if_statement", "for_statement", "while_statement",
                     "switch_statement", "do_statement"},
            "kotlin": {"if_statement", "for_statement", "while_statement",
                      "when_statement", "try_statement"},
            "scala": {"if_statement", "for_statement", "while_statement",
                     "match_statement", "try_statement"},
            "bash": {"if_statement", "for_statement", "while_statement", "case_statement"},
            "lua": {"if_statement", "for_statement", "while_statement", "repeat_statement"},
            "r": {"if_statement", "for_statement", "while_statement", "repeat_statement"},
            "dart": {"if_statement", "for_statement", "while_statement",
                    "switch_statement", "try_statement"},
            "elm": {"if_expression", "case_expression"},
            "haskell": {"if_expression", "case_expression"},
            "ocaml": {"if_expression", "match_expression"},
            "fsharp": {"if_expression", "match_expression"},
            "erlang": {"if_expression", "case_expression"},
            "elixir": {"if_expression", "case_expression"},
            "clojure": {"if_expression", "case_expression"},
            "cmake": {"if_statement", "foreach_statement", "while_statement"},
            "make": {"if_statement", "foreach_statement"},
        }
        types = control_types.get(lang, set())
        return node.type in types

    def _get_function_name(self, node, lang: str) -> Optional[str]:
        """获取函数名（根据语言定制）。"""
        # 通用：查找 identifier 或 property_identifier
        for child in node.children:
            if child.type in {"identifier", "property_identifier", "name", "method_name"}:
                return child.text.decode('utf-8')
        # 特殊处理某些语言
        if lang in {"ruby"}:
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode('utf-8')
        if lang in {"bash"}:
            for child in node.children:
                if child.type == "word":
                    return child.text.decode('utf-8')
        if lang in {"lua"}:
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode('utf-8')
        return None


# 全局 Tree-sitter 分析器实例
_ts_analyzer = TreeSitterAnalyzer()


# ============================================================
# 扫描器
# ============================================================

class Scanner:
    """合规扫描器。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._script_path = Path(__file__).resolve()
        self._ts_analyzer = _ts_analyzer
        logger.info("Scanner initialized with root: %s", self.root)

    def scan_all(self) -> ScanResult:
        logger.info("Starting full scan...")
        start_time = time.time()

        result = ScanResult(
            dir_violations=self.scan_dir_violations(),
            file_violations=self.scan_file_violations(),
            func_violations=self.scan_func_violations(),
            nesting_violations=self.scan_nesting_violations(),
            forbidden_violations=self.scan_forbidden_dir_violations(),
            empty_dir_violations=self.scan_empty_dir_violations(),
            filename_violations=self.scan_filename_violations(),
            readme_violations=self.scan_readme_violations(),
        )

        if ENABLE_SYNTAX_CHECK:
            result.syntax_violations = self.scan_syntax_errors()

        elapsed = time.time() - start_time
        logger.info("Scan completed in %.2fs, found %d violations", elapsed, result.total)
        return result

    def _is_src_package_wrapper(self, directory: Path, children: List[Path]) -> bool:
        """豁免形如 src/<与根目录同名的包> 的单子项目录。

        仅当以下条件同时满足时才生效：
        1. directory 是根目录下名为 "src" 的直接子目录；
        2. 该 "src" 目录下恰好存在一个与检查器根目录名相同（大小写、"-"/"_" 视为等价）的子文件夹。
        """
        if directory.name != "src":
            return False
        if directory.parent.resolve() != self.root.resolve():
            return False
        if len(children) != 1:
            return False
        only_child = children[0]
        if not only_child.is_dir():
            return False
        normalize = lambda s: s.lower().replace("-", "").replace("_", "")
        return normalize(only_child.name) == normalize(self.root.name)

    def scan_dir_violations(self) -> List[DirViolation]:
        violations: List[DirViolation] = []
        for dirpath, _dirnames, _filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            if is_exempt_dir(self.root, current):
                continue
            children = countable_children(current)
            if self._is_src_package_wrapper(current, children):
                continue
            if len(children) > MAX_CHILDREN:
                violations.append(
                    DirViolation(
                        path=current,
                        child_count=len(children),
                        children=tuple(item.name for item in children),
                        kind="over_max",
                    )
                )
            elif not satisfies_min_children(children):
                violations.append(
                    DirViolation(
                        path=current,
                        child_count=len(children),
                        children=tuple(item.name for item in children),
                        kind="under_min",
                    )
                )
        violations.sort(key=lambda item: (item.kind != "over_max", -item.child_count, str(item.path).lower()))
        logger.debug("Dir violations: %d", len(violations))
        return violations

    def scan_file_violations(self) -> List[FileLineViolation]:
        violations: List[FileLineViolation] = []
        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in EXT_TO_TS_LANG:
                    continue
                filepath = current / fname
                if filepath.resolve() == self._script_path:
                    continue
                if _is_file_exempt_for_lines(self.root, filepath):
                    continue
                try:
                    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError as e:
                    logger.warning("Failed to read file %s: %s", filepath, e)
                    continue
                count = len(lines)
                is_init = _is_init_py(filepath)
                ext = filepath.suffix.lower()
                is_lower_exempt = is_init or ext in FILE_LOWER_EXEMPT_EXTS
                is_upper_exempt = ext in FILE_UPPER_EXEMPT_EXTS
                if not is_lower_exempt and count < FILE_MIN_LINES:
                    violations.append(
                        FileLineViolation(
                            path=filepath,
                            line_count=count,
                            kind="under_min"
                        )
                    )
                if not is_upper_exempt and count > FILE_HARD_MAX_LINES:
                    violations.append(
                        FileLineViolation(
                            path=filepath,
                            line_count=count,
                            kind="over_hard"
                        )
                    )
                elif not is_upper_exempt and count > FILE_MAX_LINES:
                    violations.append(
                        FileLineViolation(
                            path=filepath,
                            line_count=count,
                            kind="over_soft"
                        )
                    )
        violations.sort(key=lambda v: (v.kind != "over_hard", -v.line_count, str(v.path).lower()))
        logger.debug("File line violations: %d", len(violations))
        return violations

    def scan_func_violations(self) -> List[FuncLengthViolation]:
        violations: List[FuncLengthViolation] = []
        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in EXT_TO_TS_LANG:
                    continue
                filepath = current / fname
                if filepath.resolve() == self._script_path:
                    continue
                if _is_file_exempt_for_lines(self.root, filepath):
                    continue
                try:
                    source = filepath.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    logger.warning("Failed to read file %s: %s", filepath, e)
                    continue
                lang = EXT_TO_TS_LANG[ext]
                if not is_ts_language_supported(lang):
                    raise RuntimeError(
                        f"No tree-sitter language for {filepath} ({lang!r}); no fallback"
                    )
                fv, _ = self._ts_analyzer.analyze(source, filepath, lang)
                violations.extend(fv)

        violations.sort(key=lambda v: (-v.line_count, str(v.path).lower()))
        logger.debug("Function length violations: %d", len(violations))
        return violations

    def scan_nesting_violations(self) -> List[NestingViolation]:
        violations: List[NestingViolation] = []
        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in EXT_TO_TS_LANG:
                    continue
                filepath = current / fname
                if filepath.resolve() == self._script_path:
                    continue
                if _is_file_exempt_for_lines(self.root, filepath):
                    continue
                try:
                    source = filepath.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    logger.warning("Failed to read file %s: %s", filepath, e)
                    continue
                lang = EXT_TO_TS_LANG[ext]
                if not is_ts_language_supported(lang):
                    raise RuntimeError(
                        f"No tree-sitter language for {filepath} ({lang!r}); no fallback"
                    )
                _, nv = self._ts_analyzer.analyze(source, filepath, lang)
                violations.extend(nv)

        violations.sort(key=lambda v: (-v.depth, str(v.path).lower()))
        logger.debug("Nesting violations: %d", len(violations))
        return violations

    def scan_forbidden_dir_violations(self) -> List[ForbiddenDirViolation]:
        violations: List[ForbiddenDirViolation] = []
        for dirpath, dirnames, _filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            for d in dirnames:
                if d.lower() in FORBIDDEN_DIR_NAMES:
                    violations.append(
                        ForbiddenDirViolation(
                            path=current / d,
                            name=d
                        )
                    )
        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("Forbidden dir violations: %d", len(violations))
        return violations

    def scan_empty_dir_violations(self) -> List[EmptyDirViolation]:
        violations: List[EmptyDirViolation] = []
        for dirpath, _dirnames, _filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            if self._is_empty_dir(current) and not self._is_empty_dir_exempt(current):
                violations.append(EmptyDirViolation(path=current))
        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("Empty dir violations: %d", len(violations))
        return violations

    def scan_filename_violations(self) -> List[FileNameViolation]:
        violations: List[FileNameViolation] = []
        import unicodedata
        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            if is_exempt_subtree(self.root, current):
                continue
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in EXT_TO_TS_LANG:
                    continue
                filepath = current / fname
                if filepath.resolve() == self._script_path:
                    continue
                if _is_file_exempt_for_lines(self.root, filepath):
                    continue
                if fname in FILENAME_EXEMPT_NAMES:
                    continue
                if fname.startswith("."):
                    continue
                stem = Path(fname).stem
                reasons = []
                if "-" in stem:
                    reasons.append("contains '-'")
                if "." in stem:
                    reasons.append("contains '.'")
                if stem.count("_") > 1:
                    reasons.append("more than 1 '_'")
                if sum(1 for ch in stem if ch.isdigit()) >= 3:
                    reasons.append("contains >= 3 digit chars")
                if "mixin" in stem.lower():
                    reasons.append("contains 'mixin'")
                if "impl" in stem.lower():
                    reasons.append("contains 'impl'")
                if stem.startswith("_"):
                    reasons.append("starts with '_'")
                if any(ch.isupper() for ch in stem):
                    reasons.append("contains uppercase letter")
                if not stem or not (stem[0].isalpha() and stem[0].islower()):
                    reasons.append("first char is not a lowercase letter")
                segments = stem.split("_")
                for seg in segments:
                    seg_len = len(seg)
                    if not (1 <= seg_len <= 14):
                        reasons.append(
                            "segment '{0}' length {1} not in [1,14]".format(seg, seg_len)
                        )
                run_len = 1
                for i in range(1, len(stem)):
                    if stem[i] == stem[i - 1]:
                        run_len += 1
                        if run_len > 5:
                            reasons.append("contains run of same char > 5")
                            break
                    else:
                        run_len = 1
                for ch in stem:
                    cat = unicodedata.category(ch)
                    if cat.startswith("So") or cat.startswith("Sk"):
                        reasons.append("contains emoji/symbol")
                        break
                if reasons:
                    violations.append(
                        FileNameViolation(
                            path=filepath,
                            reason="; ".join(reasons)
                        )
                    )
        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("Filename violations: %d", len(violations))
        return violations

    def scan_readme_violations(self) -> List[ReadmeViolation]:
        violations: List[ReadmeViolation] = []
        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            if is_exempt_subtree(self.root, current):
                continue
            for fname in filenames:
                if fname != "README.md":
                    continue
                filepath = current / fname
                if filepath.resolve() == self._script_path:
                    continue
                if _is_file_exempt_for_lines(self.root, filepath):
                    continue
                if is_readme_path_allowed(self.root, filepath):
                    continue
                violations.append(ReadmeViolation(path=filepath))
        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("README violations: %d", len(violations))
        return violations

    def scan_syntax_errors(self) -> List[SyntaxViolation]:
        violations: List[SyntaxViolation] = []
        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in EXT_TO_TS_LANG:
                    continue
                filepath = current / fname
                if filepath.resolve() == self._script_path:
                    continue
                if _is_file_exempt_for_lines(self.root, filepath):
                    continue
                try:
                    source = filepath.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    logger.warning("Failed to read file %s: %s", filepath, e)
                    continue
                lang = EXT_TO_TS_LANG[ext]
                if not is_ts_language_supported(lang):
                    raise RuntimeError(
                        f"No tree-sitter language for {filepath} ({lang!r}); no fallback"
                    )
                try:
                    parser = _TS_GET_PARSER(lang)
                    tree = parser.parse(source.encode("utf-8"))
                except Exception as e:
                    raise RuntimeError(
                        f"tree-sitter syntax check failed for {filepath} (no fallback): {e}"
                    ) from e
                if self._has_error_node(tree.root_node):
                    violations.append(
                        SyntaxViolation(
                            path=filepath,
                            language=lang,
                            line=None,
                            message="Syntax error detected by tree-sitter",
                        )
                    )

        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("Syntax violations: %d", len(violations))
        return violations

    def _has_error_node(self, node) -> bool:
        if node.type == "ERROR":
            return True
        for child in node.children:
            if self._has_error_node(child):
                return True
        return False

    # ===== 私有方法（回退逻辑） =====

    def _check_python_func_bodies(
        self,
        tree: ast.Module,
        filepath: Path
    ) -> Tuple[List[FuncLengthViolation], List[NestingViolation]]:
        func_violations: List[FuncLengthViolation] = []
        nesting_violations: List[NestingViolation] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                func_lines = node.end_lineno - node.lineno + 1
                if func_lines > FUNC_MAX_LINES:
                    func_violations.append(
                        FuncLengthViolation(
                            path=filepath,
                            func_name=node.name,
                            line_count=func_lines,
                            start_line=node.lineno,
                        )
                    )
            max_depth = self._max_nesting_depth(node)
            if max_depth > MAX_NESTING_DEPTH:
                nesting_violations.append(
                    NestingViolation(
                        path=filepath,
                        func_name=node.name,
                        depth=max_depth,
                        line=node.lineno,
                    )
                )
        return func_violations, nesting_violations

    def _max_nesting_depth(self, func_node: ast.AST) -> int:
        max_depth = 0
        def _walk(node: ast.AST, depth: int) -> None:
            nonlocal max_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    _walk(child, depth + 1)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                else:
                    _walk(child, depth)
            if depth > max_depth:
                max_depth = depth
        _walk(func_node, 0)
        return max_depth

    def _check_js_functions(
        self,
        source: str,
        filepath: Path
    ) -> Tuple[List[FuncLengthViolation], List[NestingViolation]]:
        func_v: List[FuncLengthViolation] = []
        nesting_v: List[NestingViolation] = []
        lines = source.splitlines()
        total = len(lines)
        js_func_re = re.compile(
            r"(?:^|[\s;])(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)"
            r"|(?:^|[\s;,])(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)"
            r"|(?:^|\s)(\w+)\s*\(.*\)\s*\{",
            re.MULTILINE,
        )
        js_ctrl_re = re.compile(
            r"^\s*(?:if|else\s+if|for|while|switch|try|catch|do)\s*[\({]",
        )
        i = 0
        while i < total:
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                i += 1
                continue
            m = js_func_re.search(line)
            if not m:
                i += 1
                continue
            func_name = m.group(1) or m.group(2) or m.group(3) or "anon"
            start_line = i + 1
            brace_line = i
            for j in range(i, min(i + 5, total)):
                if "{" in lines[j]:
                    brace_line = j
                    break
            else:
                i += 1
                continue
            depth = 0
            end_line = brace_line
            for j in range(brace_line, total):
                for ch in lines[j]:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end_line = j
                            break
                if depth == 0:
                    break
            func_lines = end_line - start_line + 1
            if func_lines > FUNC_MAX_LINES:
                func_v.append(
                    FuncLengthViolation(
                        path=filepath,
                        func_name=func_name,
                        line_count=func_lines,
                        start_line=start_line,
                    )
                )
            max_indent = 0
            for j in range(brace_line + 1, end_line):
                ln = lines[j]
                stripped_line = ln.lstrip()
                if not stripped_line or stripped_line.startswith("//"):
                    continue
                indent = len(ln) - len(stripped_line)
                if js_ctrl_re.match(stripped_line) and indent > max_indent:
                    max_indent = indent
            if max_indent >= 8:
                est_depth = max_indent // 2
                if est_depth > MAX_NESTING_DEPTH:
                    nesting_v.append(
                        NestingViolation(
                            path=filepath,
                            func_name=func_name,
                            depth=est_depth,
                            line=start_line,
                        )
                    )
            i = end_line + 1
        return func_v, nesting_v

    def _is_empty_dir(self, path: Path) -> bool:
        try:
            for item in path.iterdir():
                if item.name in RUNTIME_DIR_NAMES:
                    continue
                if item.is_file() and item.suffix in RUNTIME_FILE_SUFFIXES:
                    continue
                if item.resolve() == self._script_path:
                    continue
                return False
            return True
        except OSError:
            return False

    def _is_empty_dir_exempt(self, directory: Path) -> bool:
        if directory.resolve() == self.root:
            return True
        if directory.name in EMPTY_DIR_EXEMPT_NAMES:
            return True
        if directory.name.startswith("."):
            return True
        return False

    def _check_python_syntax(self, filepath: Path, source: str, violations: List[SyntaxViolation]) -> None:
        try:
            ast.parse(source, filename=str(filepath))
        except SyntaxError as e:
            line_no = e.lineno if hasattr(e, 'lineno') else None
            violations.append(
                SyntaxViolation(
                    path=filepath,
                    language="python",
                    line=line_no,
                    message=str(e)
                )
            )

    def _check_javascript_syntax(self, filepath: Path, source: str, violations: List[SyntaxViolation]) -> None:
        try:
            result = subprocess.run(
                ["node", "--check", str(filepath)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=5
            )
            if result.returncode != 0:
                err_output = result.stderr or ''
                line_no = None
                msg = err_output.strip()
                match = re.search(r"\((\d+):\d+\)", msg)
                if match:
                    line_no = int(match.group(1))
                violations.append(
                    SyntaxViolation(
                        path=filepath,
                        language="javascript",
                        line=line_no,
                        message=msg
                    )
                )
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("node not found or timed out, using simple regex syntax check for JS")
            self._check_js_syntax_simple(filepath, source, violations)

    def _check_js_syntax_simple(self, filepath: Path, source: str, violations: List[SyntaxViolation]) -> None:
        stack = []
        line_no = None
        for i, line in enumerate(source.splitlines(), 1):
            for ch in line:
                if ch in "({[":
                    stack.append((ch, i))
                elif ch in ")}]":
                    if not stack:
                        line_no = i
                        break
                    last, _ = stack.pop()
                    if (ch == ')' and last != '(') or \
                       (ch == '}' and last != '{') or \
                       (ch == ']' and last != '['):
                        line_no = i
                        break
            if line_no:
                break
        if line_no:
            violations.append(
                SyntaxViolation(
                    path=filepath,
                    language="javascript",
                    line=line_no,
                    message="Unmatched bracket or parenthesis"
                )
            )


# ============================================================
# 报告生成器
# ============================================================

class ReportGenerator:
    """合规报告生成器。"""

    def __init__(self, root: Path, result: ScanResult):
        self.root = root
        self.result = result

    def generate(self) -> str:
        lines = [
            "=" * 70,
            "项目合规检查报告",
            "=" * 70,
            f"根目录: {self.root.resolve()}",
            f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== 目录子项 ===",
            f"约束: {MIN_CHILDREN} <= 受检目录每层直接子项数 <= {MAX_CHILDREN}",
            f"违规数: {len(self.result.dir_violations)}",
            "",
            "=== 文件行数 ===",
            f"约束: {FILE_MIN_LINES} <= 行数 <= {FILE_MAX_LINES}（硬上限 {FILE_HARD_MAX_LINES}），__init__.py 豁免下限",
            f"违规数: {len(self.result.file_violations)}",
            "",
            "=== 函数长度 ===",
            f"约束: 函数 <= {FUNC_MAX_LINES} 行",
            f"违规数: {len(self.result.func_violations)}",
            "",
            "=== 嵌套深度 ===",
            f"约束: 控制流嵌套 <= {MAX_NESTING_DEPTH} 层",
            f"违规数: {len(self.result.nesting_violations)}",
            "",
            "=== 禁用目录名 ===",
            f"约束: 目录名不得为 {', '.join(sorted(FORBIDDEN_DIR_NAMES))}",
            f"违规数: {len(self.result.forbidden_violations)}",
            "",
            "=== 空目录 ===",
            f"约束: 不允许存在空目录（豁免：{', '.join(sorted(EMPTY_DIR_EXEMPT_NAMES))}）",
            f"违规数: {len(self.result.empty_dir_violations)}",
            "",
        ]

        lines.extend([
            "=== 文件命名 ===",
            "约束: 文件名(不含扩展名)不得含 - . emoji mixin impl 数字，下划线最多1个",
            "违规数: %d" % len(self.result.filename_violations),
            "",
            "=== README 位置 ===",
            "约束: README.md 仅允许在各 Git 仓库根目录",
            "违规数: %d" % len(self.result.readme_violations),
            "",
        ])

        if ENABLE_SYNTAX_CHECK:
            lines.extend([
                "=== 语法错误 ===",
                f"约束: 所有支持的语言文件必须语法正确",
                f"违规数: {len(self.result.syntax_violations)}",
                "",
            ])

        if self.result.is_clean:
            lines.append("全部合规。")
            return "\n".join(lines) + "\n"

        lines.append(f"共 {self.result.total} 个违规，请修复后重新检查。")
        lines.append(f"IMPORTANT: You need to adjust the project code by refactoring to ensure no code logic is missed, fix a total of {self.result.total} violations, and start full implementation")
        lines.append("")

        if self.result.dir_violations:
            lines.append("── 目录子项违规 ──")
            for item in self.result.dir_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                label = "子项过多" if item.kind == "over_max" else "子项过少"
                lines.append(f"  {rel} ({item.child_count} 项) [{label}]")
                for name in item.children:
                    lines.append(f"    - {name}")
            lines.append("")

        if self.result.file_violations:
            lines.append("── 文件行数违规 ──")
            for item in self.result.file_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                label = {
                    "under_min": f"过少(< {FILE_MIN_LINES})",
                    "over_soft": f"过多(> {FILE_MAX_LINES})",
                    "over_hard": f"严重过多(> {FILE_HARD_MAX_LINES})",
                }[item.kind]
                lines.append(f"  {rel} : {item.line_count} 行 [{label}]")
            lines.append("")

        if self.result.func_violations:
            lines.append("── 函数长度违规 ──")
            for item in self.result.func_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel} : {item.func_name}() = {item.line_count} 行 (L{item.start_line})")
            lines.append("")

        if self.result.nesting_violations:
            lines.append("── 嵌套深度违规 ──")
            for item in self.result.nesting_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel} : {item.func_name}() 深度 {item.depth} (L{item.line})")
            lines.append("")

        if self.result.forbidden_violations:
            lines.append("── 禁用目录名违规 ──")
            for item in self.result.forbidden_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel} [名称: {item.name}]")
            lines.append("")

        if self.result.empty_dir_violations:
            lines.append("── 空目录违规 ──")
            for item in self.result.empty_dir_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel} (空目录)")
            lines.append("")

        if ENABLE_SYNTAX_CHECK and self.result.syntax_violations:
            lines.append("── 语法错误违规 ──")
            for item in self.result.syntax_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                line_info = f"L{item.line}" if item.line is not None else ""
                lines.append(f"  {rel} [{item.language}] {line_info}: {item.message}")
            lines.append("")

        if self.result.filename_violations:
            lines.append("── 文件命名违规 ──")
            for item in self.result.filename_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel} : {item.reason}")
            lines.append("")

        if self.result.readme_violations:
            lines.append("── README 位置违规 ──")
            for item in self.result.readme_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ============================================================
# 主程序
# ============================================================

def main() -> int:
    global MAX_CHILDREN, ENABLE_SYNTAX_CHECK

    parser = argparse.ArgumentParser(
        description="项目合规检查：目录子项、文件行数、函数长度、嵌套深度、禁用目录名、空目录、语法错误",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    # 检查项目根目录
  %(prog)s --max 5            # 自定义子项限制
  %(prog)s --output report.txt # 指定输出文件
  %(prog)s --no-syntax        # 禁用语法检查
        """
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="检查根路径，默认项目根目录"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=MAX_CHILDREN,
        help=f"每层最大子项数，默认 {MAX_CHILDREN}"
    )
    parser.add_argument(
        "--output",
        default=REPORT_OUTPUT,
        help=f"报告输出路径，默认 {REPORT_OUTPUT}"
    )
    parser.add_argument(
        "--no-syntax",
        action="store_true",
        help="禁用语法错误检查"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只输出报告文件"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式，显示详细日志"
    )

    args = parser.parse_args()

    if args.debug:
        configure(level="DEBUG")
        logger.setLevel("DEBUG")
        logger.debug("Debug mode enabled")

    if args.no_syntax:
        ENABLE_SYNTAX_CHECK = False
        logger.info("Syntax check disabled")

    try:
        root = _resolve_scan_root()
        if args.path != ".":
            target = Path(args.path)
            if target.is_absolute():
                root = target
            else:
                root = root / args.path

        if not root.is_dir():
            logger.error("Specified path is not a directory: %s", root)
            return 1

        logger.info("Scanning: %s", root)

        scanner = Scanner(root)
        original_max_children = MAX_CHILDREN
        MAX_CHILDREN = args.max
        result = scanner.scan_all()
        MAX_CHILDREN = original_max_children

        generator = ReportGenerator(root, result)
        report = generator.generate()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        for line in report.splitlines():
            logger.info(line)

        if result.is_clean:
            logger.debug("All checks passed. Report written to %s", output_path)
            return 0
        else:
            logger.info("Found %d violations. Report written to %s", result.total, output_path)
            return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
