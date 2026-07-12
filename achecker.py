#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目合规检查：目录子项数量、文件行数、函数长度、嵌套深度、禁用目录名、空目录、语法错误。"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from echotools.logger import configure, get_logger

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

# 目录豁免配置
EXEMPT_SUBTREE_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("provider-self", "scripts"),
    ("provider-self", "plugins"),
    ("provider-self", "persist"),
    ("logs",),
    ("pre-plugin",),
    ("persist",),
    ("template",),
    ("tmp",),
)

EXEMPT_SHALLOW_REL_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("provider-plugin",),
    ("provider-docs",),
    ("provider-self",),
)

# 运行时目录/文件（自动忽略）
RUNTIME_DIR_NAMES: Set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

RUNTIME_FILE_SUFFIXES: Tuple[str, ...] = (".pyc", ".pyo")

# 文件行数约束
FILE_MIN_LINES: int = 200
FILE_MAX_LINES: int = 400
FILE_HARD_MAX_LINES: int = 800
FILE_EXEMPT_NAMES: Set[str] = {"__init__.py", "conftest.py", "init.js"}

# 文件排除配置
EXEMPT_FILE_REL_PATHS: Tuple[str, ...] = ()  # 相对路径排除
EXEMPT_FILE_NAMES: Set[str] = {"accounts.py", "file_merger.py"}  # 文件名排除（不看路径）

# 函数长度约束
FUNC_MAX_LINES: int = 50

# 嵌套深度约束
MAX_NESTING_DEPTH: int = 4

# 禁用目录名
FORBIDDEN_DIR_NAMES: Set[str] = {
    "infra", "domain", "application",
    "infrastructure", "interfaces",
    "js", "css", "static", "assets", "public",
}

# 豁免子树（不扫描文件行数等）
FILE_EXEMPT_SUBTREE_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("provider-self", "scripts"),
    ("provider-self", "plugins"),
    ("provider-self", "persist"),
    ("logs",),
    ("persist",),
    ("template",),
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
    "persist",
}

# 报告输出配置
REPORT_OUTPUT: str = "abnormis.txt"

# 语法检查配置
ENABLE_SYNTAX_CHECK: bool = True  # 是否启用语法错误检查

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
    language: str  # "python" or "javascript"
    line: Optional[int]
    message: str


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

    def __post_init__(self):
        self.dir_violations = self.dir_violations or []
        self.file_violations = self.file_violations or []
        self.func_violations = self.func_violations or []
        self.nesting_violations = self.nesting_violations or []
        self.forbidden_violations = self.forbidden_violations or []
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
            len(self.empty_dir_violations)
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
    """解析扫描根目录。
    
    使用 SCAN_ROOT 配置，如果未定义或不存在则回退到脚本所在目录。
    """
    fallback = Path(__file__).resolve().parent

    try:
        candidate = SCAN_ROOT
    except NameError:
        return fallback

    if candidate is None:
        return fallback

    if not isinstance(candidate, str) or not candidate.strip():
        return fallback

    try:
        resolved = Path(candidate).resolve()
        if resolved.is_dir():
            return resolved
        else:
            logger.warning("SCAN_ROOT '%s' is not a directory, falling back to script directory", candidate)
            return fallback
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Failed to resolve SCAN_ROOT '%s': %s, falling back to script directory", candidate, e)
        return fallback


def _relative_parts(root: Path, path: Path) -> Optional[Tuple[str, ...]]:
    """获取路径相对于根目录的 parts。"""
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return None


# ============================================================
# 路径检查逻辑
# ============================================================

def is_shallow_exempt(root: Path, directory: Path) -> bool:
    """检查目录是否在浅层豁免列表中。"""
    parts = _relative_parts(root, directory)
    if parts is None:
        return False
    return parts in EXEMPT_SHALLOW_REL_PATHS


def is_exempt_subtree(root: Path, directory: Path) -> bool:
    """检查目录是否在子树豁免列表中。"""
    root_resolved = root.resolve()
    directory_resolved = directory.resolve()

    if directory_resolved == root_resolved:
        return True

    parts = _relative_parts(root, directory)
    if parts is None:
        return False

    if any(part.startswith(".") for part in parts):
        return True

    exempt_dir_names = {"logs", "docs-src", "tests", "template", "tmp"}
    if directory.name in exempt_dir_names:
        return True

    for prefix in EXEMPT_SUBTREE_PREFIXES:
        if len(parts) >= len(prefix) and parts[:len(prefix)] == prefix:
            return True

    return False


def is_exempt_dir(root: Path, directory: Path) -> bool:
    """检查目录是否豁免（组合检查）。"""
    return is_exempt_subtree(root, directory) or is_shallow_exempt(root, directory)


def _is_file_exempt_for_lines(root: Path, filepath: Path) -> bool:
    """检查文件是否从行数/函数/嵌套检查中豁免。"""
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

    for prefix in FILE_EXEMPT_SUBTREE_PREFIXES:
        if len(parts) >= len(prefix) and parts[:len(prefix)] == prefix:
            return True

    return False


def _is_init_py(filepath: Path) -> bool:
    """检查是否为 __init__.py 文件。"""
    return filepath.name == "__init__.py"


def countable_children(directory: Path) -> List[Path]:
    """获取目录中可计数的子项（排除运行时文件和目录）。"""
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


def _walk_dirs(root: Path):
    """遍历目录树，生成目录路径和内容信息。"""
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
            if is_exempt_subtree(root, item):
                continue
            stack.append(item)


# ============================================================
# 扫描器
# ============================================================

class Scanner:
    """合规扫描器。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._script_path = Path(__file__).resolve()
        logger.info("Scanner initialized with root: %s", self.root)

    def scan_all(self) -> ScanResult:
        """执行所有扫描，返回完整结果。"""
        logger.info("Starting full scan...")
        start_time = time.time()

        result = ScanResult(
            dir_violations=self.scan_dir_violations(),
            file_violations=self.scan_file_violations(),
            func_violations=self.scan_func_violations(),
            nesting_violations=self.scan_nesting_violations(),
            forbidden_violations=self.scan_forbidden_dir_violations(),
            empty_dir_violations=self.scan_empty_dir_violations(),
        )

        if ENABLE_SYNTAX_CHECK:
            result.syntax_violations = self.scan_syntax_errors()

        elapsed = time.time() - start_time
        logger.info("Scan completed in %.2fs, found %d violations", elapsed, result.total)

        return result

    def scan_dir_violations(self) -> List[DirViolation]:
        """扫描目录子项违规。"""
        violations: List[DirViolation] = []

        for dirpath, _dirnames, _filenames in _walk_dirs(self.root):
            current = Path(dirpath)

            if is_exempt_dir(self.root, current):
                continue

            children = countable_children(current)
            if len(children) > MAX_CHILDREN:
                violations.append(
                    DirViolation(
                        path=current,
                        child_count=len(children),
                        children=tuple(item.name for item in children),
                    )
                )

        violations.sort(key=lambda item: (-item.child_count, str(item.path).lower()))
        logger.debug("Dir violations: %d", len(violations))
        return violations

    def scan_file_violations(self) -> List[FileLineViolation]:
        """扫描文件行数违规。"""
        violations: List[FileLineViolation] = []

        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)

            for fname in filenames:
                if not (fname.endswith(".py") or fname.endswith(".js")):
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

                if not is_init and count < FILE_MIN_LINES:
                    violations.append(
                        FileLineViolation(
                            path=filepath,
                            line_count=count,
                            kind="under_min"
                        )
                    )

                if count > FILE_HARD_MAX_LINES:
                    violations.append(
                        FileLineViolation(
                            path=filepath,
                            line_count=count,
                            kind="over_hard"
                        )
                    )
                elif count > FILE_MAX_LINES:
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
        """扫描函数长度违规。"""
        violations: List[FuncLengthViolation] = []

        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)

            for fname in filenames:
                if not (fname.endswith(".py") or fname.endswith(".js")):
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

                if fname.endswith(".py"):
                    try:
                        tree = ast.parse(source, filename=str(filepath))
                    except SyntaxError:
                        continue
                    fv, _ = self._check_python_func_bodies(tree, filepath)
                else:
                    fv, _ = self._check_js_functions(source, filepath)

                violations.extend(fv)

        violations.sort(key=lambda v: (-v.line_count, str(v.path).lower()))
        logger.debug("Function length violations: %d", len(violations))
        return violations

    def scan_nesting_violations(self) -> List[NestingViolation]:
        """扫描嵌套深度违规。"""
        violations: List[NestingViolation] = []

        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)

            for fname in filenames:
                if not (fname.endswith(".py") or fname.endswith(".js")):
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

                if fname.endswith(".py"):
                    try:
                        tree = ast.parse(source, filename=str(filepath))
                    except SyntaxError:
                        continue
                    _, nv = self._check_python_func_bodies(tree, filepath)
                else:
                    _, nv = self._check_js_functions(source, filepath)

                violations.extend(nv)

        violations.sort(key=lambda v: (-v.depth, str(v.path).lower()))
        logger.debug("Nesting violations: %d", len(violations))
        return violations

    def scan_forbidden_dir_violations(self) -> List[ForbiddenDirViolation]:
        """扫描禁用目录名违规。"""
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
        """扫描空目录违规。"""
        violations: List[EmptyDirViolation] = []

        for dirpath, _dirnames, _filenames in _walk_dirs(self.root):
            current = Path(dirpath)

            if self._is_empty_dir(current) and not self._is_empty_dir_exempt(current):
                violations.append(EmptyDirViolation(path=current))

        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("Empty dir violations: %d", len(violations))
        return violations

    def scan_syntax_errors(self) -> List[SyntaxViolation]:
        """扫描所有 .py 和 .js 文件的语法错误。"""
        violations: List[SyntaxViolation] = []

        for dirpath, _dirnames, filenames in _walk_dirs(self.root):
            current = Path(dirpath)

            for fname in filenames:
                if not (fname.endswith(".py") or fname.endswith(".js")):
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

                if fname.endswith(".py"):
                    self._check_python_syntax(filepath, source, violations)
                else:
                    self._check_javascript_syntax(filepath, source, violations)

        violations.sort(key=lambda v: str(v.path).lower())
        logger.debug("Syntax violations: %d", len(violations))
        return violations

    # ===== 私有方法 =====

    def _check_python_func_bodies(
        self,
        tree: ast.Module,
        filepath: Path
    ) -> Tuple[List[FuncLengthViolation], List[NestingViolation]]:
        """检查 Python 文件的函数长度和嵌套深度。"""
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
        """计算函数体内最深的控制流嵌套层级。"""
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
        """用正则检查 JS 文件的函数长度和嵌套深度。"""
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
        """检查目录是否为空（忽略运行时目录和文件）。"""
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
        """判断空目录检查是否豁免。"""
        if directory.resolve() == self.root:
            return True
        if directory.name in EMPTY_DIR_EXEMPT_NAMES:
            return True
        if directory.name.startswith("."):
            return True
        return False

    def _check_python_syntax(self, filepath: Path, source: str, violations: List[SyntaxViolation]) -> None:
        """检查 Python 文件语法。"""
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
        """检查 JavaScript 文件语法。"""
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
                err_lines = err_output.splitlines()
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
        """使用简单的括号匹配检查 JS 语法。"""
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
        """生成完整报告。"""
        lines = [
            "=" * 70,
            "项目合规检查报告",
            "=" * 70,
            f"根目录: {self.root.resolve()}",
            f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== 目录子项 ===",
            f"约束: 受检目录每层直接子项 <= {MAX_CHILDREN}",
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

        if ENABLE_SYNTAX_CHECK:
            lines.extend([
                "=== 语法错误 ===",
                f"约束: 所有 .py 和 .js 文件必须语法正确",
                f"违规数: {len(self.result.syntax_violations)}",
                "",
            ])

        if self.result.is_clean:
            lines.append("全部合规。")
            return "\n".join(lines) + "\n"

        lines.append(f"共 {self.result.total} 个违规，请修复后重新检查。")
        lines.append("")

        if self.result.dir_violations:
            lines.append("── 目录子项违规 ──")
            for item in self.result.dir_violations:
                rel = item.path.resolve().relative_to(self.root.resolve())
                lines.append(f"  {rel} ({item.child_count} 项)")
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

        return "\n".join(lines).rstrip() + "\n"


# ============================================================
# 主程序
# ============================================================

def main() -> int:
    """主入口函数。"""
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

    # 调整日志级别
    if args.debug:
        configure(level="DEBUG")
        logger.setLevel("DEBUG")
        logger.debug("Debug mode enabled")

    # 调整语法检查配置
    if args.no_syntax:
        ENABLE_SYNTAX_CHECK = False
        logger.info("Syntax check disabled")

    try:
        # 解析根目录
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

        # 执行扫描
        scanner = Scanner(root)
        original_max_children = MAX_CHILDREN
        MAX_CHILDREN = args.max
        result = scanner.scan_all()
        MAX_CHILDREN = original_max_children

        # 生成报告
        generator = ReportGenerator(root, result)
        report = generator.generate()

        # 保存报告
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        # 输出报告
        for line in report.splitlines():
            logger.info(line)

        # 返回状态
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
