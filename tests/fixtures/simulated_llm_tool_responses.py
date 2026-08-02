"""模拟 LLM 工具调用响应语料（非真实模型输出，仅用于解析回归）。

覆盖常见模型写法：thinking 前置、function_calls 外壳、属性乱序、
单双引号、markdown 围栏、转义下划线、并行多工具、JSON parameters、
type 注解、中英混排、正文夹杂等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SimulatedCase:
    """一条模拟 LLM 完整回复及其期望解析结果。"""

    id: str
    description: str
    response: str
    expect_names: List[str]
    expect_args: List[Dict[str, Any]]
    expect_clean_substrings: List[str] = field(default_factory=list)
    expect_clean_absent: List[str] = field(default_factory=lambda: ["entml:invoke", "entml:parameter", "entml:function_calls"])
    expect_thinking: Optional[str] = None
    # 若为 False，表示本条允许解析失败（仍不得标签泄露）
    expect_success: bool = True
    # 模型输出分支标签（用于矩阵测试分组）
    branch: str = "misc"
    # 除 TOOLS 外还需挂载的工具 schema 名
    extra_tools: tuple = ()


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Query weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "filters": {"type": "object"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run shell",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                    "env": {"type": "object"},
                },
                "required": ["command"],
            },
        },
    },
]

# Anthropic 兼容客户端 / rogator 链路常见工具（不在默认 TOOLS 内）
AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["file_path", "contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PowerShell",
            "description": "Run PowerShell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoList",
            "description": "Manage todos",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "status": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
]

_TOOL_BY_NAME: Dict[str, Dict[str, Any]] = {
    t["function"]["name"]: t for t in TOOLS + AGENT_TOOLS
}


def tools_for_case(case: SimulatedCase) -> List[Dict[str, Any]]:
    """返回解析/流式测试应使用的完整 tools 列表。"""
    names = {t["function"]["name"] for t in TOOLS}
    out = list(TOOLS)
    for name in case.extra_tools:
        if name not in names and name in _TOOL_BY_NAME:
            out.append(_TOOL_BY_NAME[name])
            names.add(name)
    return out


SIMULATED_LLM_RESPONSES: List[SimulatedCase] = [
    SimulatedCase(
        id="canonical_bare_invoke",
        description="规范裸 invoke，无 thinking",
        branch="canonical_bare",
        response=(
            "我先查一下杭州天气。\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            '<entml:parameter name="unit">c</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "杭州", "unit": "c"}],
        expect_clean_substrings=["我先查一下杭州天气。"],
    ),
    SimulatedCase(
        id="parallel_two_tools_bare",
        description="规范裸 invoke 并行两工具（与提示词示例一致）",
        branch="parallel_multi_invoke",
        response=(
            "稍等，我同时查天气和景点。\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            "</entml:invoke>\n"
            '<entml:invoke name="search_web">\n'
            '<entml:parameter name="query">杭州西湖 周边景点</entml:parameter>\n'
            '<entml:parameter name="limit">5</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather", "search_web"],
        expect_args=[
            {"city": "杭州"},
            {"query": "杭州西湖 周边景点", "limit": 5},
        ],
        expect_clean_substrings=["稍等，我同时查天气和景点。"],
    ),
    SimulatedCase(
        id="thinking_then_bare_invoke",
        description="thinking 前置 + 裸 invoke（无 function_calls 外壳）",
        response=(
            "<entml:thinking>\n"
            "用户要杭州天气，应调用 get_weather，unit 用 c。\n"
            "</entml:thinking>\n"
            "好的，我来查询。\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            '<entml:parameter name="days">3</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "杭州", "days": 3}],
        expect_clean_substrings=["好的，我来查询。"],
        expect_thinking="用户要杭州天气，应调用 get_weather，unit 用 c。",
    ),
    SimulatedCase(
        id="thinking_then_wrapper",
        description="thinking + function_calls 外壳（最常见线上形态）",
        response=(
            "<entml:thinking>\n"
            "用户要杭州天气，应调用 get_weather，unit 用 c。\n"
            "</entml:thinking>\n"
            "好的，我来查询。\n"
            "<entml:function_calls>\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            '<entml:parameter name="days">3</entml:parameter>\n'
            "</entml:invoke>\n"
            "</entml:function_calls>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "杭州", "days": 3}],
        expect_clean_substrings=["好的，我来查询。"],
        expect_thinking="用户要杭州天气，应调用 get_weather，unit 用 c。",
    ),
    SimulatedCase(
        id="parallel_two_tools",
        description="同一回复并行两个工具",
        response=(
            "<entml:thinking>\n并行查天气和搜索景点。\n</entml:thinking>\n"
            "稍等，我同时查天气和景点。\n"
            "<entml:function_calls>\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            "</entml:invoke>\n"
            '<entml:invoke name="search_web">\n'
            '<entml:parameter name="query">杭州西湖 周边景点</entml:parameter>\n'
            '<entml:parameter name="limit">5</entml:parameter>\n'
            "</entml:invoke>\n"
            "</entml:function_calls>"
        ),
        expect_names=["get_weather", "search_web"],
        expect_args=[
            {"city": "杭州"},
            {"query": "杭州西湖 周边景点", "limit": 5},
        ],
        expect_clean_substrings=["稍等，我同时查天气和景点。"],
        expect_thinking="并行查天气和搜索景点。",
    ),
    SimulatedCase(
        id="type_attrs_reordered",
        description="模型常把 type 写在 name 前，并混用 int/str",
        response=(
            "检索中。\n"
            '<entml:invoke name="search_web">\n'
            '<entml:parameter type="str" name="query">上海 降雨 预报</entml:parameter>\n'
            '<entml:parameter type="int" name="limit">3</entml:parameter>\n'
            '<entml:parameter name="tags" type="array">["weather","shanghai"]</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["search_web"],
        expect_args=[
            {
                "query": "上海 降雨 预报",
                "limit": 3,
                "tags": ["weather", "shanghai"],
            }
        ],
        expect_clean_substrings=["检索中。"],
    ),
    SimulatedCase(
        id="single_quotes_everywhere",
        description="整段单引号属性（部分模型/转义产物）",
        response=(
            "调用工具：\n"
            "<entml:invoke name='get_weather'>\n"
            "<entml:parameter name='city'>北京</entml:parameter>\n"
            "<entml:parameter name='unit'>c</entml:parameter>\n"
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "北京", "unit": "c"}],
        expect_clean_substrings=["调用工具："],
    ),
    SimulatedCase(
        id="markdown_fenced_xml",
        description="把 invoke 包进 ```xml 代码块",
        response=(
            "按规范调用：\n"
            "```xml\n"
            '<entml:invoke name="read_file">\n'
            '<entml:parameter name="path">src/main.py</entml:parameter>\n'
            '<entml:parameter name="offset">0</entml:parameter>\n'
            '<entml:parameter name="limit">80</entml:parameter>\n'
            "</entml:invoke>\n"
            "```"
        ),
        expect_names=["read_file"],
        expect_args=[{"path": "src/main.py", "offset": 0, "limit": 80}],
        expect_clean_substrings=["按规范调用："],
        expect_clean_absent=["entml:invoke", "entml:parameter", "```"],
    ),
    SimulatedCase(
        id="escaped_underscore_name",
        description="markdown 转义工具名 get\\_weather",
        response=(
            '<entml:invoke name="get\\_weather">\n'
            '<entml:parameter name="city">深圳</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "深圳"}],
    ),
    SimulatedCase(
        id="extra_attrs_on_invoke",
        description="invoke 带多余 id/index 属性",
        response=(
            '<entml:invoke name="search_web" id="call_1" index="0">\n'
            '<entml:parameter name="query">echotools sdk</entml:parameter>\n'
            '<entml:parameter name="limit">2</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["search_web"],
        expect_args=[{"query": "echotools sdk", "limit": 2}],
    ),
    SimulatedCase(
        id="parameters_json_block",
        description="使用 entml:parameters JSON 整包",
        response=(
            "执行搜索。\n"
            '<entml:invoke name="search_web">\n'
            "<entml:parameters>\n"
            '{"query":"西湖门票","limit":4,"tags":["travel"],"filters":{"lang":"zh"}}\n'
            "</entml:parameters>\n"
            "</entml:invoke>"
        ),
        expect_names=["search_web"],
        expect_args=[
            {
                "query": "西湖门票",
                "limit": 4,
                "tags": ["travel"],
                "filters": {"lang": "zh"},
            }
        ],
        expect_clean_substrings=["执行搜索。"],
    ),
    SimulatedCase(
        id="parameters_sub_tags_fallback",
        description="parameters 内非 JSON，回退子标签",
        response=(
            '<entml:invoke name="get_weather">\n'
            "<entml:parameters>\n"
            "<city>成都</city>\n"
            "<unit>c</unit>\n"
            "<days>2</days>\n"
            "</entml:parameters>\n"
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "成都", "unit": "c", "days": 2}],
    ),
    SimulatedCase(
        id="multiline_shell_command",
        description="多行命令参数 + 对象 env",
        response=(
            "<entml:thinking>\n需要跑一段检查脚本。\n</entml:thinking>\n"
            "我先跑检查。\n"
            '<entml:invoke name="run_shell">\n'
            '<entml:parameter name="command">\n'
            "python -m pytest src/tests -q\n"
            "</entml:parameter>\n"
            '<entml:parameter name="timeout_ms">60000</entml:parameter>\n'
            '<entml:parameter name="env">{"PYTHONPATH":"src","LANG":"C"}</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["run_shell"],
        expect_args=[
            {
                "command": "\npython -m pytest src/tests -q\n",
                "timeout_ms": 60000,
                "env": {"PYTHONPATH": "src", "LANG": "C"},
            }
        ],
        expect_clean_substrings=["我先跑检查。"],
        expect_thinking="需要跑一段检查脚本。",
    ),
    SimulatedCase(
        id="path_with_angle_brackets_noise",
        description="参数值含尖括号噪声",
        response=(
            '<entml:invoke name="read_file">\n'
            '<entml:parameter name="path">docs/<draft>.md</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["read_file"],
        expect_args=[{"path": "docs/<draft>.md"}],
    ),
    SimulatedCase(
        id="prose_then_tool_then_prose_attempt",
        description="工具前后都有可见正文",
        response=(
            "第一步先读文件。\n"
            '<entml:invoke name="read_file">\n'
            '<entml:parameter name="path">README.md</entml:parameter>\n'
            "</entml:invoke>\n"
            "读完再继续分析。"
        ),
        expect_names=["read_file"],
        expect_args=[{"path": "README.md"}],
        expect_clean_substrings=["第一步先读文件。", "读完再继续分析。"],
    ),
    SimulatedCase(
        id="only_thinking_no_tool",
        description="仅思考无工具——不得误解析",
        response=(
            "<entml:thinking>\n还需要用户确认城市。\n</entml:thinking>\n"
            "请问你要查哪个城市的天气？"
        ),
        expect_names=[],
        expect_args=[],
        expect_clean_substrings=["请问你要查哪个城市的天气？"],
        expect_thinking="还需要用户确认城市。",
        expect_success=True,
    ),
    SimulatedCase(
        id="orphan_close_tags_noise",
        description="模型胡写残留闭合标签，无有效 invoke",
        response=(
            "解析失败样例：</entml:invoke>\n"
            '<entml:parameter name="city">幽灵</entml:parameter>\n'
            "请重试。"
        ),
        expect_names=[],
        expect_args=[],
        expect_clean_substrings=["请重试。"],
        expect_success=True,
    ),
    SimulatedCase(
        id="three_tools_mixed_styles",
        description="三条调用混用不同写法",
        response=(
            "<entml:thinking>\n需要天气、搜索、读文件。\n</entml:thinking>\n"
            "开始。\n"
            "<entml:function_calls>\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">广州</entml:parameter>\n'
            "</entml:invoke>\n"
            '<entml:invoke name="search_web">\n'
            "<entml:parameters>"
            '{"query":"广州塔 开放时间","limit":1}'
            "</entml:parameters>\n"
            "</entml:invoke>\n"
            '<entml:invoke name="read\\_file">\n'
            '<entml:parameter type="str" name="path">notes.txt</entml:parameter>\n'
            "</entml:invoke>\n"
            "</entml:function_calls>"
        ),
        expect_names=["get_weather", "search_web", "read_file"],
        expect_args=[
            {"city": "广州"},
            {"query": "广州塔 开放时间", "limit": 1},
            {"path": "notes.txt"},
        ],
        expect_clean_substrings=["开始。"],
        expect_thinking="需要天气、搜索、读文件。",
    ),
    SimulatedCase(
        id="english_assistant_style",
        description="英文助手口吻 + wrapper",
        response=(
            "<entml:thinking>\nI should search the web for the SDK docs.\n</entml:thinking>\n"
            "I'll look that up.\n"
            "<entml:function_calls>\n"
            '<entml:invoke name="search_web">\n'
            '<entml:parameter name="query">echotools inject_fncall</entml:parameter>\n'
            '<entml:parameter name="limit">10</entml:parameter>\n'
            '<entml:parameter name="filters">{"site":"github.com"}</entml:parameter>\n'
            "</entml:invoke>\n"
            "</entml:function_calls>"
        ),
        expect_names=["search_web"],
        expect_args=[
            {
                "query": "echotools inject_fncall",
                "limit": 10,
                "filters": {"site": "github.com"},
            }
        ],
        expect_clean_substrings=["I'll look that up."],
        expect_thinking="I should search the web for the SDK docs.",
    ),
    SimulatedCase(
        id="boolean_like_strings_stay_string_when_schema_string",
        description="string 字段写入 true/false 字面量应保持语义正确",
        response=(
            '<entml:invoke name="search_web">\n'
            '<entml:parameter name="query">true</entml:parameter>\n'
            '<entml:parameter name="limit">1</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["search_web"],
        expect_args=[{"query": "true", "limit": 1}],
    ),
    SimulatedCase(
        id="dense_no_newlines",
        description="无换行压缩输出（部分模型）",
        response=(
            "查一下。"
            '<entml:invoke name="get_weather">'
            '<entml:parameter name="city">南京</entml:parameter>'
            '<entml:parameter name="days">1</entml:parameter>'
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "南京", "days": 1}],
        expect_clean_substrings=["查一下。"],
    ),
    SimulatedCase(
        id="history_style_tool_block_must_not_parse",
        description="历史 <tool> 伪代码不得被当成 invoke",
        response=(
            "参考历史：\n"
            "<tool>\n"
            "{get_weather: {\"city\": \"杭州\", \"unit\": \"c\"}}\n"
            "晴 26°C\n"
            "</tool>\n"
            "我再确认一次实时天气。\n"
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            '<entml:parameter name="unit">c</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "杭州", "unit": "c"}],
        expect_clean_substrings=["参考历史：", "我再确认一次实时天气。"],
    ),
    # --- 模型分支：Anthropic 兼容客户端 / rogator 常见 agent 工具 ---
    SimulatedCase(
        id="agent_write_windows_path",
        description="Write：Windows 绝对路径 + 含引号 contents",
        branch="agent_write",
        extra_tools=("Write",),
        response=(
            "写入文件。\n"
            '<entml:invoke name="Write">\n'
            '<entml:parameter name="file_path">C:\\Users\\dev\\project\\main.py</entml:parameter>\n'
            '<entml:parameter name="contents">print("hello")</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["Write"],
        expect_args=[
            {
                "file_path": "C:\\Users\\dev\\project\\main.py",
                "contents": 'print("hello")',
            }
        ],
        expect_clean_substrings=["写入文件。"],
    ),
    SimulatedCase(
        id="agent_bash_pipeline",
        description="Bash：管道 + 引号 + 反斜杠",
        branch="agent_bash",
        extra_tools=("Bash",),
        response=(
            '<entml:invoke name="Bash">\n'
            '<entml:parameter name="command">cd /tmp && grep -r "foo\\bar" . | head -5</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["Bash"],
        expect_args=[{"command": 'cd /tmp && grep -r "foo\\bar" . | head -5'}],
    ),
    SimulatedCase(
        id="agent_powershell_cmdlet",
        description="PowerShell：Write-Output + 路径",
        branch="agent_powershell",
        extra_tools=("PowerShell",),
        response=(
            '<entml:invoke name="PowerShell">\n'
            '<entml:parameter name="command">Write-Output "C:\\Users\\dev"</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["PowerShell"],
        expect_args=[{"command": 'Write-Output "C:\\Users\\dev"'}],
    ),
    SimulatedCase(
        id="agent_todolist_array",
        description="TodoList：todos 必须为 JSON array",
        branch="agent_todolist",
        extra_tools=("TodoList",),
        response=(
            '<entml:invoke name="TodoList">\n'
            '<entml:parameter name="todos">[{"title": "测试 Bash 工具", "status": "in_progress"}]</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["TodoList"],
        expect_args=[
            {"todos": [{"title": "测试 Bash 工具", "status": "in_progress"}]},
        ],
    ),
    SimulatedCase(
        id="agent_parallel_write_bash",
        description="并行 Write + Bash（规范双 invoke）",
        branch="parallel_agent",
        extra_tools=("Write", "Bash"),
        response=(
            '<entml:invoke name="Write">\n'
            '<entml:parameter name="file_path">notes.txt</entml:parameter>\n'
            '<entml:parameter name="contents">ok</entml:parameter>\n'
            "</entml:invoke>\n"
            '<entml:invoke name="Bash">\n'
            '<entml:parameter name="command">echo ok</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["Write", "Bash"],
        expect_args=[
            {"file_path": "notes.txt", "contents": "ok"},
            {"command": "echo ok"},
        ],
    ),
    SimulatedCase(
        id="agent_thinking_then_bash",
        description="thinking 闭合后再 Bash",
        branch="thinking_then_agent",
        extra_tools=("Bash",),
        response=(
            "<entml:thinking>\n需要先执行 echo。\n</entml:thinking>\n"
            "开始执行。\n"
            '<entml:invoke name="Bash">\n'
            '<entml:parameter name="command">echo hello</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["Bash"],
        expect_args=[{"command": "echo hello"}],
        expect_clean_substrings=["开始执行。"],
        expect_thinking="需要先执行 echo。",
    ),
    SimulatedCase(
        id="agent_bash_inside_thinking",
        description="thinking 块内 hold 工具前缀；闭合后块外 invoke 才解析",
        branch="thinking_invoke_hold",
        extra_tools=("Bash",),
        response=(
            "<entml:thinking>\n计划：\n"
            "将在块外执行 echo in-thinking\n"
            "</entml:thinking>\n"
            '<entml:invoke name="Bash">\n'
            '<entml:parameter name="command">echo in-thinking</entml:parameter>\n'
            "</entml:invoke>\n"
            "可见回复。"
        ),
        expect_names=["Bash"],
        expect_args=[{"command": "echo in-thinking"}],
        expect_clean_substrings=["可见回复。"],
        expect_thinking="计划：",
    ),
    SimulatedCase(
        id="prompt_canonical_two_invoke_template",
        description="与提示词模板完全一致的双 invoke 串",
        branch="canonical_bare",
        response=(
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">上海</entml:parameter>\n'
            "</entml:invoke>\n"
            '<entml:invoke name="search_web">\n'
            '<entml:parameter name="query">上海 天气</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather", "search_web"],
        expect_args=[{"city": "上海"}, {"query": "上海 天气"}],
    ),
    SimulatedCase(
        id="model_type_str_overrides_schema_int",
        description="模型 type=str 优先于 schema integer",
        branch="type_hint_priority",
        response=(
            '<entml:invoke name="get_weather">\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n'
            '<entml:parameter type="str" name="days">3</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["get_weather"],
        expect_args=[{"city": "杭州", "days": "3"}],
    ),
    SimulatedCase(
        id="model_type_int_on_limit",
        description="模型 type=int 与 schema integer 一致",
        branch="type_hint_priority",
        response=(
            '<entml:invoke name="search_web">\n'
            '<entml:parameter type="int" name="limit">7</entml:parameter>\n'
            "</entml:invoke>"
        ),
        expect_names=["search_web"],
        expect_args=[{"limit": 7}],
    ),
    SimulatedCase(
        id="real_world_thinking_then_get_time",
        description="真实语料：thinking 长文 + 正文推荐 + get\\_time invoke",
        branch="thinking_then_agent",
        extra_tools=("get_time",),
        response=(
            "<entml:thinking>\n\n"
            "用户要求给出上午和下午各一个景点，同时并行调用工具获取当前时间。"
            "根据历史对话，我已经知道了杭州天气和几个热门景点。"
            "现在需要给出两个具体建议，并调用 get\\_time 工具确认当前时间。"
            "我会并行调用 get\\_time 获取当前时间，然后给出推荐。\n\n"
            "</entml:thinking>\n\n\n\n"
            "好的，我推荐上午去**灵隐寺**（清净幽深，适合清晨游览），"
            "下午去**雷峰塔**（俯瞰西湖全景，傍晚时分尤其美）。"
            "我先确认一下当前时间，方便您安排行程。\n\n\n\n"
            '<entml:invoke name="get\\_time">\n\n'
            '<entml:parameter name="city">杭州</entml:parameter>\n\n'
            "</entml:invoke>\n"
        ),
        expect_names=["get_time"],
        expect_args=[{"city": "杭州"}],
        expect_clean_substrings=["灵隐寺", "雷峰塔", "确认一下当前时间"],
        expect_thinking="get\\_time",
    ),
]

# 矩阵测试必须覆盖的分支
REQUIRED_MODEL_BRANCHES = frozenset(
    {
        "canonical_bare",
        "parallel_multi_invoke",
        "parallel_agent",
        "agent_write",
        "agent_bash",
        "agent_powershell",
        "agent_todolist",
        "thinking_then_agent",
        "thinking_invoke_hold",
        "type_hint_priority",
    }
)


def iter_simulated_cases():
    return list(SIMULATED_LLM_RESPONSES)


def iter_cases_with_tools() -> List[SimulatedCase]:
    """至少解析出一个 tool 的语料。"""
    return [c for c in SIMULATED_LLM_RESPONSES if c.expect_names]


def iter_cases_by_branch(branch: str) -> List[SimulatedCase]:
    return [c for c in SIMULATED_LLM_RESPONSES if c.branch == branch]


def covered_model_branches() -> frozenset:
    return frozenset(c.branch for c in SIMULATED_LLM_RESPONSES if c.expect_names)


def iter_bare_invoke_cases():
    """仅含裸 invoke 语料（无 legacy function_calls 外壳）。"""
    banned = "<entml:function_calls"
    return [c for c in SIMULATED_LLM_RESPONSES if banned not in c.response]
