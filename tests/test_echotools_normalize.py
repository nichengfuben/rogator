from __future__ import annotations

"""验证 echotools 内置 tool args 规范化。"""

import json
import unittest

from echotools import get_protocol


TODO_TOOLS = [{
    "type": "function",
    "function": {
        "name": "todo_write",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
}]


class TestEchotoolsNormalize(unittest.TestCase):
    def test_entml_python_literal_array(self) -> None:
        protocol = get_protocol("entml")
        text = """
<entml:function_calls>
<entml:invoke name="todo_write">
<entml:parameters>
<todos>[{'id': '1', 'content': '收集系统负载信息', 'status': 'in_progress'}]</todos>
</entml:parameters>
</entml:invoke>
</entml:function_calls>
"""
        _, calls = protocol.parse(text, TODO_TOOLS)
        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertIsInstance(args["todos"], list)
        self.assertEqual(args["todos"][0]["id"], "1")
        self.assertEqual(args["todos"][0]["content"], "收集系统负载信息")


if __name__ == "__main__":
    unittest.main()
