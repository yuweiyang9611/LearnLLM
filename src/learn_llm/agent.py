"""A fully local, deterministic tool-agent for the systems lesson.

This module demonstrates the *shape* of an agent: planner, registered tools,
state/trace, a bounded control loop and explicit stop conditions.  The planner
is a small rule-based program, not an LLM.  No network API, ``eval`` or
``exec`` is used.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .rag import ExtractiveRag


AgentStatus = Literal["completed", "error", "step_limit"]
ToolHandler = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Tool:
    """A named capability that the deterministic controller may execute."""

    name: str
    description: str
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name or not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("tool name must use lowercase letters, numbers and underscores")


@dataclass(frozen=True, slots=True)
class AgentAction:
    """The planner's next requested action; ``finish`` is a control action."""

    tool_name: str
    tool_input: str

    @classmethod
    def finish(cls, answer: str) -> "AgentAction":
        return cls("finish", answer)


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """A deterministic record of one tool execution."""

    success: bool
    output: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One planner decision and its optional tool observation."""

    number: int
    action: AgentAction
    observation: ToolObservation | None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Final status plus a complete, inspectable execution trace."""

    task: str
    status: AgentStatus
    answer: str
    steps: tuple[AgentStep, ...]
    stop_reason: str

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


Planner = Callable[[str, Sequence[AgentStep]], AgentAction]


class RuleBasedPlanner:
    """Route explicit calculation tasks to math and everything else to search."""

    _CALCULATE_PREFIX = re.compile(
        r"^\s*(?:计算|算一下|calculate)\s*[:：]?\s*", flags=re.IGNORECASE
    )
    _SEARCH_PREFIX = re.compile(
        r"^\s*(?:检索|搜索|查询|search)\s*[:：]?\s*", flags=re.IGNORECASE
    )

    def __call__(self, task: str, steps: Sequence[AgentStep]) -> AgentAction:
        if steps:
            observation = steps[-1].observation
            if observation is None:
                return AgentAction.finish("控制器没有收到工具结果。")
            if observation.success:
                return AgentAction.finish(observation.output)
            return AgentAction.finish(f"工具执行失败：{observation.error}")

        calculation = self._CALCULATE_PREFIX.match(task)
        if calculation:
            expression = task[calculation.end() :].strip()
            return AgentAction("calculator", expression)

        search = self._SEARCH_PREFIX.match(task)
        query = task[search.end() :].strip() if search else task.strip()
        return AgentAction("knowledge_search", query)


class LocalAgent:
    """Execute planner actions through a bounded and inspectable control loop."""

    def __init__(self, tools: Iterable[Tool], *, planner: Planner | None = None) -> None:
        tool_map: dict[str, Tool] = {}
        for tool in tools:
            if tool.name == "finish":
                raise ValueError("'finish' is reserved for the controller")
            if tool.name in tool_map:
                raise ValueError(f"duplicate tool: {tool.name}")
            tool_map[tool.name] = tool
        if not tool_map:
            raise ValueError("at least one tool is required")
        self.tools = tool_map
        self.planner = planner or RuleBasedPlanner()

    def run(self, task: str, *, max_steps: int = 4) -> AgentResult:
        if not task.strip():
            raise ValueError("task cannot be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        steps: list[AgentStep] = []
        for number in range(1, max_steps + 1):
            try:
                action = self.planner(task, tuple(steps))
            except Exception as error:  # A planner is an extension boundary.
                message = f"规划器异常：{type(error).__name__}: {error}"
                return AgentResult(task, "error", message, tuple(steps), "planner_error")

            if action.tool_name == "finish":
                steps.append(AgentStep(number, action, None))
                return AgentResult(
                    task,
                    "completed",
                    action.tool_input,
                    tuple(steps),
                    "finish_action",
                )

            tool = self.tools.get(action.tool_name)
            if tool is None:
                error = f"未知工具：{action.tool_name}"
                observation = ToolObservation(False, "", error)
                steps.append(AgentStep(number, action, observation))
                return AgentResult(task, "error", error, tuple(steps), "unknown_tool")

            try:
                output = tool.handler(action.tool_input)
                if not isinstance(output, str):
                    output = str(output)
                observation = ToolObservation(True, output)
            except Exception as error:  # Tool errors become state, not crashes.
                message = f"{type(error).__name__}: {error}"
                observation = ToolObservation(False, "", message)
                steps.append(AgentStep(number, action, observation))
                return AgentResult(
                    task,
                    "error",
                    f"工具 {tool.name} 执行失败：{message}",
                    tuple(steps),
                    "tool_error",
                )
            steps.append(AgentStep(number, action, observation))

        message = f"达到最大步骤数 {max_steps}，控制循环已停止。"
        return AgentResult(task, "step_limit", message, tuple(steps), "max_steps")


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 128
_MAX_AST_NODES = 50
_MAX_ABSOLUTE_RESULT = 1_000_000_000_000.0
_MAX_EXPONENT = 12.0


def _checked_number(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("只允许整数和小数")
    if not math.isfinite(float(value)):
        raise ValueError("计算结果必须是有限数")
    if abs(float(value)) > _MAX_ABSOLUTE_RESULT:
        raise ValueError("计算结果超出教学工具的安全范围")
    return value


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant):
        return _checked_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _evaluate_node(node.operand)
        return _checked_number(_UNARY_OPERATORS[type(node.op)](value))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > _MAX_EXPONENT:
            raise ValueError("指数绝对值不能超过 12")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
        return _checked_number(value)
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Evaluate basic arithmetic through an AST whitelist (never ``eval``)."""

    if not expression.strip():
        raise ValueError("算术表达式不能为空")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("算术表达式过长")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("算术表达式语法错误") from error
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("算术表达式过于复杂")
    value = _evaluate_node(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return format(value, ".12g")


class KnowledgeSearchTool:
    """Adapt :class:`ExtractiveRag` to the agent's string tool interface."""

    def __init__(self, rag: ExtractiveRag) -> None:
        self.rag = rag

    def __call__(self, query: str) -> str:
        return self.rag.answer(query).answer


def build_local_agent(rag: ExtractiveRag, *, planner: Planner | None = None) -> LocalAgent:
    """Build the lesson's safe calculator + grounded-search agent."""

    return LocalAgent(
        [
            Tool("calculator", "用 AST 白名单计算基础算术", safe_calculate),
            Tool(
                "knowledge_search",
                "在本地知识库检索，并返回抽取式答案与来源",
                KnowledgeSearchTool(rag),
            ),
        ],
        planner=planner,
    )
