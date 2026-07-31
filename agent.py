"""
agent.py

A small tool-calling agent for the AI Tutor project.

Given a conversation, the model decides -- via Ollama's native tool-calling
(the "tools" field on /api/chat, not hand-rolled prompt parsing) -- whether
to call a tool from tools.py before answering. This module executes any
requested calls, feeds the results back as "tool" role messages, and
repeats (bounded) until the model produces a plain text answer.

Today there's one tool (create_study_plan), but nothing here is specific
to it -- new tools just need to be added to tools.py's TOOL_SCHEMAS /
TOOL_FUNCTIONS to become available.
"""

from __future__ import annotations

import json

from llm_client import OllamaClient
from tools import TOOL_SCHEMAS, call_tool

MAX_TOOL_ROUNDS = 4  # guards against a runaway model <-> tool loop

SYSTEM_PROMPT = (
    "You are an AI tutor agent. You have access to tools -- use one "
    "whenever it would give a more accurate or reliable answer than "
    "writing it out yourself (e.g. exact day-by-day counts for a study "
    "plan). After a tool returns a result, explain it to the student "
    "clearly in your own words rather than dumping the raw data."
)


def run_turn(client: OllamaClient, messages: list[dict]) -> str:
    """
    Run one user turn to completion: ask the model, execute any tool
    calls it requests, feed the results back, and repeat (bounded) until
    it returns a final text answer. Appends every assistant/tool message
    generated along the way to `messages` in place, and returns the final
    reply text.

    Raises OllamaConnectionError / OllamaResponseError on failure (from
    the underlying OllamaClient calls) -- callers should catch these the
    same way they do for plain client.chat() calls.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        message = client.chat_with_tools(messages, tools=TOOL_SCHEMAS)
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content", "") or ""

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            result = call_tool(name, arguments)
            messages.append({"role": "tool", "content": result})

    # Ran out of rounds without a plain answer (unusual) -- ask once more
    # with no tools available so the model is forced to just answer,
    # rather than leaving the student with nothing.
    reply = client.chat(messages)
    messages.append({"role": "assistant", "content": reply})
    return reply
