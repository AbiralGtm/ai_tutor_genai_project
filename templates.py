"""
templates.py

Prompt templates for Step 2: Prompting.

Each template takes a student's topic or question and returns a list of
chat messages -- the same shape OllamaClient.chat() expects -- built
according to a specific prompting technique:

- Zero-shot         ask directly, no examples, no reasoning scaffold
- Few-shot          show a couple of worked examples first, so the model
                    mirrors their structure and tone
- Chain-of-thought  explicitly ask the model to reason step by step before
                    giving a final answer
- Role              give the model a persona (supplied by the user) to
                    answer from

Each template is deliberately tutoring-flavoured (explaining concepts,
working through problems), since that's the domain of this project, but the
underlying pattern behind each technique is generic and reusable elsewhere.

No topic, example, or role content is hardcoded anywhere in this file. The
topic always comes from the user at runtime; for few-shot, the worked
examples do too; for role, so does the persona description (see chat.py,
which collects these interactively) -- this module only supplies the
structure each technique wraps that input in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptTechnique:
    """A named prompting technique and the function that builds its prompt."""

    name: str
    description: str
    build: Callable[..., list[dict]]


def zero_shot(topic: str) -> list[dict]:
    """No examples, no reasoning scaffold -- just ask directly."""
    return [
        {
            "role": "system",
            "content": "You are an AI tutor. Answer clearly and directly.",
        },
        {
            "role": "user",
            "content": f"Explain the concept of {topic} to a student in simple, clear terms.",
        },
    ]


def few_shot(topic: str, examples: list[tuple[str, str]]) -> list[dict]:
    """
    Show the model a handful of worked (question, answer) example pairs
    before the real question, so it mirrors their style and structure.

    Unlike the other techniques, few-shot needs example content beyond the
    topic itself -- but that content is supplied by the user at runtime
    (see chat.py's collect_examples()), never hardcoded here. With no
    examples supplied, this degrades gracefully to a zero-shot-style ask.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI tutor. Answer the student's question in the "
                "same style, tone, and structure as the example answers "
                "shown to you."
            ),
        }
    ]

    for question, answer in examples:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    messages.append({"role": "user", "content": topic})
    return messages


def chain_of_thought(topic: str) -> list[dict]:
    """Explicitly ask the model to reason step by step before concluding."""
    return [
        {
            "role": "system",
            "content": (
                "You are an AI tutor. When solving a problem, think through "
                "it step by step out loud, then clearly state the final "
                "answer at the end."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{topic}\n\n"
                "Work through this step by step, showing your reasoning, "
                "then give the final answer on its own line."
            ),
        },
    ]


def role_based(topic: str, role: str) -> list[dict]:
    """
    Give the model a persona to answer from -- shapes tone and approach,
    not just content.

    The role is supplied by the user at runtime (see chat.py), not
    hardcoded here, so it can be anything from "a patient elementary school
    teacher" to "a no-nonsense drill sergeant."
    """
    return [
        {
            "role": "system",
            "content": f"You are {role}. Answer the student's question from that perspective.",
        },
        {
            "role": "user",
            "content": f"Can you help me understand {topic}?",
        },
    ]


TECHNIQUES: dict[str, PromptTechnique] = {
    "1": PromptTechnique(
        name="Zero-Shot",
        description="Ask directly, no examples given",
        build=zero_shot,
    ),
    "2": PromptTechnique(
        name="Few-Shot",
        description="Provide your own worked examples to set the format",
        build=few_shot,
    ),
    "3": PromptTechnique(
        name="Chain-of-Thought",
        description="Ask the model to reason step by step",
        build=chain_of_thought,
    ),
    "4": PromptTechnique(
        name="Role",
        description="Answer in-character as a persona you provide",
        build=role_based,
    ),
}
