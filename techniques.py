"""
techniques.py

Steps 2-5: Prompting techniques.

Each technique takes a fully-formed instruction -- produced by a template
in templates.py -- and wraps it the way that technique demands:

- Zero-shot          send the instruction directly, no examples
- Few-shot           precede it with worked examples the user supplies
- Chain-of-thought   ask the model to reason step by step before concluding
- Role               answer from one of a fixed set of tutor personas

A technique knows nothing about *what* is being asked (that's templates.py's
job) -- only *how* to ask it. Any template can be combined with any
technique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptTechnique:
    """A named prompting technique and the function that applies it."""

    name: str
    description: str
    build: Callable[..., list[dict]]


def zero_shot(instruction: str) -> list[dict]:
    """No examples, no reasoning scaffold -- ask directly."""
    return [
        {
            "role": "system",
            "content": "You are an AI tutor. Answer clearly and directly.",
        },
        {"role": "user", "content": instruction},
    ]


def few_shot(instruction: str, examples: list[tuple[str, str]]) -> list[dict]:
    """
    Precede the instruction with (question, answer) example pairs supplied
    by the user at runtime, so the model mirrors their style and structure.
    With no examples supplied, this degrades to a zero-shot-style ask.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI tutor. Answer the student's request in the "
                "same style, tone, and structure as the example answers "
                "shown to you."
            ),
        }
    ]

    for question, answer in examples:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    messages.append({"role": "user", "content": instruction})
    return messages


def chain_of_thought(instruction: str) -> list[dict]:
    """Ask the model to reason step by step before giving a final answer."""
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
                f"{instruction}\n\n"
                "Work through this step by step, showing your reasoning, "
                "then give the final answer on its own line."
            ),
        },
    ]


def role_based(instruction: str, role_description: str) -> list[dict]:
    """Answer from a persistent persona's perspective."""
    return [
        {
            "role": "system",
            "content": f"You are {role_description}. Answer the student's request from that perspective.",
        },
        {"role": "user", "content": instruction},
    ]


# Step 5: fixed tutor personas. Each maps a menu key to (display name,
# description injected into the system prompt).
ROLES: dict[str, tuple[str, str]] = {
    "1": (
        "Teacher",
        "a supportive classroom teacher who explains things clearly and "
        "checks that the student is following along",
    ),
    "2": (
        "Examiner",
        "a formal examiner who evaluates answers rigorously, precisely, "
        "and points out gaps or mistakes",
    ),
    "3": (
        "Study Coach",
        "an encouraging study coach who helps a student plan their work, "
        "stay motivated, and build good study habits",
    ),
    "4": (
        "Subject Expert",
        "a deeply knowledgeable subject-matter expert who gives precise, "
        "technically accurate answers",
    ),
}


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
        description="Answer as one of four tutor personas",
        build=role_based,
    ),
}
