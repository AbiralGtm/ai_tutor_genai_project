"""
templates.py

Step 6: Prompt Templates.

A small library of reusable, tutoring-specific templates. Each template
declares the fields it needs (dynamic input substitution) and a function
that turns those field values into a plain-language instruction -- the
"what" of the request.

Templates know nothing about prompting technique (that's techniques.py's
job -- the "how"). Any template can be paired with any technique in
chat.py, so e.g. "Generate a Quiz" can be asked zero-shot, few-shot,
chain-of-thought, or in a role, without templates.py changing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Field:
    """One piece of input a template needs from the user."""

    name: str
    prompt: str
    multiline: bool = False
    default: str | None = None
    choices: dict[str, str] | None = None  # menu key -> substituted value


@dataclass(frozen=True)
class Template:
    """A reusable task template: what fields it needs, and how to turn
    those fields into a plain-language instruction for the model."""

    name: str
    description: str
    fields: list[Field] = field(default_factory=list)
    build_instruction: Callable[[dict[str, str]], str] = lambda values: ""


def _explain_instruction(values: dict[str, str]) -> str:
    return f"Explain the concept of {values['topic']} to a student in clear, complete terms."


def _summarize_instruction(values: dict[str, str]) -> str:
    return (
        "Summarize the following study material for a student, capturing "
        f"the key points:\n\n{values['material']}"
    )


def _simplify_instruction(values: dict[str, str]) -> str:
    return (
        f"Simplify the concept of {values['topic']} using plain language, "
        "short sentences, and a relatable analogy, for a student who is "
        "struggling to understand it."
    )


def _quiz_instruction(values: dict[str, str]) -> str:
    count = values.get("count") or "5"
    if values["format"] == "multiple-choice":
        return (
            f"Generate {count} multiple-choice questions about "
            f"{values['topic']}. For each question, provide four options "
            "labeled A-D and clearly state the correct answer."
        )
    return (
        f"Generate {count} open-ended quiz questions about "
        f"{values['topic']}, each followed by a model answer."
    )


def _solve_instruction(values: dict[str, str]) -> str:
    return f"Solve the following problem:\n\n{values['problem']}"


def _revision_notes_instruction(values: dict[str, str]) -> str:
    return (
        f"Create concise revision notes on {values['topic']} for a student "
        "preparing for an exam. Use short bullet points and highlight the "
        "most exam-relevant details."
    )


def _study_plan_instruction(values: dict[str, str]) -> str:
    return (
        f"Create a study plan for {values['subject']}, to be completed "
        f"within {values['timeframe']}. Break it into a day-by-day or "
        "session-by-session schedule with a clear goal for each session."
    )


TEMPLATES: dict[str, Template] = {
    "1": Template(
        name="Explain a Concept",
        description="Generate an explanation of an academic topic",
        fields=[Field("topic", "What topic or concept should be explained?")],
        build_instruction=_explain_instruction,
    ),
    "2": Template(
        name="Summarize Study Material",
        description="Summarize pasted study material, no examples",
        fields=[
            Field("material", "Paste the study material to summarize:", multiline=True)
        ],
        build_instruction=_summarize_instruction,
    ),
    "3": Template(
        name="Simplify a Concept",
        description="Simplify a complex concept for a struggling student",
        fields=[Field("topic", "What concept should be simplified?")],
        build_instruction=_simplify_instruction,
    ),
    "4": Template(
        name="Generate a Quiz",
        description="Generate quiz questions, multiple-choice or open-ended",
        fields=[
            Field("topic", "What topic should the quiz cover?"),
            Field("count", "How many questions? (default 5)", default="5"),
            Field(
                "format",
                "Quiz format:",
                choices={"1": "multiple-choice", "2": "open-ended"},
            ),
        ],
        build_instruction=_quiz_instruction,
    ),
    "5": Template(
        name="Solve a Problem",
        description="Work through an academic or reasoning problem",
        fields=[Field("problem", "Enter the problem to solve:", multiline=True)],
        build_instruction=_solve_instruction,
    ),
    "6": Template(
        name="Create Revision Notes",
        description="Turn a topic into concise, exam-focused revision notes",
        fields=[Field("topic", "What topic should the revision notes cover?")],
        build_instruction=_revision_notes_instruction,
    ),
    "7": Template(
        name="Create a Study Plan",
        description="Build a day-by-day or session-by-session study schedule",
        fields=[
            Field("subject", "What subject or topics should the plan cover?"),
            Field("timeframe", "How much time is available (e.g. '2 weeks')?"),
        ],
        build_instruction=_study_plan_instruction,
    ),
}
