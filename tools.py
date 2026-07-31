"""
tools.py

The tool(s) the AI Tutor agent can call.

Each tool is a plain, deterministic Python function -- the kind of thing a
small local model does *not* reliably get right on its own (exact counts,
even distribution across days) -- paired with a JSON-schema description in
Ollama's tool-calling format, so the model can decide when to call it and
with what arguments, then explain the result rather than compute it.

To add a new tool: write the function, describe it in TOOL_SCHEMAS, and
register it in TOOL_FUNCTIONS.
"""

from __future__ import annotations

import json
from typing import Callable


def create_study_plan(
    subject: str,
    duration_days: int,
    topics: list[str] | None = None,
    hours_per_day: float = 1.0,
) -> dict:
    """
    Deterministically spread a subject's topics across `duration_days`,
    cycling through them and inserting a review day every 4th day. This is
    the part a language model tends to fumble on its own (miscounting
    days, uneven coverage) -- the arithmetic here is exact, and the model's
    job is just to present the result well.
    """
    duration_days = max(1, int(duration_days))
    hours_per_day = max(0.25, float(hours_per_day))
    clean_topics = [t.strip() for t in (topics or []) if t and t.strip()]
    if not clean_topics:
        clean_topics = [subject]

    review_every = 4
    schedule = []
    topic_index = 0

    for day in range(1, duration_days + 1):
        if day % review_every == 0:
            schedule.append(
                {
                    "day": day,
                    "focus": "Review & practice",
                    "detail": (
                        f"Revisit earlier topics in {subject}; do practice "
                        "questions or a self-quiz on what's been covered so far."
                    ),
                }
            )
            continue

        topic = clean_topics[topic_index % len(clean_topics)]
        topic_index += 1
        schedule.append(
            {
                "day": day,
                "focus": topic,
                "detail": (
                    f'Study "{topic}" for about {hours_per_day:g} hour(s): '
                    "read/take notes, then a short recall check."
                ),
            }
        )

    return {
        "subject": subject,
        "duration_days": duration_days,
        "hours_per_day": hours_per_day,
        "topics_covered": clean_topics,
        "schedule": schedule,
    }


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_study_plan",
            "description": (
                "Generate an exact day-by-day study schedule for a subject: "
                "evenly cycles through the given topics across the "
                "available days and inserts a review day every 4th day. "
                "Call this whenever the student asks for a study plan, "
                "schedule, or timetable -- don't lay out the day-by-day "
                "breakdown yourself, call this tool so the day counts and "
                "coverage are exact, then explain/present the result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject or exam being studied for.",
                    },
                    "duration_days": {
                        "type": "integer",
                        "description": "How many days the plan should cover.",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The specific topics to cover, in any order. Use "
                            "topics the student mentioned; if they gave none, "
                            "propose a reasonable breakdown of the subject."
                        ),
                    },
                    "hours_per_day": {
                        "type": "number",
                        "description": "Roughly how many hours per day the student can study. Defaults to 1.",
                    },
                },
                "required": ["subject", "duration_days"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "create_study_plan": create_study_plan,
}


def call_tool(name: str, arguments: dict) -> str:
    """
    Execute a registered tool by name and return its result as a JSON
    string -- this is what gets sent back to the model as a "tool" role
    message's content.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})

    try:
        result = func(**arguments)
    except TypeError as exc:
        return json.dumps({"error": f"Bad arguments for '{name}': {exc}"})

    return json.dumps(result)
