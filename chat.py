#!/usr/bin/env python3
"""
chat.py

Console entry point for the AI Tutor project (Steps 2-6: Prompting).

Flow:
    1. Pick a template -- what you want help with (explain, summarize,
       simplify, quiz, solve a problem, revision notes, study plan).
       Templates live in templates.py and declare the fields they need.
    2. Fill in that template's fields.
    3. Pick a prompting technique -- how the request should be asked
       (zero-shot, few-shot, chain-of-thought, role). Techniques live in
       techniques.py.
    4. See the exact prompt built from that combination, then the model's
       response.

Any template can be paired with any technique -- e.g. "Generate a Quiz"
can be asked zero-shot, few-shot, chain-of-thought, or in a role.

Usage:
    python chat.py
"""

from __future__ import annotations

import sys

from config import Settings
from llm_client import OllamaClient, OllamaConnectionError, OllamaResponseError
from templates import TEMPLATES, Template, Field
from techniques import TECHNIQUES, ROLES, PromptTechnique


def print_header(settings: Settings) -> None:
    print("=" * 64)
    print("AI Tutor -- Prompting Console")
    print(f"Model : {settings.model}")
    print(f"Host  : {settings.host}")
    print("=" * 64)


def print_menu(title: str, items: dict) -> None:
    print(f"\n{title}")
    width = max(len(item.name) for item in items.values()) + 2
    for key, item in items.items():
        print(f"  {key}) {item.name:<{width}}{item.description}")
    print("  q) Quit")


def choose_from_menu(title: str, items: dict):
    """Show a menu of Template or PromptTechnique entries and return the
    chosen one, or None if the user wants to quit."""
    print_menu(title, items)
    while True:
        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice in {"q", "quit", "exit"}:
            return None

        item = items.get(choice)
        if item is None:
            print("Not a valid option, try again.")
            continue
        return item


def read_field(field: Field) -> str:
    if field.choices:
        print(f"\n{field.prompt}")
        for key, label in field.choices.items():
            print(f"  {key}) {label}")
        while True:
            choice = input("> ").strip()
            if choice in field.choices:
                return field.choices[choice]
            print("Not a valid option, try again.")

    if field.multiline:
        print(f"\n{field.prompt} (finish with an empty line)")
        lines: list[str] = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        value = "\n".join(lines).strip()
    else:
        value = input(f"\n{field.prompt} ").strip()

    if not value and field.default is not None:
        return field.default
    return value


def collect_fields(template: Template) -> dict[str, str]:
    return {f.name: read_field(f) for f in template.fields}


def collect_examples() -> list[tuple[str, str]]:
    """
    Interactively collect (question, answer) example pairs for few-shot
    prompting. Entered live, every time -- nothing here is pre-written.
    """
    print("\nProvide a few worked examples so the model can learn the style.")
    print("Leave the question blank when you're done adding examples.")

    examples: list[tuple[str, str]] = []
    while True:
        question = input(f"Example {len(examples) + 1} question (blank to stop): ").strip()
        if not question:
            break
        answer = input(f"Example {len(examples) + 1} answer: ").strip()
        if not answer:
            print("(answer was empty, example discarded)")
            continue
        examples.append((question, answer))

    return examples


def choose_role() -> str | None:
    print("\nChoose a tutor role:")
    for key, (name, _description) in ROLES.items():
        print(f"  {key}) {name}")
    print("  q) Quit")

    while True:
        choice = input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice in ROLES:
            return ROLES[choice][1]
        print("Not a valid option, try again.")


def print_prompt(messages: list[dict]) -> None:
    """Show exactly what is sent to the model, so the technique is visible."""
    print("\n--- Prompt sent to the model ---")
    for message in messages:
        print(f"[{message['role']}]")
        print(message["content"].strip())
        print()
    print("---------------------------------")


def build_messages(technique: PromptTechnique, instruction: str) -> list[dict] | None:
    """Collect any extra input a technique needs, then build its messages.
    Returns None if the user quits mid-flow."""
    if technique.name == "Few-Shot":
        examples = collect_examples()
        return technique.build(instruction, examples=examples)

    if technique.name == "Role":
        role_description = choose_role()
        if role_description is None:
            return None
        return technique.build(instruction, role_description=role_description)

    return technique.build(instruction)


def main() -> int:
    settings = Settings()
    client = OllamaClient(settings)

    if not client.health_check():
        print(
            f"\nCould not reach Ollama at {settings.host}.\n"
            "Make sure it's running, then try again:\n"
            "    ollama serve\n"
            f"    ollama pull {settings.model}\n",
            file=sys.stderr,
        )
        return 1

    print_header(settings)

    while True:
        try:
            template = choose_from_menu("Step 1 -- choose what you want help with:", TEMPLATES)
            if template is None:
                print("Goodbye.")
                return 0

            values = collect_fields(template)
            instruction = template.build_instruction(values)

            technique = choose_from_menu("\nStep 2 -- choose a prompting technique:", TECHNIQUES)
            if technique is None:
                print("Goodbye.")
                return 0

            messages = build_messages(technique, instruction)
            if messages is None:
                print("Goodbye.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        print_prompt(messages)

        print("Thinking...")
        try:
            reply = client.chat(messages)
        except (OllamaConnectionError, OllamaResponseError) as exc:
            print(f"[error] {exc}")
            continue

        print("\n--- Response ---")
        print(reply)
        print("----------------")


if __name__ == "__main__":
    sys.exit(main())
