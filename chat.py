#!/usr/bin/env python3
"""
chat.py

Console entry point for the AI Tutor project.

Step 2: Prompting -- pick a prompting technique (zero-shot, few-shot,
chain-of-thought, or role) and see exactly how it shapes the prompt sent to
the local LLM, alongside the model's response. The prompt is always printed
before the request is sent, so the difference between techniques is visible,
not just their output.

Usage:
    python chat.py
"""

from __future__ import annotations

import sys

from config import Settings
from llm_client import OllamaClient, OllamaConnectionError, OllamaResponseError
from templates import TECHNIQUES, PromptTechnique


def print_header(settings: Settings) -> None:
    print("=" * 64)
    print("AI Tutor -- Step 2: Prompting")
    print(f"Model : {settings.model}")
    print(f"Host  : {settings.host}")
    print("=" * 64)


def print_menu() -> None:
    print("\nChoose a prompting technique:")
    for key, technique in TECHNIQUES.items():
        print(f"  {key}) {technique.name:<16} {technique.description}")
    print("  q) Quit")


def print_prompt(messages: list[dict]) -> None:
    """Show exactly what is sent to the model, so the technique is visible."""
    print("\n--- Prompt sent to the model ---")
    for message in messages:
        print(f"[{message['role']}]")
        print(message["content"].strip())
        print()
    print("---------------------------------")


def collect_examples() -> list[tuple[str, str]]:
    """
    Interactively collect (question, answer) example pairs from the user for
    few-shot prompting. Entered live, every time -- nothing here is
    pre-written.
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


def collect_role() -> str:
    """
    Interactively collect the persona the model should adopt for role
    prompting. Entered live, every time -- nothing here is pre-written.
    """
    print("\nDescribe the role the model should play, e.g. 'a patient")
    print("elementary school teacher' or 'a strict university professor'.")
    return input("Role: ").strip()


def run_technique(client: OllamaClient, technique: PromptTechnique) -> None:
    if technique.name == "Few-Shot":
        examples = collect_examples()
        topic = input("\nNow enter your actual topic or question: ").strip()
        if not topic:
            print("(nothing entered, back to menu)")
            return
        messages = technique.build(topic, examples=examples)

    elif technique.name == "Role":
        role = collect_role()
        if not role:
            print("(no role entered, back to menu)")
            return
        topic = input("\nNow enter your actual topic or question: ").strip()
        if not topic:
            print("(nothing entered, back to menu)")
            return
        messages = technique.build(topic, role=role)

    else:
        topic = input(f"\nEnter a topic or question for {technique.name}: ").strip()
        if not topic:
            print("(nothing entered, back to menu)")
            return
        messages = technique.build(topic)

    print_prompt(messages)

    print("Thinking...")
    try:
        reply = client.chat(messages)
    except (OllamaConnectionError, OllamaResponseError) as exc:
        print(f"[error] {exc}")
        return

    print("\n--- Response ---")
    print(reply)
    print("----------------")


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
        print_menu()
        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if choice in {"q", "quit", "exit"}:
            print("Goodbye.")
            return 0

        technique = TECHNIQUES.get(choice)
        if technique is None:
            print("Not a valid option, try again.")
            continue

        run_technique(client, technique)


if __name__ == "__main__":
    sys.exit(main())
