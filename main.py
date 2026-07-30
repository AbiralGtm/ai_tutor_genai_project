#!/usr/bin/env python3
"""
main.py

Top-level launcher for the AI Tutor console tools.

    1) Prompting Console  -- chat.py's manual templates + techniques menu
                             (Steps 2-6)
    2) LangChain Workflow -- chains.py's Topic -> Explanation -> Notes ->
                             Quiz pipeline (Step 7)

This file is only a dispatcher -- it holds no prompting or chain logic of
its own. Each tool's module is imported lazily, right before it's run, so
choosing the Prompting Console never requires LangChain to be installed.
Both chat.py and chains.py remain fully usable on their own too:

    python chat.py
    python chains.py

The Memory System (memory.py) logs you in once, here, and passes that
profile down to whichever tool you pick -- so switching between the
Prompting Console and the LangChain Workflow in the same session shares
one identity and one learning history, instead of asking for your name
twice.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from typing import Callable

import memory


def run_prompting_console(user: memory.User) -> None:
    import chat

    chat.main(user)


def run_langchain_workflow(user: memory.User) -> None:
    try:
        import chains
    except ImportError as exc:
        print(f"\n[error] The LangChain Workflow needs extra packages: {exc}")
        print("Install them with: pip install -r requirements.txt")
        return

    chains.main(user)


TOOLS: dict[str, tuple[str, str, Callable[[memory.User], None]]] = {
    "1": (
        "Prompting Console",
        "Manual templates + prompting techniques",
        run_prompting_console,
    ),
    "2": (
        "LangChain Workflow",
        "Topic -> Explanation -> Notes -> Quiz pipeline",
        run_langchain_workflow,
    ),
}


def print_menu() -> None:
    print("\nChoose a tool:")
    width = max(len(name) for name, _description, _run in TOOLS.values()) + 2
    for key, (name, description, _run) in TOOLS.items():
        print(f"  {key}) {name:<{width}}{description}")
    print("  q) Quit")


def main() -> int:
    print("=" * 64)
    print("AI Tutor")
    print("=" * 64)

    user = memory.login()
    if user is None:
        print("Goodbye.")
        return 0

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

        tool = TOOLS.get(choice)
        if tool is None:
            print("Not a valid option, try again.")
            continue

        _name, _description, run = tool
        run(user)
        print()  # spacing before the launcher menu reappears


if __name__ == "__main__":
    sys.exit(main())
