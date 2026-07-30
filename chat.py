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

import memory
from config import Settings
from llm_client import OllamaClient, OllamaConnectionError, OllamaResponseError
from templates import TEMPLATES, Template, Field
from techniques import TECHNIQUES, ROLES, PromptTechnique

# Step 1 also offers two history-backed actions alongside the 7 templates.
# They're plain sentinel strings rather than Template objects, since they
# don't have fields or an instruction builder -- main() branches on them
# explicitly before falling through to the normal template flow.
VIEW_HISTORY = "VIEW_HISTORY"
CONTINUE_CONVERSATION = "CONTINUE_CONVERSATION"


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


def choose_step1() -> Template | str | None:
    """
    Step 1's menu: the 7 templates, plus two history-backed actions. Returns
    a Template, one of the VIEW_HISTORY/CONTINUE_CONVERSATION sentinels, or
    None if the user wants to quit.
    """
    extra = [
        ("8", "View History", "See your past topics and techniques", VIEW_HISTORY),
        ("9", "Continue a Conversation", "Resume a past conversation where you left off", CONTINUE_CONVERSATION),
    ]

    print("\nStep 1 -- choose what you want help with:")
    width = max([len(t.name) for t in TEMPLATES.values()] + [len(label) for _, label, _, _ in extra]) + 2
    for key, template in TEMPLATES.items():
        print(f"  {key}) {template.name:<{width}}{template.description}")
    for key, label, description, _sentinel in extra:
        print(f"  {key}) {label:<{width}}{description}")
    print("  q) Quit")

    extra_by_key = {key: sentinel for key, _label, _description, sentinel in extra}

    while True:
        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice in {"q", "quit", "exit"}:
            return None
        if choice in extra_by_key:
            return extra_by_key[choice]

        template = TEMPLATES.get(choice)
        if template is None:
            print("Not a valid option, try again.")
            continue
        return template


def show_history(user: memory.User) -> None:
    history = memory.recent_history(user.id, limit=10)
    if not history:
        print("\nNo history yet -- this is your first session!")
        return

    print(f"\n--- {user.name}'s Learning History (most recent first) ---")
    for i, interaction in enumerate(history, start=1):
        note = " (resumable)" if interaction.transcript else ""
        print(f"{i}. [{interaction.created_at}] {interaction.summary}{note}")
    print("-" * 40)


def pick_conversation(user: memory.User) -> memory.Interaction | None:
    """
    List resumable past conversations (Prompting Console runs with a saved
    transcript) and let the user pick one. Returns None if there's nothing
    to resume or the user backs out.
    """
    resumable = [h for h in memory.recent_history(user.id, limit=20) if h.transcript]
    if not resumable:
        print("\nNo resumable conversations yet -- have one first, then come back.")
        return None

    print("\nPick a conversation to continue:")
    for i, interaction in enumerate(resumable, start=1):
        print(f"  {i}) [{interaction.created_at}] {interaction.summary}")
    print("  b) Back")

    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice in {"b", "back", "q", "quit", "exit"}:
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(resumable):
            return resumable[int(choice) - 1]

        print("Not a valid option, try again.")


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


def extract_topic(values: dict[str, str]) -> str:
    """
    Best-effort short label for what a template run was "about", used for
    the learning history and memory context. Different templates name
    their main field differently (topic, subject, problem, material).
    """
    for key in ("topic", "subject"):
        if values.get(key):
            return values[key]

    for key in ("problem", "material"):
        text = values.get(key)
        if text:
            text = text.strip().replace("\n", " ")
            return text[:60] + ("..." if len(text) > 60 else "")

    return "(no topic)"


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


def run_conversation(
    client: OllamaClient,
    user: memory.User,
    messages: list[dict],
    activity: str,
    technique_name: str | None,
    topic: str,
    resumed: bool = False,
) -> bool:
    """
    Send `messages`, print the response, then keep the conversation going
    with follow-ups (reusing the same message list, so the model keeps
    prior turns as context) until the user asks for a new topic or quits.
    Finally records the whole exchange -- including the full transcript,
    so it can be resumed later -- to the user's learning history.

    Returns True if the user asked to quit outright.
    """
    follow_up_count = 0

    if resumed:
        print(f'\nResuming your conversation about "{topic}"...')
        for message in messages:
            if message["role"] != "system":
                print(f"\n[{message['role']}] {message['content']}")
    else:
        print_prompt(messages)
        print("Thinking...")
        try:
            reply = client.chat(messages)
        except (OllamaConnectionError, OllamaResponseError) as exc:
            print(f"[error] {exc}")
            return False  # nothing to record, back to Step 1

        print("\n--- Response ---")
        print(reply)
        print("----------------")
        messages.append({"role": "assistant", "content": reply})

    quit_requested = False
    while True:
        try:
            follow_up = input(
                "\nAsk a follow-up, Enter for a new topic, or 'q' to quit: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            quit_requested = True
            break

        if follow_up.lower() in {"q", "quit", "exit"}:
            quit_requested = True
            break

        if not follow_up:
            break  # back to Step 1 for a new topic

        messages.append({"role": "user", "content": follow_up})
        print("Thinking...")
        try:
            reply = client.chat(messages)
        except (OllamaConnectionError, OllamaResponseError) as exc:
            print(f"[error] {exc}")
            messages.pop()  # drop the follow-up that failed, keep history clean
            continue

        print("\n--- Response ---")
        print(reply)
        print("----------------")
        messages.append({"role": "assistant", "content": reply})
        follow_up_count += 1

    summary = f'{activity} on "{topic}"'
    if technique_name:
        summary += f" using {technique_name}"
    if resumed:
        summary += " (continued from history)"
    if follow_up_count:
        summary += f", with {follow_up_count} follow-up{'s' if follow_up_count != 1 else ''}"

    memory.record_interaction(
        user_id=user.id,
        tool="prompting_console",
        activity=activity,
        topic=topic,
        technique=technique_name,
        summary=summary + ".",
        transcript=messages,
    )

    return quit_requested


def main(user: memory.User | None = None) -> int:
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

    # When launched via main.py, `user` is already logged in and shared
    # across tools. Standalone (`python chat.py`), log in here instead.
    if user is None:
        user = memory.login()
        if user is None:
            print("Goodbye.")
            return 0

    print_header(settings)

    while True:
        try:
            choice = choose_step1()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if choice is None:
            print("Goodbye.")
            return 0

        if choice == VIEW_HISTORY:
            show_history(user)
            continue

        if choice == CONTINUE_CONVERSATION:
            interaction = pick_conversation(user)
            if interaction is None:
                continue
            quit_requested = run_conversation(
                client,
                user,
                interaction.transcript,
                interaction.activity,
                interaction.technique,
                interaction.topic or "(no topic)",
                resumed=True,
            )
            if quit_requested:
                print("Goodbye.")
                return 0
            continue

        template = choice
        try:
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

        # The memory system: fold recent history into the system message so
        # the response can actually be personalized, not just persisted.
        context = memory.build_context(user, memory.recent_history(user.id, limit=5))
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{context}"

        topic = extract_topic(values)
        quit_requested = run_conversation(client, user, messages, template.name, technique.name, topic)
        if quit_requested:
            print("Goodbye.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
