#!/usr/bin/env python3
"""
chains.py

LangChain Chains -- a sequential workflow: Topic -> Explanation -> Notes -> Quiz.

Each stage is its own modular LLMChain (explanation_chain, notes_chain,
quiz_chain) and can be run entirely on its own -- e.g.
notes_chain.invoke({"explanation": "..."}) works without the topic or the
full pipeline. build_pipeline() composes the three into a SequentialChain,
which passes each stage's output into the next by matching an LLMChain's
output_key to the next prompt's input variable: explanation_chain's
"explanation" output becomes notes_chain's "explanation" input, and
notes_chain's "notes" output becomes quiz_chain's "notes" input.

This uses LangChain's classic LLMChain/SequentialChain API. As of
LangChain 1.0, that API lives in the separate `langchain-classic`
compatibility package (it's deprecated upstream in favor of LCEL's
`prompt | llm` syntax, but still fully functional) -- see requirements.txt.

This is a separate subsystem from chat.py's manual templates/techniques
console: chat.py builds raw message lists by hand and talks to Ollama
through llm_client.OllamaClient, while this module lets LangChain manage
prompting and chain composition, via LangChain's own Ollama integration
(langchain_ollama.OllamaLLM). Both talk to the same local model.

Usage:
    python chains.py
"""

from __future__ import annotations

import sys
import warnings

from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import memory
from config import Settings
from llm_client import OllamaClient

# LLMChain/SequentialChain are intentionally used here (see module
# docstring); silence the upstream nudge toward LCEL so the console output
# stays clean. Must be added after the langchain_classic import above --
# LangChain registers its own "always show this" filter for the warning
# as part of that import, which would otherwise take priority.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

EXPLANATION_PROMPT = PromptTemplate(
    input_variables=["topic", "memory_context"],
    template=(
        "{memory_context}\n\n"
        "Explain the concept of {topic} to a student in clear, complete terms."
    ),
)

NOTES_PROMPT = PromptTemplate(
    input_variables=["explanation"],
    template=(
        "Turn the following explanation into concise, exam-focused revision "
        "notes. Use short bullet points and highlight the most important "
        "details.\n\nExplanation:\n{explanation}"
    ),
)

QUIZ_PROMPT = PromptTemplate(
    input_variables=["notes"],
    template=(
        "Based on the following revision notes, write 5 multiple-choice "
        "quiz questions. For each question, provide four options labeled "
        "A-D and clearly state the correct answer.\n\nRevision notes:\n{notes}"
    ),
)


def build_chains(llm: OllamaLLM) -> tuple[LLMChain, LLMChain, LLMChain]:
    """
    Build each stage as an independent, modular chain. Every chain here can
    be run on its own -- this function doesn't wire them together, that's
    build_pipeline()'s job.
    """
    explanation_chain = LLMChain(llm=llm, prompt=EXPLANATION_PROMPT, output_key="explanation")
    notes_chain = LLMChain(llm=llm, prompt=NOTES_PROMPT, output_key="notes")
    quiz_chain = LLMChain(llm=llm, prompt=QUIZ_PROMPT, output_key="quiz")
    return explanation_chain, notes_chain, quiz_chain


def build_pipeline(llm: OllamaLLM) -> SequentialChain:
    """
    Compose the three stages into one sequential workflow: Topic ->
    Explanation -> Notes -> Quiz. SequentialChain passes each chain's
    output to the next automatically, by matching output_key to the
    following prompt's input_variables.

    "memory_context" is an external input alongside "topic" -- the Memory
    System's personalization summary (see memory.build_context()), only
    consumed by the explanation stage; it flows into notes and quiz
    implicitly, through the explanation text they're built from.
    """
    explanation_chain, notes_chain, quiz_chain = build_chains(llm)
    return SequentialChain(
        chains=[explanation_chain, notes_chain, quiz_chain],
        input_variables=["topic", "memory_context"],
        output_variables=["explanation", "notes", "quiz"],
    )


def print_stage(title: str, content: str) -> None:
    print(f"\n--- {title} ---")
    print(content.strip())
    print("-" * (len(title) + 8))


def main(user: memory.User | None = None) -> int:
    settings = Settings()

    # Reuse Step 1's connection + error handling for the pre-flight check,
    # even though generation itself goes through LangChain's own client.
    if not OllamaClient(settings).health_check():
        print(
            f"\nCould not reach Ollama at {settings.host}.\n"
            "Make sure it's running, then try again:\n"
            "    ollama serve\n"
            f"    ollama pull {settings.model}\n",
            file=sys.stderr,
        )
        return 1

    # When launched via main.py, `user` is already logged in and shared
    # across tools. Standalone (`python chains.py`), log in here instead.
    if user is None:
        user = memory.login()
        if user is None:
            print("Goodbye.")
            return 0

    llm = OllamaLLM(
        base_url=settings.host,
        model=settings.model,
        client_kwargs={"timeout": settings.timeout},
    )
    pipeline = build_pipeline(llm)

    print("=" * 64)
    print("AI Tutor -- LangChain Sequential Workflow")
    print("Topic -> Explanation -> Notes -> Quiz")
    print(f"Model : {settings.model}")
    print(f"Host  : {settings.host}")
    print("=" * 64)

    while True:
        try:
            topic = input("\nEnter a topic (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not topic:
            print("Goodbye.")
            return 0

        context = memory.build_context(user, memory.recent_history(user.id, limit=5))

        print("\nRunning pipeline...")
        try:
            result = pipeline.invoke({"topic": topic, "memory_context": context})
        except Exception as exc:
            print(
                f"[error] {exc}\n"
                "(Check that Ollama is still running and the model is pulled.)"
            )
            continue

        print_stage("Explanation", result["explanation"])
        print_stage("Revision Notes", result["notes"])
        print_stage("Quiz", result["quiz"])

        memory.record_interaction(
            user_id=user.id,
            tool="langchain_workflow",
            activity="Topic -> Explanation -> Notes -> Quiz",
            topic=topic,
            technique=None,
            summary=f'Ran the LangChain pipeline on "{topic}".',
        )


if __name__ == "__main__":
    sys.exit(main())
