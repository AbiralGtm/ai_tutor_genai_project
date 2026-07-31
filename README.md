# AI Tutor

- **Step 1: LLM Integration** -- a minimal, working connection to a locally
  running Llama 3.2 model via [Ollama](https://ollama.com).
- **Steps 2-6: Prompting** -- a two-step console menu. First pick a task
  template (explain, summarize, simplify, quiz, solve a problem, revision
  notes, study plan), then pick a prompting technique (zero-shot, few-shot,
  chain-of-thought, role) to apply to it. Any template can be combined with
  any technique, and the constructed prompt is always printed before it's
  sent.
- **Step 7: LangChain Chains** -- a sequential workflow (Topic ->
  Explanation -> Notes -> Quiz) built with LangChain, where each stage's
  output feeds the next.
- **Step 8: Memory System** -- a local SQLite profile per student, tracking
  learning history across runs and feeding it back into prompts so
  responses adapt to what's already known about that student.
- **Step 9: RAG** -- index your own PDF/TXT/MD study material and get
  answers grounded in it, instead of just whatever the model already knows.
- **Step 10: Agent** -- a tool-calling agent that can reach for a
  deterministic tool (e.g. an exact day-by-day study plan) instead of
  writing out arithmetic it tends to get wrong itself.

## Prerequisites

1. [Install Ollama](https://ollama.com/download).
2. Pull the chat model:
   ```bash
   ollama pull llama3.2
   ```
3. Pull the embedding model (needed for RAG -- Step 9):
   ```bash
   ollama pull nomic-embed-text
   ```
4. Make sure the server is running (Ollama usually starts this automatically;
   otherwise):
   ```bash
   ollama serve
   ```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # adjust if your Ollama host/model differ
```

## Run

```bash
python main.py
```

This first asks who's studying:

```
Who's studying today?
Enter your name:
```

A new name creates a profile; an existing one (matched case-insensitively)
logs back in with a short recap of what you last studied. Then the
top-level launcher opens:

```
Choose a tool:
  1) Prompting Console   Manual templates + prompting techniques
  2) LangChain Workflow  Topic -> Explanation -> Notes -> Quiz pipeline
  q) Quit
```

Quitting a tool (its own `q`) returns you to this menu rather than exiting
the program, and you stay logged in as the same profile the whole time.
`main.py` is only a dispatcher -- it imports `chat` or `chains` lazily,
right when you pick that option, and holds no prompting logic of its own.
Both tools also run standalone if you'd rather skip the launcher (each
will ask you to log in itself, since there's no shared session to hand
off from):

```bash
python chat.py      # Prompting Console directly
python chains.py    # LangChain Workflow directly
```

### Prompting Console

**Step 1 -- choose what you want help with:**

```
  1) Explain a Concept         Generate an explanation of an academic topic
  2) Summarize Study Material  Summarize pasted study material, no examples
  3) Simplify a Concept        Simplify a complex concept for a struggling student
  4) Generate a Quiz           Generate quiz questions, multiple-choice or open-ended
  5) Solve a Problem           Work through an academic or reasoning problem
  6) Create Revision Notes     Turn a topic into concise, exam-focused revision notes
  7) Create a Study Plan       Build a day-by-day or session-by-session study schedule
  8) View History                  See your past topics and techniques
  9) Continue a Conversation       Resume a past conversation where you left off
  10) Index Study Material          Upload a PDF/TXT/MD file for grounded Q&A
  11) Ask About My Study Materials  Answer questions using your indexed material (RAG)
  12) Ask the Agent                 Chat with a tool-using agent (e.g. exact day-by-day study plans)
  q) Quit
```

You'll then be asked to fill in that template's fields (e.g. a topic, or
pasted study material -- multi-line fields end on a blank line).

**Step 2 -- choose a prompting technique:**

```
  1) Zero-Shot         Ask directly, no examples given
  2) Few-Shot          Provide your own worked examples to set the format
  3) Chain-of-Thought  Ask the model to reason step by step
  4) Role              Answer as one of four tutor personas
  q) Quit
```

Picking **Few-Shot** prompts you to type your own example question/answer
pairs first (leave the question blank to stop adding examples) -- nothing
is pre-written, the examples come from you, live, every run.

Picking **Role** prompts you to choose one of four fixed personas:
Teacher, Examiner, Study Coach, or Subject Expert.

The app then prints the exact prompt it built from your template + technique
combination, sends it to the model, and shows the response.

**After a response**, you can keep the conversation going:

```
Ask a follow-up, Enter for a new topic, or 'q' to quit:
```

A follow-up is appended to the same message list (so the model still has
the full conversation as context) and sent again. Press Enter with nothing
typed to head back to Step 1 for an unrelated topic, or `q` to quit
outright. The whole exchange -- including how many follow-ups you asked,
and the full message transcript -- is recorded as one entry in your
learning history.

**Options 8 and 9 at Step 1** work with that history:

- **View History** lists your recent entries, most recent first, each
  tagged `(resumable)` if it has a saved transcript.
- **Continue a Conversation** lists only the resumable ones; picking one
  reloads its exact message transcript and drops you straight into the
  same follow-up loop, so the model still has everything said in that
  conversation as context. Continuing creates a new history entry marked
  "(continued from history)" rather than editing the original.

### LangChain Workflow (Step 7)

Enter a topic and the app runs a three-stage pipeline, printing each stage
as it completes:

```
Topic -> Explanation -> Notes -> Quiz
```

1. **Explanation** -- explains the topic.
2. **Revision Notes** -- built from stage 1's explanation, not the raw topic.
3. **Quiz** -- 5 multiple-choice questions, built from stage 2's notes.

Each stage is its own `LLMChain` (`explanation_chain`, `notes_chain`,
`quiz_chain` in `chains.py`) and can be run standalone -- e.g.
`notes_chain.invoke({"explanation": "..."})` works on its own, without a
topic or the rest of the pipeline. `build_pipeline()` composes all three
into a `SequentialChain`, which passes outputs between chains by matching
each chain's `output_key` to the next chain's prompt input: `explanation_chain`
outputs `"explanation"`, which `notes_chain`'s prompt consumes; `notes_chain`
outputs `"notes"`, which `quiz_chain`'s prompt consumes.

This uses LangChain's classic `LLMChain`/`SequentialChain` API rather than
the newer LCEL (`prompt | llm`) style. As of LangChain 1.0, that API moved
out of the core `langchain` package into `langchain-classic` -- still fully
supported, just no longer the default recommendation upstream (which is why
`chains.py` suppresses the resulting deprecation warning).

`chains.py` is a separate subsystem from `chat.py`: `chat.py` builds message
lists by hand and talks to Ollama through `llm_client.OllamaClient`, while
`chains.py` lets LangChain manage prompting and chain composition via its
own Ollama integration (`langchain_ollama.OllamaLLM`). Both point at the
same local model and the same `config.py` settings; `chains.py` also reuses
`OllamaClient.health_check()` for its pre-flight connectivity check.

## Memory System (Step 8)

`memory.py` stores two things in a local SQLite file (`memory.db` by
default, no server or extra dependency -- `sqlite3` is in Python's standard
library):

- **users** -- one row per profile (name, created/last-seen timestamps),
  matched case-insensitively so "Abiral" and "abiral" are the same person.
- **interactions** -- one row per completed request: which tool, which
  template/technique (or "LangChain pipeline"), the topic, and a short
  summary. Written after every successful response in both `chat.py` and
  `chains.py`. For the Prompting Console specifically, the full message
  transcript is saved too (as JSON), which is what makes "Continue a
  Conversation" possible -- `chains.py`'s pipeline runs don't save one,
  since there's no equivalent open-ended conversation to resume.

Logging in (`memory.login()`) resolves a typed name to a profile, creating
one if it doesn't exist, and prints a short recap of recent topics for a
returning user. `main.py` calls this once and shares the profile with
whichever tool you pick; `chat.py` and `chains.py` call it themselves when
run standalone.

Before every request, `memory.build_context()` turns a user's recent
history into a short paragraph (e.g. *"You are tutoring Abiral. They have
recently studied: 'recursion'; 'photosynthesis'. Use this history to
personalize your response..."*) which gets folded into the prompt:

- In `chat.py`, it's appended to whichever technique's system message was
  built (all four put a system message first).
- In `chains.py`, it's passed as a `memory_context` input to the
  explanation stage's `PromptTemplate`; personalization then flows into
  the notes and quiz stages implicitly, since they're built from the
  (already personalized) explanation text rather than the raw topic again.

The database file is gitignored (`*.db`) since it's local, per-user state,
not project source.

## RAG (Step 9)

`rag.py` lets you index your own study material and get answers grounded
in it, via Step 1's options 10 and 11.

**10) Index Study Material** asks for a path to a `.pdf`, `.txt`, or `.md`
file, then:

1. Extracts its text (`pypdf` for PDFs, plain read for `.txt`/`.md`).
2. Splits it into ~800-character overlapping chunks (`chunk_text()` --
   simple and dependency-free, doesn't try to respect sentence boundaries).
3. Embeds each chunk with a local Ollama embedding model (`nomic-embed-text`
   by default) via `OllamaClient.embed()`.
4. Stores the chunks and their embeddings in the same local SQLite file
   `memory.py` uses, scoped to your user profile.

**11) Ask About My Study Materials** embeds your question the same way,
scores it against every stored chunk for your profile with cosine
similarity computed directly in Python (`retrieve()` -- no vector
database, brute-force search over your own notes), and takes the top 4
matches. Those chunks get built into a grounded prompt (`build_rag_messages()`)
that instructs the model to answer from that context and say so plainly if
the material doesn't cover the question, rather than guessing. From there
it's the same `run_conversation()` follow-up loop, memory personalization,
and history recording as every other option.

This is a genuinely simple, custom retrieval store -- not Chroma/FAISS/a
production vector DB -- which is a deliberate fit for the scale here (one
student's own notes, dozens to a few hundred chunks) and keeps the "one
local SQLite file, no server" approach consistent with `memory.py`.

## Agent (Step 10)

**12) Ask the Agent** starts a conversation with a tool-calling agent, via
Ollama's *native* tool-calling (the `tools` field on `/api/chat` -- the
model itself decides when to call a tool, no hand-rolled prompt parsing).

Right now there's one tool, `create_study_plan` (`tools.py`): given a
subject, a number of days, optional topics, and hours/day, it
deterministically cycles through the topics day by day and inserts a
review day every 4th day. This is the kind of thing a small local model
tends to get wrong on its own -- miscounting days, uneven topic coverage --
so the tool does the arithmetic exactly, and the model's job is just to
present the result well.

The loop (`agent.run_turn()` in `agent.py`):

1. Send the conversation to the model with `tools=TOOL_SCHEMAS`.
2. If the response requests a tool call, run it (`tools.call_tool()`) and
   append the result as a `"tool"` role message.
3. Send the conversation again -- repeat until the model returns a plain
   answer (bounded at 4 rounds, to avoid a runaway loop).

`chat.py` shows this happening rather than hiding it -- you'll see
`[agent calling tool: create_study_plan(...)]` and `[tool result] ...`
printed before the final `--- Response ---`. From there it's the same
follow-up loop, memory personalization, and history recording as every
other option; resuming an agent conversation from **9) Continue a
Conversation** correctly routes back through the tool-calling loop (not
the plain one), since each interaction records which kind it was.

Adding a new tool means writing the function and its JSON-schema
description in `tools.py` (`TOOL_SCHEMAS` / `TOOL_FUNCTIONS`) -- `agent.py`
itself has no tool-specific code. No new dependency was needed for this --
tool-calling goes through the same `requests`-based `OllamaClient` as
everything else (`OllamaClient.chat_with_tools()`).

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Top-level launcher -- logs in once, dispatches to `chat.py` or `chains.py` |
| `config.py` | Environment-driven settings (host, model, timeout, memory DB path) |
| `llm_client.py` | `OllamaClient` -- generic, reusable wrapper around Ollama's chat API |
| `memory.py` | Step 8: Memory System -- user profiles + learning history (SQLite) |
| `rag.py` | Step 9: RAG -- chunking, embedding, storage, and retrieval over study material (SQLite) |
| `tools.py` | Step 10: Agent tools -- deterministic functions + their JSON-schema descriptions |
| `agent.py` | Step 10: Agent -- the tool-calling loop (`run_turn()`), model-agnostic to which tools exist |
| `templates.py` | Step 6: reusable task templates (the "what") -- fields + instruction builders |
| `techniques.py` | Steps 2-5: prompting techniques (the "how") -- zero-shot, few-shot, chain-of-thought, role |
| `chat.py` | Prompting Console -- ties templates + techniques + client + memory + RAG + agent together |
| `chains.py` | Step 7: LangChain sequential workflow (Topic -> Explanation -> Notes -> Quiz), memory-aware |

`llm_client.py` stays fully generic -- no tutoring-specific code. Templates
and techniques are deliberately separate: a template decides *what* is
being asked, a technique decides *how* it's asked, and `chat.py` combines
whichever pair the user picks. This keeps every template/technique
independent and means adding a new one doesn't touch the others.

## Task templates (Step 6)

| Template | Fields | Purpose |
|---|---|---|
| Explain a Concept | topic | Generate an explanation of an academic topic |
| Summarize Study Material | material (multi-line) | Summarize study material without examples |
| Simplify a Concept | topic | Simplify a complex concept for a struggling student |
| Generate a Quiz | topic, count, format | Multiple-choice or open-ended quiz questions |
| Solve a Problem | problem (multi-line) | Work through an academic/reasoning problem |
| Create Revision Notes | topic | Concise, exam-focused revision notes |
| Create a Study Plan | subject, timeframe | Day-by-day or session-by-session study schedule |

## Prompting techniques (Steps 2-5)

| Technique | Idea |
|---|---|
| Zero-shot | Ask directly, no examples, no reasoning scaffold |
| Few-shot | You supply worked examples at runtime so the model mirrors their structure |
| Chain-of-thought | Explicitly ask the model to reason step by step before the final answer |
| Role | Pick one of four fixed personas (Teacher, Examiner, Study Coach, Subject Expert) for the model to answer from |

No topic, example, or template content is hardcoded anywhere. Every
template's fields, few-shot's example pairs, and the topic itself all come
from whatever the user types in at the console; only the four role
personas are fixed, by design, per the project spec.

## Configuration

Set via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL of the Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Model tag to use |
| `OLLAMA_TIMEOUT` | `60` | Request timeout in seconds |
| `MEMORY_DB_PATH` | `memory.db` | Path to the Memory System / RAG SQLite file |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model used for RAG embeddings |
