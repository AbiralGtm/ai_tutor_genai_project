# AI Tutor

- **Step 1: LLM Integration** -- a minimal, working connection to a locally
  running Llama 3.2 model via [Ollama](https://ollama.com).
- **Steps 2-6: Prompting** -- a two-step console menu. First pick a task
  template (explain, summarize, simplify, quiz, solve a problem, revision
  notes, study plan), then pick a prompting technique (zero-shot, few-shot,
  chain-of-thought, role) to apply to it. Any template can be combined with
  any technique, and the constructed prompt is always printed before it's
  sent.

## Prerequisites

1. [Install Ollama](https://ollama.com/download).
2. Pull the model:
   ```bash
   ollama pull llama3.2
   ```
3. Make sure the server is running (Ollama usually starts this automatically;
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
python chat.py
```

**Step 1 -- choose what you want help with:**

```
  1) Explain a Concept         Generate an explanation of an academic topic
  2) Summarize Study Material  Summarize pasted study material, no examples
  3) Simplify a Concept        Simplify a complex concept for a struggling student
  4) Generate a Quiz           Generate quiz questions, multiple-choice or open-ended
  5) Solve a Problem           Work through an academic or reasoning problem
  6) Create Revision Notes     Turn a topic into concise, exam-focused revision notes
  7) Create a Study Plan       Build a day-by-day or session-by-session study schedule
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

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Environment-driven settings (host, model, timeout) |
| `llm_client.py` | `OllamaClient` -- generic, reusable wrapper around Ollama's chat API |
| `templates.py` | Step 6: reusable task templates (the "what") -- fields + instruction builders |
| `techniques.py` | Steps 2-5: prompting techniques (the "how") -- zero-shot, few-shot, chain-of-thought, role |
| `chat.py` | Console app that ties templates + techniques + client together |

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
