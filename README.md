# AI Tutor

- **Step 1: LLM Integration** -- a minimal, working connection to a locally
  running Llama 3.2 model via [Ollama](https://ollama.com).
- **Step 2: Prompting** -- a console menu for comparing four prompting
  techniques (zero-shot, few-shot, chain-of-thought, role) on the same
  local model, with the constructed prompt printed before it's sent.

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

You'll see a menu of prompting techniques. Pick one, enter a topic or
question, and the app prints the exact prompt it built for that technique
before sending it to the model, followed by the response.

```
Choose a prompting technique:
  1) Zero-Shot         Ask directly, no examples given
  2) Few-Shot          Provide your own worked examples to set the format
  3) Chain-of-Thought  Ask the model to reason step by step
  4) Role               Answer in-character as a persona you provide
  q) Quit
```

Picking Few-Shot prompts you to type your own example question/answer
pairs first (leave the question blank to stop adding examples), then your
real topic. Picking Role prompts you to describe the persona (e.g. "a
patient elementary school teacher") before your topic. Nothing is
pre-written -- both come from you, live, every run.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | Environment-driven settings (host, model, timeout) |
| `llm_client.py` | `OllamaClient` -- generic, reusable wrapper around Ollama's chat API |
| `templates.py` | Prompt templates for zero-shot, few-shot, chain-of-thought, and role prompting |
| `chat.py` | Console menu that ties the templates and client together |

`llm_client.py` stays generic by design -- no tutoring-specific code. All
tutoring flavor (concept explanations, worked examples, tutor persona)
lives in `templates.py`, so it's easy to add new techniques or tasks
without touching the client.

## Prompting techniques (Step 2)

| Technique | Idea |
|---|---|
| Zero-shot | Ask directly, no examples, no reasoning scaffold |
| Few-shot | You supply worked examples at runtime so the model mirrors their structure |
| Chain-of-thought | Explicitly ask the model to reason step by step before the final answer |
| Role | You supply a persona at runtime for the model to answer from |

No topic, example, or role content is hardcoded in `templates.py`. Every
technique's input -- including few-shot's example pairs and role's persona
description -- comes from whatever the user types in at the console.

## Configuration

Set via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL of the Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Model tag to use |
| `OLLAMA_TIMEOUT` | `60` | Request timeout in seconds |
