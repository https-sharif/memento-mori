# Offline chatbot

A fully offline memory assistant: Streamlit for the UI, a local Ollama model for
generation, Chroma for retrieval. Nothing here calls a cloud API.

**This is a separate prototype.** It shares no code, ports, or data with
`vision_service` and `backend`. There is no camera, no face recognition, and no
connection to the live pipeline — it answers questions from documents and
caregiver-entered notes instead of from a video scene. It lives in this
repository because it explores the same problem from a different angle.

---

## Prerequisites

Python 3.10+, and **a running Ollama daemon** with the `llama3.2` model pulled:

```bash
# https://ollama.com/download
ollama pull llama3.2
ollama serve            # must stay running on http://localhost:11434
```

> **If Ollama is not running, the app does not tell you.** The LLM call is
> wrapped in a bare `except` (`app.py:653`), so a connection failure is
> indistinguishable from "no relevant memory found" — the patient just sees
> *"I do not remember that right now."* If answers seem uniformly blank, check
> the daemon first.

The first run also downloads the `all-MiniLM-L6-v2` embedding model from Hugging
Face. That one download is the only network access the app ever makes; after it,
the app runs with no internet at all.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use `.venv\Scripts\activate` instead.

This pulls in `torch`, `transformers`, and `chromadb` — expect a multi-hundred-MB
install.

## Run

Two Streamlit apps share one data directory. Run each in its own terminal:

```bash
# Caregiver admin UI — add people, objects, routines, reminders; upload documents
python -m streamlit run admin.py --server.port 8502

# Patient chat UI
python -m streamlit run app.py --server.port 8501
```

Start with the admin app. Until something has been added and indexed, the chat
app has nothing to retrieve and will answer every question with the fallback
message.

`ingest.py` is a standalone CLI batch loader — an alternative to the admin app's
upload page for bulk PDF/TXT ingestion.

---

## How it works

1. The admin app writes structured entries to `memory_data/memories.json` and
   indexes them, along with any uploaded PDFs and text files, into a Chroma
   store at `memory_data/chroma_memory/`.
2. A patient question runs a similarity search over that store — top 4 results,
   discarding anything below a `0.35` relevance score.
3. If the top result is a person, routine, object, or reminder, a **templated**
   answer is returned directly. This is the common case, and it involves **no
   LLM call at all**.
4. Only when no template applies does the app make a single Ollama call,
   constrained to the retrieved context.
5. The answer is rejected and replaced with the fallback message if it contains
   hedging language ("I think", "probably", "as an AI"), on the theory that an
   uncertain answer is worse than no answer for this audience.

All paths derive from the file's own location, so the directory can be moved or
cloned anywhere.

## Data and privacy

`memory_data/` holds caregiver-entered personal information and the vector index
built from it. It is gitignored and never leaves your machine. Delete the
directory to erase everything; both apps recreate it empty on next start.

See the repository's [PRIVACY.md](../PRIVACY.md) and
[DISCLAIMER.md](../DISCLAIMER.md) — the disclaimer applies to this app in full.

## Known limitations

- **Silent failures.** Ollama being down, a retrieval error, and a genuinely
  unknown question all produce the same fallback message.
- **No authentication.** The admin app is a full CRUD interface on the patient's
  memory data, reachable by anyone who can open port 8502.
- **No tests.**
