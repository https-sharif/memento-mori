# Privacy and biometric data

This project processes **face biometrics** and a **live camera feed**. That carries real legal and
ethical weight, so this document states plainly what is captured, where it goes, and how to delete
it.

Read this alongside [DISCLAIMER.md](DISCLAIMER.md).

## What the system captures

| Data | Where it comes from | Where it is stored |
|---|---|---|
| **Face embeddings** — a 512-float vector per registered face | `POST /register` and `POST /register/capture` in `vision_service` | `vision_service/storage/data/embeddings.json`, on your machine |
| **Name, relationship, note** for each registered person | The registration form | Same file |
| **Live camera frames** | Your webcam, while `vision_service` runs | In memory only. Served on request via `GET /frame`; never written to disk |
| **Patient profile** — name, condition, family, preferences | `patient_profile.json`, edited by hand | That file, on your machine |
| **Scene descriptions and caretaker questions** | The running pipeline | Sent to the Gemini API (see below); held in memory, never written to disk |

Face embeddings are **biometric identifiers**. Under the GDPR they are special-category data
(Art. 9), and jurisdictions such as Illinois (BIPA) and Texas (CUBI) impose specific consent and
retention duties on anyone who collects them. Treat the embeddings file accordingly.

## What leaves your machine

**`vision_service` is fully offline.** Face recognition and object detection run locally. No image,
frame, or embedding is ever transmitted anywhere. Model weights are downloaded once on first run.

**`backend` calls the Google Gemini API.** What is sent: the text description of the current scene
(names, relationships, object labels), retrieved lines from `patient_profile.json`, and caretaker
questions. **No images are ever sent** — only text. Your prompts are subject to
[Google's Gemini API terms](https://ai.google.dev/gemini-api/terms). If you do not want any data
leaving your machine, run with `MOCK_LLM=true`, which makes no network calls at all.

**`offline-chatbot/` is fully offline.** It uses a local Ollama model and a local Chroma store.

## Consent

**Register only people who have knowingly agreed to it.** Enrolling someone's face without their
informed consent is unlawful in many jurisdictions, regardless of intent.

Where the person cannot meaningfully consent — which includes many people living with advanced
dementia — consent must come from whoever holds legal authority for their care decisions, in line
with local law.

## Deleting biometric data

Remove one person:

```bash
curl -X DELETE http://localhost:8000/people/NAME
```

Remove everyone, permanently:

```bash
rm vision_service/storage/data/embeddings.json
```

The file is recreated empty on the next registration. There is no backup and no recovery — that is
deliberate.

## Retention

The system applies **no automatic retention limit**. Embeddings persist until you delete them. If
you deploy this anywhere real, set and enforce a retention policy; the code will not do it for you.

## Committing data by accident

`.gitignore` excludes `vision_service/storage/data/embeddings.json`, `**/memory_data/`, and `.env`.
Verify before pushing:

```bash
git ls-files | grep -E 'embeddings\.json|memory_data|\.env$'   # must print nothing
```

## Known limitations

This is a prototype, and its security posture reflects that:

- **No authentication on any endpoint.** Anyone who can reach the port can register a face, delete
  a person, or pull a live camera still from `GET /frame`.
- **CORS is fully open** (`allow_origins=["*"]`) on both services.
- **No encryption at rest.** `embeddings.json` is plain, readable JSON.
- **No audit log.** Nothing records who registered or deleted whom.

Both services therefore default to binding `127.0.0.1`. Do not expose either to a network, and do
not deploy this as-is.
