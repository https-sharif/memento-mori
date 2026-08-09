# Demo guide

How to run all three pieces locally and record a walkthrough.

Full setup instructions are in the [README](../README.md); this file covers the
demo flow itself.

---

## Before you record

**Register a face.** Face recognition only matches people enrolled on the
machine doing the recording. Without this step the demo will show "no one
recognized" for its entire duration.

1. Start the vision service and open <http://localhost:8000/static/register.html>.
2. Enter a name and relationship — "Self" is fine for a solo demo.
3. Capture 3–5 photos from slightly different angles, then submit.
4. Confirm: `curl http://localhost:8000/people` should list the name.

This stores a face embedding on your machine. See [PRIVACY.md](../PRIVACY.md).

**Decide on Gemini or mock mode.** The free Gemini tier allows roughly 5
requests per minute, which a live demo can exhaust. Running with
`MOCK_LLM=true` gives canned responses, no network calls, and no rate limit —
the architecture and every response shape are identical. Mock cards are labelled
`(mock)` so it is obvious on camera.

**Match the profile to your script.** `patient_profile.json` ships with a
fictional patient named Arthur. If you plan to ask "what does Arthur like?",
either keep it as-is or edit the file first — the answers come from it.

---

## Flow (3–5 minutes)

1. **Sketch the architecture.** Camera → vision service (8000) → backend
   (8001) → browser page. Mention that face recognition runs locally and only
   text is sent to Gemini.

2. **Show the live scene.** Open `frontend/index.html`, point the camera at the
   registered face plus a detectable object — a cup, a bottle, a phone. The
   memory card fills in with the name, relationship, and note; the object chips
   update below it.

3. **Hold still, then move.** Worth calling out: the card does not regenerate
   every second. The backend hashes the scene and only calls the LLM when it
   actually changes. Step out of frame and back to trigger a fresh card.

4. **Ask a caregiver question.** Type something like "How do I keep him calm at
   dinner?" into the question box and send. The answer appears in its own panel —
   the patient's card above is untouched, which is deliberate.

5. **Show the analysis endpoint.**

   ```bash
   curl -s -X POST http://localhost:8001/api/caregiver/analyze \
     -H "Content-Type: application/json" \
     -d '{"message":"Arthur got agitated and started pacing before dinner."}'
   ```

   Returns a structured `BehavioralAnalysis`: category, observed triggers,
   rationale, a non-pharmacological intervention, and a crisis flag. Note that
   the rationale is model-generated text, not a clinical assessment — see
   [DISCLAIMER.md](../DISCLAIMER.md).

6. **Close on the data source.** Open `patient_profile.json` and explain that
   the backend retrieves matching lines from it and passes only those to the
   model. Every card and answer in the demo traces back to this file.

---

## URLs to keep open

| URL | Shows |
|---|---|
| <http://localhost:8000/health> | Vision service is up |
| <http://localhost:8000/people> | Registered faces |
| <http://localhost:8000/static/register.html> | Registration UI |
| <http://localhost:8001/health> | Backend is up |
| <http://localhost:8001/latest> | Current memory card, as JSON |
| `frontend/index.html` | The demo page |

---

## Troubleshooting

**"No one recognized" throughout.** The face on camera was never registered on
this machine. See "Before you record" above.

**Gemini errors mentioning quota or 429.** The free tier is about 5 requests per
minute. Wait ~15 seconds, or restart the backend with `MOCK_LLM=true`.

**No card appears.** Check that both services are running and the browser console
is clean. The page polls `http://localhost:8001/latest` every 2 seconds — hitting
that URL directly tells you whether the backend or the page is at fault.

**No webcam.** Run `python backend/mock_vision.py` instead of the real vision
service. It serves the same websocket contract with synthetic scenes, so the
backend and frontend behave normally.

**Camera is black or permission was denied.** Browsers only grant camera access
on `localhost` or HTTPS, and only one process can hold the camera. Close other
apps using it, then reload.
