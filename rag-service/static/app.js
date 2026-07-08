let lastVoiceText = "";
let reconnectTimer = null;

const els = {
  status: document.getElementById("connection-status"),
  title: document.getElementById("card-title"),
  body: document.getElementById("card-body"),
  voice: document.getElementById("voice-text"),
  updated: document.getElementById("last-updated"),
  avatar: document.getElementById("avatar-initial"),
  replay: document.getElementById("replay"),
  demo: document.getElementById("demo-cue"),
  caregiverInput: document.getElementById("caregiver-input"),
  analyze: document.getElementById("analyze"),
  analysisCard: document.getElementById("analysis-card"),
  ragInput: document.getElementById("rag-input"),
  ragGenerate: document.getElementById("rag-generate"),
  retrievalCard: document.getElementById("retrieval-card"),
};

function setStatus(text, state) {
  els.status.className = `status-pill ${state || ""}`.trim();
  els.status.lastChild.textContent = ` ${text}`;
}

function speak(text) {
  lastVoiceText = text || "";
  if (!lastVoiceText || !("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(lastVoiceText);
  utterance.rate = 0.82;
  utterance.pitch = 0.95;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
}

function initials(title) {
  const clean = (title || "").replace(/[^a-zA-Z\s]/g, "").trim();
  if (!clean) return "♡";
  return clean.split(/\s+/).slice(0, 2).map(word => word[0]).join("").toUpperCase();
}

function applyCue(cue) {
  const title = cue.card_title || "A gentle reminder";
  const body = cue.card_body || "You are safe. Take your time.";
  const voice = cue.voice_guidance || body;

  els.title.textContent = title;
  els.body.textContent = body;
  els.voice.textContent = voice;
  els.avatar.textContent = initials(title);
  els.updated.textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;

  speak(voice);
}

function connectFrontendSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws/frontend`);

  ws.onopen = () => setStatus("Live", "connected");
  ws.onmessage = (event) => applyCue(JSON.parse(event.data));
  ws.onerror = () => setStatus("Connection issue", "disconnected");
  ws.onclose = () => {
    setStatus("Reconnecting", "disconnected");
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectFrontendSocket, 1500);
  };
}


function renderRetrieval(result) {
  const context = result?.debug?.retrieved_context || "No retrieved context returned.";
  const output = result?.output;
  els.retrievalCard.className = "retrieval-card";
  els.retrievalCard.innerHTML = `
    <h3>Retrieved context sent to the LLM</h3>
    <pre>${escapeHtml(context)}</pre>
  `;
  if (output) applyCue(output);
}

function renderAnalysis(result) {
  const triggers = (result.observed_triggers || []).map(item => `<li>${escapeHtml(item)}</li>`).join("");
  els.analysisCard.className = "analysis-card";
  els.analysisCard.innerHTML = `
    <h3>${escapeHtml(result.category || "Caregiver guidance")}</h3>
    ${result.is_crisis ? '<p class="crisis">Crisis risk detected. Prioritize immediate safety.</p>' : ""}
    <p><strong>Rationale:</strong> ${escapeHtml(result.clinical_rationale || "No rationale returned.")}</p>
    <p><strong>Try this:</strong> ${escapeHtml(result.actionable_intervention || "No intervention returned.")}</p>
    ${triggers ? `<p><strong>Observed triggers:</strong></p><ul>${triggers}</ul>` : ""}
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

els.replay.addEventListener("click", () => speak(lastVoiceText));

els.demo.addEventListener("click", async () => {
  const cue = {
    card_title: "Sarah, your daughter",
    card_body: "Sarah is here with you. She visits often and cares about you very much.",
    voice_guidance: "Hi Arthur, Sarah is here with you. You are safe, and she is happy to see you."
  };
  applyCue(cue);
  await fetch("/api/cue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cue)
  }).catch(() => {});
});


els.ragGenerate.addEventListener("click", async () => {
  const message = els.ragInput.value.trim();
  if (!message) return;

  els.retrievalCard.className = "retrieval-card";
  els.retrievalCard.innerHTML = "<p>Retrieving patient memory and generating structured output...</p>";

  try {
    const response = await fetch("/api/rag/orientation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    renderRetrieval(await response.json());
  } catch (error) {
    els.retrievalCard.innerHTML = "<p>Could not run RAG right now. Make sure the backend is running.</p>";
  }
});

els.analyze.addEventListener("click", async () => {
  const message = els.caregiverInput.value.trim();
  if (!message) return;

  els.analysisCard.className = "analysis-card";
  els.analysisCard.innerHTML = "<p>Analyzing...</p>";

  try {
    const response = await fetch("/api/caregiver/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });
    renderAnalysis(await response.json());
  } catch (error) {
    els.analysisCard.innerHTML = "<p>Could not analyze right now. Make sure the backend is running.</p>";
  }
});

connectFrontendSocket();
