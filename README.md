# Small O — Local Voice AI Assistant

A fully local, privacy-first voice AI assistant that runs entirely on your machine.
No cloud APIs, no data leaving your device.

---

## What It Does

You speak → it listens → it thinks → it talks back.

Small O captures your voice in the browser, transcribes it locally with Whisper,
reasons with a local LLM via Ollama, and speaks the response with Piper TTS — all
in real time, streamed token by token to a cartoon-styled frontend.

---

## Architecture

```
Browser (React)
│
│  getUserMedia() → AudioWorklet → silence detection
│  Float32 audio chunks ──────────────────────────────────► WebSocket (ws://localhost:8765)
│                                                                        │
│  ◄── JSON events (VOICE_STATE, STT_RESULT, LLM_TOKEN, ...) ───────────┤
│                                                               Python Backend
│                                                               │
│                                                               ├─ faster-whisper  (STT)
│                                                               ├─ Ollama HTTP API (LLM)
│                                                               ├─ piper-tts        (TTS → speakers)
│                                                               ├─ Memory System    (FAISS + SQLite)
│                                                               └─ Plugin Router    (web / computer / security)
```

### Audio pipeline (browser → backend)

| Step | Where | What happens |
|------|-------|-------------|
| Capture | Browser `AudioWorklet` | Runs in the audio thread — never throttled by tab focus |
| Silence detection | Browser main thread | RMS energy threshold; 1.5 s silence → clip end |
| Send | WebSocket binary frame | `[4-byte uint32 sampleRate][Float32[] samples]` |
| Resample | Python (numpy) | Converts any browser sample rate to 16 kHz for Whisper |
| Transcribe | faster-whisper | Local `base` model; returns text + latency |
| LLM | Ollama (`phi3`) | Streamed token-by-token via HTTP; `num_predict: -1` (unlimited) |
| TTS | piper-tts | Sentence-level streaming; plays through system speakers |

### WebSocket event types (backend → frontend)

| Event | Payload | Purpose |
|-------|---------|---------|
| `VOICE_STATE` | `{state}` | idle / listening / thinking / speaking |
| `STT_RESULT` | `{text, recording_time, transcription_time}` | User utterance + timing |
| `LLM_TOKEN` | `{token, done}` | Stream tokens; `done:true` finalises the bubble |
| `PLUGIN_ACTION` | `{plugin, action, result, direct}` | Web / computer / security plugin result |
| `MEMORY_EVENT` | `{type, summary, importance, id, retrieved}` | New memory or retrieval |
| `SYSTEM_STATS` | `{cpu, ram, battery}` | Live system stats (every 2 s) |
| `pong` | `{}` | Heartbeat reply to frontend ping |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | bundled with Node.js |
| Ollama | latest | [ollama.ai](https://ollama.ai) — must be running |
| phi3 model | — | `ollama pull phi3` |

> **macOS only** — the computer plugin uses PyAutoGUI and Quartz bindings.
> Linux support is possible with minor changes to the system plugin.

---

## Setup

### 1. Clone

```bash
git clone <repo-url> smallO_v2
cd smallO_v2
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Piper TTS model

Piper needs a voice model (`.onnx` + `.onnx.json`) placed in `backend/tts/models/`.
Download from [piper-samples](https://rhasspy.github.io/piper-samples/):

```bash
mkdir -p backend/tts/models
# Example: English (US) — Ryan medium
curl -L -o backend/tts/models/en_US-ryan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
curl -L -o backend/tts/models/en_US-ryan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
```

### 4. Initialise the memory database

```bash
cd backend
python memory_system/db/init_db.py
cd ..
```

### 5. Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 6. Start Ollama

```bash
ollama serve        # if not already running as a service
ollama pull phi3    # first time only
```

---

## Running

```bash
# Standard run — opens browser automatically
python main.py

# With detailed logs — shows every pipeline step + latency
python main_with_logs.py
```

`main.py` starts both processes silently and opens `http://localhost:5173`.
`main_with_logs.py` pipes all backend events to the terminal with timestamps and colour.

Press **Ctrl+C** to stop — both frontend and backend (and all child processes) are shut down cleanly.

### First-time startup sequence

```
python main.py
  └── backend/main.py   →  WebSocket server  ws://localhost:8765
  └── npm run dev       →  Vite dev server   http://localhost:5173
```

1. Browser opens at `http://localhost:5173`
2. Page shows **"Please wait while the server starts"** (WebSocket not yet connected)
3. Transitions to **"Loading models..."** (Whisper + Piper warming up, ~10–20 s)
4. Transitions to **"Tap to Start 🎤"** (backend ready, waiting for mic permission)
5. Click the button → browser asks for microphone permission → click Allow
6. Shows **"Say something!"** — start talking

---

## Project Structure

```
smallO_v2/
├── main.py                    # Root launcher — starts backend + frontend
├── main_with_logs.py          # Same but with detailed terminal logs
├── requirements.txt           # Python dependencies
│
├── backend/
│   ├── main.py                # WebSocket server + pipeline orchestration
│   ├── stt/
│   │   └── main_stt.py        # faster-whisper transcription (receives audio from browser)
│   ├── llm/
│   │   ├── main_llm.py        # Ollama streaming LLM client
│   │   └── SYSTEM_PROMPT.py   # System prompt for the assistant
│   ├── tts/
│   │   └── main_tts.py        # Piper TTS — sentence-level streaming to speakers
│   ├── memory_system/
│   │   ├── db/                # SQLite schema + connection
│   │   ├── embeddings/        # FAISS vector store + sentence-transformers embedder
│   │   ├── retrieval/         # Semantic search over memories
│   │   ├── core/              # Importance scoring + insert pipeline
│   │   ├── entities/          # Named entity extractor + registry
│   │   └── lifecycle/         # Memory consolidation worker
│   ├── plugins/
│   │   ├── router.py          # Routes queries to the right plugin
│   │   ├── internet/          # Web search + scraping (requests + BeautifulSoup)
│   │   ├── computer/          # macOS automation (PyAutoGUI + Quartz)
│   │   └── security/          # System security checks
│   └── utils/
│       └── latency.py         # Step-level latency tracker
│
└── frontend/
    ├── public/
    │   └── smallo-pcm.js      # AudioWorklet processor (runs in audio thread)
    └── src/
        ├── App.tsx             # Root layout + cartoon side decorations
        ├── hooks/
        │   ├── useWebSocket.ts # WS client — events, heartbeat, backoff reconnect
        │   └── useMicrophone.ts# AudioWorklet mic capture + silence detection
        ├── store/
        │   └── appStore.ts     # Zustand global state
        ├── components/
        │   ├── StatusBar.tsx          # Top bar — model, memory count, latency, clock
        │   ├── ConversationStream.tsx # Chat bubbles + empty states
        │   ├── InfoDrawer.tsx         # Slide-down memory + system stats panel
        │   ├── PluginNotifications.tsx# Toast notifications for plugin actions
        │   ├── StatusRing.tsx         # Animated voice state indicator
        │   └── ErrorBoundary.tsx      # React error boundary with styled fallback
        ├── lib/
        │   └── wsRef.ts        # Shared WebSocket ref (accessible outside React tree)
        └── types/
            └── events.ts       # TypeScript types for all WS events
```

---

## Configuration

### Change the LLM model

Edit `backend/llm/main_llm.py`:
```python
_MODEL = "phi3"   # change to any model installed in Ollama
```

### Adjust silence detection sensitivity

Edit `frontend/src/hooks/useMicrophone.ts`:
```ts
const SILENCE_RMS   = 0.015   // lower = more sensitive (picks up quieter speech)
const SILENCE_MS    = 1500    // ms of silence before clip is sent (lower = faster response)
const MIN_RECORD_MS = 400     // minimum clip length (filters out accidental sounds)
```

### Change TTS voice

Edit `backend/tts/main_tts.py` to point to a different `.onnx` model file.

### System prompt

Edit `backend/llm/SYSTEM_PROMPT.py` to change Small O's personality and behaviour.

---

## How Memory Works

Every conversation is automatically analysed:

1. **Retrieval** — before LLM call, top-5 semantically similar memories are fetched from FAISS and injected into context
2. **Insertion** — after the conversation, the LLM extracts key facts, decisions, and ideas; these are embedded and stored in SQLite + FAISS
3. **Importance scoring** — each memory gets a score 1–10; higher-importance memories survive consolidation longer
4. **Consolidation** — a background worker periodically merges similar low-importance memories to prevent unbounded growth

Memory types: `personal` · `project` · `decision` · `idea` · `reflection` · `action`

---

## Plugins

The plugin router intercepts queries before the LLM when a direct tool action is more appropriate.

| Plugin | Trigger examples | What it does |
|--------|-----------------|-------------|
| `internet` | "search for…", "what's the latest…" | Web search + content scraping |
| `computer` | "open…", "type…", "screenshot" | macOS GUI automation via PyAutoGUI |
| `security` | "check my…", "scan…" | System security status |

---

## Roadmap

| Feature | Status | Notes |
|---------|--------|-------|
| Browser mic capture | ✅ Done | AudioWorklet — background-tab safe |
| Token streaming frontend | ✅ Done | Real-time bubble with cursor |
| Memory system | ✅ Done | FAISS + SQLite + entity extraction |
| Plugin router | ✅ Done | Web, computer, security |
| VAD (voice activity detection) | 🔜 Planned | Browser-side; will replace silence threshold |
| Speaker / person recognition | 🔜 Planned | Voice embedding → identity; slots in after VAD |
| Remote deployment | 🔜 Planned | HTTPS + WSS so any device on the network can connect |
| Wake word | 🔜 Planned | "Hey Small O" to avoid manual Tap-to-Start |
| Multi-turn context window | 🔜 Planned | Sliding window with memory injection |

---

## Known Limitations

- **macOS only** — the computer plugin uses Quartz/PyAutoGUI macOS bindings
- **Ollama must be running** before starting `python main.py`
- **First response is slow** (~10–20 s) while Whisper and Piper load their models into memory; subsequent responses are fast
- **One speaker at a time** — the silence detector does not differentiate between speakers yet (VAD + speaker recognition are on the roadmap)
- **No wake word** — you must tap the 🎤 button once per session to grant mic access
