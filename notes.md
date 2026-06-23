# Small O — Backlog & Open Questions

Notes captured during refactor planning. Items here are not in the 
current refactor stages — they're future-stage candidates, latent 
bugs to revisit, or open questions to think about.

## UI / Toggle behavior issues (deferred to stage 8)

- **MIC pill** doesn't actually mute the microphone — the AudioWorklet 
  keeps capturing when MIC is off, only the send is gated by 
  voiceState + sttEnabled. The worklet should stop on MIC=off.
- **VOICE pill** doesn't fully mute TTS — `_tts_enabled` only stops 
  the WS audio broadcast; `sd.play` may still play locally. TTS 
  synthesis still runs (wasted CPU). VOICE=off should skip the 
  entire TTS pipeline, not just suppress the network frames.
- **BROWSER pill** label is misleading — it only controls WebViewer 
  panel visibility on the frontend. Backend still emits 
  WEB_SCREENSHOT events. Either rename the pill ("VIEWER") or add a 
  backend handler that gates screenshot emission.
- **Mode pill** click guard depends on voiceState — if voiceState 
  gets stuck (e.g. backend disconnects mid-turn), pill becomes 
  permanently unclickable. Need a better stuck-detection or escape 
  hatch.

## Audit drift (still unresolved)

- SYSTEM_EVENT `model_swap` is handled by frontend but never emitted 
  by backend. Either wire the emitter or remove the handler.
- StatusBar still shows `—` for model name (was hardcoded `phi3 · 
  Ollama`). Wire a real MODEL_INFO event from backend in stage 2 or 
  later.
- FAISS `1/(1+distance)` similarity wrapping at 
  `memory_system/retrieval/search.py:268` compresses cosine 
  similarity. Memory retrieval quality issue. Fix during stage 6 or 
  earlier if memory feels wrong.

## Stage 1 limitations (known, accepted)

- Multi-step plan summaries are NOT appended to session_history in 
  normal mode. The "On it…" ack is appended; the eventual summary 
  arrives async after the turn returns. Resolves itself in stage 3 
  when the planner is deleted.
- `ReflectionMemory` and `ActionMemory` are gated, which means 
  super-mode users no longer get reflection-based context unless 
  they switch back. Acceptable trade-off.

## Future stage ideas

- Settings/preferences UI (`user_settings.py`) — runtime-mutable 
  config for active provider, voice, default mode. Deferred until 
  after the structural refactor stabilizes.
- Mid-conversation provider switching (e.g., switch to a smarter 
  model for one hard turn).
- VAD threshold tuning if barge-in feels off.
- WebMonitor polling interval per-target customization.
- Coding agent v2 (multi-file edits, diffs, plan mode, undo) — 
  after coding agent v1 in stage 5 is stable.

## Investigations / open questions

- Confirm Ollama Turbo model names you actually want 
  (gpt-oss, deepseek-v3.1, qwen3-coder, kimi-k2, glm-4.6, etc).
- Verify which providers actually support which feature 
  (tools, streaming, multimodal) before stage 2 implementation.
