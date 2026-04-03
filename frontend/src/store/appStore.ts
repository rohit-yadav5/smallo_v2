import { create } from 'zustand'
import type { VoiceState, ConversationMessage, MemoryNode, PluginNotification, SystemStats } from '../types/events'

interface AppState {
  voiceState:           VoiceState
  wsConnected:          boolean
  micActive:            boolean          // true once AudioWorklet pipeline is running
  messages:             ConversationMessage[]
  partialUserText:      string           // confirmed words from streaming STT
  partialHypothesis:    string           // unconfirmed trailing words (faded in UI)
  memoryNodes:          MemoryNode[]
  pluginNotifications:  PluginNotification[]
  systemStats:          SystemStats
  memoryCount:          number
  latency:              number
  currentStreamId:      string | null

  setVoiceState:            (state: VoiceState) => void
  setWsConnected:           (connected: boolean) => void
  setMicActive:             (v: boolean) => void
  setPartialUserText:       (text: string, hypothesis: string) => void
  clearPartialUserText:     () => void
  addUserMessage:           (text: string, stt_time: number, transcription_time: number) => void
  startAssistantMessage:    () => string
  appendToken:              (id: string, token: string) => void
  finalizeMessage:          (id: string, llm_time?: number, tts_time?: number, plugin?: string, text?: string) => void
  addMemoryNode:            (node: Omit<MemoryNode, 'x' | 'y'>) => void
  glowMemoryNode:           (id: string) => void
  addPluginNotification:    (notif: Omit<PluginNotification, 'id' | 'timestamp'>) => void
  removePluginNotification: (id: string) => void
  setSystemStats:           (stats: SystemStats) => void
  setLatency:               (ms: number) => void
}

// Module-level map — cancels stale glow timers when same node glows again
const _glowTimers = new Map<string, ReturnType<typeof setTimeout>>()

export const useAppStore = create<AppState>((set, get) => ({
  voiceState:          'idle',
  wsConnected:         false,
  micActive:           false,
  messages:            [],
  partialUserText:     '',
  partialHypothesis:   '',
  memoryNodes:         [],
  pluginNotifications: [],
  systemStats:         { cpu: 0, ram: 0, battery: 0 },
  memoryCount:         0,
  latency:             0,
  currentStreamId:     null,

  setVoiceState:        (state)     => set({ voiceState: state }),
  setWsConnected:       (connected) => set({ wsConnected: connected }),
  setMicActive:         (v)         => set({ micActive: v }),
  setPartialUserText:   (text, hyp) => set({ partialUserText: text, partialHypothesis: hyp }),
  clearPartialUserText: ()          => set({ partialUserText: '', partialHypothesis: '' }),

  addUserMessage: (text, stt_time, transcription_time) => {
    const id = crypto.randomUUID()
    set((s) => ({
      messages: [...s.messages, {
        id, role: 'user', text,
        stt_time,           // recording duration
        transcription_time, // whisper processing time
      }],
    }))
  },

  startAssistantMessage: () => {
    const id = crypto.randomUUID()
    set((s) => ({
      messages: [...s.messages, { id, role: 'assistant', text: '', streaming: true }],
      currentStreamId: id,
    }))
    return id
  },

  appendToken: (id, token) => {
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, text: m.text + token } : m
      ),
    }))
  },

  finalizeMessage: (id, llm_time, tts_time, plugin, text) => {
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? { ...m, streaming: false, llm_time, tts_time, plugin, ...(text !== undefined && { text }) }
          : m
      ),
      currentStreamId: null,
    }))
  },

  addMemoryNode: (node) => {
    const angle  = Math.random() * Math.PI * 2
    const radius = 60 + Math.random() * 120
    const x = 50 + Math.cos(angle) * radius * 0.4
    const y = 50 + Math.sin(angle) * radius * 0.3
    set((s) => ({
      memoryNodes: [...s.memoryNodes.slice(-40), { ...node, x, y }],
      memoryCount: s.memoryCount + 1,
    }))
  },

  glowMemoryNode: (id) => {
    // Cancel any existing timer for this node before starting a new one
    const existing = _glowTimers.get(id)
    if (existing !== undefined) clearTimeout(existing)

    set((s) => ({
      memoryNodes: s.memoryNodes.map((n) => (n.id === id ? { ...n, glowing: true } : n)),
    }))

    const t = setTimeout(() => {
      _glowTimers.delete(id)
      set((s) => ({
        memoryNodes: s.memoryNodes.map((n) => (n.id === id ? { ...n, glowing: false } : n)),
      }))
    }, 1500)
    _glowTimers.set(id, t)
  },

  addPluginNotification: (notif) => {
    const id   = crypto.randomUUID()
    const full = { ...notif, id, timestamp: Date.now() }
    set((s) => ({ pluginNotifications: [...s.pluginNotifications, full] }))
    setTimeout(() => get().removePluginNotification(id), 4000)
  },

  removePluginNotification: (id) => {
    set((s) => ({ pluginNotifications: s.pluginNotifications.filter((n) => n.id !== id) }))
  },

  setSystemStats: (stats) => set({ systemStats: stats }),
  setLatency:     (ms)    => set({ latency: ms }),
}))
