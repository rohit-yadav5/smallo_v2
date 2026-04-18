import { create } from 'zustand'
import type { VoiceState, ConversationMessage, MemoryNode, PluginNotification, SystemStats, PlanEvent, WebScreenshot, SystemEvent, FileCreatedEvent } from '../types/events'

export interface PlanStepState {
  text:    string
  result?: string
  done:    boolean
}

export interface ActivePlan {
  goal:        string
  steps:       PlanStepState[]
  currentStep: number           // -1 = decomposing
  phase:       PlanEvent['phase']
  summary?:    string
  reason?:     string
  finishedAt?: number           // timestamp for auto-hide
}

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
  activePlan:           ActivePlan | null  // running / recently finished plan

  latestScreenshot:         WebScreenshot | null  // most recent browser viewport

  /** RAM available in GB from last ram_report event */
  availableRamGb:           number | null
  /** 'low' | 'medium' | 'high' — null until first ram_report */
  ramPressure:              'low' | 'medium' | 'high' | null
  /** Transient toast for low_memory / model_swap events */
  systemToast:              { message: string; kind: 'warning' | 'info' } | null

  /** STT toggle — false disables mic capture entirely */
  sttEnabled:               boolean
  /** TTS toggle — false skips audio synthesis; text still streams */
  ttsEnabled:               boolean
  /** Browser viewer panel open/closed state */
  browserViewerOpen:        boolean

  /** Files created this session, newest first */
  sessionFiles:             FileCreatedEvent[]
  addSessionFile:           (f: FileCreatedEvent) => void
  setSessionFiles:          (files: FileCreatedEvent[]) => void

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
  handlePlanEvent:          (ev: PlanEvent) => void
  setScreenshot:            (shot: WebScreenshot) => void
  handleSystemEvent:        (ev: SystemEvent) => void
  clearSystemToast:         () => void
  toggleSTT:                () => void
  toggleTTS:                () => void
  setBrowserViewerOpen:     (v: boolean) => void
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
  activePlan:          null,
  latestScreenshot:    null,
  availableRamGb:      null,
  ramPressure:         null,
  systemToast:         null,
  sttEnabled:          true,
  ttsEnabled:          true,
  browserViewerOpen:   false,
  sessionFiles:        [],

  addSessionFile:  (f) => set((s) => ({ sessionFiles: [f, ...s.sessionFiles] })),
  setSessionFiles: (files) => set({ sessionFiles: files }),

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

  setSystemStats:  (stats) => set({ systemStats: stats }),
  setLatency:      (ms)    => set({ latency: ms }),
  setScreenshot:   (shot)  => set({ latestScreenshot: shot }),

  handlePlanEvent: (ev) => {
    set((s) => {
      switch (ev.phase) {
        case 'decomposed': {
          const steps: PlanStepState[] = (ev.steps ?? []).map((text) => ({
            text, done: false,
          }))
          return {
            activePlan: {
              goal:        ev.goal ?? '',
              steps,
              currentStep: -1,
              phase:       'decomposed',
            },
          }
        }
        case 'step_start': {
          if (!s.activePlan) return {}
          return {
            activePlan: {
              ...s.activePlan,
              currentStep: ev.step_index ?? 0,
              phase:       'step_start',
            },
          }
        }
        case 'step_done': {
          if (!s.activePlan) return {}
          const idx = ev.step_index ?? 0
          const steps = s.activePlan.steps.map((step, i) =>
            i === idx ? { ...step, done: true, result: ev.result } : step
          )
          return {
            activePlan: { ...s.activePlan, steps, phase: 'step_done' },
          }
        }
        case 'complete': {
          if (!s.activePlan) return {}
          const plan: ActivePlan = {
            ...s.activePlan,
            phase:       'complete',
            summary:     ev.summary,
            finishedAt:  Date.now(),
          }
          // Auto-hide after 10 s
          setTimeout(() => {
            const cur = get().activePlan
            if (cur?.finishedAt === plan.finishedAt) {
              set({ activePlan: null })
            }
          }, 10_000)
          return { activePlan: plan }
        }
        case 'failed': {
          const plan: ActivePlan = {
            ...(s.activePlan ?? { goal: ev.goal ?? '', steps: [], currentStep: -1 }),
            phase:      'failed',
            reason:     ev.reason,
            finishedAt: Date.now(),
          }
          setTimeout(() => {
            const cur = get().activePlan
            if (cur?.finishedAt === plan.finishedAt) set({ activePlan: null })
          }, 10_000)
          return { activePlan: plan }
        }
        case 'cancelled': {
          return { activePlan: null }
        }
        default:
          return {}
      }
    })
  },

  handleSystemEvent: (ev) => {
    if (ev.event === 'ram_report') {
      // Derive pressure level from available_gb using the same thresholds as backend
      const gb = ev.available_gb ?? null
      const pressure: 'low' | 'medium' | 'high' | null =
        gb === null ? null : gb < 1.5 ? 'high' : gb < 3.0 ? 'medium' : 'low'
      set({ availableRamGb: gb, ramPressure: pressure })
    } else if (ev.event === 'low_memory') {
      const gb = ev.available_gb ?? null
      if (gb !== null) set({ availableRamGb: gb, ramPressure: 'high' })
      // Show a transient amber warning toast — auto-cleared after 6s
      set({ systemToast: { message: ev.message, kind: 'warning' } })
      setTimeout(() => set({ systemToast: null }), 6_000)
    } else if (ev.event === 'model_swap') {
      // Show a brief info toast (model change notification)
      set({ systemToast: { message: ev.message, kind: 'info' } })
      setTimeout(() => set({ systemToast: null }), 4_000)
    }
  },

  clearSystemToast:      () => set({ systemToast: null }),
  toggleSTT:             () => set((s) => ({ sttEnabled: !s.sttEnabled })),
  toggleTTS:             () => set((s) => ({ ttsEnabled: !s.ttsEnabled })),
  setBrowserViewerOpen:  (v) => set({ browserViewerOpen: v }),
}))
