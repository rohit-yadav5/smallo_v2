export type VoiceState = 'idle' | 'listening' | 'thinking' | 'speaking'

export interface STTResult {
  text: string
  recording_time: number
  transcription_time: number
}

export interface PluginAction {
  plugin: 'web' | 'computer' | 'security' | string
  action: string
  result: string
  direct?: boolean
}

export interface LLMToken {
  token: string
  done: boolean
}

export interface VoiceStateEvent {
  state: VoiceState
}

export interface MemoryEvent {
  type: 'personal' | 'project' | 'decision' | 'idea' | 'reflection' | 'action'
  importance: number
  summary: string
  id: string
  retrieved?: boolean
}

export interface STTPartial {
  text:       string   // confirmed words (solid display)
  hypothesis: string   // unconfirmed trailing words (faded display)
}

export interface AudioChunkMeta {
  format:      'pcm16' | 'opus'
  sample_rate: number
  chunk_count: number
  chunk_ms:    number
}

export type WSEvent =
  | { event: 'STT_RESULT';    data: STTResult }
  | { event: 'STT_PARTIAL';   data: STTPartial }
  | { event: 'PLUGIN_ACTION'; data: PluginAction }
  | { event: 'LLM_TOKEN';     data: LLMToken }
  | { event: 'VOICE_STATE';   data: VoiceStateEvent }
  | { event: 'MEMORY_EVENT';  data: MemoryEvent }
  | { event: 'SYSTEM_STATS';  data: SystemStats }
  | { event: 'AUDIO_CHUNK';   data: AudioChunkMeta }
  | { event: 'AUDIO_DONE';    data: Record<string, never> }
  | { event: 'AUDIO_ABORT';   data: Record<string, never> }
  | { event: 'pong';          data: Record<string, never> }

export interface SystemStats {
  cpu: number
  ram: number
  battery: number
}

export interface ConversationMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  streaming?: boolean
  stt_time?: number            // recording duration (how long user spoke)
  transcription_time?: number  // whisper processing time
  llm_time?: number            // LLM generation time (assistant only)
  tts_time?: number
  plugin?: string
}

export interface MemoryNode {
  id: string
  type: MemoryEvent['type']
  summary: string
  importance: number
  x: number
  y: number
  glowing?: boolean
}

export interface PluginNotification {
  id: string
  plugin: string
  action: string
  result: string
  timestamp: number
}
