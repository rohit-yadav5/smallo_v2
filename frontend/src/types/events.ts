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

export interface ProactiveEvent {
  /** Sub-type of proactive event, e.g. "reminder" | "web_monitor" */
  event: string
  /** Human-readable message to surface to the user (reminders) */
  message?: string
  // web_monitor fields
  target_id?:   string
  description?: string
  summary?:     string
  url?:         string
}

/** Live browser viewport screenshot streamed from the web agent. */
export interface WebScreenshot {
  /** Base64-encoded JPEG image of the browser viewport */
  image:     string
  /** Current URL loaded in the browser */
  url:       string
  /** Unix epoch timestamp (seconds) */
  timestamp: number
}

/** Sent from the frontend to inject typed text into the pipeline. */
export interface TextInputSent {
  text: string
}

/** Autonomous planner progress events. */
export interface PlanEvent {
  phase:      'decomposed' | 'step_start' | 'step_done' | 'complete' | 'failed' | 'cancelled'
  goal?:      string
  steps?:     string[]       // decomposed
  step_index?: number        // step_start / step_done
  step_text?:  string        // step_start
  total?:      number        // step_start
  result?:     string        // step_done
  summary?:    string        // complete
  reason?:     string        // failed
}

/** RAM pressure system events emitted by the backend. */
export interface SystemEvent {
  /** Sub-type of system event */
  event: 'low_memory' | 'model_swap' | 'ram_report'
  /** Human-readable message */
  message: string
  /** Available RAM in GB at time of event */
  available_gb?: number
}

export interface FileCreatedEvent {
  uid:        string
  title:      string
  filename:   string
  extension:  string
  size_bytes: number
  created_at: number
}

export interface FileListEvent {
  files: FileCreatedEvent[]
}

export interface FileContentEvent {
  uid:       string
  title:     string
  filename:  string
  content:   string
  extension: string
}

export type WSEvent =
  | { event: 'STT_RESULT';       data: STTResult }
  | { event: 'STT_PARTIAL';      data: STTPartial }
  | { event: 'PLUGIN_ACTION';    data: PluginAction }
  | { event: 'LLM_TOKEN';        data: LLMToken }
  | { event: 'VOICE_STATE';      data: VoiceStateEvent }
  | { event: 'MEMORY_EVENT';     data: MemoryEvent }
  | { event: 'SYSTEM_STATS';     data: SystemStats }
  | { event: 'SYSTEM_EVENT';     data: SystemEvent }
  | { event: 'AUDIO_CHUNK';      data: AudioChunkMeta }
  | { event: 'AUDIO_DONE';       data: Record<string, never> }
  | { event: 'AUDIO_ABORT';      data: Record<string, never> }
  | { event: 'PROACTIVE_EVENT';  data: ProactiveEvent }
  | { event: 'PLAN_EVENT';       data: PlanEvent }
  | { event: 'WEB_SCREENSHOT';   data: WebScreenshot }
  | { event: 'FILE_CREATED';     data: FileCreatedEvent }
  | { event: 'FILE_LIST';        data: FileListEvent }
  | { event: 'FILE_CONTENT';     data: FileContentEvent }
  | { event: 'pong';             data: Record<string, never> }

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
