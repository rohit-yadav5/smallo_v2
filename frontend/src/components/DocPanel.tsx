import { useState, useEffect, useRef } from 'react'
import { useAppStore } from '../store/appStore'
import { wsRef } from '../lib/wsRef'
import type { FileCreatedEvent } from '../types/events'

// ── Extension badge config ────────────────────────────────────────────────────
const EXT_BADGES: Record<string, { label: string; bg: string; color: string }> = {
  '.txt':  { label: 'TXT',  bg: '#E5E7EB', color: '#374151' },
  '.md':   { label: 'MD',   bg: '#DBEAFE', color: '#1D4ED8' },
  '.py':   { label: 'PY',   bg: '#DCFCE7', color: '#15803D' },
  '.json': { label: 'JSON', bg: '#FEF3C7', color: '#B45309' },
  '.html': { label: 'HTML', bg: '#FFE4E6', color: '#BE123C' },
}

function getBadge(ext: string) {
  return EXT_BADGES[ext] ?? { label: 'FILE', bg: '#E5E7EB', color: '#374151' }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  return `${(bytes / 1024).toFixed(1)} KB`
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000 - ts)
  if (diff < 60)  return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  return `${Math.floor(diff / 3600)}h ago`
}

// ── FileRow ───────────────────────────────────────────────────────────────────
function FileRow({ file, isNew }: { file: FileCreatedEvent; isNew: boolean }) {
  const badge = getBadge(file.extension)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // Tiny delay so CSS transition fires on mount
    const t = setTimeout(() => setVisible(true), 20)
    return () => clearTimeout(t)
  }, [])

  function handleDownload() {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ event: 'DOWNLOAD_FILE', data: { uid: file.uid } }))
  }

  return (
    <div
      style={{
        opacity:    visible ? 1 : 0,
        transform:  visible ? 'translateY(0)' : 'translateY(-8px)',
        transition: 'opacity 0.25s ease, transform 0.25s ease',
        background: isNew ? 'rgba(123,47,190,0.05)' : 'transparent',
        borderRadius: 8,
        padding: '6px 8px',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        borderBottom: '1px solid rgba(0,0,0,0.06)',
      }}
    >
      {/* Extension badge */}
      <span
        style={{
          display:      'inline-flex',
          alignItems:   'center',
          justifyContent: 'center',
          minWidth:     36,
          padding:      '2px 4px',
          borderRadius: 4,
          fontSize:     10,
          fontWeight:   800,
          fontFamily:   'JetBrains Mono, monospace',
          background:   badge.bg,
          color:        badge.color,
          border:       '1.5px solid rgba(0,0,0,0.12)',
          flexShrink:   0,
        }}
      >
        {badge.label}
      </span>

      {/* Title + meta */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            fontSize:     13,
            fontWeight:   700,
            fontFamily:   'Inter, sans-serif',
            color:        '#111',
            whiteSpace:   'nowrap',
            overflow:     'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {file.title}
        </div>
        <div
          style={{
            fontSize:   11,
            color:      '#6B7280',
            fontFamily: 'Inter, sans-serif',
            marginTop:  1,
          }}
        >
          {formatSize(file.size_bytes)} · {timeAgo(file.created_at)}
        </div>
      </div>

      {/* Download button */}
      <button
        onClick={handleDownload}
        title="Download file"
        style={{
          flexShrink:   0,
          width:        28,
          height:       28,
          borderRadius: 6,
          border:       '2px solid #000',
          background:   '#FFD60A',
          boxShadow:    '2px 2px 0px #000',
          cursor:       'pointer',
          display:      'flex',
          alignItems:   'center',
          justifyContent: 'center',
          fontSize:     14,
          fontWeight:   900,
          lineHeight:   1,
          transition:   'transform 0.1s',
        }}
        onMouseDown={(e) => (e.currentTarget.style.transform = 'translate(1px,1px)')}
        onMouseUp={(e) => (e.currentTarget.style.transform = '')}
        onMouseLeave={(e) => (e.currentTarget.style.transform = '')}
      >
        ↓
      </button>
    </div>
  )
}

// ── DocPanel ──────────────────────────────────────────────────────────────────
export function DocPanel() {
  const sessionFiles = useAppStore((s) => s.sessionFiles)
  const [collapsed, setCollapsed]   = useState(false)
  const [pulsing,   setPulsing]     = useState(false)
  const prevCountRef                = useRef(0)
  const [newUids,   setNewUids]     = useState<Set<string>>(new Set())

  // Track new files for animation + pulse
  useEffect(() => {
    if (sessionFiles.length > prevCountRef.current) {
      const added = sessionFiles.slice(0, sessionFiles.length - prevCountRef.current)
      setNewUids((prev) => {
        const next = new Set(prev)
        added.forEach((f) => next.add(f.uid))
        return next
      })
      // Clear "new" highlight after 3 s
      const t = setTimeout(() => {
        setNewUids((prev) => {
          const next = new Set(prev)
          added.forEach((f) => next.delete(f.uid))
          return next
        })
      }, 3_000)

      if (collapsed) {
        setPulsing(true)
        setTimeout(() => setPulsing(false), 1_500)
      }

      prevCountRef.current = sessionFiles.length
      return () => clearTimeout(t)
    }
    prevCountRef.current = sessionFiles.length
  }, [sessionFiles.length, collapsed])

  // Don't render at all when no files exist yet
  if (sessionFiles.length === 0) return null

  return (
    <div
      style={{
        position:     'fixed',
        bottom:       16,
        right:        16,
        zIndex:       50,
        width:        320,
        fontFamily:   'Inter, sans-serif',
        // Slide in from right
        animation:    'docPanelSlideIn 0.3s ease',
      }}
    >
      <style>{`
        @keyframes docPanelSlideIn {
          from { opacity: 0; transform: translateX(20px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes docBadgePulse {
          0%   { transform: scale(1); }
          40%  { transform: scale(1.35); }
          100% { transform: scale(1); }
        }
      `}</style>

      {/* ── Header ── */}
      <div
        onClick={() => setCollapsed((c) => !c)}
        style={{
          display:       'flex',
          alignItems:    'center',
          justifyContent:'space-between',
          padding:       '8px 12px',
          background:    '#7B2FBE',
          border:        '3px solid #000',
          borderRadius:  collapsed ? 12 : '12px 12px 0 0',
          boxShadow:     collapsed ? '4px 4px 0px #000' : '4px 0px 0px #000',
          cursor:        'pointer',
          userSelect:    'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 14 }}>📄</span>
          <span
            style={{
              fontSize:   13,
              fontWeight: 800,
              color:      '#fff',
            }}
          >
            Files this session
          </span>
          {/* Count badge */}
          <span
            style={{
              display:       'inline-flex',
              alignItems:    'center',
              justifyContent:'center',
              minWidth:      20,
              height:        20,
              borderRadius:  10,
              background:    '#FFD60A',
              border:        '2px solid #000',
              fontSize:      11,
              fontWeight:    900,
              color:         '#000',
              fontFamily:    'JetBrains Mono, monospace',
              animation:     pulsing ? 'docBadgePulse 0.5s ease 3' : 'none',
            }}
          >
            {sessionFiles.length}
          </span>
        </div>
        <span style={{ color: '#FFD60A', fontSize: 12, fontWeight: 700 }}>
          {collapsed ? '▲' : '▼'}
        </span>
      </div>

      {/* ── File list ── */}
      {!collapsed && (
        <div
          style={{
            background:  '#fff',
            border:      '3px solid #000',
            borderTop:   'none',
            borderRadius:'0 0 12px 12px',
            boxShadow:   '4px 4px 0px #000',
            maxHeight:   320,
            overflowY:   'auto',
            padding:     '4px 6px',
          }}
        >
          {sessionFiles.map((f) => (
            <FileRow key={f.uid} file={f} isNew={newUids.has(f.uid)} />
          ))}
        </div>
      )}
    </div>
  )
}
