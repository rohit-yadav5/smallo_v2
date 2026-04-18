/**
 * WebViewer.tsx — Live browser viewport panel for the web agent.
 *
 * Visibility is controlled by `browserViewerOpen` in appStore.
 * App.tsx renders this component only when that flag is true.
 * Screenshots still arrive and are stored in appStore regardless.
 *
 * Includes:
 *   • LIVE badge (pulsing)
 *   • Current URL display (truncated)
 *   • Minimize / expand toggle (chevron button)
 *   • × close button (sets browserViewerOpen = false)
 *   • Timestamp of last capture
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../store/appStore'

// ── Inline chevron SVGs ────────────────────────────────────────────────────────

function ChevronUp() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M2 9.5L7 4.5L12 9.5" stroke="currentColor" strokeWidth="2.2"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function ChevronDown() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M2 4.5L7 9.5L12 4.5" stroke="currentColor" strokeWidth="2.2"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

// ── Component ──────────────────────────────────────────────────────────────────

export function WebViewer() {
  const screenshot         = useAppStore((s) => s.latestScreenshot)
  const setBrowserViewerOpen = useAppStore((s) => s.setBrowserViewerOpen)
  const [minimized, setMinimized] = useState(false)

  // No screenshot yet — render nothing but stay mounted so the store update
  // triggers a re-render once a screenshot arrives.
  if (!screenshot) return null

  const ts   = new Date(screenshot.timestamp * 1000)
  const time = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const displayUrl = screenshot.url.length > 60
    ? screenshot.url.slice(0, 57) + '…'
    : screenshot.url

  const miniUrl = screenshot.url.length > 40
    ? screenshot.url.slice(0, 37) + '…'
    : screenshot.url

  return (
    <AnimatePresence>
      <motion.div
        key="web-viewer"
        initial={{ opacity: 0, y: 20, scale: 0.97 }}
        animate={{ opacity: 1, y: 0,  scale: 1 }}
        exit={{    opacity: 0, y: 20, scale: 0.97 }}
        transition={{ duration: 0.25, ease: 'easeOut' }}
        style={{
          border:         '3px solid #FF6B35',
          borderRadius:   '12px',
          background:     'rgba(0,0,0,0.85)',
          backdropFilter: 'blur(12px)',
          overflow:       'hidden',
          fontFamily:     'monospace',
          boxShadow:      '4px 4px 0 #FF6B35',
        }}
      >
        {/* ── Header bar ─────────────────────────────────────────── */}
        <div style={{
          display:    'flex',
          alignItems: 'center',
          gap:        '10px',
          padding:    '8px 12px',
          background: '#FF6B35',
        }}>
          {/* LIVE badge */}
          <motion.div
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              background:    '#FF4D6D',
              color:         '#fff',
              fontSize:      '10px',
              fontWeight:    '900',
              padding:       '2px 7px',
              borderRadius:  '4px',
              letterSpacing: '0.1em',
              border:        '1.5px solid #fff',
              flexShrink:    0,
            }}
          >
            LIVE
          </motion.div>

          {/* Browser icon */}
          <span style={{ fontSize: '14px' }}>🌐</span>

          {/* Title */}
          <span style={{
            color:      '#000',
            fontWeight: '800',
            fontSize:   '12px',
            flexShrink: 0,
          }}>
            BROWSER
          </span>

          {/* When minimized, show the URL inline in the header */}
          {minimized && (
            <span style={{
              color:        'rgba(0,0,0,0.65)',
              fontSize:     '11px',
              overflow:     'hidden',
              textOverflow: 'ellipsis',
              whiteSpace:   'nowrap',
              flex:         1,
              minWidth:     0,
            }}>
              {miniUrl || 'about:blank'}
            </span>
          )}

          {/* Spacer (only when expanded) */}
          {!minimized && <div style={{ flex: 1 }} />}

          {/* Timestamp — only visible when expanded */}
          {!minimized && (
            <span style={{
              color:      'rgba(0,0,0,0.7)',
              fontSize:   '11px',
              flexShrink: 0,
            }}>
              {time}
            </span>
          )}

          {/* Minimize / expand toggle */}
          <button
            onClick={() => setMinimized((m) => !m)}
            aria-label={minimized ? 'Expand browser panel' : 'Minimize browser panel'}
            style={{
              display:        'flex',
              alignItems:     'center',
              justifyContent: 'center',
              background:     'rgba(0,0,0,0.18)',
              border:         '1.5px solid rgba(0,0,0,0.25)',
              borderRadius:   '5px',
              color:          '#000',
              cursor:         'pointer',
              padding:        '3px 5px',
              flexShrink:     0,
              transition:     'background 0.15s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(0,0,0,0.32)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(0,0,0,0.18)')}
          >
            {minimized ? <ChevronDown /> : <ChevronUp />}
          </button>

          {/* × Close button — collapses panel back (sets browserViewerOpen=false) */}
          <button
            onClick={() => setBrowserViewerOpen(false)}
            aria-label="Close browser panel"
            style={{
              display:        'flex',
              alignItems:     'center',
              justifyContent: 'center',
              background:     'rgba(0,0,0,0.18)',
              border:         '1.5px solid rgba(0,0,0,0.25)',
              borderRadius:   '5px',
              color:          '#000',
              cursor:         'pointer',
              padding:        '2px 6px',
              flexShrink:     0,
              fontSize:       '13px',
              fontWeight:     '700',
              lineHeight:     1,
              transition:     'background 0.15s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(0,0,0,0.32)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(0,0,0,0.18)')}
          >
            ×
          </button>
        </div>

        {/* ── Collapsible body ───────────────────────────────────── */}
        <motion.div
          initial={false}
          animate={{
            height:  minimized ? 0 : 'auto',
            opacity: minimized ? 0 : 1,
          }}
          transition={{ duration: 0.2, ease: 'easeInOut' }}
          style={{ overflow: 'hidden' }}
        >
          {/* URL bar */}
          <div style={{
            display:      'flex',
            alignItems:   'center',
            gap:          '8px',
            padding:      '6px 12px',
            background:   'rgba(255,255,255,0.04)',
            borderBottom: '1px solid rgba(255,107,53,0.3)',
          }}>
            <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)' }}>🔒</span>
            <span style={{
              flex:         1,
              fontSize:     '11px',
              color:        'rgba(255,255,255,0.75)',
              fontFamily:   'monospace',
              overflow:     'hidden',
              textOverflow: 'ellipsis',
              whiteSpace:   'nowrap',
            }}>
              {displayUrl || 'about:blank'}
            </span>
          </div>

          {/* Screenshot image */}
          <div style={{ position: 'relative' }}>
            <img
              src={`data:image/jpeg;base64,${screenshot.image}`}
              alt="Browser viewport"
              style={{
                display:        'block',
                width:          '100%',
                height:         'auto',
                maxHeight:      '400px',
                objectFit:      'cover',
                objectPosition: 'top',
              }}
            />
            <div style={{
              position:      'absolute',
              inset:         0,
              background:    'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.03) 3px, rgba(0,0,0,0.03) 4px)',
              pointerEvents: 'none',
            }} />
          </div>

          {/* Footer */}
          <div style={{
            display:        'flex',
            alignItems:     'center',
            justifyContent: 'space-between',
            padding:        '6px 12px',
            background:     'rgba(255,255,255,0.02)',
            borderTop:      '1px solid rgba(255,107,53,0.2)',
          }}>
            <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.3)' }}>
              1280 × 720 · JPEG
            </span>
            <span style={{ fontSize: '10px', color: 'rgba(255,107,53,0.7)' }}>
              Small O Web Agent
            </span>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
