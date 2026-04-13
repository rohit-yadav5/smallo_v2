/**
 * WebViewer.tsx — Live browser viewport panel for the web agent.
 *
 * Shows the most-recent WEB_SCREENSHOT frame received over WebSocket.
 * Includes:
 *   • LIVE badge (pulsing) when a screenshot is present
 *   • Current URL display (truncated)
 *   • Manual "Take screenshot" button
 *   • Timestamp of last capture
 *
 * Hidden entirely when no screenshot has been received yet (activates the
 * first time the agent navigates somewhere).
 *
 * Neobrutalist palette:
 *   - Panel border:  #FF6B35 (orange) — distinct from plan purple & memory teal
 *   - LIVE badge:    #FF4D6D (red) with pulse animation
 *   - URL bar:       rgba(0,0,0,0.6) dark glass
 *   - Screenshot btn:#FFD60A (yellow)
 */

import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../store/appStore'

export function WebViewer() {
  const screenshot = useAppStore((s) => s.latestScreenshot)

  if (!screenshot) return null

  const ts   = new Date(screenshot.timestamp * 1000)
  const time = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  // Truncate URL for display
  const displayUrl = screenshot.url.length > 60
    ? screenshot.url.slice(0, 57) + '…'
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
          border:       '3px solid #FF6B35',
          borderRadius: '12px',
          background:   'rgba(0,0,0,0.85)',
          backdropFilter: 'blur(12px)',
          overflow:     'hidden',
          fontFamily:   'monospace',
          boxShadow:    '4px 4px 0 #FF6B35',
        }}
      >
        {/* ── Header bar ─────────────────────────────────────────── */}
        <div style={{
          display:        'flex',
          alignItems:     'center',
          gap:            '10px',
          padding:        '8px 12px',
          background:     '#FF6B35',
          borderBottom:   '2px solid #FF6B35',
        }}>
          {/* LIVE badge */}
          <motion.div
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
            style={{
              background:   '#FF4D6D',
              color:        '#fff',
              fontSize:     '10px',
              fontWeight:   '900',
              padding:      '2px 7px',
              borderRadius: '4px',
              letterSpacing: '0.1em',
              border:       '1.5px solid #fff',
              flexShrink:   0,
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

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Timestamp */}
          <span style={{
            color:    'rgba(0,0,0,0.7)',
            fontSize: '11px',
          }}>
            {time}
          </span>
        </div>

        {/* ── URL bar ────────────────────────────────────────────── */}
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
            flex:       1,
            fontSize:   '11px',
            color:      'rgba(255,255,255,0.75)',
            fontFamily: 'monospace',
            overflow:   'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {displayUrl || 'about:blank'}
          </span>
        </div>

        {/* ── Screenshot image ────────────────────────────────────── */}
        <div style={{ position: 'relative' }}>
          <img
            src={`data:image/jpeg;base64,${screenshot.image}`}
            alt="Browser viewport"
            style={{
              display:   'block',
              width:     '100%',
              height:    'auto',
              maxHeight: '400px',
              objectFit: 'cover',
              objectPosition: 'top',
            }}
          />

          {/* Subtle scan-line overlay for neobrutalist feel */}
          <div style={{
            position:        'absolute',
            inset:           0,
            background:      'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.03) 3px, rgba(0,0,0,0.03) 4px)',
            pointerEvents:   'none',
          }} />
        </div>

        {/* ── Footer ──────────────────────────────────────────────── */}
        <div style={{
          display:      'flex',
          alignItems:   'center',
          justifyContent: 'space-between',
          padding:      '6px 12px',
          background:   'rgba(255,255,255,0.02)',
          borderTop:    '1px solid rgba(255,107,53,0.2)',
        }}>
          <span style={{
            fontSize: '10px',
            color:    'rgba(255,255,255,0.3)',
          }}>
            1280 × 720 · JPEG
          </span>

          <span style={{
            fontSize: '10px',
            color:    'rgba(255,107,53,0.7)',
          }}>
            Small O Web Agent
          </span>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
