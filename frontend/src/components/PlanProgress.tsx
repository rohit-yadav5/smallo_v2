/**
 * PlanProgress.tsx — Real-time autonomous planner status panel.
 *
 * Displays the decomposed steps, active step (spinner), completed steps
 * (checkmark + collapsed result), and final summary or failure reason.
 * Disappears automatically 10 s after completion (managed by appStore).
 *
 * Styled with the same neobrutalist palette as the rest of the UI:
 *   - Plan header:      #7B2FBE (purple)
 *   - Step pending:     rgba(0,0,0,0.08) grey
 *   - Step active:      #FFD60A (yellow) + spinner
 *   - Step done:        #4CC9F0 (cyan) + ✓
 *   - Complete bubble:  #4361EE (blue)
 *   - Failed bubble:    #FF4D6D (red)
 */

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../store/appStore'

// ── Spinner SVG (simple rotating arc) ────────────────────────────────────────
function Spinner() {
  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
      className="inline-block w-3.5 h-3.5 shrink-0"
      style={{
        border:       '2.5px solid rgba(0,0,0,0.2)',
        borderTop:    '2.5px solid #000',
        borderRadius: '50%',
      }}
    />
  )
}

// ── Single step row ───────────────────────────────────────────────────────────
function StepRow({
  text,
  done,
  active,
  result,
}: {
  text:    string
  done:    boolean
  active:  boolean
  result?: string
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className="flex flex-col gap-0.5 px-2.5 py-1.5 rounded-lg"
      style={{
        background: active ? '#FFD60A'
          : done    ? 'rgba(76,201,240,0.18)'
          :           'rgba(0,0,0,0.06)',
        border:     active ? '2px solid #000' : '2px solid transparent',
        transition: 'background 0.2s',
      }}
    >
      <div className="flex items-center gap-2">
        {/* Status icon */}
        <span className="shrink-0 text-sm leading-none">
          {active ? <Spinner /> : done ? '✓' : '·'}
        </span>

        {/* Step text */}
        <span
          className="text-xs font-semibold leading-tight flex-1 truncate"
          style={{ color: active ? '#000' : done ? '#0a4a5c' : 'rgba(0,0,0,0.55)' }}
        >
          {text}
        </span>

        {/* Expand result toggle */}
        {done && result && (
          <button
            className="text-xs opacity-50 hover:opacity-100 shrink-0"
            onClick={() => setExpanded((e) => !e)}
            title={expanded ? 'Collapse' : 'Show result'}
          >
            {expanded ? '⌃' : '⌄'}
          </button>
        )}
      </div>

      {/* Expanded result */}
      <AnimatePresence>
        {expanded && result && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <p
              className="text-xs pl-5 pt-0.5 leading-relaxed"
              style={{ color: '#0a4a5c', fontFamily: 'JetBrains Mono, monospace' }}
            >
              {result.slice(0, 300)}{result.length > 300 ? '…' : ''}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────
export function PlanProgress() {
  const plan = useAppStore((s) => s.activePlan)

  if (!plan) return null

  const isTerminal = plan.phase === 'complete' || plan.phase === 'failed' || plan.phase === 'cancelled'

  return (
    <AnimatePresence>
      <motion.div
        key="plan-panel"
        initial={{ opacity: 0, y: 8, scale: 0.97 }}
        animate={{ opacity: 1, y: 0,  scale: 1 }}
        exit={{ opacity: 0, y: -8, scale: 0.97 }}
        transition={{ duration: 0.22, type: 'spring', stiffness: 300, damping: 24 }}
        className="flex flex-col gap-2 rounded-2xl overflow-hidden shrink-0"
        style={{
          border:    '3px solid #000',
          boxShadow: '5px 5px 0px #000',
        }}
      >
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div
          className="flex items-center gap-2 px-3 py-2"
          style={{ background: '#7B2FBE' }}
        >
          <span className="text-base">🗺</span>
          <span className="font-black text-sm" style={{ color: '#FFD60A' }}>
            Autonomous Plan
          </span>
          {!isTerminal && (
            <motion.div
              className="ml-auto w-2 h-2 rounded-full"
              style={{ background: '#FFD60A' }}
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
          )}
          {isTerminal && (
            <span className="ml-auto text-xs font-bold" style={{ color: 'rgba(255,255,255,0.6)' }}>
              {plan.phase === 'complete' ? '✅ done' : plan.phase === 'failed' ? '✗ failed' : '⛔ cancelled'}
            </span>
          )}
        </div>

        {/* ── Goal ───────────────────────────────────────────────────── */}
        <div className="px-3" style={{ background: '#F8F0FF' }}>
          <p className="text-xs font-semibold py-1.5 truncate" style={{ color: 'rgba(0,0,0,0.5)' }}>
            Goal: {plan.goal}
          </p>
        </div>

        {/* ── Steps list ─────────────────────────────────────────────── */}
        {plan.steps.length > 0 && (
          <div
            className="flex flex-col gap-1 px-2 pb-2"
            style={{ background: '#F8F0FF', maxHeight: '180px', overflowY: 'auto' }}
          >
            {plan.steps.map((step, i) => (
              <StepRow
                key={i}
                text={step.text}
                done={step.done}
                active={i === plan.currentStep && !isTerminal}
                result={step.result}
              />
            ))}
          </div>
        )}

        {/* ── Decomposing spinner (before steps arrive) ───────────────── */}
        {plan.steps.length === 0 && !isTerminal && (
          <div
            className="flex items-center gap-2 px-3 pb-3"
            style={{ background: '#F8F0FF' }}
          >
            <Spinner />
            <span className="text-xs" style={{ color: 'rgba(0,0,0,0.5)' }}>
              Decomposing goal into steps…
            </span>
          </div>
        )}

        {/* ── Terminal states ─────────────────────────────────────────── */}
        {plan.phase === 'complete' && plan.summary && (
          <div
            className="px-3 py-2.5"
            style={{ background: '#4361EE', borderTop: '2px solid #000' }}
          >
            <p className="text-sm font-medium leading-relaxed" style={{ color: '#fff' }}>
              {plan.summary}
            </p>
          </div>
        )}

        {plan.phase === 'failed' && plan.reason && (
          <div
            className="px-3 py-2.5"
            style={{ background: '#FF4D6D', borderTop: '2px solid #000' }}
          >
            <p className="text-sm font-medium" style={{ color: '#fff' }}>
              ✗ {plan.reason}
            </p>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  )
}
