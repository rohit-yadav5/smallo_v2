import { motion, AnimatePresence } from 'framer-motion'
import { useAppStore } from '../store/appStore'

const TYPE_COLORS: Record<string, string> = {
  personal:   '#f97316',
  project:    '#3b82f6',
  decision:   '#eab308',
  reflection: '#8b5cf6',
  action:     '#10b981',
  idea:       '#06b6d4',
}

function StatItem({ label, value, warn }: { label: string; value: number; warn?: boolean }) {
  const high = warn && value > 80
  return (
    <span style={{ fontFamily: 'JetBrains Mono, monospace', color: high ? '#FFD60A' : '#ffffff', fontWeight: 700 }}>
      {label} <strong style={{ color: high ? '#FFD60A' : '#ffffff' }}>{value.toFixed(0)}%</strong>
    </span>
  )
}

interface Props {
  open: boolean
}

export function InfoDrawer({ open }: Props) {
  const memoryNodes = useAppStore((s) => s.memoryNodes)
  const systemStats = useAppStore((s) => s.systemStats)

  const recent = [...memoryNodes].reverse().slice(0, 10)

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.25, ease: 'easeInOut' }}
          style={{ overflow: 'hidden' }}
        >
          <div
            className="rounded-2xl p-4 flex flex-col gap-3"
            style={{
              background: '#FF006E',
              border: '3px solid #000000',
              boxShadow: '5px 5px 0px #000000',
            }}
          >
            {/* Stats row */}
            <div className="flex items-center gap-4 text-xs pb-2" style={{ borderBottom: '2px dashed rgba(255,255,255,0.4)' }}>
              <StatItem label="🖥 CPU" value={systemStats.cpu} warn />
              <StatItem label="💾 RAM" value={systemStats.ram} warn />
              <StatItem label="🔋 Bat" value={systemStats.battery} />
            </div>

            {/* Memory list */}
            {recent.length === 0 ? (
              <p className="text-xs text-center py-2" style={{ color: 'rgba(255,255,255,0.7)' }}>
                No memories yet
              </p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {recent.map((node) => (
                  <div
                    key={node.id}
                    className="flex items-start gap-2.5 px-2 py-1.5 rounded-lg text-sm transition-colors"
                    style={{
                      background: node.glowing ? '#FFD60A' : 'rgba(255,255,255,0.15)',
                      border: node.glowing ? '2px solid #000000' : '1px solid rgba(255,255,255,0.2)',
                    }}
                  >
                    <div
                      className="rounded-full mt-1.5 shrink-0"
                      style={{
                        width: 8,
                        height: 8,
                        background: TYPE_COLORS[node.type] ?? '#d1c9be',
                        boxShadow: node.glowing ? `0 0 6px ${TYPE_COLORS[node.type]}` : 'none',
                      }}
                    />
                    <span
                      className="flex-1 leading-snug truncate font-medium"
                      style={{ color: '#ffffff' }}
                      title={node.summary}
                    >
                      {node.summary}
                    </span>
                    <span
                      className="text-xs shrink-0 font-bold"
                      style={{ fontFamily: 'JetBrains Mono, monospace', color: 'rgba(255,255,255,0.6)' }}
                    >
                      {node.type}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
