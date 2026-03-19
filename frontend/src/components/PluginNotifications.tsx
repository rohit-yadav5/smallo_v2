import { AnimatePresence, motion } from 'framer-motion'
import { useAppStore } from '../store/appStore'

const PLUGIN_META: Record<string, { icon: string; label: string; color: string }> = {
  web: { icon: '🌐', label: 'Web Search', color: '#3b82f6' },
  internet: { icon: '🌐', label: 'Web Search', color: '#3b82f6' },
  computer: { icon: '💻', label: 'Computer', color: '#8b5cf6' },
  security: { icon: '🔒', label: 'Security', color: '#ef4444' },
}

export function PluginNotifications() {
  const notifications = useAppStore((s) => s.pluginNotifications)
  const remove = useAppStore((s) => s.removePluginNotification)

  return (
    <div className="fixed right-4 top-16 flex flex-col gap-2 z-50 pointer-events-none" style={{ width: '240px' }}>
      <AnimatePresence>
        {notifications.map((n) => {
          const meta = PLUGIN_META[n.plugin] || { icon: '⚡', label: n.plugin, color: '#8b5cf6' }
          return (
            <motion.div
              key={n.id}
              initial={{ opacity: 0, x: 40 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 40 }}
              transition={{ duration: 0.3 }}
              className="rounded border px-3 py-2 text-xs pointer-events-auto cursor-pointer"
              style={{
                background: 'rgba(8,8,20,0.95)',
                borderColor: `${meta.color}40`,
                backdropFilter: 'blur(8px)',
              }}
              onClick={() => remove(n.id)}
            >
              <div className="flex items-center gap-2 mb-1">
                <span>{meta.icon}</span>
                <span style={{ fontFamily: 'Orbitron, sans-serif', color: meta.color, fontSize: '10px', letterSpacing: '0.1em' }}>
                  {meta.label.toUpperCase()}
                </span>
              </div>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717a' }}>
                {n.action}
              </div>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}
