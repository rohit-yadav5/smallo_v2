import { AnimatePresence, motion } from 'framer-motion'
import { useAppStore } from '../store/appStore'

const PLUGIN_META: Record<string, { icon: string; label: string; bg: string; text: string }> = {
  web:      { icon: '🌐', label: 'Web Search', bg: '#4CC9F0', text: '#000000' },
  internet: { icon: '🌐', label: 'Web Search', bg: '#4CC9F0', text: '#000000' },
  computer: { icon: '💻', label: 'Computer',   bg: '#7B2FBE', text: '#ffffff' },
  security: { icon: '🔒', label: 'Security',   bg: '#F72585', text: '#ffffff' },
}

export function PluginNotifications() {
  const notifications = useAppStore((s) => s.pluginNotifications)
  const remove = useAppStore((s) => s.removePluginNotification)

  return (
    <div className="fixed right-4 top-4 flex flex-col gap-2 z-50 pointer-events-none" style={{ width: '240px' }}>
      <AnimatePresence>
        {notifications.map((n) => {
          const meta = PLUGIN_META[n.plugin] ?? { icon: '⚡', label: n.plugin, bg: '#FFD60A', text: '#000000' }
          return (
            <motion.div
              key={n.id}
              initial={{ opacity: 0, x: 40, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 40, scale: 0.9 }}
              transition={{ duration: 0.25, type: 'spring', stiffness: 300, damping: 24 }}
              className="rounded-2xl px-4 py-3 text-sm pointer-events-auto cursor-pointer"
              style={{
                background: meta.bg,
                border: '3px solid #000000',
                boxShadow: '5px 5px 0px #000000',
                fontFamily: 'Inter, sans-serif',
              }}
              onClick={() => remove(n.id)}
            >
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-xl">{meta.icon}</span>
                <span className="font-black text-sm" style={{ color: meta.text }}>
                  {meta.label}
                </span>
              </div>
              <div className="text-xs font-bold" style={{ color: meta.text, opacity: 0.7, fontFamily: 'JetBrains Mono, monospace' }}>
                {n.action}
              </div>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}
