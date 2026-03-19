import { useAppStore } from '../store/appStore'

interface RadialGaugeProps {
  value: number
  label: string
  color: string
}

function RadialGauge({ value, label, color }: RadialGaugeProps) {
  const radius = 28
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value / 100) * circumference

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative w-16 h-16">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 72 72">
          <circle cx="36" cy="36" r={radius} fill="none" stroke="rgba(139,92,246,0.1)" strokeWidth="4" />
          <circle
            cx="36"
            cy="36"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="4"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />
        </svg>
        <div
          className="absolute inset-0 flex items-center justify-center text-xs font-bold"
          style={{ fontFamily: 'JetBrains Mono, monospace', color }}
        >
          {value}%
        </div>
      </div>
      <span
        className="text-xs tracking-widest"
        style={{ fontFamily: 'Orbitron, sans-serif', color: '#52525b' }}
      >
        {label}
      </span>
    </div>
  )
}

export function SystemStats() {
  const { cpu, ram, battery } = useAppStore((s) => s.systemStats)

  return (
    <div className="flex flex-col h-full gap-3">
      <div
        className="text-xs tracking-widest text-zinc-500 px-1"
        style={{ fontFamily: 'Orbitron, sans-serif' }}
      >
        SYSTEM TELEMETRY
      </div>

      <div className="flex flex-col items-center gap-6 flex-1 justify-center">
        <RadialGauge value={cpu} label="CPU" color="#8b5cf6" />
        <RadialGauge value={ram} label="RAM" color="#a855f7" />
        <RadialGauge value={battery} label="PWR" color="#c084fc" />
      </div>

      {/* Connection status */}
      <div
        className="px-1 text-xs"
        style={{ fontFamily: 'JetBrains Mono, monospace', color: '#3f3f46' }}
      >
        <div>OLLAMA ● LOCAL</div>
        <div>WHISPER ● TINY</div>
        <div>PIPER ● AMY-MD</div>
      </div>
    </div>
  )
}
