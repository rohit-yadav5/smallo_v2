import { useRef } from 'react'
import { useAppStore } from '../store/appStore'

const TYPE_COLORS: Record<string, string> = {
  personal: '#8b5cf6',
  project: '#3b82f6',
  decision: '#eab308',
  idea: '#a855f7',
  reflection: '#6366f1',
  action: '#10b981',
}

export function MemoryNetwork() {
  const nodes = useAppStore((s) => s.memoryNodes)
  const memoryCount = useAppStore((s) => s.memoryCount)
  const svgRef = useRef<SVGSVGElement>(null)

  return (
    <div className="flex flex-col h-full gap-3">
      <div
        className="text-xs tracking-widest text-zinc-500 px-1"
        style={{ fontFamily: 'Orbitron, sans-serif' }}
      >
        MEMORY NETWORK
        <span className="ml-2 text-purple-500">{memoryCount}</span>
      </div>

      <div className="flex-1 relative overflow-hidden">
        <svg
          ref={svgRef}
          className="w-full h-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          style={{
            WebkitMaskImage: 'linear-gradient(to right, black 40%, transparent 100%)',
            maskImage: 'linear-gradient(to right, black 40%, transparent 100%)',
          }}
        >
          {/* Connection lines between nearby nodes */}
          {nodes.map((node, i) =>
            nodes.slice(i + 1, i + 4).map((other) => {
              const dist = Math.hypot(node.x - other.x, node.y - other.y)
              if (dist > 35) return null
              return (
                <line
                  key={`${node.id}-${other.id}`}
                  x1={node.x}
                  y1={node.y}
                  x2={other.x}
                  y2={other.y}
                  stroke={TYPE_COLORS[node.type] || '#8b5cf6'}
                  strokeWidth="0.3"
                  strokeOpacity="0.25"
                />
              )
            })
          )}

          {/* Nodes */}
          {nodes.map((node) => (
            <g key={node.id}>
              {node.glowing && (
                <circle
                  cx={node.x}
                  cy={node.y}
                  r="3.5"
                  fill={TYPE_COLORS[node.type] || '#8b5cf6'}
                  opacity="0.3"
                >
                  <animate attributeName="r" values="3.5;6;3.5" dur="1s" repeatCount="2" />
                  <animate attributeName="opacity" values="0.3;0;0.3" dur="1s" repeatCount="2" />
                </circle>
              )}
              <circle
                cx={node.x}
                cy={node.y}
                r={1.2 + node.importance * 1.5}
                fill={TYPE_COLORS[node.type] || '#8b5cf6'}
                opacity={node.glowing ? 1 : 0.7}
              >
                <title>{node.summary}</title>
              </circle>
            </g>
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-col gap-1 px-1">
        {Object.entries(TYPE_COLORS).slice(0, 4).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
            <span
              className="text-xs capitalize"
              style={{ fontFamily: 'JetBrains Mono, monospace', color: '#52525b' }}
            >
              {type}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
