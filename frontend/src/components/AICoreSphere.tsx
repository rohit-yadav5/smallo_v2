import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { useAppStore } from '../store/appStore'

const STATE_COLORS = {
  idle: new THREE.Color(0x8b5cf6),
  listening: new THREE.Color(0xa855f7),
  thinking: new THREE.Color(0xc084fc),
  speaking: new THREE.Color(0xd946ef),
}

export function AICoreSphere() {
  const mountRef = useRef<HTMLDivElement>(null)
  const voiceState = useAppStore((s) => s.voiceState)
  const voiceStateRef = useRef(voiceState)

  useEffect(() => {
    voiceStateRef.current = voiceState
  }, [voiceState])

  useEffect(() => {
    if (!mountRef.current) return
    const container = mountRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Scene
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000)
    camera.position.z = 3.5

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setClearColor(0x000000, 0)
    container.appendChild(renderer.domElement)

    // Particle sphere
    const PARTICLE_COUNT = 1200
    const positions = new Float32Array(PARTICLE_COUNT * 3)
    const colors = new Float32Array(PARTICLE_COUNT * 3)
    const sizes = new Float32Array(PARTICLE_COUNT)
    const velocities: THREE.Vector3[] = []
    const basePositions: THREE.Vector3[] = []

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const phi = Math.acos(2 * Math.random() - 1)
      const theta = Math.random() * Math.PI * 2
      const r = 1 + (Math.random() - 0.5) * 0.3

      const x = r * Math.sin(phi) * Math.cos(theta)
      const y = r * Math.sin(phi) * Math.sin(theta)
      const z = r * Math.cos(phi)

      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z

      basePositions.push(new THREE.Vector3(x, y, z))
      velocities.push(new THREE.Vector3(
        (Math.random() - 0.5) * 0.002,
        (Math.random() - 0.5) * 0.002,
        (Math.random() - 0.5) * 0.002,
      ))

      colors[i * 3] = 0.54
      colors[i * 3 + 1] = 0.36
      colors[i * 3 + 2] = 0.96

      sizes[i] = Math.random() * 3 + 1
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1))

    const material = new THREE.PointsMaterial({
      size: 0.04,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      sizeAttenuation: true,
    })

    const particles = new THREE.Points(geometry, material)
    scene.add(particles)

    // Connection lines (subset of particles)
    const LINE_PARTICLES = 80
    const linePositions = new Float32Array(LINE_PARTICLES * LINE_PARTICLES * 6)
    const lineGeometry = new THREE.BufferGeometry()
    const linePosAttr = new THREE.BufferAttribute(linePositions, 3)
    linePosAttr.setUsage(THREE.DynamicDrawUsage)
    lineGeometry.setAttribute('position', linePosAttr)

    const lineMaterial = new THREE.LineBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.12 })

    const lines = new THREE.LineSegments(lineGeometry, lineMaterial)
    scene.add(lines)

    // Glow sphere
    const glowGeo = new THREE.SphereGeometry(1.15, 32, 32)
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x8b5cf6,
      transparent: true,
      opacity: 0.04,
      side: THREE.BackSide,
    })
    const glowMesh = new THREE.Mesh(glowGeo, glowMat)
    scene.add(glowMesh)

    let frame = 0
    let animId: number

    function animate() {
      animId = requestAnimationFrame(animate)
      frame++

      const state = voiceStateRef.current
      const targetColor = STATE_COLORS[state]
      const posAttr = geometry.attributes.position as THREE.BufferAttribute
      const colAttr = geometry.attributes.color as THREE.BufferAttribute

      const speedMult = state === 'thinking' ? 3 : state === 'speaking' ? 2 : state === 'listening' ? 1.5 : 1
      const expandMult = state === 'listening' ? 1.08 : state === 'speaking' ? 1.12 : 1

      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const base = basePositions[i]
        const vel = velocities[i]

        const noise = Math.sin(frame * 0.02 * speedMult + i * 0.1) * 0.015
        const nx = base.x * expandMult + vel.x * frame * speedMult + noise
        const ny = base.y * expandMult + vel.y * frame * speedMult + noise
        const nz = base.z * expandMult + vel.z * frame * speedMult + noise

        posAttr.setXYZ(i, nx, ny, nz)

        // Lerp color toward target
        const cr = colAttr.getX(i)
        const cg = colAttr.getY(i)
        const cb = colAttr.getZ(i)
        colAttr.setXYZ(
          i,
          cr + (targetColor.r - cr) * 0.02,
          cg + (targetColor.g - cg) * 0.02,
          cb + (targetColor.b - cb) * 0.02,
        )
      }
      posAttr.needsUpdate = true
      colAttr.needsUpdate = true

      // Update connection lines
      let lineIdx = 0
      const linePosData = linePosAttr.array as Float32Array
      for (let i = 0; i < LINE_PARTICLES; i++) {
        for (let j = i + 1; j < LINE_PARTICLES; j++) {
          const ax = posAttr.getX(i), ay = posAttr.getY(i), az = posAttr.getZ(i)
          const bx = posAttr.getX(j), by = posAttr.getY(j), bz = posAttr.getZ(j)
          const dist = Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
          if (dist < 0.6) {
            linePosData[lineIdx++] = ax
            linePosData[lineIdx++] = ay
            linePosData[lineIdx++] = az
            linePosData[lineIdx++] = bx
            linePosData[lineIdx++] = by
            linePosData[lineIdx++] = bz
          }
        }
      }
      lineGeometry.setDrawRange(0, lineIdx / 3)
      linePosAttr.needsUpdate = true

      particles.rotation.y += 0.003 * speedMult
      particles.rotation.x += 0.001 * speedMult

      const glowOpacity = state === 'speaking' ? 0.1 : state === 'listening' ? 0.07 : 0.04
      glowMat.opacity += (glowOpacity - glowMat.opacity) * 0.05
      glowMesh.rotation.y -= 0.002

      renderer.render(scene, camera)
    }

    animate()

    const handleResize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', handleResize)
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  return <div ref={mountRef} className="w-full h-full" />
}
