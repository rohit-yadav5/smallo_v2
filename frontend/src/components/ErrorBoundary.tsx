import { Component, ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    console.error('[ErrorBoundary] caught:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          minHeight: '100vh', display: 'flex', alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #FFD60A 0%, #FF4D6D 50%, #7B2FBE 100%)',
          fontFamily: 'Inter, sans-serif',
        }}>
          <div style={{
            background: '#FF4D6D', border: '4px solid #000',
            boxShadow: '8px 8px 0px #000', borderRadius: '16px',
            padding: '32px', maxWidth: '480px', textAlign: 'center',
          }}>
            <p style={{ fontSize: '3rem', marginBottom: '16px' }}>💥</p>
            <p style={{ fontWeight: 900, fontSize: '1.2rem', color: '#fff', marginBottom: '12px' }}>
              Something crashed
            </p>
            <pre style={{
              background: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '12px',
              fontSize: '0.75rem', color: '#fff', textAlign: 'left',
              overflow: 'auto', maxHeight: '200px',
            }}>
              {this.state.error.message}
            </pre>
            <button
              onClick={() => this.setState({ error: null })}
              style={{
                marginTop: '16px', background: '#FFD60A', border: '3px solid #000',
                boxShadow: '4px 4px 0px #000', borderRadius: '10px',
                padding: '8px 20px', fontWeight: 900, cursor: 'pointer', fontSize: '0.9rem',
              }}
            >
              Try again
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
