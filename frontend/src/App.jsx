import { useState, useRef, useEffect } from 'react'

// ── colour tokens ─────────────────────────────────────────────────────────────
const PURPLE     = '#1a0a2e'
const PURPLE_MID = '#2d1b69'
const GOLD       = '#FFD700'
const GOLD_DIM   = '#c9a227'

// ── WebSocket URL (convert http→ws, https→wss automatically) ─────────────────
const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
const WS_URL  = BACKEND.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws'

// =============================================================================
// Landing page
// =============================================================================
function Landing({ onStart }) {
  return (
    <div
      className="d-flex flex-column align-items-center justify-content-center vh-100"
      style={{ background: `linear-gradient(135deg, ${PURPLE} 0%, ${PURPLE_MID} 100%)` }}
    >
      <div className="text-center px-4">
        <div style={{ fontSize: '4.5rem', lineHeight: 1 }}>🐟</div>

        <h1
          className="display-3 fw-bold mt-2 mb-1"
          style={{ color: GOLD, letterSpacing: '3px', textShadow: `0 0 24px ${GOLD}55` }}
        >
          Guppy Folds
        </h1>

        <p className="fs-5 mb-5" style={{ color: GOLD_DIM, letterSpacing: '1px' }}>
          ✨ AI Origami ✨
        </p>

        <button
          onClick={onStart}
          className="btn btn-lg px-5 py-3 fw-semibold"
          style={{
            background: `linear-gradient(135deg, ${GOLD} 0%, ${GOLD_DIM} 100%)`,
            color: PURPLE,
            border: 'none',
            borderRadius: '50px',
            fontSize: '1.25rem',
            boxShadow: `0 4px 24px ${GOLD}44`,
          }}
        >
          🖐 Try Origami
        </button>

        <p className="mt-4 small" style={{ color: GOLD_DIM + '88' }}>
          Camera required · Nothing is stored
        </p>
      </div>
    </div>
  )
}

// =============================================================================
// Origami view
// =============================================================================
function OrigamiView({ onBack }) {
  const videoRef   = useRef(null)
  const canvasRef  = useRef(null)
  const wsRef      = useRef(null)
  const sendingRef = useRef(false)   // throttle: only one frame in-flight at a time

  const [frame,  setFrame]  = useState(null)
  const [folds,  setFolds]  = useState(0)
  const [status, setStatus] = useState('Connecting…')
  const [error,  setError]  = useState(null)
  const [retryKey, setRetryKey] = useState(0)   // incrementing this remounts the effect

  function retry() {
    setError(null)
    setFrame(null)
    setStatus('Connecting…')
    setRetryKey(k => k + 1)
  }

  useEffect(() => {
    let ws, intervalId, stream

    async function start() {
      // ── request camera ────────────────────────────────────────────────────
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true })
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      } catch {
        setError('Camera access denied. Grant camera access in your browser\u2019s address bar, then click Try Again.')
        return
      }

      // ── open WebSocket ────────────────────────────────────────────────────
      ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen  = () => setStatus('Connected ✓')
      ws.onerror = () => setStatus('Connection error')
      ws.onclose = () => setStatus('Disconnected')

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'frame') {
          setFrame('data:image/jpeg;base64,' + msg.data)
          setFolds(msg.folds ?? 0)
          sendingRef.current = false  // ready for the next frame
        }
      }

      // ── frame sending loop (~30 fps ceiling, throttled by backend reply) ─
      intervalId = setInterval(() => {
        if (
          ws.readyState !== WebSocket.OPEN ||
          sendingRef.current ||
          !videoRef.current?.videoWidth
        ) return

        const canvas = canvasRef.current
        canvas.width  = videoRef.current.videoWidth
        canvas.height = videoRef.current.videoHeight
        canvas.getContext('2d').drawImage(videoRef.current, 0, 0)

        const b64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
        sendingRef.current = true
        ws.send(JSON.stringify({ type: 'frame', data: b64 }))
      }, 33)
    }

    start()

    return () => {
      clearInterval(intervalId)
      ws?.close()
      stream?.getTracks().forEach(t => t.stop())
    }
  }, [retryKey])

  const sendCmd = (key) => {
    if (wsRef.current?.readyState === WebSocket.OPEN)
      wsRef.current.send(JSON.stringify({ type: 'cmd', key }))
  }

  // ── error screen ──────────────────────────────────────────────────────────
  if (error) {
    return (
      <div
        className="d-flex flex-column align-items-center justify-content-center vh-100"
        style={{ background: PURPLE }}
      >
        <div style={{ fontSize: '2.5rem' }}>📷</div>
        <p className="fs-5 text-center px-4 mt-3" style={{ color: GOLD, maxWidth: 420 }}>{error}</p>
        <div className="d-flex gap-3 mt-4">
          <button
            onClick={retry}
            className="btn px-4 fw-semibold"
            style={{ background: GOLD, color: PURPLE, borderRadius: '20px', border: 'none' }}
          >
            Try Again
          </button>
          <button
            onClick={onBack}
            className="btn px-4"
            style={{ color: GOLD, border: `1px solid ${GOLD}`, borderRadius: '20px' }}
          >
            ← Back
          </button>
        </div>
        <p className="mt-4 small text-center px-4" style={{ color: GOLD_DIM }}>
          Click the camera icon 🔒 in your browser's address bar → allow camera → click Try Again
        </p>
      </div>
    )
  }

  return (
    <div
      className="min-vh-100 py-3 px-2"
      style={{ background: `linear-gradient(135deg, ${PURPLE} 0%, ${PURPLE_MID} 100%)` }}
    >
      {/* ── header ─────────────────────────────────────────────────────── */}
      <div className="d-flex align-items-center justify-content-between mb-3 px-1">
        <button
          onClick={onBack}
          className="btn btn-sm"
          style={{ color: GOLD, border: `1px solid ${GOLD}` }}
        >
          ← Back
        </button>

        <h4 className="mb-0 fw-bold" style={{ color: GOLD, letterSpacing: '2px' }}>
          Guppy Folds
        </h4>

        <span
          className="badge"
          style={{ background: PURPLE_MID, color: GOLD, border: `1px solid ${GOLD}44`, fontSize: '0.75rem' }}
        >
          {status}
        </span>
      </div>

      {/* ── video feed ─────────────────────────────────────────────────── */}
      <div className="d-flex justify-content-center mb-3">
        <div
          style={{
            borderRadius: '14px',
            overflow: 'hidden',
            border: `2px solid ${GOLD}`,
            boxShadow: `0 0 30px ${GOLD}22`,
            lineHeight: 0,
            width: '100%',
            maxWidth: 1100,
          }}
        >
          {frame ? (
            <img
              src={frame}
              alt="Origami live feed"
              style={{ display: 'block', width: '100%' }}
            />
          ) : (
            <div
              className="d-flex align-items-center justify-content-center"
              style={{ width: '100%', aspectRatio: '16/9', background: PURPLE_MID, color: GOLD }}
            >
              Loading camera…
            </div>
          )}
        </div>
      </div>

      {/* ── fold counter ───────────────────────────────────────────────── */}
      <div className="d-flex justify-content-center mb-3">
        <span
          style={{
            background: PURPLE_MID,
            border: `1px solid ${GOLD}55`,
            borderRadius: '30px',
            padding: '6px 22px',
            color: GOLD_DIM,
            fontSize: '1rem',
          }}
        >
          Folds committed:&nbsp;
          <strong style={{ color: GOLD, fontSize: '1.2rem' }}>{folds}</strong>
        </span>
      </div>

      {/* ── control buttons ────────────────────────────────────────────── */}
      <div className="d-flex justify-content-center gap-2 mb-4">
        <button
          onClick={() => sendCmd('c')}
          className="btn px-4 fw-semibold"
          style={{ background: GOLD, color: PURPLE, borderRadius: '20px', border: 'none' }}
        >
          🔄 Unfold
        </button>
        <button
          onClick={() => sendCmd('r')}
          className="btn px-4"
          style={{ border: `1px solid ${GOLD}`, color: GOLD, borderRadius: '20px' }}
        >
          ↩ Reset
        </button>
      </div>

      {/* ── instructions card ──────────────────────────────────────────── */}
      <div className="row justify-content-center mx-0">
        <div className="col-12 col-sm-10 col-md-7 col-lg-5">
          <div
            className="p-3"
            style={{
              background: PURPLE_MID,
              border: `1px solid ${GOLD}44`,
              borderRadius: '16px',
            }}
          >
            <h6 className="fw-bold mb-3" style={{ color: GOLD }}>✦ How to fold</h6>

            <div
              className="d-flex flex-column gap-2"
              style={{ color: '#f0e6c8', fontSize: '0.9rem' }}
            >
              <div>👌&nbsp; <strong>Pinch</strong> thumb + index finger on the paper</div>
              <div>✋&nbsp; <strong>Drag</strong> toward the centre to fold</div>
              <div>🖐&nbsp; <strong>Release</strong> to lock the fold (drag &gt;30% to commit)</div>
              <div>🔄&nbsp; Click <strong>Unfold</strong> or press&nbsp;
                <kbd style={{ background: PURPLE, color: GOLD, border: `1px solid ${GOLD}55` }}>C</kbd>
                &nbsp;to undo all folds
              </div>
            </div>

            <div className="text-center mt-3 pt-2" style={{ borderTop: `1px solid ${GOLD}22` }}>
              <span style={{ color: GOLD_DIM }}>
                Folds committed:&nbsp;
                <strong style={{ color: GOLD, fontSize: '1.1rem' }}>{folds}</strong>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* hidden capture elements */}
      <video ref={videoRef} style={{ display: 'none' }} autoPlay playsInline muted />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  )
}

// =============================================================================
// Root
// =============================================================================
export default function App() {
  const [page, setPage] = useState('landing')
  return page === 'landing'
    ? <Landing onStart={() => setPage('origami')} />
    : <OrigamiView onBack={() => setPage('landing')} />
}
