import React, { useEffect, useRef, useState } from 'react'
import { autoPredict } from '../utils/autoPredict'
import { drawBoxes } from '../utils'

export default function Camera() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isRunning, setIsRunning] = useState(false)
  const [status, setStatus] = useState('')
  const [result, setResult] = useState('')
  const intervalRef = useRef(null)

  const startCamera = async () => {
    try {
      setStatus('Initializing...')
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      videoRef.current.srcObject = stream
      
      videoRef.current.onloadedmetadata = () => {
        setIsRunning(true)
        setStatus('Running')
        const v = videoRef.current
        const c = canvasRef.current
        c.width = v.clientWidth || 320
        c.height = v.clientHeight || 240
        intervalRef.current = setInterval(captureAndPredict, 1000)
      }
    } catch (err) {
      console.error(err)
      setStatus('Camera access denied or error.')
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setIsRunning(false)
    setStatus('Stopped')
    setResult('')
  }

  const captureAndPredict = async () => {
    if (!videoRef.current) return
    try {
      const v = videoRef.current
      const c = canvasRef.current
      const boxes = await autoPredict(v)
      drawBoxes(c, boxes)
      setStatus('Classification Result')
      setResult(`Faces: ${boxes.length}`)
    } catch (e) {
      drawBoxes(canvasRef.current, [])
      setStatus('Classification Result')
      setResult('Uncertain prediction')
    }
  }

  useEffect(() => {
    return () => stopCamera()
  }, [])

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative rounded-lg overflow-hidden bg-black shadow-lg" style={{ minWidth: 320, minHeight: 240 }}>
        {!isRunning && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                Camera Off
            </div>
        )}
        <video ref={videoRef} autoPlay playsInline className="block max-w-full" muted />
        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
      </div>

      <div className="flex gap-4">
        {!isRunning ? (
          <button onClick={startCamera} className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-full font-semibold transition-colors">
            Start Camera
          </button>
        ) : (
          <button onClick={stopCamera} className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-full font-semibold transition-colors">
            Stop Camera
          </button>
        )}
      </div>
      
      {status && (
        <div className="text-center">
          <span className={`px-3 py-1 rounded-full text-sm font-bold ${
            status === 'Classification Result' ? 'bg-green-900 text-green-300' :
            status === 'Error' ? 'bg-red-900 text-red-300' : 'text-slate-400 bg-slate-700'
          }`}>{status}</span>
          {result && <div className="mt-2 text-lg font-medium">{result}</div>}
        </div>
      )}
    </div>
  )
}
