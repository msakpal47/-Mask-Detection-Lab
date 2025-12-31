import React, { useEffect, useRef, useState } from 'react'
import { Upload as UploadIcon } from 'lucide-react'
import { autoPredict } from '../utils/autoPredict'
import { drawBoxes } from '../utils'

export default function Upload({ initialImage }) {
  const [imageSrc, setImageSrc] = useState(initialImage || null)
  const [status, setStatus] = useState('')
  const [result, setResult] = useState('')
  const [scores, setScores] = useState(null)
  const imgRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    if (initialImage) {
        handleUrlPrediction(initialImage)
    }
  }, [initialImage])

  const handleUrlPrediction = async (url) => {
      setImageSrc(url)
      setStatus('Loading...')
      try {
          const res = await fetch(url)
          const blob = await res.blob()
          processBlob(blob)
      } catch (e) {
          setStatus('Error loading image')
      }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setImageSrc(URL.createObjectURL(file))
      processBlob(file)
    }
  }

  const processBlob = (blob) => {
      setStatus('Processing...')
      setResult('')
      setTimeout(() => {
          if (!imgRef.current || imgRef.current.naturalWidth === 0) return
          ;(async () => {
            try {
              const img = imgRef.current
              const canvas = canvasRef.current
              canvas.width = img.clientWidth
              canvas.height = img.clientHeight
              const boxes = await autoPredict(img)
              drawBoxes(canvas, boxes)
              setStatus('Classification Result')
              setResult(`Faces: ${boxes.length}`)
              setScores(null)
            } catch (e) {
              drawBoxes(canvasRef.current, [])
              setStatus('Error')
              setResult('Failed to process')
              setScores(null)
            }
          })()
      }, 100)
  }

  // Review save removed for auto prediction only

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="w-full flex justify-center">
        <label className="cursor-pointer bg-slate-700 hover:bg-slate-600 transition-colors px-6 py-8 rounded-xl border-2 border-dashed border-slate-500 flex flex-col items-center gap-2">
          <UploadIcon size={32} className="text-blue-400" />
          <span className="text-lg font-medium">Click to Upload Photo</span>
          <span className="text-sm text-slate-400">JPG, PNG supported</span>
          <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
        </label>
      </div>

      {imageSrc && (
        <div className="relative rounded-lg overflow-hidden border border-slate-700 shadow-lg max-w-full">
          <img ref={imgRef} id="previewImage" src={imageSrc} alt="Upload" className="max-w-full max-h-[500px] block" />
          <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" />
        </div>
      )}

      {status && (
        <div className="text-center">
            <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                status === 'Classification Result' ? 'bg-green-900 text-green-300' : 
                status === 'Error' ? 'bg-red-900 text-red-300' : 'bg-blue-900 text-blue-300'
            }`}>
                {status}
            </span>
            {result && <div className="mt-2 text-lg font-medium">{result}</div>}
            {scores && (
              <div className="mt-1 text-xs text-slate-400">
                Scores — Proper: {Math.round((scores.mask ?? 0)*100)}%{scores.improper!==undefined ? ` | Improper: ${Math.round((scores.improper ?? 0)*100)}%` : ''} | Without: {Math.round((scores.noMask ?? 0)*100)}%
              </div>
            )}
            
        </div>
      )}
    </div>
  )
}
