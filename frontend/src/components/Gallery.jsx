import React, { useEffect, useState } from 'react'

export default function Gallery({ onSelect }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    const key = 'reviews'
    const reviews = JSON.parse(localStorage.getItem(key) || '[]')
    setItems(reviews.reverse())
    setLoading(false)
  }, [])

  if (loading) return <div className="text-center p-10">Loading gallery...</div>

  if (items.length === 0) return <div className="text-center p-10 text-slate-400">No review images saved yet.</div>

  const filtered = items.filter(i => filter === 'all' || i.label === filter)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2 items-center">
        <button onClick={() => setFilter('all')} className={`px-3 py-1 rounded ${filter==='all'?'bg-blue-600 text-white':'bg-slate-700 text-slate-300'}`}>All</button>
        <button onClick={() => setFilter('Proper Mask')} className={`px-3 py-1 rounded ${filter==='Proper Mask'?'bg-green-600 text-white':'bg-slate-700 text-slate-300'}`}>Proper Mask</button>
        <button onClick={() => setFilter('Improper Mask')} className={`px-3 py-1 rounded ${filter==='Improper Mask'?'bg-yellow-600 text-white':'bg-slate-700 text-slate-300'}`}>Improper Mask</button>
        <button onClick={() => setFilter('Without Mask')} className={`px-3 py-1 rounded ${filter==='Without Mask'?'bg-red-600 text-white':'bg-slate-700 text-slate-300'}`}>Without Mask</button>
        <div className="flex-1" />
        <button onClick={() => {localStorage.removeItem('reviews'); setItems([])}} className="px-3 py-1 rounded bg-slate-700 text-slate-200 hover:bg-slate-600">Clear All</button>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filtered.map((it, idx) => (
          <div key={idx} className="rounded-lg overflow-hidden border border-slate-700 bg-slate-800">
            <img src={it.image} alt={it.label} className="w-full h-40 object-cover" />
            <div className="p-2 text-sm text-slate-300 flex items-center justify-between">
              <span>{it.label}</span>
              <button onClick={() => onSelect?.(it.image)} className="px-2 py-1 rounded bg-blue-600 text-white">Open</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
