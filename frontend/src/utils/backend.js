const API =
  (typeof import.meta !== "undefined" &&
    import.meta.env &&
    import.meta.env.VITE_API_URL) ||
  (typeof window !== "undefined" && window.location
    ? window.location.origin
    : "http://localhost:8000")

export async function health() {
  const res = await fetch(API + "/health").catch(() => null)
  if (!res || !res.ok) return { ok: false, model_loaded: false }
  const data = await res.json().catch(() => ({}))
  return { ok: true, model_loaded: !!data.model_loaded }
}

export async function predictImageFile(file) {
  const fd = new FormData()
  fd.append("file", file)
  const res = await fetch(API + "/predict-image", { method: "POST", body: fd })
  if (!res.ok) throw new Error("predict failed")
  return res.json()
}
