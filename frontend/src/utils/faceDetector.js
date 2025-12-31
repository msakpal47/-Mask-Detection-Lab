import * as blazeface from "@tensorflow-models/blazeface"

let faceModel = null

export async function loadFaceModel() {
  if (!faceModel) {
    faceModel = await blazeface.load()
  }
  return faceModel
}

export async function estimateFaces(element) {
  const fm = await loadFaceModel()
  return fm.estimateFaces(element, false)
}

export async function detectFaces(element) {
  const preds = await estimateFaces(element)
  return preds.map(p => {
    const [x1, y1] = p.topLeft
    const [x2, y2] = p.bottomRight
    const w = Math.max(1, Math.floor(x2 - x1))
    const h = Math.max(1, Math.floor(y2 - y1))
    const x = Math.max(0, Math.floor(x1))
    const y = Math.max(0, Math.floor(y1))
    return { box: { x, y, width: w, height: h } }
  })
}
