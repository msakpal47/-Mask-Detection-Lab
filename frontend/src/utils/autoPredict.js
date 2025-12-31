import { predictMask } from "./predict"
import { drawBoxes } from "./drawBox"

export async function autoPredict({ image, canvas, faceDetections, maskModel }) {
  const detections = []
  if (!faceDetections || faceDetections.length === 0) {
    drawBoxes(canvas, detections)
    return
  }
  for (const face of faceDetections) {
    if (!face || !face.box) continue
    const { x, y, width, height } = face.box
    const faceCanvas = document.createElement("canvas")
    faceCanvas.width = width
    faceCanvas.height = height
    faceCanvas.getContext("2d").drawImage(image, x, y, width, height, 0, 0, width, height)
    const result = await predictMask(faceCanvas, maskModel)
    detections.push({ box: { x, y, width, height }, label: result.label, score: result.score })
  }
  drawBoxes(canvas, detections)
}
