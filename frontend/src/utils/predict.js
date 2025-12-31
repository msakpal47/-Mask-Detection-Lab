import * as tf from "@tensorflow/tfjs"

const LABELS = ["With Mask", "Without Mask"]

export async function predictMask(faceCanvas, model) {
  const textureScore = await computeLowerTexture(faceCanvas)
  const tensor = tf.browser
    .fromPixels(faceCanvas)
    .resizeBilinear([224, 224])
    .toFloat()
    .sub(127.5)
    .div(127.5)
    .expandDims(0)

  const pred = model.predict(tensor)
  const scores = await pred.data()
  tf.dispose([tensor, pred])

  const arr = Array.from(scores)
  const maxIndex = arr.indexOf(Math.max(...arr))
  const confidence = arr[maxIndex]
  let label = LABELS[maxIndex]
  if (label === "With Mask" && textureScore > 0.12) {
    label = "Without Mask"
  }
  return { label, score: confidence }
}

async function computeLowerTexture(faceCanvas) {
  const img = tf.browser.fromPixels(faceCanvas).toFloat().div(255)
  const shape = img.shape
  const h = shape[0]
  const w = shape[1]
  const half = Math.floor(h / 2)
  const lower = img.slice([half, 0, 0], [h - half, w, 3])
  const gray = lower.mul(tf.tensor([0.299, 0.587, 0.114])).sum(2)
  const moments = tf.moments(gray)
  const std = moments.variance.sqrt()
  const v = await std.data()
  tf.dispose([img, lower, gray, moments.mean, moments.variance, std])
  return v[0]
}
