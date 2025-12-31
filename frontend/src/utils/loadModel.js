import * as tf from "@tensorflow/tfjs";

let model = null;

export async function loadModel() {
  if (model) return model;

  try {
    model = await tf.loadGraphModel(window.location.origin + "/model/model.json");
    console.log("✅ Mask graph model loaded");
    return model;
  } catch (err) {
    console.error("❌ Failed to load model:", err);
    throw err;
  }
}
