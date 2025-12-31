import * as tf from "@tensorflow/tfjs";

let modelPromise = null;

export async function loadMaskModel() {
  if (!modelPromise) {
    console.log("Loading mask model...");
    // Changed to loadGraphModel as the model.json format is "graph-model"
    modelPromise = tf.loadGraphModel(
      window.location.origin + "/model/model.json"
    );
    modelPromise
      .then(() => {
        console.log("Mask model loaded ✅");
      })
      .catch((err) => {
        console.error("Mask model load failed ❌", err);
      });
  }
  return modelPromise;
}
