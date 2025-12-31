import * as tf from "@tensorflow/tfjs";

export function preprocessImage(img) {
  return tf.tidy(() => {
    return tf.browser
      .fromPixels(img)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(127.5)
      .sub(1)
      .expandDims(0);
  });
}

export function preprocess(img) {
  return tf.tidy(() =>
    tf.browser
      .fromPixels(img)
      .resizeBilinear([224, 224])
      .toFloat()
      .div(127.5)
      .sub(1)
      .expandDims(0)
  );
}
