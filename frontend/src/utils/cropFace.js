export function cropFace(img, face) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  let [x1, y1] = face.topLeft;
  let [x2, y2] = face.bottomRight;

  // 🔒 Clamp values inside image
  x1 = Math.max(0, Math.floor(x1));
  y1 = Math.max(0, Math.floor(y1));
  x2 = Math.min(img.width, Math.floor(x2));
  y2 = Math.min(img.height, Math.floor(y2));

  const width = x2 - x1;
  const height = y2 - y1;

  if (width <= 0 || height <= 0) {
    throw new Error("Invalid face crop");
  }

  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(
    img,
    x1, y1, width, height,
    0, 0, width, height
  );

  return { canvas, x: x1, y: y1, w: width, h: height };
}
