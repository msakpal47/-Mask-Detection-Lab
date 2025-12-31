export function drawBox(ctx, x, y, w, h, label, conf) {
  const color =
    label === "With Mask"
      ? "green"
      : label === "Without Mask"
      ? "red"
      : "orange";
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.fillText(`${label} ${conf}%`, x, y - 5);
}

export function drawBoxes(canvas, detections) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  console.log("🟩 Drawing boxes", detections);
  detections.forEach((det) => {
    const { box, label, score } = det;
    const color =
      label === "With Mask"
        ? "green"
        : label === "Without Mask"
        ? "red"
        : label === "Improper Mask"
        ? "orange"
        : "yellow";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = color;
    ctx.font = "16px Arial";
    ctx.fillText(`${label} ${(score * 100).toFixed(1)}%`, box.x, box.y - 8);
  });
}
