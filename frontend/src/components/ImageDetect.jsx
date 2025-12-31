import { useRef } from "react";
import { autoPredict } from "../utils/autoPredict";
import { detectFaces } from "../utils/faceDetector";
import { health, predictImageFile } from "../utils/backend";
import { drawBoxes } from "../utils/drawBox";

export default function ImageDetect({ maskModel, onBack }) {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const inputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = imgRef.current;
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      const canvas = canvasRef.current;
      canvas.width = img.width;
      canvas.height = img.height;
      try {
        const h = await health();
        if (h.ok && h.model_loaded) {
          const res = await predictImageFile(file);
          const rw = img.naturalWidth || img.width;
          const rh = img.naturalHeight || img.height;
          const sx = img.width / rw;
          const sy = img.height / rh;
          const dets = (res.boxes || []).map(b => ({
            box: { x: Math.round(b.x * sx), y: Math.round(b.y * sy), width: Math.round(b.w * sx), height: Math.round(b.h * sy) },
            label: b.label,
            score: b.score
          }));
          drawBoxes(canvas, dets);
        } else {
          const faces = await detectFaces(img);
          await autoPredict({ image: img, canvas, faceDetections: faces, maskModel });
        }
      } catch {
        const faces = await detectFaces(img);
        await autoPredict({ image: img, canvas, faceDetections: faces, maskModel });
      }
    };
  };

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
        <button
          onClick={() => {
            if (imgRef.current) imgRef.current.src = "";
            const c = canvasRef.current;
            if (c) {
              c.width = 0; 
              c.height = 0;
            }
            onBack && onBack();
          }}
          style={{
            background: "#334155",
            color: "#e2e8f0",
            padding: "8px 16px",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Back
        </button>
        <button
          onClick={() => inputRef.current.click()}
          style={{
            background: "#5b5cff",
            color: "#fff",
            padding: "10px 20px",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Choose Image
        </button>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      <img
        ref={imgRef}
        alt=""
        style={{ width: "100%", borderRadius: "10px" }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
