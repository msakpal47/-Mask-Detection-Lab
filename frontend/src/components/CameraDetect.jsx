import React, { useEffect, useRef, useState } from "react";
import { Video } from "lucide-react";
import { loadMaskModel } from "../utils/loadMaskModel";
import { autoPredict } from "../utils/autoPredict";
import { loadFaceModel, detectFaces } from "../utils/faceDetector";

export default function CameraDetect({ onBack }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [running, setRunning] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => stop();
  }, []);

  const stop = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    const stream = videoRef.current?.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setRunning(false);
  };

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    videoRef.current.srcObject = stream;
    videoRef.current.onloadedmetadata = async () => {
      setRunning(true);
      await loadFaceModel();
      let maskModel;
      try {
        maskModel = await loadMaskModel();
      } catch (err) {
        console.error("MODEL LOAD FAILED", err);
        alert("Model not loaded. Check model.json");
        return;
      }
      const v = videoRef.current;
      const c = canvasRef.current;
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      intervalRef.current = setInterval(async () => {
        const ctx = c.getContext("2d");
        const faces = await detectFaces(v);
        ctx.clearRect(0, 0, c.width, c.height);
        await autoPredict({ image: v, canvas: c, faceDetections: faces, maskModel });
      }, 800);
    };
  };

  return (
    <div className="flex flex-col items-center w-full">
      <div className="w-full max-w-3xl flex justify-between mb-3">
        <button 
          onClick={() => { stop(); onBack && onBack(); }}
          className="px-4 py-2 bg-slate-700 text-slate-200 rounded-lg hover:bg-slate-600 transition-all"
        >
          Back
        </button>
        {!running && (
          <button 
            onClick={start}
            className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-all duration-200 shadow-md font-medium group"
          >
            <Video className="w-5 h-5 group-hover:scale-110 transition-transform" />
            <span>Start Camera</span>
          </button>
        )}
      </div>
      
      <div className="relative inline-block max-w-full rounded-lg overflow-hidden shadow-2xl border border-slate-700 bg-black">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          className="block max-w-full h-auto"
          style={{ maxHeight: '600px' }}
        />
        <canvas 
          ref={canvasRef} 
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>
    </div>
  );
}
