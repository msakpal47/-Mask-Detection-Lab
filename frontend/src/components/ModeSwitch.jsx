import React, { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { loadMaskModel } from "../utils/loadMaskModel";
import { loadFaceModel, estimateFaces } from "../utils/faceDetector";
import ImageDetect from "./ImageDetect";
import CameraDetect from "./CameraDetect";

let faceModelPromise = null;
let maskModelPromise = null;

export default function ModeSwitch() {
  const [ready, setReady] = useState(false);
  const [maskModel, setMaskModel] = useState(null);
  const [mode, setMode] = useState(null); // null | 'image' | 'camera'

  React.useEffect(() => {
    (async () => {
      faceModelPromise = loadFaceModel();
      maskModelPromise = loadMaskModel();
      const [_, m] = await Promise.all([faceModelPromise, maskModelPromise]);
      setMaskModel(m);
      setReady(true);
      console.log("Models loaded once ✅");
    })();
  }, []);

  return (
    <div className="flex flex-col items-center w-full">
      {!ready && <p className="text-slate-400">Loading models...</p>}
      {ready && (
        <div className="flex items-center gap-3 mb-4">
          <button
            onClick={() => setMode("image")}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              mode === "image"
                ? "bg-indigo-600 text-white"
                : "bg-slate-700 text-slate-200 hover:bg-slate-600"
            }`}
          >
            Image Upload
          </button>
          <button
            onClick={() => setMode("camera")}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              mode === "camera"
                ? "bg-indigo-600 text-white"
                : "bg-slate-700 text-slate-200 hover:bg-slate-600"
            }`}
          >
            Live Camera
          </button>
        </div>
      )}
      {ready && maskModel && mode === "image" && (
        <ImageDetect maskModel={maskModel} onBack={() => setMode(null)} />
      )}
      {ready && mode === "camera" && (
        <CameraDetect onBack={() => setMode(null)} />
      )}
    </div>
  );
}
