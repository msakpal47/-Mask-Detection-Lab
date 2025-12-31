import React from "react";
import ModeSwitch from "./components/ModeSwitch";

export default function App() {
  return (
    <div style={{ background: "#0f172a", minHeight: "100vh", color: "#fff" }}>
      <h2 style={{ padding: 16 }}>Mask Detection System</h2>
      <ModeSwitch />
    </div>
  );
}
