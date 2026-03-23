import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import IranGeopolDashboard from "./dashboard.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <IranGeopolDashboard />
  </StrictMode>
);
