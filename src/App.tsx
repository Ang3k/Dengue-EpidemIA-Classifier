import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Triage from "./pages/Triage";
import Graphics from "./pages/Graphics";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/triagem" element={<Triage />} />
        <Route path="/graphics" element={<Graphics />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;