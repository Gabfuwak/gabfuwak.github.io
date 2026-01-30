import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Nav from './components/Nav';
import Home from './pages/Home';
import Projects from './pages/Projects';
import Experiments from './pages/Experiments';
import WebGPUTriangle from './pages/experiments/WebGPUTriangle';
import Assignment from './pages/ImSynthesis/Assignment';
import NotFound from './pages/NotFound';

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/experiments/webgpu-triangle" element={<WebGPUTriangle />} />
        <Route path="/secret/assignment" element={<Assignment />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}
