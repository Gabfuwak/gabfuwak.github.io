export default function Projects() {
  return (
    <div>
      <h1>Projects</h1>

      <h2>I'm currently working on:</h2>
      <ul>
        <li>Expressive bipedal locomotion through RL</li>
        <li>Stress detection in VR using physiological markers (HRV, EDA)</li>
        <li>Technical consulting for a short 3D animation film at Hongik University (tooling, shaders, tech-art guidance)</li>
      </ul>

      <h2>In the past, I worked on:</h2>
      <ul>
        <li>
          <a href="https://github.com/Gabfuwak/difference-voxelization" target="_blank" rel="noopener noreferrer">
            A multi-view vision system based on space voxelization to detect subpixel moving objects
          </a>
          <br />
          <em>My (messy) introduction to WebGPU for the simulator, with a vision algorithm detecting sub-pixel motion in real time.</em>
        </li>
        <li>
          <a href="https://hackaday.io/project/185729-upkie-wheeled-biped-robots/log/245395-articulated-head-and-marker-tracking" target="_blank" rel="noopener noreferrer">
            A head attachment for Upkie, an open-source wheel-legged robot
          </a>
          <br />
          <em>My primer in robotics at INRIA Paris, supervised by <a href="https://scaron.info/" target="_blank" rel="noopener noreferrer">Stéphane Caron</a>. This is where I learned the basics of CAD, 3D printing, soldering, electronics.</em>
        </li>
        <li>
          <a href="https://github.com/Gabfuwak/minirust" target="_blank" rel="noopener noreferrer">
            A borrowchecker and RISC-V compiler backend for minirust, a toy language based on Rust
          </a>
          <br />
          <em>The reason I know assembly beyond MIPS!</em>
        </li>
      </ul>
    </div>
  );
}
