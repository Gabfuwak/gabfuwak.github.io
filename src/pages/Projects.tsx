export default function Projects() {
  return (
    <div>
      <h1>Projects</h1>

      <h2>I'm currently working on:</h2>
      <ul>
        <li><strong>Bipedal locomotion through RL </strong>
          <br />
          <em>Non-humanoid bipeds (think legs without a torso) in MuJoCo with the goal of making the gait expressive.</em>
        </li>
        <li><strong>Stress detection in VR using physiological markers (HRV, EDA)</strong></li>
        <li><strong>Technical consulting for a 3D animation short film for students at Hongik University</strong>
          <br />
          <em>Tooling (set up a P4 server), shader programming, tech art guidance</em>
        </li>
        <li><strong>Implementing the <a href="https://cs.dartmouth.edu/~wjarosz/publications/bitterli20spatiotemporal.html">ReSTIR</a> paper for direct illumination in real time</strong>
          <br />
          <em>University project in my Image Synthesis class, it's supposed to handle millions of light sources!</em>
        </li>

        <li><strong><a href='https://github.com/Gabfuwak/gabfuwak.github.io'>This website!!</a></strong>
          <br />
          <em>With WebGPU more impressive than my web design skills! If you get lost, you might find a secret...</em> </li>
      </ul>

      <h2>In the past, I worked on:</h2>
          <em>A selection of past projects with public documentation and/or that I'm proud of. More to come as I find time to clean up code and write some documentation!</em>
      <ul>
        <li>
          <a href="https://github.com/Gabfuwak/difference-voxelization" target="_blank" rel="noopener noreferrer">
            <strong>A multi-view vision system based on space voxelization to detect subpixel moving objects</strong>
          </a>
          <br />
          <em>My (messy) introduction to WebGPU for the simulator, with a vision algorithm detecting sub-pixel motion in real time.</em>
        </li>
        <li>
          <a href="https://hackaday.io/project/185729-upkie-wheeled-biped-robots/log/245395-articulated-head-and-marker-tracking" target="_blank" rel="noopener noreferrer">
            <strong>A head attachment for Upkie, an open-source wheel-legged robot</strong>
          </a>
          <br />
          <em>A 45 hours project over the last semester of my bachelor's degree. It is my primer in robotics at INRIA Paris, supervised by <a href="https://scaron.info/" target="_blank" rel="noopener noreferrer">Stéphane Caron</a>. This is where I learned the basics of CAD, 3D printing, soldering, electronics.</em>
        </li>
        <li>
          <a href="https://github.com/Gabfuwak/minirust" target="_blank" rel="noopener noreferrer">
            <strong>A borrowchecker and RISC-V compiler backend for minirust, a toy language based on Rust</strong>
          </a>
          <br />
          <em>The reason I know assembly beyond MIPS!</em>
        </li>
        <li><strong>Building RX1, an open-source humanoid robot designed by <a href="https://red-rabbit-robotics.ghost.io/">Red Rabbit Robotics </a></strong>
          <br />
          <em>This project is on hold due to lack of time and funding. More news this summer, it should go up a category...</em>
        </li>
      </ul>
    </div>
  );
}
