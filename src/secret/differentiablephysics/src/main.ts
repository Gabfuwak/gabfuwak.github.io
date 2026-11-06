import "./style.css";
import { initCamera, moveForward, pan, rotateYaw, rotatePitch } from "./camera";
import {
  initWebGPU, createPipeline, createComputePipeline, createUniformBuffer,
  createAccumulationBuffer, createSphereBuffer, createTriangleBuffer, createComputeBindGroup,
  createDisplayBindGroup, packUniforms, packSpheres,
  renderFrame,
} from "./renderer";
import { materials, lights, triangles } from "./scene";
import { createParticles, createDistanceConstraints, updateParticles } from "./physics";
import { initDisplayControls } from "./ui";

async function main() {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.height = Math.floor(window.innerHeight / 8) * 8;
  const renderOptions = initDisplayControls();
  const { device, context, format } = await initWebGPU(canvas);

  const camera = initCamera(canvas);

  const [commonCode, raytracerCode, displayCode] = await Promise.all([
    fetch("shaders/common.wgsl").then(r => r.text()),
    fetch("shaders/raytracer.wgsl").then(r => r.text()),
    fetch("shaders/display.wgsl").then(r => r.text()),
  ]);
  const displayPipeline    = createPipeline(device, format, displayCode);
  const raytracerPipeline  = createComputePipeline(device, format, commonCode + raytracerCode);

  const uniformBuffer = createUniformBuffer(device);
  const accumulationBuffer = createAccumulationBuffer(device, canvas.width, canvas.height);
  const particles = createParticles();
  const constraints = createDistanceConstraints(particles);


  const particlesBuffer = createSphereBuffer(device, particles);
  const trianglesBuffer = createTriangleBuffer(device, triangles);
  const displayBindGroup = createDisplayBindGroup(device, displayPipeline, accumulationBuffer, uniformBuffer);
  const raytracerBindGroup = createComputeBindGroup(device, raytracerPipeline, uniformBuffer, particlesBuffer, trianglesBuffer, accumulationBuffer);

  function uploadAndRender() {
    device.queue.writeBuffer(uniformBuffer, 0, packUniforms(camera, lights, materials, canvas.width, canvas.height, renderOptions));
    device.queue.writeBuffer(particlesBuffer, 0, packSpheres(particles));
    renderFrame(device, context, displayPipeline, raytracerPipeline, displayBindGroup, raytracerBindGroup, canvas.width, canvas.height);
  }

  uploadAndRender();

  // --- Keyboard controls ---
  const keys = new Set<string>();
  const MOVE_SPEED = 0.2;
  const ROT_SPEED  = 1.2;

  window.addEventListener('keydown', e => {
    keys.add(e.code);
    if (e.code.startsWith('Arrow')) e.preventDefault();
  });
  window.addEventListener('keyup', e => keys.delete(e.code));

  let lastTime = performance.now();
  let accumulator = 0;

  function tick(now: number) {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    const sp = MOVE_SPEED * dt;

    if (keys.has('KeyW')) { console.log("W");moveForward(camera,  sp); }
    if (keys.has('KeyS')) { moveForward(camera, -sp); }
    if (keys.has('KeyA')) { pan(camera, -sp, 0); }
    if (keys.has('KeyD')) { pan(camera,  sp, 0); }
    if (keys.has('KeyR')) { pan(camera, 0,  sp); }
    if (keys.has('KeyF')) { pan(camera, 0, -sp); }
    if (keys.has('ArrowLeft'))  { rotateYaw(camera,   ROT_SPEED * dt); }
    if (keys.has('ArrowRight')) { rotateYaw(camera,  -ROT_SPEED * dt); }
    if (keys.has('ArrowUp'))    { rotatePitch(camera,  ROT_SPEED * dt); }
    if (keys.has('ArrowDown'))  { rotatePitch(camera, -ROT_SPEED * dt); }

    const physicsDt = 1 / 120;

    accumulator += dt;

    while (accumulator >= physicsDt) {
      updateParticles(particles, constraints, physicsDt);
      accumulator -= physicsDt;
    }

    uploadAndRender();

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

main().catch(e => console.error(e));
