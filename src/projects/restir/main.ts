// === main.ts ===

import commonCode    from './shaders/common.wgsl?raw';
import raytracerCode from './shaders/raytracer.wgsl?raw';
import pathtracerCode from './shaders/pathtracer.wgsl?raw';
import displayCode   from './shaders/display.wgsl?raw';

import { initCamera, pan, moveForward, rotateYaw, rotatePitch } from './camera.ts';
import type { Camera } from './camera.ts';
import { buildBVH, buildTLAS } from './bvh.ts';
import type { AABB } from './bvh.ts';
import {
  buildScene, extractSceneData, buildEmissiveTriangleList,
  SCENE_CAMERAS, _mat4InverseTRS,
} from './scene.ts';
import type { Scene, SceneData, EmissiveList } from './scene.ts';
import {
  initWebGPU, createTimestampResources, createAccumulationBuffer,
  createUniformBuffer, packUniforms, createPipeline, createComputePipeline,
  createSceneBuffers, destroySceneBuffers, createComputeBindGroup,
  createDisplayBindGroup, renderFrame,
} from './renderer.ts';
import type { SceneBuffers, TimestampResources } from './renderer.ts';
import { createBrdfViz, createCdfViz, createReservoirViz } from './viz.ts';

// Encode a BVH { data, nodeCount } into a GPU-ready ArrayBuffer (32 bytes/node).
function encodeBVH(bvh: { data: number[]; nodeCount: number }): ArrayBuffer {
  const buf = new ArrayBuffer(bvh.nodeCount * 32);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  for (let i = 0; i < bvh.nodeCount; i++) {
    const s = i * 8;
    f32[s]   = bvh.data[s];   f32[s+1] = bvh.data[s+1]; f32[s+2] = bvh.data[s+2];
    u32[s+3] = bvh.data[s+3];
    f32[s+4] = bvh.data[s+4]; f32[s+5] = bvh.data[s+5]; f32[s+6] = bvh.data[s+6];
    u32[s+7] = bvh.data[s+7];
  }
  return buf;
}

function bvhMaxDepthOf(bvhData: number[]): number {
  let max = 0;
  const stack: { idx: number; depth: number }[] = [{idx: 0, depth: 0}];
  while (stack.length > 0) {
    const {idx, depth} = stack.pop()!;
    if (depth > max) max = depth;
    const b = idx * 8;
    if (bvhData[b+3] === 0) {
      stack.push({idx: idx + 1,            depth: depth + 1});
      stack.push({idx: bvhData[b+7],       depth: depth + 1});
    }
  }
  return max;
}

// Logarithmic slider: maps [0, steps] -> [min, max] on a log scale
function logSliderToValue(sliderVal: number, steps: number, min: number, max: number): number {
  const t = sliderVal / steps;
  return Math.round(min * Math.pow(max / min, t));
}
function valueToLogSlider(value: number, steps: number, min: number, max: number): number {
  return Math.round(steps * Math.log(value / min) / Math.log(max / min));
}

// --- Viz initialization ---
const dragonMat = { roughness: 0.3, metalness: 0.0, fresnel: [0.04, 0.04, 0.04] as [number,number,number], baseColor: [1.0, 0.71, 0.29] as [number,number,number] };

createBrdfViz(document.getElementById('viz-raytracer')!, {
  ...dragonMat,
  incomingAngle: 55,
  lightAngle: 30,
  areaLight: true,
});

createBrdfViz(document.getElementById('viz-naive')!, {
  ...dragonMat,
  incomingAngle: 55,
  lightAngle: 30,
  showLightRay: false,
  swatchLightAngle: 0,
  showSamples: true,
  sampling: 'uniform',
  numSamples: 9,
  areaLight: true,
});

createBrdfViz(document.getElementById('viz-nee-direct')!, {
  ...dragonMat,
  incomingAngle: 55,
  lightAngle: 30,
  showSwatch: false,
  showLightRay: false,
  showNeeRays: true,
  areaLight: true,
  areaLightWidth: 140,
  draggableLight: false,
  width: 340,
});

createBrdfViz(document.getElementById('viz-nee-indirect')!, {
  ...dragonMat,
  incomingAngle: 55,
  lightAngle: 30,
  showSwatch: false,
  showLightRay: false,
  showSamples: true,
  sampling: 'uniform',
  numSamples: 9,
  crossOutAreaHits: true,
  areaLight: true,
  areaLightWidth: 140,
  draggableLight: false,
  width: 340,
});

createBrdfViz(document.getElementById('viz-brdfis')!, {
  ...dragonMat,
  metalness: 0.1,
  incomingAngle: 55,
  lightAngle: 30,
  showLobe: true,
  showSwatch: false,
  showLightRay: false,
  showSamples: true,
  sampling: 'brdf',
  numSamples: 9,
  areaLight: true,
  showSlider: true,
});

createBrdfViz(document.getElementById('viz-mis-naive')!, {
  ...dragonMat,
  metalness: 1.0,
  baseColor: [0.8, 0.8, 0.8],
  fresnel: [0.95, 0.95, 0.95],
  roughness: 0.15,
  incomingAngle: 55,
  lightAngle: 30,
  showLobe: true,
  showSwatch: false,
  showLightRay: false,
  showSamples: true,
  sampling: 'brdf',
  numSamples: 9,
  crossOutAreaHits: true,
  showNeeRays: true,
  areaLight: true,
  areaLightWidth: 140,
  draggableLight: false,
  showSlider: true,
});

createBrdfViz(document.getElementById('viz-mis-balanced')!, {
  ...dragonMat,
  metalness: 0.0,
  baseColor: [0.8, 0.8, 0.8],
  fresnel: [0.04, 0.04, 0.04],
  roughness: 0.1,
  incomingAngle: 55,
  lightAngle: 30,
  showLobe: true,
  showSwatch: false,
  showLightRay: false,
  showSamples: true,
  sampling: 'brdf',
  numSamples: 9,
  showNeeRays: true,
  areaLight: true,
  areaLightWidth: 140,
  draggableLight: false,
  showSlider: true,
  misWeights: true,
  misWeightMode: 'length',
  neeRayCount: 1,
});

createCdfViz(document.getElementById('viz-nee-cdf')!, {
  entries: [
    { flux: 0.20, type: 'emissive' },
    { flux: 0.20, type: 'emissive' },
    { flux: 0.38, type: 'point' },
  ],
});

createReservoirViz(document.getElementById('viz-ris-reservoir')!);

// --- Main renderer ---

async function main(): Promise<void> {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.height = Math.floor(window.innerHeight / 8) * 8;
  const { device, context, format, hasTimestamps } = await initWebGPU(canvas);

  const camera: Camera = initCamera(canvas);

  const displayPipeline    = createPipeline(device, format, displayCode);
  const raytracerPipeline  = createComputePipeline(device, commonCode + raytracerCode);
  const pathtracerPipeline = createComputePipeline(device, commonCode + pathtracerCode);

  const uniformBuffer = createUniformBuffer(device);
  const accumulationBuffer = createAccumulationBuffer(device, canvas.width, canvas.height);
  const reservoirBuffer = createAccumulationBuffer(device, canvas.width, canvas.height);
  const accumClearData = new Uint8Array(canvas.width * canvas.height * 16);
  const displayBindGroup = createDisplayBindGroup(device, displayPipeline, accumulationBuffer, uniformBuffer);

  let frame_count = 0;
  function resetAccumulation(): void {
    frame_count = 0;
    device.queue.writeBuffer(accumulationBuffer, 0, accumClearData);
    device.queue.writeBuffer(reservoirBuffer, 0, accumClearData);
  }

  // --- Mutable scene state (swapped on scene change) ---
  let scene: Scene | null = null;
  let sceneData: SceneData | null = null;
  let emissiveList: EmissiveList = { buffer: new ArrayBuffer(0), count: 0, totalPower: 0 };
  let sceneBuffers: SceneBuffers | null = null;
  let raytracerBindGroup: GPUBindGroup | null = null;
  let pathtracerBindGroup: GPUBindGroup | null = null;
  let bvhMaxDepth = 0;
  let loading = false;
  let currentSceneName: string | null = null;

  const loadingOverlay = document.getElementById("loading-overlay") as HTMLElement;
  function showLoading(msg: string): Promise<void> {
    loadingOverlay.textContent = msg;
    loadingOverlay.style.display = "";
    // Yield so the browser can paint
    return new Promise(r => setTimeout(r, 0));
  }
  function hideLoading(): void { loadingOverlay.style.display = "none"; }

  // Scene-specific control panels
  const sceneCornellDiv = document.getElementById("scene-cornell") as HTMLElement;
  const sceneSponzaDiv  = document.getElementById("scene-sponza") as HTMLElement;
  const dragonResSelect  = document.getElementById("dragon-res") as HTMLSelectElement;
  const lightCountSlider = document.getElementById("light-count") as HTMLInputElement;
  const lightCountVal    = document.getElementById("light-count-val") as HTMLElement;
  const lightCountApply  = document.getElementById("light-count-apply") as HTMLButtonElement;
  let sponzaLightCount = 100;

  dragonResSelect.addEventListener("change", async () => {
    await loadScene("cornell");
    uploadAndRender();
  });

  const lightEmissionSlider = document.getElementById("light-emission") as HTMLInputElement;
  const lightEmissionVal    = document.getElementById("light-emission-val") as HTMLElement;

  lightCountSlider.addEventListener("input", () => {
    lightCountVal.textContent = lightCountSlider.value;
  });
  lightCountApply.addEventListener("click", async () => {
    sponzaLightCount = parseInt(lightCountSlider.value);
    await loadScene("sponza");
    uploadAndRender();
  });
  lightEmissionSlider.addEventListener("input", () => {
    const em = parseFloat(lightEmissionSlider.value);
    lightEmissionVal.textContent = em.toFixed(1);
    if (sceneData && currentSceneName === "sponza") {
      // Material id 1 = light in sponza scene
      sceneData.materials[1].emission = em;
      resetAccumulation();
      uploadAndRender();
    }
  });

  function updateSceneControls(sceneName: string): void {
    sceneCornellDiv.style.display = sceneName === "cornell" ? "" : "none";
    sceneSponzaDiv.style.display  = sceneName === "sponza"  ? "" : "none";
  }

  async function loadScene(sceneName: string): Promise<void> {
    loading = true;
    currentSceneName = sceneName;
    updateSceneControls(sceneName);
    console.log(`Loading scene: ${sceneName}...`);
    const t0 = performance.now();

    // Set camera for this scene
    const cam = SCENE_CAMERAS[sceneName] || SCENE_CAMERAS.cornell;
    camera.position = [...cam.position];
    camera.target   = [...cam.target];

    const sceneOpts: Record<string, unknown> = sceneName === "sponza"
      ? { lightCount: sponzaLightCount }
      : { dragonRes: dragonResSelect.value };
    await showLoading(`Loading mesh...`);
    scene = await buildScene(sceneName, sceneOpts);
    sceneData = extractSceneData(scene);

    // Build one BLAS per unique mesh (not per instance)
    const numMeshes = sceneData.meshes.length;
    const numInstances = sceneData.instances.length;
    await showLoading(`Building BLAS (${numMeshes} unique mesh${numMeshes > 1 ? "es" : ""}, ${numInstances} instances)...`);
    const blases = sceneData.meshes.map(mesh =>
      buildBVH({ positions: mesh.positions, indices: mesh.indices })
    );

    // Per-mesh offsets into combined BLAS buffer and unified geometry
    const meshBlasOffsets:   number[] = []; // node offset in combined BLAS buffer
    const meshIndexOffsets:  number[] = []; // triangle offset in unified index buffer
    const meshVertexOffsets: number[] = [];
    let totalBlasNodes = 0, totalVertices = 0, totalTriangles = 0;
    for (let mi = 0; mi < numMeshes; mi++) {
      meshBlasOffsets.push(totalBlasNodes);
      meshIndexOffsets.push(totalTriangles);
      meshVertexOffsets.push(totalVertices);
      totalBlasNodes  += blases[mi].nodeCount;
      totalVertices   += sceneData.meshes[mi].positions.length / 3;
      totalTriangles  += sceneData.meshes[mi].indices.length / 3;
    }

    // Build unified geometry buffers (one copy per unique mesh)
    const allPositions = new Float32Array(totalVertices * 3);
    const allNormals   = new Float32Array(totalVertices * 3);
    const allUvs       = new Float32Array(totalVertices * 2);
    const allIndices   = new Uint32Array(totalTriangles * 3);
    for (let mi = 0; mi < numMeshes; mi++) {
      const mesh = sceneData.meshes[mi];
      const blas = blases[mi];
      const vBase = meshVertexOffsets[mi];
      const iOff  = meshIndexOffsets[mi];
      allPositions.set(mesh.positions, vBase * 3);
      allNormals.set(mesh.normals,     vBase * 3);
      allUvs.set(mesh.uvs,             vBase * 2);
      for (let i = 0; i < blas.primitives.length; i++) {
        const src = blas.primitives[i] * 3;
        allIndices[(iOff + i) * 3]     = mesh.indices[src]     + vBase;
        allIndices[(iOff + i) * 3 + 1] = mesh.indices[src + 1] + vBase;
        allIndices[(iOff + i) * 3 + 2] = mesh.indices[src + 2] + vBase;
      }
    }

    // Emissive triangle list
    emissiveList = buildEmissiveTriangleList(sceneData, allIndices, allPositions, meshIndexOffsets);

    // Instance AABBs -> TLAS (transform each mesh's BLAS root AABB)
    const instanceAABBs: AABB[] = sceneData.instances.map(inst => {
      const d = blases[inst.meshIndex].data;
      const [x0,y0,z0, x1,y1,z1] = [d[0],d[1],d[2], d[4],d[5],d[6]];
      const corners: [number,number,number][] = [[x0,y0,z0],[x1,y0,z0],[x0,y1,z0],[x1,y1,z0],
                       [x0,y0,z1],[x1,y0,z1],[x0,y1,z1],[x1,y1,z1]];
      const m = inst.transform;
      const min = new Float32Array([1e30,1e30,1e30]), max = new Float32Array([-1e30,-1e30,-1e30]);
      for (const [x,y,z] of corners) {
        const wx = m[0]*x + m[4]*y + m[8]*z  + m[12];
        const wy = m[1]*x + m[5]*y + m[9]*z  + m[13];
        const wz = m[2]*x + m[6]*y + m[10]*z + m[14];
        if (wx < min[0]) min[0]=wx; if (wx > max[0]) max[0]=wx;
        if (wy < min[1]) min[1]=wy; if (wy > max[1]) max[1]=wy;
        if (wz < min[2]) min[2]=wz; if (wz > max[2]) max[2]=wz;
      }
      return { minCorner: min, maxCorner: max };
    });

    await showLoading(`Building TLAS (${numInstances} instances)...`);
    const tlas = buildTLAS(instanceAABBs);

    // BVH depth for slider
    const tlasMaxDepth = bvhMaxDepthOf(tlas.data);
    const blasMaxDepthVal = Math.max(...blases.map(b => bvhMaxDepthOf(b.data)));
    bvhMaxDepth = tlasMaxDepth + blasMaxDepthVal + 1;
    bvhSlider.max = String(bvhMaxDepth);

    console.log(`BVH build: ${(performance.now()-t0).toFixed(1)}ms | TLAS: ${tlas.nodeCount} nodes (depth ${tlasMaxDepth}) | BLASes: ${totalBlasNodes} nodes total (max depth ${blasMaxDepthVal}) | ${numMeshes} unique meshes, ${numInstances} instances`);

    // Encode TLAS + BLASes
    const tlasBytes = encodeBVH(tlas);
    const blasBytes = new ArrayBuffer(totalBlasNodes * 32);
    for (let mi = 0; mi < blases.length; mi++) {
      new Uint8Array(blasBytes, meshBlasOffsets[mi] * 32).set(new Uint8Array(encodeBVH(blases[mi])));
    }

    // Encode instance table — each instance references its mesh's shared BLAS/index offsets
    const INSTANCE_STRIDE = 36;
    const instanceData = new Float32Array(numInstances * INSTANCE_STRIDE);
    const instanceDataU32 = new Uint32Array(instanceData.buffer);
    for (let ii = 0; ii < numInstances; ii++) {
      const inst = sceneData.instances[ii];
      const invT = _mat4InverseTRS(inst.transform);
      const base = ii * INSTANCE_STRIDE;
      for (let j = 0; j < 16; j++) instanceData[base + j]      = inst.transform[j];
      for (let j = 0; j < 16; j++) instanceData[base + 16 + j] = invT[j];
      instanceDataU32[base + 32] = meshBlasOffsets[inst.meshIndex];
      instanceDataU32[base + 33] = meshIndexOffsets[inst.meshIndex];
      instanceDataU32[base + 34] = inst.materialId;
      instanceDataU32[base + 35] = 0;
    }

    await showLoading(`Uploading to GPU...`);
    // Destroy old GPU buffers, create new ones
    destroySceneBuffers(sceneBuffers);
    const geo = { positions: allPositions, normals: allNormals, uvs: allUvs, indices: allIndices };
    sceneBuffers = createSceneBuffers(device, geo, tlasBytes, blasBytes, instanceData.buffer, emissiveList.buffer);

    // Recreate bind groups (they reference scene buffers)
    raytracerBindGroup  = createComputeBindGroup(device, raytracerPipeline,  uniformBuffer, sceneBuffers, accumulationBuffer, reservoirBuffer);
    pathtracerBindGroup = createComputeBindGroup(device, pathtracerPipeline, uniformBuffer, sceneBuffers, accumulationBuffer, reservoirBuffer);

    resetAccumulation();
    loading = false;
    hideLoading();
    console.log(`Scene "${sceneName}" ready (${(performance.now()-t0).toFixed(0)}ms)`);
  }

  // --- Stats ---
  const stats = document.getElementById("stats") as HTMLElement;
  const cpuTimes: number[] = [];
  const gpuTimes: number[] = [];
  const MAX_SAMPLES = 1000;

  const ts: TimestampResources | null = createTimestampResources(device, hasTimestamps);
  if (ts) {
    ts.onGpuTime = (ms: number) => {
      gpuTimes.push(ms);
      if (gpuTimes.length > MAX_SAMPLES) gpuTimes.shift();
      updateStats();
    };
  }

  let lastCpuMs = 0;

  function updateStats(): void {
    const cpuAvg = cpuTimes.reduce((a, b) => a + b, 0) / (cpuTimes.length || 1);
    let text = `CPU  last: ${lastCpuMs.toFixed(2)}ms  avg(${cpuTimes.length}): ${cpuAvg.toFixed(2)}ms`;
    if (ts) {
      const gpuAvg = gpuTimes.reduce((a, b) => a + b, 0) / (gpuTimes.length || 1);
      const gpuLast = gpuTimes.at(-1) ?? 0;
      text += `\nGPU  last: ${gpuLast.toFixed(2)}ms  avg(${gpuTimes.length}): ${gpuAvg.toFixed(2)}ms`;
    }
    stats.textContent = text;
  }

  // --- Controls ---
  const POINT_LIGHT = { position: [200, 400, 200] as [number,number,number], color: [1,1,1] as [number,number,number], intensity: 0.03, direction: [0,-1,0] as [number,number,number], angle: 360 };

  const controlsToggle = document.getElementById("controls-toggle") as HTMLElement;
  const controlsBody = document.getElementById("controls-body") as HTMLElement;
  controlsToggle.addEventListener("click", () => {
    const open = controlsBody.style.display !== "none";
    controlsBody.style.display = open ? "none" : "";
    controlsToggle.textContent = open ? "CONTROLS ▸" : "CONTROLS ▾";
  });

  let usePointLight = true;
  const pointLightCb = document.getElementById("point-light") as HTMLInputElement;
  pointLightCb.addEventListener("change", () => {
    usePointLight = pointLightCb.checked;
    resetAccumulation();
    uploadAndRender();
  });

  // ptMode bitflags: bit0 = NEE, bit1 = BRDF IS, bit2 = MIS, bit3 = RIS, bit4 = ReSTIR
  let usePathTracer = false;
  let ptMode = 0;
  const modeCb = document.getElementById("render-mode") as HTMLInputElement;
  modeCb.addEventListener("change", () => {
    usePathTracer = modeCb.checked;
    resetAccumulation();
    uploadAndRender();
  });

  const neeCb = document.getElementById("nee-mode") as HTMLInputElement;
  const brdfIsCb = document.getElementById("brdfis-mode") as HTMLInputElement;
  neeCb.addEventListener("change", () => {
    ptMode = neeCb.checked ? (ptMode | 1) : (ptMode & ~1);
    resetAccumulation();
    uploadAndRender();
  });
  brdfIsCb.addEventListener("change", () => {
    ptMode = brdfIsCb.checked ? (ptMode | 2) : (ptMode & ~2);
    resetAccumulation();
    uploadAndRender();
  });
  const misCb = document.getElementById("mis-mode") as HTMLInputElement;
  misCb.addEventListener("change", () => {
    ptMode = misCb.checked ? (ptMode | 4) : (ptMode & ~4);
    resetAccumulation();
    uploadAndRender();
  });
  const risCb = document.getElementById("ris-mode") as HTMLInputElement;
  const risSamplesSlider = document.getElementById("ris-samples") as HTMLInputElement;
  const risSamplesVal = document.getElementById("ris-samples-val") as HTMLElement;
  const RIS_MIN = 1, RIS_MAX = 256, RIS_STEPS = 1000;
  let risSamples = 32;
  let risTarget = 2;
  const risTargetSelect = document.getElementById("ris-target") as HTMLSelectElement;
  risSamplesSlider.value = String(valueToLogSlider(risSamples, RIS_STEPS, RIS_MIN, RIS_MAX));
  risCb.addEventListener("change", () => {
    ptMode = risCb.checked ? (ptMode | 8) : (ptMode & ~8);
    resetAccumulation();
    uploadAndRender();
  });
  const restirCb = document.getElementById("restir-mode") as HTMLInputElement;
  restirCb.addEventListener("change", () => {
    ptMode = restirCb.checked ? (ptMode | 16) : (ptMode & ~16);
    resetAccumulation();
    uploadAndRender();
  });
  risSamplesSlider.addEventListener("input", () => {
    risSamples = logSliderToValue(parseInt(risSamplesSlider.value), RIS_STEPS, RIS_MIN, RIS_MAX);
    risSamplesVal.textContent = String(risSamples);
    resetAccumulation();
    uploadAndRender();
  });
  risTargetSelect.addEventListener("change", () => {
    risTarget = parseInt(risTargetSelect.value);
    resetAccumulation();
    uploadAndRender();
  });

  // Scene selector
  const sceneSelect = document.getElementById("scene-select") as HTMLSelectElement;
  sceneSelect.addEventListener("change", async () => {
    await loadScene(sceneSelect.value);
    uploadAndRender();
  });

  function uploadAndRender(): void {
    if (loading || !sceneData || !scene) return;
    const t = performance.now();
    const computePipeline   = usePathTracer ? pathtracerPipeline  : raytracerPipeline;
    const computeBindGroup  = usePathTracer ? pathtracerBindGroup : raytracerBindGroup;
    if (!computeBindGroup) return;
    const lights = (usePointLight && currentSceneName === "cornell") ? [...scene.lights, POINT_LIGHT] : scene.lights;
    const useRis = (ptMode & 8) !== 0;
    const effectiveRisSamples = useRis ? risSamples : 1;
    device.queue.writeBuffer(uniformBuffer, 0, packUniforms(camera, lights, sceneData.materials, frame_count, bvhVisDepth, bvhHeatMax, bvhEarlyStop, canvas.width, canvas.height, ptMode, emissiveList.count, effectiveRisSamples, risTarget));
    renderFrame(device, context, displayPipeline, computePipeline, displayBindGroup, computeBindGroup, canvas.width, canvas.height, ts);
    frame_count++;
    lastCpuMs = performance.now() - t;

    cpuTimes.push(lastCpuMs);
    if (cpuTimes.length > MAX_SAMPLES) cpuTimes.shift();
    updateStats();
  }

  // SPP (logarithmic: 1–4096)
  const SPP_MIN = 1, SPP_MAX = 4096, SPP_STEPS = 1000;
  let spp = 32;
  const sppSlider = document.getElementById("spp-slider") as HTMLInputElement;
  const sppLabel  = document.getElementById("spp-val") as HTMLElement;
  sppSlider.value = String(valueToLogSlider(spp, SPP_STEPS, SPP_MIN, SPP_MAX));
  sppSlider.addEventListener("input", () => {
    const newSpp = logSliderToValue(parseInt(sppSlider.value), SPP_STEPS, SPP_MIN, SPP_MAX);
    if (newSpp < spp) resetAccumulation();
    spp = newSpp;
    sppLabel.textContent = String(spp);
    if (frame_count < spp) uploadAndRender();
  });

  // BVH visualization controls
  let bvhVisDepth  = -1;
  let bvhHeatMax   = 4;
  let bvhEarlyStop = 1;
  const bvhSlider  = document.getElementById("bvh-depth") as HTMLInputElement;
  const bvhLabel   = document.getElementById("bvh-depth-val") as HTMLElement;
  const heatSlider = document.getElementById("bvh-heat") as HTMLInputElement;
  const heatLabel  = document.getElementById("bvh-heat-val") as HTMLElement;
  bvhSlider.addEventListener("input", () => {
    bvhVisDepth = parseInt(bvhSlider.value);
    bvhLabel.textContent = bvhVisDepth < 0 ? "off" : String(bvhVisDepth);
    resetAccumulation();
    uploadAndRender();
  });
  heatSlider.addEventListener("input", () => {
    bvhHeatMax = parseInt(heatSlider.value);
    heatLabel.textContent = String(bvhHeatMax);
    resetAccumulation();
    uploadAndRender();
  });
  const earlyStopCb = document.getElementById("bvh-early-stop") as HTMLInputElement;
  earlyStopCb.addEventListener("change", () => {
    bvhEarlyStop = earlyStopCb.checked ? 1 : 0;
    resetAccumulation();
    uploadAndRender();
  });

  // --- Initial scene load ---
  await loadScene(sceneSelect.value);
  uploadAndRender();

  // --- Keyboard controls ---
  const keys = new Set<string>();
  const MOVE_SPEED = 200;
  const ROT_SPEED  = 1.2;

  window.addEventListener('keydown', e => {
    keys.add(e.code);
    if (e.code.startsWith('Arrow')) e.preventDefault();
  });
  window.addEventListener('keyup', e => keys.delete(e.code));

  let lastTime = performance.now();

  function tick(now: number): void {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;

    const sp = MOVE_SPEED * dt;
    let moved = false;

    if (keys.has('KeyW')) { moveForward(camera,  sp);            moved = true; }
    if (keys.has('KeyS')) { moveForward(camera, -sp);            moved = true; }
    if (keys.has('KeyA')) { pan(camera, -sp, 0);                 moved = true; }
    if (keys.has('KeyD')) { pan(camera,  sp, 0);                 moved = true; }
    if (keys.has('KeyR')) { pan(camera, 0,  sp);                 moved = true; }
    if (keys.has('KeyF')) { pan(camera, 0, -sp);                 moved = true; }
    if (keys.has('ArrowLeft'))  { rotateYaw(camera,   ROT_SPEED * dt); moved = true; }
    if (keys.has('ArrowRight')) { rotateYaw(camera,  -ROT_SPEED * dt); moved = true; }
    if (keys.has('ArrowUp'))    { rotatePitch(camera,  ROT_SPEED * dt); moved = true; }
    if (keys.has('ArrowDown'))  { rotatePitch(camera, -ROT_SPEED * dt); moved = true; }

    if (moved) resetAccumulation();
    if (frame_count < spp) uploadAndRender();

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

main().catch(e => console.error(e));
