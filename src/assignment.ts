import { load_mesh, create_quad, get_mat } from './utils/mesh_gen';
import { initWebGPU, initCamera, getCameraBasis, extractSceneData, getMVP, pan, moveForward, rotateYaw, rotatePitch } from './utils/webgpu';
import { type Scene, type Light, type Material } from './utils/scene';
import { type AABB, createAABBWireframe, buildSceneBVH, type BVHNode, flattenBVH } from './utils/cpuBVH';
import shaderCode from './shaders/assignment.wgsl?raw';
import noiseShaderCode from './shaders/noise.wgsl?raw';
import aabbShaderCode from './shaders/aabb.wgsl?raw';

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

const MODELS = {
  'Dragon (2k tri)': 'dragon_2348.obj',
  'Dragon (117k tri)': 'dragon_117452.obj',
} as const;

type ModelName = keyof typeof MODELS;

// ---------------------------------------------------------------------------
// Scene definition
// ---------------------------------------------------------------------------

function buildScene(canvas: HTMLCanvasElement, objContent: string): Scene {
  const camera = initCamera(canvas,
    [278, 273, -800],
    [278, 273, -799],
    [0, 1, 0],
    3.1415 / 3.0,
    0.1,
    2000,
  );

  const boxCenter = [278, 274, 280];

  const normalize = (x: number, y: number, z: number): Float32Array => {
    const len = Math.sqrt(x * x + y * y + z * z);
    return new Float32Array([x / len, y / len, z / len]);
  };

  const lights: Light[] = [
    {
      position: new Float32Array([71, 412, 140]),
      color: new Float32Array([1.0, 1.0, 1.0]),
      intensity: 0.8,
      direction: normalize(boxCenter[0] - 71, boxCenter[1] - 412, boxCenter[2] - 140),
      angle: 90.0,
    },
    {
      position: new Float32Array([485, 137, 140]),
      color: new Float32Array([0.0, 0.2, 0.8]),
      intensity: 0.6,
      direction: normalize(boxCenter[0] - 485, boxCenter[1] - 137, boxCenter[2] - 140),
      angle: 360.0,
    },
    {
      position: new Float32Array([71, 137, 70]),
      color: new Float32Array([0.7, 0.3, 0.0]),
      intensity: 0.2,
      direction: normalize(boxCenter[0] - 71, boxCenter[1] - 137, boxCenter[2] - 70),
      angle: 360.0,
    },
    {
      position: new Float32Array([554, 494, 280]),
      color: new Float32Array([1.0, 1.0, 1.0]),
      intensity: 0.2,
      direction: normalize(boxCenter[0] - 554, boxCenter[1] - 494, boxCenter[2] - 280),
      angle: 360.0,
    },
  ];

  const whiteMaterial: Material = {
    id: 0,
    diffuseAlbedo: new Float32Array([1.0, 1.0, 1.0]),
    roughness: 0, metalness: 0,
    fresnel: new Float32Array([0.05, 0.05, 0.05]),
    emission: 0.0,
    basePerlinFreq: 0, perlinOctaveAmp: 0, perlinOctaveNb: 0,
  };
  const redMaterial: Material = {
    id: 1,
    diffuseAlbedo: new Float32Array([0.65, 0.05, 0.05]),
    roughness: 0, metalness: 0,
    fresnel: new Float32Array([0.05, 0.05, 0.05]),
    emission: 0.0,
    basePerlinFreq: 0, perlinOctaveAmp: 0, perlinOctaveNb: 0,
  };
  const greenMaterial: Material = {
    id: 2,
    diffuseAlbedo: new Float32Array([0.12, 0.45, 0.15]),
    roughness: 0, metalness: 0,
    fresnel: new Float32Array([0.05, 0.05, 0.05]),
    emission: 0.0,
    basePerlinFreq: 0, perlinOctaveAmp: 0, perlinOctaveNb: 0,
  };
  const noisyDragonMaterial: Material = {
    id: 3,
    diffuseAlbedo: new Float32Array([0.8, 0.0, 0.0]),
    roughness: 0.5, metalness: 1.0,
    fresnel: new Float32Array([1.0, 0.71, 0.29]),
    emission: 0.0,
    basePerlinFreq: 10, perlinOctaveAmp: 0.9, perlinOctaveNb: 5,
  };
  const lightMaterial: Material = {
    id: 4,
    diffuseAlbedo: new Float32Array([1.0, 1.0, 1.0]),
    roughness: 0.0, metalness: 0.0,
    fresnel: new Float32Array([0.05, 0.05, 0.05]),
    emission: 3.0,
    basePerlinFreq: 0, perlinOctaveAmp: 0, perlinOctaveNb: 0,
  };

  const identityTransform = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);

  const objects = [
    { mesh: create_quad([552.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 559.2], [549.6, 0.0, 559.2], [1.0, 1.0, 1.0]), material: whiteMaterial, transform: identityTransform, label: "Floor" },
    { mesh: create_quad([556.0, 548.8, 0.0], [556.0, 548.8, 559.2], [0.0, 548.8, 559.2], [0.0, 548.8, 0.0], [1.0, 1.0, 1.0]), material: whiteMaterial, transform: identityTransform, label: "Ceiling" },
    { mesh: create_quad([343.0, 548.0, 227.0], [343.0, 548.0, 332.0], [213.0, 548.0, 332.0], [213.0, 548.0, 227.0], [1.0, 1.0, 1.0]), material: lightMaterial, transform: identityTransform, label: "Light" },
    { mesh: create_quad([549.6, 0.0, 559.2], [0.0, 0.0, 559.2], [0.0, 548.8, 559.2], [556.0, 548.8, 559.2], [1.0, 1.0, 1.0]), material: whiteMaterial, transform: identityTransform, label: "Back wall" },
    { mesh: create_quad([0.0, 0.0, 559.2], [0.0, 0.0, 0.0], [0.0, 548.8, 0.0], [0.0, 548.8, 559.2], [0.12, 0.45, 0.15]), material: greenMaterial, transform: identityTransform, label: "Right wall" },
    { mesh: create_quad([552.8, 0.0, 0.0], [549.6, 0.0, 559.2], [556.0, 548.8, 559.2], [556.0, 548.8, 0.0], [0.65, 0.05, 0.05]), material: redMaterial, transform: identityTransform, label: "Left wall" },
    { mesh: create_quad([556.0, 548.8, 0.0], [0.0, 548.8, 0.0], [0.0, 0.0, 0.0], [552.8, 0.0, 0.0], [1.0, 1.0, 1.0]), material: whiteMaterial, transform: identityTransform, label: "Front wall" },
    {
      mesh: load_mesh(objContent, [0.8, 0.2, 0.2]),
      material: noisyDragonMaterial,
      transform: get_mat({ translation: [279, 115, 269], rotation: [0, Math.PI / 4, 0], scale: 2 }),
      label: "Dragon",
    },
  ];

  return { objects, lights, camera };
}

// ---------------------------------------------------------------------------
// GPU engine
// ---------------------------------------------------------------------------

interface Engine {
  scene: Scene;
  render(): void;
  startAnimation(): void;
  stopAnimation(): void;
  setUseRaytracer(val: boolean): void;
  setSpp(val: number): void;
  resetAccum(): void;
  updateMaterial(idx: number, rgb: [number, number, number], roughness?: number, metalness?: number, basePerlinFreq?: number, perlinOctaveAmp?: number, perlinOctaveNb?: number): void;
  bvhRoot: BVHNode;
  setDebugAABBs(aabbs: AABB[]): void;
  pickObject(sx: number, sy: number): Promise<number>;
  startFPSMonitor(onResult: (fps: number) => void): void;
  stopFPSMonitor(): void;
  destroy(): void;
}

async function createEngine(canvas: HTMLCanvasElement, scene: Scene): Promise<Engine> {
  // @ts-ignore
  const { device, context, _adapter } = await initWebGPU(canvas);

  const merged = extractSceneData(scene);
  const { positions: vertexPositions, indices: indexData, uvs: vertexUVs, objectIds, normals: vertexNormals, materials } = merged;

  const vertexBuffer = device.createBuffer({ label: "Vertices", size: vertexPositions.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  const indexBuffer = device.createBuffer({ label: "Vertex indices", size: indexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  const objectIdBuffer = device.createBuffer({ label: "Object IDs", size: objectIds.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  const normalBuffer = device.createBuffer({ label: "Vertex normals", size: vertexNormals.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  const uvBuffer = device.createBuffer({ label: "Vertex UVs", size: vertexUVs.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });

  const BVH_FLOATS_PER_NODE = 8;

  const t0 = performance.now();
  const bvhRoot = buildSceneBVH(scene);
  console.log(`BVH build: ${(performance.now() - t0).toFixed(1)}ms`);
  const { nodes: flat, primitives } = flattenBVH(bvhRoot);

  const reorderedIndex = new Uint32Array(indexData.length);
  for (let i = 0; i < primitives.length; i++) {
    const src = primitives[i] * 3;
    reorderedIndex[i * 3]     = indexData[src];
    reorderedIndex[i * 3 + 1] = indexData[src + 1];
    reorderedIndex[i * 3 + 2] = indexData[src + 2];
  }

  device.queue.writeBuffer(vertexBuffer, 0, vertexPositions.buffer, vertexPositions.byteOffset, vertexPositions.byteLength);
  device.queue.writeBuffer(indexBuffer, 0, reorderedIndex.buffer, reorderedIndex.byteOffset, reorderedIndex.byteLength);
  device.queue.writeBuffer(objectIdBuffer, 0, objectIds.buffer, objectIds.byteOffset, objectIds.byteLength);
  device.queue.writeBuffer(normalBuffer, 0, vertexNormals.buffer, vertexNormals.byteOffset, vertexNormals.byteLength);
  device.queue.writeBuffer(uvBuffer, 0, vertexUVs.buffer, vertexUVs.byteOffset, vertexUVs.byteLength);

  const bvhBytes = new ArrayBuffer(flat.length * BVH_FLOATS_PER_NODE * 4);
  {
    const view = new DataView(bvhBytes);
    for (let i = 0; i < flat.length; i++) {
      const n = flat[i];
      const b = i * BVH_FLOATS_PER_NODE * 4;
      view.setFloat32(b +  0, n.minCorner[0], true);
      view.setFloat32(b +  4, n.minCorner[1], true);
      view.setFloat32(b +  8, n.minCorner[2], true);
      if (!n.isLeaf) {
        view.setUint32(b + 12, 0, true);
      } else {
        view.setUint32(b + 12, n.nbTris, true);
      }
      view.setFloat32(b + 16, n.maxCorner[0], true);
      view.setFloat32(b + 20, n.maxCorner[1], true);
      view.setFloat32(b + 24, n.maxCorner[2], true);
      if (n.isLeaf) {
        view.setUint32(b + 28, n.triangleIndex >= 0 ? n.triangleIndex : 0, true);
      } else {
        view.setUint32(b + 28, n.rightChild >= 0 ? n.rightChild : 0, true);
      }
    }
  }

  const bvhBuffer = device.createBuffer({ label: "BVH nodes", size: bvhBytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bvhBuffer, 0, bvhBytes);

  const MAX_LIGHTS = 4;
  const MAX_MATERIALS = 16;
  const MATERIAL_SIZE = 12;
  const MVP_SIZE = 16;
  const SHARED_HEADER = 4;
  const LIGHTS_SIZE = MAX_LIGHTS * 12;
  const MATERIALS_HEADER = 4;
  const RAY_CAMERA_SIZE = 12;
  const RAY_OFFSET = MVP_SIZE + SHARED_HEADER + LIGHTS_SIZE + MATERIALS_HEADER + MAX_MATERIALS * MATERIAL_SIZE;
  const UNIFORM_LENGTH = RAY_OFFSET + RAY_CAMERA_SIZE;

  let spp = 1;

  const packLightsAndMaterials = (out: Float32Array, frame_count = 0) => {
    out.set(scene.camera.position, 16);
    out[19] = scene.lights.length;
    for (let i = 0; i < scene.lights.length; i++) {
      out.set(scene.lights[i].position, 20 + i * 12);
      out[23 + i * 12] = scene.lights[i].intensity;
      out.set(scene.lights[i].color, 24 + i * 12);
      out.set(scene.lights[i].direction, 28 + i * 12);
      out[31 + i * 12] = scene.lights[i].angle * Math.PI / 180;
    }
    const matOffset = 20 + MAX_LIGHTS * 12;
    out[matOffset] = materials.length;
    out[matOffset + 1] = spp;
    out[matOffset + 2] = frame_count;
    out[matOffset + 3] = canvas.width;
    for (let i = 0; i < materials.length; i++) {
      const baseIdx = matOffset + 4 + i * MATERIAL_SIZE;
      out.set(materials[i].diffuseAlbedo, baseIdx);
      out[baseIdx + 3] = materials[i].roughness ?? 0;
      out.set(materials[i].fresnel, baseIdx + 4);
      out[baseIdx + 7] = materials[i].metalness ?? 0;
      out[baseIdx + 8] = materials[i].emission;
      out[baseIdx + 9] = materials[i].basePerlinFreq;
      out[baseIdx + 10] = materials[i].perlinOctaveAmp;
      out[baseIdx + 11] = materials[i].perlinOctaveNb;
    }
  };

  const packUniforms = (time = 0, frame_count = 0): Float32Array<ArrayBuffer> => {
    const data = new Float32Array(UNIFORM_LENGTH);
    data.set(getMVP(scene.camera), 0);
    packLightsAndMaterials(data, frame_count);
    const basis = getCameraBasis(scene.camera);
    const fovFactor = Math.tan(scene.camera.fov / 2);
    data.set([...basis.forward, fovFactor], RAY_OFFSET);
    data.set([...basis.right, scene.camera.aspect], RAY_OFFSET + 4);
    data.set([...basis.up, time], RAY_OFFSET + 8);
    return data;
  };

  const uniformBuffer = device.createBuffer({ size: UNIFORM_LENGTH * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniformBuffer, 0, packUniforms());

  const accumBuffer = device.createBuffer({ size: canvas.width * canvas.height * 4 * 4, usage: GPUBufferUsage.STORAGE });

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const shaderModule = device.createShaderModule({ label: "Shader", code: noiseShaderCode.concat(shaderCode) });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "storage" } },
    ] as GPUBindGroupLayoutEntry[],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: vertexBuffer } },
      { binding: 2, resource: { buffer: indexBuffer } },
      { binding: 3, resource: { buffer: objectIdBuffer } },
      { binding: 4, resource: { buffer: normalBuffer } },
      { binding: 5, resource: { buffer: uvBuffer } },
      { binding: 6, resource: { buffer: bvhBuffer } },
      { binding: 7, resource: { buffer: accumBuffer } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const pickCoordsBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pickResultBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readbackBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const pickPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'pick_main' },
  });

  const pickBindGroup = device.createBindGroup({
    layout: pickPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: vertexBuffer } },
      { binding: 2, resource: { buffer: indexBuffer } },
      { binding: 3, resource: { buffer: objectIdBuffer } },
      { binding: 6, resource: { buffer: bvhBuffer } },
      { binding: 8, resource: { buffer: pickCoordsBuffer } },
      { binding: 9, resource: { buffer: pickResultBuffer } },
    ],
  });

  const rastPipeline = device.createRenderPipeline({
    label: "Rasterizer pipeline",
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "rastVertexMain" },
    fragment: { module: shaderModule, entryPoint: "rastFragmentMain", targets: [{ format: canvasFormat }] },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });

  const rayPipeline = device.createRenderPipeline({
    label: "Raytraced pipeline",
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: "rayVertexMain" },
    fragment: { module: shaderModule, entryPoint: "rayFragmentMain", targets: [{ format: canvasFormat }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
  });

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const MAX_AABB = 1200000;
  const aabbVertexBuffer = device.createBuffer({ label: "AABB vertices", size: MAX_AABB * 8 * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const aabbIndexBuffer = device.createBuffer({ label: "AABB indices", size: MAX_AABB * 24 * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });

  const aabbShaderModule = device.createShaderModule({ label: "AABB shader", code: aabbShaderCode });

  const aabbBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ] as GPUBindGroupLayoutEntry[],
  });

  const aabbBindGroup = device.createBindGroup({
    layout: aabbBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: aabbVertexBuffer } },
    ],
  });

  const aabbPipeline = device.createRenderPipeline({
    label: "AABB pipeline",
    layout: device.createPipelineLayout({ bindGroupLayouts: [aabbBindGroupLayout] }),
    vertex: { module: aabbShaderModule, entryPoint: "aabbVertexMain" },
    fragment: { module: aabbShaderModule, entryPoint: "aabbFragmentMain", targets: [{ format: canvasFormat }] },
    primitive: { topology: "line-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
  });

  let destroyed = false;
  let useRaytracer = false;
  let animating = false;
  let animFrameId = 0;
  let aabbIndexCount = 0;
  let frameCount = 1;
  const startTime = performance.now();

  const updateUniforms = (time = 0) => {
    device.queue.writeBuffer(uniformBuffer, 0, packUniforms(time, frameCount));
  };

  const renderFrame = (timestamp: number) => {
    if (destroyed) return;

    const elapsed = (timestamp - startTime) / 1000;

    if (animating) {
      const angle = elapsed * 1.25;
      const lx = 278 + 220 * Math.cos(angle);
      const lz = 280 + 220 * Math.sin(angle);
      const ly = 274 + 180 * Math.sin(angle);
      scene.lights[3].position.set([lx, ly, lz]);
      const dx = 278 - lx, dy = 274 - ly, dz = 280 - lz;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      scene.lights[3].direction.set([dx / len, dy / len, dz / len]);
    }

    updateUniforms(elapsed);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0.4, a: 1 },
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    if (useRaytracer) {
      pass.setPipeline(rayPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6);
    } else {
      pass.setPipeline(rastPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(indexData.length);
    }

    if (!useRaytracer && aabbIndexCount > 0) {
      pass.setPipeline(aabbPipeline);
      pass.setBindGroup(0, aabbBindGroup);
      pass.setIndexBuffer(aabbIndexBuffer, 'uint32');
      pass.drawIndexed(aabbIndexCount);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    frameCount++;
  };

  const animationLoop = (timestamp: number) => {
    if (destroyed) return;
    renderFrame(timestamp);
    animFrameId = requestAnimationFrame(animationLoop);
  };

  animFrameId = requestAnimationFrame(animationLoop);

  let fpsTimeout = 0;
  let fpsMeasuring = false;

  const startFPSMonitor = (onResult: (fps: number) => void) => {
    stopFPSMonitor();
    fpsMeasuring = true;

    const measure = () => {
      if (destroyed || !fpsMeasuring) return;
      const startCount = frameCount;
      const measureStart = performance.now();
      const duration = 500;

      fpsTimeout = window.setTimeout(() => {
        if (destroyed || !fpsMeasuring) return;
        const elapsed = (performance.now() - measureStart) / 1000;
        const fps = (frameCount - startCount) / elapsed;
        onResult(fps);
        fpsTimeout = window.setTimeout(measure, 0);
      }, duration);
    };

    fpsTimeout = window.setTimeout(measure, 500);
  };

  const stopFPSMonitor = () => {
    fpsMeasuring = false;
    clearTimeout(fpsTimeout);
  };

  return {
    scene,
    bvhRoot,

    render() { if (!destroyed) renderFrame(performance.now()); },
    startAnimation() { animating = true; },
    stopAnimation() { animating = false; },
    setUseRaytracer(val: boolean) { useRaytracer = val; },
    setSpp(val: number) { spp = val; },
    resetAccum() { frameCount = 1; },

    async pickObject(sx: number, sy: number): Promise<number> {
      if (destroyed) return -1;
      device.queue.writeBuffer(pickCoordsBuffer, 0, new Float32Array([sx, sy]));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pickPipeline);
      pass.setBindGroup(0, pickBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(pickResultBuffer, 0, readbackBuffer, 0, 4);
      device.queue.submit([encoder.finish()]);
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const id = new Int32Array(readbackBuffer.getMappedRange())[0];
      readbackBuffer.unmap();
      return id;
    },

    updateMaterial(idx: number, rgb: [number, number, number], roughness?: number, metalness?: number, basePerlinFreq?: number, perlinOctaveAmp?: number, perlinOctaveNb?: number) {
      if (idx >= 0 && idx < scene.objects.length) {
        scene.objects[idx].material.diffuseAlbedo.set(rgb);
        if (roughness !== undefined) scene.objects[idx].material.roughness = roughness;
        if (metalness !== undefined) scene.objects[idx].material.metalness = metalness;
        if (basePerlinFreq !== undefined) scene.objects[idx].material.basePerlinFreq = basePerlinFreq;
        if (perlinOctaveAmp !== undefined) scene.objects[idx].material.perlinOctaveAmp = perlinOctaveAmp;
        if (perlinOctaveNb !== undefined) scene.objects[idx].material.perlinOctaveNb = perlinOctaveNb;
        if (!destroyed) renderFrame(performance.now());
      }
    },

    setDebugAABBs(aabbs: AABB[]) {
      const { positions, indices } = createAABBWireframe(aabbs);
      device.queue.writeBuffer(aabbVertexBuffer, 0, positions);
      device.queue.writeBuffer(aabbIndexBuffer, 0, indices);
      aabbIndexCount = indices.length;
    },

    startFPSMonitor,
    stopFPSMonitor,

    destroy() {
      destroyed = true;
      cancelAnimationFrame(animFrameId);
      stopFPSMonitor();
      device.destroy();
    },
  };
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

const sRGBToLinear = (c: number) =>
  c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);

const linearToSRGB = (c: number) => {
  const v = Math.max(0, Math.min(1, c));
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
};

const hexToLinearRGB = (hex: string): [number, number, number] => {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [sRGBToLinear(r), sRGBToLinear(g), sRGBToLinear(b)];
};

const linearRGBToHex = (r: number, g: number, b: number): string => {
  const toHex = (c: number) => Math.round(linearToSRGB(c) * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

// ---------------------------------------------------------------------------
// BVH helpers
// ---------------------------------------------------------------------------

function getAABBsAtDepth(node: BVHNode, depth: number): AABB[] {
  if (depth === 0 || node.kind === "leaf") return [node.boundingBox];
  return [
    ...getAABBsAtDepth(node.left, depth - 1),
    ...getAABBsAtDepth(node.right, depth - 1),
  ];
}

function getBVHMaxDepth(node: BVHNode): number {
  if (node.kind === "leaf") return 0;
  return 1 + Math.max(getBVHMaxDepth(node.left), getBVHMaxDepth(node.right));
}

// ---------------------------------------------------------------------------
// DOM wiring
// ---------------------------------------------------------------------------

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const noWebGPU = document.getElementById('no-webgpu')!;
const webgpuContent = document.getElementById('webgpu-content')!;
const performanceWarning = document.getElementById('performance-warning')!;
const bvhSection = document.getElementById('bvh-section')!;
const materialSection = document.getElementById('material-section')!;

const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
const animateCheckbox = document.getElementById('animate-checkbox') as HTMLInputElement;
const raytracedCheckbox = document.getElementById('raytraced-checkbox') as HTMLInputElement;
const sppInput = document.getElementById('spp-input') as HTMLInputElement;
const sppValue = document.getElementById('spp-value')!;
const bvhDepthInput = document.getElementById('bvh-depth-input') as HTMLInputElement;
const bvhDepthValue = document.getElementById('bvh-depth-value')!;
const materialObjectName = document.getElementById('material-object-name')!;
const materialColor = document.getElementById('material-color') as HTMLInputElement;
const roughnessInput = document.getElementById('roughness-input') as HTMLInputElement;
const roughnessValue = document.getElementById('roughness-value')!;
const metalnessInput = document.getElementById('metalness-input') as HTMLInputElement;
const metalnessValue = document.getElementById('metalness-value')!;
const perlinFreqInput = document.getElementById('perlin-freq-input') as HTMLInputElement;
const perlinFreqValue = document.getElementById('perlin-freq-value')!;
const perlinAmpInput = document.getElementById('perlin-amp-input') as HTMLInputElement;
const perlinAmpValue = document.getElementById('perlin-amp-value')!;
const perlinOctavesInput = document.getElementById('perlin-octaves-input') as HTMLInputElement;
const perlinOctavesValue = document.getElementById('perlin-octaves-value')!;
const vulkanSummary = document.getElementById('vulkan-summary')!;
const vulkanDetail = document.getElementById('vulkan-detail')!;
const vulkanClose = document.getElementById('vulkan-close')!;

let engine: Engine | null = null;
let selectedObject = -1;
let dragonBVH: BVHNode | null = null;
let selectedModel: ModelName = 'Dragon (2k tri)';

function getCurrentMaterialValues() {
  return {
    color: materialColor.value,
    roughness: parseFloat(roughnessInput.value),
    metalness: parseFloat(metalnessInput.value),
    basePerlinFreq: parseFloat(perlinFreqInput.value),
    perlinOctaveAmp: parseFloat(perlinAmpInput.value),
    perlinOctaveNb: parseFloat(perlinOctavesInput.value),
  };
}

function applyMaterial() {
  if (!engine || selectedObject < 0) return;
  const { color, roughness, metalness, basePerlinFreq, perlinOctaveAmp, perlinOctaveNb } = getCurrentMaterialValues();
  const [r, g, b] = hexToLinearRGB(color);
  engine.updateMaterial(selectedObject, [r, g, b], roughness, metalness, basePerlinFreq, perlinOctaveAmp, perlinOctaveNb);
  engine.resetAccum();
}

function selectObject(idx: number) {
  if (!engine) return;
  selectedObject = idx;
  const mat = engine.scene.objects[idx].material;
  materialColor.value = linearRGBToHex(mat.diffuseAlbedo[0], mat.diffuseAlbedo[1], mat.diffuseAlbedo[2]);
  roughnessInput.value = String(mat.roughness ?? 0.5);
  roughnessValue.textContent = (mat.roughness ?? 0.5).toFixed(2);
  metalnessInput.value = String(mat.metalness ?? 0);
  metalnessValue.textContent = (mat.metalness ?? 0).toFixed(2);
  perlinFreqInput.value = String(mat.basePerlinFreq);
  perlinFreqValue.textContent = mat.basePerlinFreq.toFixed(1);
  perlinAmpInput.value = String(mat.perlinOctaveAmp);
  perlinAmpValue.textContent = mat.perlinOctaveAmp.toFixed(2);
  perlinOctavesInput.value = String(mat.perlinOctaveNb);
  perlinOctavesValue.textContent = String(mat.perlinOctaveNb);
  materialObjectName.textContent = engine.scene.objects[idx].label ?? '';
  materialSection.style.display = '';
}

async function loadScene(modelName: ModelName) {
  engine?.destroy();
  engine = null;
  bvhSection.style.display = 'none';
  materialSection.style.display = 'none';
  animateCheckbox.checked = false;
  raytracedCheckbox.checked = false;
  sppInput.value = '1';
  sppValue.textContent = '1';

  const objContent = await fetch(`/assets/${MODELS[modelName]}`).then(r => r.text());
  const scene = buildScene(canvas, objContent);
  engine = await createEngine(canvas, scene);

  dragonBVH = engine.bvhRoot;
  const maxDepth = getBVHMaxDepth(engine.bvhRoot);
  bvhDepthInput.max = String(maxDepth);
  bvhDepthInput.value = '0';
  bvhDepthValue.textContent = '0';
  bvhSection.style.display = '';
  engine.setDebugAABBs(getAABBsAtDepth(engine.bvhRoot, 0));

  const firstIdx = scene.objects.findIndex(obj => obj.label);
  selectObject(firstIdx !== -1 ? firstIdx : 0);

  engine.startFPSMonitor(fps => {
    performanceWarning.style.display = fps < 30 ? '' : 'none';
  });
}

// Vulkan warning toggle
vulkanSummary.addEventListener('click', () => {
  vulkanDetail.style.display = '';
  vulkanSummary.style.display = 'none';
});
vulkanClose.addEventListener('click', () => {
  performanceWarning.style.display = 'none';
  vulkanDetail.style.display = 'none';
  vulkanSummary.style.display = '';
});

// Canvas click → object pick
canvas.addEventListener('click', async (e) => {
  if (!engine) return;
  const rect = canvas.getBoundingClientRect();
  const sx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const sy = 1 - ((e.clientY - rect.top) / rect.height) * 2;
  const id = await engine.pickObject(sx, sy);
  if (id < 0) return;
  const idx = engine.scene.objects.findIndex(obj => obj.material.id === id);
  if (idx !== -1) selectObject(idx);
});

// Model select
modelSelect.addEventListener('change', () => {
  selectedModel = modelSelect.value as ModelName;
  loadScene(selectedModel).catch(() => {
    webgpuContent.style.display = 'none';
    noWebGPU.style.display = '';
  });
});

// Animation checkbox
animateCheckbox.addEventListener('change', () => {
  if (animateCheckbox.checked) engine?.startAnimation();
  else engine?.stopAnimation();
});

// Raytraced checkbox
raytracedCheckbox.addEventListener('change', () => {
  engine?.setUseRaytracer(raytracedCheckbox.checked);
  engine?.resetAccum();
});

// SPP slider
sppInput.addEventListener('input', () => {
  sppValue.textContent = sppInput.value;
  engine?.setSpp(parseInt(sppInput.value));
  engine?.resetAccum();
});

// BVH depth slider
bvhDepthInput.addEventListener('input', () => {
  const d = parseInt(bvhDepthInput.value);
  bvhDepthValue.textContent = String(d);
  if (dragonBVH) engine?.setDebugAABBs(getAABBsAtDepth(dragonBVH, d));
});

// Material sliders
roughnessInput.addEventListener('input', () => {
  roughnessValue.textContent = parseFloat(roughnessInput.value).toFixed(2);
  applyMaterial();
});
metalnessInput.addEventListener('input', () => {
  metalnessValue.textContent = parseFloat(metalnessInput.value).toFixed(2);
  applyMaterial();
});
perlinFreqInput.addEventListener('input', () => {
  perlinFreqValue.textContent = parseFloat(perlinFreqInput.value).toFixed(1);
  applyMaterial();
});
perlinAmpInput.addEventListener('input', () => {
  perlinAmpValue.textContent = parseFloat(perlinAmpInput.value).toFixed(2);
  applyMaterial();
});
perlinOctavesInput.addEventListener('input', () => {
  perlinOctavesValue.textContent = perlinOctavesInput.value;
  applyMaterial();
});
materialColor.addEventListener('input', () => {
  applyMaterial();
});

// Keyboard camera controls
const keys = new Set<string>();
const MOVE_SPEED = 200;
const ROT_SPEED = 1.2;
let lastTime = performance.now();

window.addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code))
    e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.code));

const tick = (now: number) => {
  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  const cam = engine?.scene.camera;
  if (cam) {
    const sp = MOVE_SPEED * dt;
    let moved = false;
    if (keys.has('KeyW')) { moveForward(cam, sp); moved = true; }
    if (keys.has('KeyS')) { moveForward(cam, -sp); moved = true; }
    if (keys.has('KeyA')) { pan(cam, -sp, 0); moved = true; }
    if (keys.has('KeyD')) { pan(cam, sp, 0); moved = true; }
    if (keys.has('KeyR')) { pan(cam, 0, sp); moved = true; }
    if (keys.has('KeyF')) { pan(cam, 0, -sp); moved = true; }
    if (keys.has('ArrowLeft'))  { rotateYaw(cam, ROT_SPEED * dt); moved = true; }
    if (keys.has('ArrowRight')) { rotateYaw(cam, -ROT_SPEED * dt); moved = true; }
    if (keys.has('ArrowUp'))    { rotatePitch(cam, ROT_SPEED * dt); moved = true; }
    if (keys.has('ArrowDown'))  { rotatePitch(cam, -ROT_SPEED * dt); moved = true; }
    if (moved) engine?.resetAccum();
  }

  requestAnimationFrame(tick);
};
requestAnimationFrame(tick);

// Init
if (!navigator.gpu) {
  noWebGPU.style.display = '';
} else {
  webgpuContent.style.display = '';
  loadScene(selectedModel).catch(() => {
    webgpuContent.style.display = 'none';
    noWebGPU.style.display = '';
  });
}
