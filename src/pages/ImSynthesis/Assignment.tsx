import { useEffect, useState, useRef } from 'react';
import { HexColorPicker } from 'react-colorful';
import { load_mesh, create_quad, get_mat } from '../../utils/mesh_gen';
import dragonLowObj from '../../assets/dragon_2348.obj?raw';
import dragonHighObj from '../../assets/dragon_117452.obj?raw';

const MODELS = {
  'Dragon (2k tri)': dragonLowObj,
  'Dragon (117k tri)': dragonHighObj,
} as const;

type ModelName = keyof typeof MODELS;
import { initWebGPU, initCamera, getCameraBasis, extractSceneData, getMVP, pan, moveForward, rotateYaw, rotatePitch } from '../../utils/webgpu';
import { type Scene, type Light, type Material} from '../../utils/scene';
import { type AABB, createAABBWireframe, buildSceneBVH, type BVHNode, flattenBVH } from '../../utils/cpuBVH';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';
import shaderCode from '../../shaders/assignment.wgsl?raw';
import noiseShaderCode from '../../shaders/noise.wgsl?raw';
import aabbShaderCode from '../../shaders/aabb.wgsl?raw';

// ---------------------------------------------------------------------------
// Scene definition
// ---------------------------------------------------------------------------

function buildScene(canvas: HTMLCanvasElement, modelName: ModelName): Scene {
  const camera = initCamera(canvas,
    [278, 273, -800],
    [278, 273, -799],
    [0, 1, 0],
    //2 * Math.atan(0.025 / (2 * 0.035)),
    3.1415/3.0,
    0.1,
    2000,
  );

  const boxCenter = [278, 274, 280];

  const normalize = (x: number, y: number, z: number): Float32Array => {
    const len = Math.sqrt(x * x + y * y + z * z);
    return new Float32Array([x / len, y / len, z / len]);
  };

  const lights: Light[] = [
    // Keylight
    {
        position: new Float32Array([71, 412, 140]),
        color: new Float32Array([1.0, 1.0, 1.0]),
        intensity: 1.5,
        direction: normalize(
          boxCenter[0] - 71,
          boxCenter[1] - 412,
          boxCenter[2] - 140
        ),
        angle: 360.0
    },
    // Fill light
    {
        position: new Float32Array([485, 137, 140]),
        color: new Float32Array([0.0, 0.2, 0.8]),
        intensity: 2.5,
        direction: normalize(
          boxCenter[0] - 485,
          boxCenter[1] - 137,
          boxCenter[2] - 140
        ),
        angle: 360.0
    },
    // Back light
    {
        position: new Float32Array([71, 137, 70]),
        color: new Float32Array([0.7, 0.3, 0.0]),
        intensity: 0.8,
        direction: normalize(
          boxCenter[0] - 71,
          boxCenter[1] - 137,
          boxCenter[2] - 70
        ),
        angle: 360.0
    },
    // Rotating light
    {
        position: new Float32Array([554, 494, 280]),
        color: new Float32Array([1.0, 1.0, 1.0]),
        intensity: 0.8,
        direction: normalize(
          boxCenter[0] - 554,
          boxCenter[1] - 494,
          boxCenter[2] - 280
        ),
        angle: 360.0
    },
  ];

  const whiteMaterial: Material = {
      id: 0,
      diffuseAlbedo: new Float32Array([1.0, 1.0, 1.0]),
      roughness: 0,
      metalness: 0,
      fresnel: new Float32Array([0.9, 0.9, 0.9]), // plastic
  };
  const redMaterial:   Material = {
      id: 1,
      diffuseAlbedo: new Float32Array([0.65, 0.05, 0.05]),
      roughness: 0,
      metalness: 0,
      fresnel: new Float32Array([0.05, 0.05, 0.05]), // plastic
  };
  const greenMaterial: Material = {
      id: 2,
      diffuseAlbedo: new Float32Array([0.12, 0.45, 0.15]),
      roughness: 0,
      metalness: 0,
      fresnel: new Float32Array([0.05, 0.05, 0.05]), // plastic
  };
  const noisyDragonMaterial: Material = {
      id: 3,
      diffuseAlbedo: new Float32Array([0.8, 0.0, 0.0]),
      roughness: 0.5,
      metalness: 1.0,
      fresnel: new Float32Array([1.0, 0.71, 0.29]),
  };

  const identityTransform = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);

  const objects = [
    // Cornell Box - Floor
    {
      mesh: create_quad(
        [552.8, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 0.0, 559.2], [549.6, 0.0, 559.2],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform, label: "Floor",
    },
    // Cornell Box - Ceiling
    {
      mesh: create_quad(
        [556.0, 548.8, 0.0], [556.0, 548.8, 559.2],
        [0.0, 548.8, 559.2], [0.0, 548.8, 0.0],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform, label: "Ceiling",
    },
    // Cornell Box - Light (area on ceiling)
    {
      mesh: create_quad(
        [343.0, 548.8, 227.0], [343.0, 548.8, 332.0],
        [213.0, 548.8, 332.0], [213.0, 548.8, 227.0],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform, label: "Light",
    },
    // Cornell Box - Back wall
    {
      mesh: create_quad(
        [549.6, 0.0, 559.2], [0.0, 0.0, 559.2],
        [0.0, 548.8, 559.2], [556.0, 548.8, 559.2],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform, label: "Back wall",
    },
    // Cornell Box - Right wall (green)
    {
      mesh: create_quad(
        [0.0, 0.0, 559.2], [0.0, 0.0, 0.0],
        [0.0, 548.8, 0.0], [0.0, 548.8, 559.2],
        [0.12, 0.45, 0.15],
      ),
      material: greenMaterial, transform: identityTransform, label: "Right wall",
    },
    // Cornell Box - Left wall (red)
    {
      mesh: create_quad(
        [552.8, 0.0, 0.0], [549.6, 0.0, 559.2],
        [556.0, 548.8, 559.2], [556.0, 548.8, 0.0],
        [0.65, 0.05, 0.05],
      ),
      material: redMaterial, transform: identityTransform, label: "Left wall",
    },
    // Cornell Box - Front wall
    {
      mesh: create_quad(
        [556.0, 548.8, 0.0], [0.0, 548.8, 0.0],
        [0.0, 0.0, 0.0], [552.8, 0.0, 0.0],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform, label: "Front wall",
    },
    // Stanford dragon
    {
      mesh: load_mesh(MODELS[modelName], [0.8, 0.2, 0.2]),
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
  setNbBounces(val: number): void;
  updateMaterial(idx: number, rgb: [number, number, number], roughness?: number, metalness?: number): void;
  bvhRoot: BVHNode;
  setDebugAABBs(aabbs: AABB[]): void;
  pickObject(sx: number, sy: number): Promise<number>;
  startFPSMonitor(onResult: (fps: number) => void): void;
  stopFPSMonitor(): void;
  destroy(): void;
}

async function createEngine(canvas: HTMLCanvasElement, scene: Scene): Promise<Engine> {
  // @ts-ignore - keeping adapter for reference
  const { device, context, _adapter } = await initWebGPU(canvas);

  const merged = extractSceneData(scene);
  const { positions: vertexPositions, indices: indexData, uvs: vertexUVs, objectIds, normals: vertexNormals, materials } = merged;

  // ----- Buffers -----

  const vertexBuffer = device.createBuffer({
    label: "Vertices",
    size: vertexPositions.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  const indexBuffer = device.createBuffer({
    label: "Vertex indices",
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  const objectIdBuffer = device.createBuffer({
    label: "Object IDs",
    size: objectIds.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  const normalBuffer = device.createBuffer({
    label: "Vertex normals",
    size: vertexNormals.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  const uvBuffer = device.createBuffer({
    label: "Vertex UVs",
    size: vertexUVs.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  const BVH_FLOATS_PER_NODE = 8; // 8 data

  device.queue.writeBuffer(vertexBuffer, 0, vertexPositions.buffer, vertexPositions.byteOffset, vertexPositions.byteLength);
  device.queue.writeBuffer(indexBuffer, 0, indexData.buffer, indexData.byteOffset, indexData.byteLength);
  device.queue.writeBuffer(objectIdBuffer, 0, objectIds.buffer, objectIds.byteOffset, objectIds.byteLength);
  device.queue.writeBuffer(normalBuffer, 0, vertexNormals.buffer, vertexNormals.byteOffset, vertexNormals.byteLength);
  device.queue.writeBuffer(uvBuffer, 0, vertexUVs.buffer, vertexUVs.byteOffset, vertexUVs.byteLength);

  const bvhRoot = buildSceneBVH(scene);
  const flat = flattenBVH(bvhRoot);
  const bvhBytes = new ArrayBuffer(flat.length * BVH_FLOATS_PER_NODE * 4);
  {
    const view = new DataView(bvhBytes);
    for (let i = 0; i < flat.length; i++) {
      const n = flat[i];
      const b = i * BVH_FLOATS_PER_NODE * 4; // byte offset
      view.setFloat32(b +  0, n.minCorner[0],        true);
      view.setFloat32(b +  4, n.minCorner[1],        true);
      view.setFloat32(b +  8, n.minCorner[2],        true);
      view.setUint32 (b + 12, n.isLeaf ? 1 : 0,      true);
      view.setFloat32(b + 16, n.maxCorner[0],        true);
      view.setFloat32(b + 20, n.maxCorner[1],        true);
      view.setFloat32(b + 24, n.maxCorner[2],        true);
      if(n.isLeaf){
        view.setUint32 (b + 28, n.triangleIndex >= 0 ? n.triangleIndex : 0, true);
      }else{
        view.setUint32 (b + 28, n.rightChild     >= 0 ? n.rightChild     : 0, true);
      }
    }
  }

  const bvhBuffer = device.createBuffer({
    label: "BVH nodes",
    size: bvhBytes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bvhBuffer, 0, bvhBytes);

  // ----- Uniforms -----

  const MAX_LIGHTS = 4;
  const MAX_MATERIALS = 16;
  const MATERIAL_SIZE = 8; // vec3 baseColor + roughness + vec3 fresnel + metalness = 8 floats
  const MVP_SIZE = 16;
  const SHARED_HEADER = 4; // camera_pos (vec3) + nbLights (f32)
  const LIGHTS_SIZE = MAX_LIGHTS * 12;
  const MATERIALS_HEADER = 4; // nbMaterials + nb_bounces + 2 padding floats
  const RAY_CAMERA_SIZE = 12; // forward+fov, right+aspect, up+time
  const RAY_OFFSET = MVP_SIZE + SHARED_HEADER + LIGHTS_SIZE + MATERIALS_HEADER + MAX_MATERIALS * MATERIAL_SIZE;
  const UNIFORM_LENGTH = RAY_OFFSET + RAY_CAMERA_SIZE;

  let nbBounces = 1;

  const packLightsAndMaterials = (out: Float32Array) => {
    out.set(scene.camera.position, 16); // camera_pos at index 16-18 (shared)
    out[19] = scene.lights.length;      // nbLights at index 19 (shared)
    for (let i = 0; i < scene.lights.length; i++) {
      out.set(scene.lights[i].position, 20 + i * 12);
      out[23 + i * 12] = scene.lights[i].intensity;
      out.set(scene.lights[i].color, 24 + i * 12);
      out.set(scene.lights[i].direction, 28 + i * 12);
      out[31 + i * 12] = scene.lights[i].angle * Math.PI / 180;
    }
    const matOffset = 20 + MAX_LIGHTS * 12;
    out[matOffset] = materials.length;
    out[matOffset + 1] = nbBounces;
    for (let i = 0; i < materials.length; i++) {
      const baseIdx = matOffset + 4 + i * MATERIAL_SIZE;
      out.set(materials[i].diffuseAlbedo, baseIdx);
      out[baseIdx + 3] = materials[i].roughness ?? 0;
      out.set(materials[i].fresnel, baseIdx + 4);
      out[baseIdx + 7] = materials[i].metalness ?? 0;
    }
  };

  const packUniforms = (time = 0): Float32Array<ArrayBuffer> => {
    const data = new Float32Array(UNIFORM_LENGTH);
    // Rasterizer MVP (indices 0-15)
    data.set(getMVP(scene.camera), 0);
    // Shared lights/materials (indices 16+)
    packLightsAndMaterials(data);
    // Raytracer camera basis (at RAY_OFFSET)
    const basis = getCameraBasis(scene.camera);
    const fovFactor = Math.tan(scene.camera.fov / 2);
    data.set([...basis.forward, fovFactor], RAY_OFFSET);
    data.set([...basis.right, scene.camera.aspect], RAY_OFFSET + 4);
    data.set([...basis.up, time], RAY_OFFSET + 8);
    return data;
  };

  const uniformBuffer = device.createBuffer({
    size: UNIFORM_LENGTH * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(uniformBuffer, 0, packUniforms());

  // ----- Pipelines -----

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

  const shaderModule = device.createShaderModule({ label: "Shader", code: noiseShaderCode.concat(shaderCode)});

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT,                         buffer: { type: "read-only-storage" } },
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
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // ----- Pick buffers -----

  const pickCoordsBuffer = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const pickResultBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

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

  // ----- Depth texture -----

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ----- AABB debug pipeline -----

  const MAX_AABB = 120000;
  const aabbVertexBuffer = device.createBuffer({
    label: "AABB vertices",
    size: MAX_AABB * 8 * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const aabbIndexBuffer = device.createBuffer({
    label: "AABB indices",
    size: MAX_AABB * 24 * 4,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });

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
    vertex:   { module: aabbShaderModule, entryPoint: "aabbVertexMain" },
    fragment: { module: aabbShaderModule, entryPoint: "aabbFragmentMain", targets: [{ format: canvasFormat }] },
    primitive: { topology: "line-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
  });

  // ----- Render state -----

  let destroyed = false;
  let useRaytracer = false;
  let animating = false;
  let animFrameId = 0;
  let aabbIndexCount = 0;
  let frameCount = 0;
  const startTime = performance.now();

  const updateUniforms = (time = 0) => {
    device.queue.writeBuffer(uniformBuffer, 0, packUniforms(time));
  };

  const renderFrame = (timestamp: number) => {
    if (destroyed) return;

    const elapsed = (timestamp - startTime) / 1000;

    if (animating) {
      const angle = elapsed * 1.25; // matches solution: angle = time_ms / 800
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

  // ----- Animation loop -----

  const animationLoop = (timestamp: number) => {
    if (destroyed) return;
    renderFrame(timestamp);
    animFrameId = requestAnimationFrame(animationLoop);
  };

  animFrameId = requestAnimationFrame(animationLoop);

  // ----- FPS monitor -----

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
        console.log(`Average FPS: ${fps.toFixed(1)}`);
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

  // ----- Public API -----

  return {
    scene,
    bvhRoot,

    render() {
      if (!destroyed) renderFrame(performance.now());
    },

    startAnimation() {
      animating = true;
    },

    stopAnimation() {
      animating = false;
    },

    setUseRaytracer(val: boolean) {
      useRaytracer = val;
    },

    setNbBounces(val: number) {
      nbBounces = val;
    },

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

    updateMaterial(idx: number, rgb: [number, number, number], roughness?: number, metalness?: number) {
      if (idx >= 0 && idx < scene.objects.length) {
        scene.objects[idx].material.diffuseAlbedo.set(rgb);
        if (roughness !== undefined) scene.objects[idx].material.roughness = roughness;
        if (metalness !== undefined) scene.objects[idx].material.metalness = metalness;
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
// Color helpers (linear RGB ↔ sRGB hex)
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
// BVH debug helpers
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
// React component
// ---------------------------------------------------------------------------

export default function Playground() {
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [showPerformanceWarning, setShowPerformanceWarning] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [useRaytracer, setUseRaytracer] = useState(false);
  const [sceneReady, setSceneReady] = useState(false);
  const [selectedObject, setSelectedObject] = useState<number>(-1);
  const [color, setColor] = useState('#cc3300');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [roughness, setRoughness] = useState(0.5);
  const [metalness, setMetalness] = useState(0);
  const [nbBounces, setNbBounces] = useState(1);
  const [selectedModel, setSelectedModel] = useState<ModelName>('Dragon (2k tri)');

  const [bvhDepth, setBvhDepth] = useState(0);
  const [maxBvhDepth, setMaxBvhDepth] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const pickerRef = useRef<HTMLDivElement>(null);
  const dragonBVHRef = useRef<BVHNode | null>(null);

  const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !engineRef.current || !sceneReady) return;
    const rect = canvas.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const sy = 1 - ((e.clientY - rect.top) / rect.height) * 2;
    const id = await engineRef.current.pickObject(sx, sy);
    if (id < 0) return;
    const objIdx = engineRef.current.scene.objects.findIndex(obj => obj.material.id === id);
    if (objIdx === -1) return;
    setSelectedObject(objIdx);
    const mat = engineRef.current.scene.objects[objIdx].material;
    setColor(linearRGBToHex(mat.diffuseAlbedo[0], mat.diffuseAlbedo[1], mat.diffuseAlbedo[2]));
    setRoughness(mat.roughness ?? 0.5);
    setMetalness(mat.metalness ?? 0);
  };

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setPickerOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!navigator.gpu) {
      setWebgpuSupported(false);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        setSceneReady(false);
        setIsAnimating(false);
        setUseRaytracer(false);
        setNbBounces(1);
        setBvhDepth(0);
        setPickerOpen(false);
        const scene = buildScene(canvas, selectedModel);
        const engine = await createEngine(canvas, scene);
        if (cancelled) { engine.destroy(); return; }

        engineRef.current = engine;

        dragonBVHRef.current = engine.bvhRoot;
        setMaxBvhDepth(getBVHMaxDepth(engine.bvhRoot));
        engine.setDebugAABBs(getAABBsAtDepth(engine.bvhRoot, 0));

        // Initialize color picker with first labeled object
        const firstLabeledIdx = scene.objects.findIndex(obj => obj.label);
        const initIdx = firstLabeledIdx !== -1 ? firstLabeledIdx : 0;
        if (scene.objects.length > 0) {
          const mat = scene.objects[initIdx].material;
          setSelectedObject(initIdx);
          setColor(linearRGBToHex(mat.diffuseAlbedo[0], mat.diffuseAlbedo[1], mat.diffuseAlbedo[2]));
          setRoughness(mat.roughness ?? 0.5);
          setMetalness(mat.metalness ?? 0);
        }
        setSceneReady(true);

        engine.startFPSMonitor(fps => setShowPerformanceWarning(fps < 30));
      } catch (error) {
        console.error("WebGPU initialization failed:", error);
        setWebgpuSupported(false);
      }
    })();

    return () => {
      cancelled = true;
      engineRef.current?.destroy();
      engineRef.current = null;
    };
  }, [selectedModel]);

  // Keyboard camera controls: WASD/ZQSD = move, arrow keys = rotate
  useEffect(() => {
    if (!sceneReady) return;

    const keys = new Set<string>();
    const MOVE_SPEED = 200; // units/second (scene is ~550 units wide)
    const ROT_SPEED  = 1.2; // radians/second

    let rafId = 0;
    let lastTime = performance.now();

    const onKeyDown = (e: KeyboardEvent) => {
      keys.add(e.code);
      if (e.code === 'ArrowUp' || e.code === 'ArrowDown' || e.code === 'ArrowLeft' || e.code === 'ArrowRight')
        e.preventDefault();
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.code);

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    const tick = (now: number) => {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      const cam = engineRef.current?.scene.camera;
      if (cam) {
        const sp = MOVE_SPEED * dt;

        if (keys.has('KeyW')) moveForward(cam, sp);
        if (keys.has('KeyS')) moveForward(cam, -sp);
        if (keys.has('KeyA')) pan(cam, -sp, 0);
        if (keys.has('KeyD')) pan(cam, sp, 0);
        if (keys.has('KeyR')) pan(cam, 0, sp);
        if (keys.has('KeyF')) pan(cam, 0, -sp);

        if (keys.has('ArrowLeft'))  rotateYaw(cam, ROT_SPEED * dt);
        if (keys.has('ArrowRight')) rotateYaw(cam, -ROT_SPEED * dt);
        if (keys.has('ArrowUp'))    rotatePitch(cam, ROT_SPEED * dt);
        if (keys.has('ArrowDown'))  rotatePitch(cam, -ROT_SPEED * dt);
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      cancelAnimationFrame(rafId);
    };
  }, [sceneReady]);

  return (
    <div>
      <h1>WebGPU playground</h1>

      <p>
      This is what I'm currently working on. It might be a future experiment or a class assignment, who nose...
      </p>

      {webgpuSupported ? (
        <>
          {showPerformanceWarning && <VulkanWarning />}
          <canvas ref={canvasRef} width="1024" height="1024" style={{ background: 'black', display: 'block', margin: '0 auto', cursor: 'crosshair' }} onClick={handleCanvasClick}></canvas>
          <p style={{ textAlign: 'center', margin: '6px 0 0', fontSize: '0.85rem', opacity: 0.6 }}>
            WASD to move, R/F up/down, arrow keys to look around, click to select a material
          </p>

          <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', alignItems: 'flex-start', marginTop: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              Model:
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as ModelName)}
              >
                {Object.keys(MODELS).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>

            <label htmlFor="animatingCheckbox" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <input
                type="checkbox"
                id="animatingCheckbox"
                checked={isAnimating}
                onChange={(e) => {
                  const val = e.target.checked;
                  setIsAnimating(val);
                  if (val) engineRef.current?.startAnimation();
                  else engineRef.current?.stopAnimation();
                }}
              />
              Light animation
            </label>

            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
              <label htmlFor="raytracingCheckbox" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <input
                  type="checkbox"
                  id="raytracingCheckbox"
                  checked={useRaytracer}
                  onChange={(e) => {
                    const val = e.target.checked;
                    setUseRaytracer(val);
                    engineRef.current?.setUseRaytracer(val);
                  }}
                />
                Raytraced
              </label>
              <span style={{ color: 'red', fontSize: '0.75em' }}>at your own risk</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', minWidth: '140px' }}>
              <label htmlFor="nbBounces">Bounces: {nbBounces}</label>
              <input
                type="range"
                id="nbBounces"
                min="1"
                max="20"
                step="1"
                value={nbBounces}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  setNbBounces(val);
                  engineRef.current?.setNbBounces(val);
                }}
              />
            </div>
          </div>

          <br /><br />

          {sceneReady && dragonBVHRef.current && (
            <div style={{ maxWidth: '400px', marginBottom: '16px' }}>
              <h3>BVH Debug</h3>
              <div>
                <label htmlFor="bvhDepth">Depth: {bvhDepth}</label><br />
                <input
                  type="range"
                  id="bvhDepth"
                  min="0"
                  max={maxBvhDepth}
                  step="1"
                  value={bvhDepth}
                  onChange={(e) => {
                    const d = parseInt(e.target.value);
                    setBvhDepth(d);
                    engineRef.current?.setDebugAABBs(getAABBsAtDepth(dragonBVHRef.current!, d));
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}

          {sceneReady && (
            <div style={{ maxWidth: '400px' }}>
              <h3>Material Editor — {engineRef.current?.scene.objects[selectedObject]?.label}</h3>

              <div style={{ marginBottom: '10px' }}>
                <label>Color</label><br />
                <div ref={pickerRef} style={{ position: 'relative', display: 'inline-block', marginTop: '4px' }}>
                  <div
                    onClick={() => setPickerOpen(v => !v)}
                    style={{ width: '36px', height: '36px', background: color, border: '2px solid #555', borderRadius: '4px', cursor: 'pointer' }}
                  />
                  {pickerOpen && (
                    <div style={{ position: 'absolute', top: '44px', left: 0, zIndex: 100 }}>
                      <HexColorPicker
                        color={color}
                        onChange={(hex) => {
                          setColor(hex);
                          const [r, g, b] = hexToLinearRGB(hex);
                          engineRef.current?.updateMaterial(selectedObject, [r, g, b], roughness, metalness);
                        }}
                      />
                    </div>
                  )}
                </div>
              </div>

              <div style={{ marginBottom: '10px' }}>
                <label htmlFor="roughness">Roughness: {roughness.toFixed(2)}</label><br />
                <input
                  type="range"
                  id="roughness"
                  min="0"
                  max="1"
                  step="0.01"
                  value={roughness}
                  onChange={(e) => {
                    const r = parseFloat(e.target.value);
                    setRoughness(r);
                    const [red, green, blue] = hexToLinearRGB(color);
                    engineRef.current?.updateMaterial(selectedObject, [red, green, blue], r, metalness);
                  }}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '10px' }}>
                <label htmlFor="metalness">Metalness: {metalness.toFixed(2)}</label><br />
                <input
                  type="range"
                  id="metalness"
                  min="0"
                  max="1"
                  step="0.01"
                  value={metalness}
                  onChange={(e) => {
                    const m = parseFloat(e.target.value);
                    setMetalness(m);
                    const [red, green, blue] = hexToLinearRGB(color);
                    engineRef.current?.updateMaterial(selectedObject, [red, green, blue], roughness, m);
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}
        </>
      ) : (
        <WebGPUWarning />
      )}

    </div>
  );
}
