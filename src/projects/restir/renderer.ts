// === renderer.ts ===
// WebGPU device, pipeline, bind groups, render loop

import type { Camera } from './camera.ts';
import { getCameraBasis } from './camera.ts';
import type { Material, Light } from './scene.ts';

export interface WebGPUInit {
  device:        GPUDevice;
  context:       GPUCanvasContext;
  format:        GPUTextureFormat;
  hasTimestamps: boolean;
}

export interface TimestampResources {
  querySet:     GPUQuerySet;
  resolveBuffer: GPUBuffer;
  readBuffers:  GPUBuffer[];
  readIndex:    number;
  onGpuTime?:   (ms: number) => void;
}

export interface SceneBuffers {
  vertexBuffer:   GPUBuffer;
  indexBuffer:    GPUBuffer;
  normalBuffer:   GPUBuffer;
  uvBuffer:       GPUBuffer;
  tlasBuffer:     GPUBuffer;
  blasBuffer:     GPUBuffer;
  instanceBuffer: GPUBuffer;
  emissiveBuffer: GPUBuffer;
}

export interface Geometry {
  positions: Float32Array<ArrayBuffer>;
  normals:   Float32Array<ArrayBuffer>;
  uvs:       Float32Array<ArrayBuffer>;
  indices:   Uint32Array<ArrayBuffer>;
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<WebGPUInit> {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found");

  const hasTimestamps = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamps ? ['timestamp-query'] : [],
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16),
    },
  });

  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  return { device, context, format, hasTimestamps };
}

// Creates GPU timestamp query resources. Returns null if not supported.
export function createTimestampResources(device: GPUDevice, hasTimestamps: boolean): TimestampResources | null {
  if (!hasTimestamps) return null;

  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });

  // resolveBuffer: GPU writes raw u64 nanosecond timestamps here
  const resolveBuffer = device.createBuffer({
    size: 2 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  // Two read buffers alternating so one is always available while the other is mapped
  const readBuffers = [0, 1].map(() => device.createBuffer({
    size: 2 * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  }));

  return { querySet, resolveBuffer, readBuffers, readIndex: 0 };
}

export function createAccumulationBuffer(device: GPUDevice, screenWidth: number, screenHeight: number): GPUBuffer {
  return device.createBuffer({
    size: screenWidth*screenHeight*16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
}

// Uniform layout (must match WGSL Uniforms struct):
// Camera:          4 × vec4f =  64 bytes  (floats 0–15)
// Lights[4]:       4 × 12f   = 192 bytes  (floats 16–63)
// Materials hdr:   1 × vec4f =  16 bytes  (floats 64–67)
// Screen+pad:      1 × vec4f =  16 bytes  (floats 68–71)
// Materials[16]:  16 × 12f   = 768 bytes  (floats 72–263)
// Total: 1056 bytes = 264 floats
const UNIFORM_FLOATS = 268;

export function createUniformBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    size: UNIFORM_FLOATS * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function packUniforms(
  camera: Camera,
  lights: Light[],
  materials: Material[],
  frameCount: number,
  bvhVisDepth = -1,
  bvhHeatMax = 4,
  bvhEarlyStop = 0,
  screenWidth = 1024,
  screenHeight = 1024,
  renderMode = 0,
  emissiveCount = 0,
  risSamples = 1,
  risTarget = 1
): Float32Array<ArrayBuffer> {
  const data = new Float32Array(UNIFORM_FLOATS) as Float32Array<ArrayBuffer>;

  // Camera (floats 0–15)
  const basis = getCameraBasis(camera);
  const fovFactor = Math.tan(camera.fov / 2);
  data[0]  = camera.position[0]; data[1]  = camera.position[1]; data[2]  = camera.position[2];
  data[3]  = fovFactor;
  data[4]  = basis.forward[0];   data[5]  = basis.forward[1];   data[6]  = basis.forward[2];
  data[7]  = camera.aspect;
  data[8]  = basis.right[0];     data[9]  = basis.right[1];     data[10] = basis.right[2];
  data[11] = lights.length;
  data[12] = basis.up[0];        data[13] = basis.up[1];        data[14] = basis.up[2];
  data[15] = frameCount || 0;

  // Lights (floats 16–63, 12 floats per light)
  for (let i = 0; i < lights.length; i++) {
    const o = 16 + i * 12;
    data[o]    = lights[i].position[0];  data[o+1]  = lights[i].position[1];  data[o+2]  = lights[i].position[2];
    data[o+3]  = lights[i].intensity;
    data[o+4]  = lights[i].color[0];     data[o+5]  = lights[i].color[1];     data[o+6]  = lights[i].color[2];
    data[o+7]  = 0; // _pad1
    data[o+8]  = lights[i].direction[0]; data[o+9]  = lights[i].direction[1]; data[o+10] = lights[i].direction[2];
    data[o+11] = lights[i].angle * Math.PI / 180;
  }

  // Materials header (floats 64–67)
  data[64] = materials.length;
  data[65] = bvhVisDepth;
  data[66] = bvhHeatMax;
  data[67] = bvhEarlyStop;

  // Screen size + render mode (floats 68–71)
  data[68] = screenWidth;
  data[69] = screenHeight;
  data[70] = renderMode;
  data[71] = emissiveCount;

  // RIS (floats 72–75)
  data[72] = risSamples;
  data[73] = risTarget;
  // data[74..75] pad

  // Materials (floats 76–267, 12 floats per material)
  for (let i = 0; i < materials.length; i++) {
    const o = 76 + i * 12;
    data[o]   = materials[i].diffuseAlbedo[0]; data[o+1] = materials[i].diffuseAlbedo[1]; data[o+2] = materials[i].diffuseAlbedo[2];
    data[o+3] = materials[i].roughness || 0;
    data[o+4] = materials[i].fresnel[0];       data[o+5] = materials[i].fresnel[1];       data[o+6] = materials[i].fresnel[2];
    data[o+7] = materials[i].metalness || 0;
    data[o+8] = materials[i].emission || 0;
  }

  return data;
}

export function createPipeline(device: GPUDevice, format: GPUTextureFormat, shaderCode: string): GPURenderPipeline {
  const module = device.createShaderModule({ code: shaderCode });
  return device.createRenderPipeline({
    layout: "auto",
    vertex:   { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
  });
}

export function createComputePipeline(device: GPUDevice, shaderCode: string): GPUComputePipeline {
  const module = device.createShaderModule({ code: shaderCode });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "computeMain" },
  });
}

// geo: { positions, normals, uvs, indices } — unified geometry buffers
// tlasBytes, blasBytes, instanceBytes: ArrayBuffers from main.ts
export function createSceneBuffers(
  device: GPUDevice,
  geo: Geometry,
  tlasBytes: ArrayBuffer,
  blasBytes: ArrayBuffer,
  instanceBytes: ArrayBuffer,
  emissiveBytes: ArrayBuffer
): SceneBuffers {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

  const vertexBuffer   = device.createBuffer({ label: "vertices",  size: geo.positions.byteLength, usage });
  const indexBuffer    = device.createBuffer({ label: "indices",   size: geo.indices.byteLength,   usage });
  const normalBuffer   = device.createBuffer({ label: "normals",   size: geo.normals.byteLength,   usage });
  const uvBuffer       = device.createBuffer({ label: "uvs",       size: geo.uvs.byteLength,       usage });
  const tlasBuffer     = device.createBuffer({ label: "tlas",      size: tlasBytes.byteLength,     usage });
  const blasBuffer     = device.createBuffer({ label: "blases",    size: blasBytes.byteLength,     usage });
  const instanceBuffer = device.createBuffer({ label: "instances", size: instanceBytes.byteLength, usage });
  // Emissive triangle list — minimum 16 bytes (WebGPU requires non-zero)
  const emissiveBuffer = device.createBuffer({ label: "emissive",  size: Math.max(emissiveBytes.byteLength, 16), usage });

  device.queue.writeBuffer(vertexBuffer,   0, geo.positions);
  device.queue.writeBuffer(indexBuffer,    0, geo.indices);
  device.queue.writeBuffer(normalBuffer,   0, geo.normals);
  device.queue.writeBuffer(uvBuffer,       0, geo.uvs);
  device.queue.writeBuffer(tlasBuffer,     0, tlasBytes);
  device.queue.writeBuffer(blasBuffer,     0, blasBytes);
  device.queue.writeBuffer(instanceBuffer, 0, instanceBytes);
  if (emissiveBytes.byteLength > 0) {
    device.queue.writeBuffer(emissiveBuffer, 0, emissiveBytes);
  }

  return { vertexBuffer, indexBuffer, normalBuffer, uvBuffer, tlasBuffer, blasBuffer, instanceBuffer, emissiveBuffer };
}

export function destroySceneBuffers(sceneBuffers: SceneBuffers | null): void {
  if (!sceneBuffers) return;
  for (const key of Object.keys(sceneBuffers) as (keyof SceneBuffers)[]) {
    sceneBuffers[key].destroy();
  }
}

export function createComputeBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  uniformBuffer: GPUBuffer,
  sceneBuffers: SceneBuffers,
  accumulationBuffer: GPUBuffer,
  reservoirBuffer: GPUBuffer
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: sceneBuffers.vertexBuffer } },
      { binding: 2, resource: { buffer: sceneBuffers.indexBuffer } },
      { binding: 3, resource: { buffer: sceneBuffers.normalBuffer } },
      { binding: 4, resource: { buffer: sceneBuffers.uvBuffer } },
      { binding: 5, resource: { buffer: sceneBuffers.tlasBuffer } },
      { binding: 6, resource: { buffer: sceneBuffers.blasBuffer } },
      { binding: 7, resource: { buffer: sceneBuffers.instanceBuffer } },
      { binding: 8, resource: { buffer: accumulationBuffer } },
      { binding: 9, resource: { buffer: sceneBuffers.emissiveBuffer } },
      { binding: 10, resource: { buffer: reservoirBuffer } },
    ],
  });
}

export function createDisplayBindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  accumulationBuffer: GPUBuffer,
  uniformBuffer: GPUBuffer
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: accumulationBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
    ],
  });
}

// ts: timestamp resources from createTimestampResources(), or null
export function renderFrame(
  device: GPUDevice,
  context: GPUCanvasContext,
  displayPipeline: GPURenderPipeline,
  computePipeline: GPUComputePipeline,
  displayBindGroup: GPUBindGroup,
  computeBindGroup: GPUBindGroup,
  width: number,
  height: number,
  ts: TimestampResources | null
): void {
  const encoder = device.createCommandEncoder();

  const computePassDesc: GPUComputePassDescriptor = {};
  if (ts) {
    computePassDesc.timestampWrites = {
      querySet: ts.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1
    };
  }

  const computePass = encoder.beginComputePass(computePassDesc);
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  computePass.end();

  const displayPassDesc: GPURenderPassDescriptor = {
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  };

  const displayPass = encoder.beginRenderPass(displayPassDesc);
  displayPass.setPipeline(displayPipeline);
  displayPass.setBindGroup(0, displayBindGroup);
  displayPass.draw(3);
  displayPass.end();

  if (ts) {
    encoder.resolveQuerySet(ts.querySet, 0, 2, ts.resolveBuffer, 0);
    const readBuf = ts.readBuffers[ts.readIndex];
    if (readBuf.mapState === 'unmapped') {
      encoder.copyBufferToBuffer(ts.resolveBuffer, 0, readBuf, 0, 2 * 8);
    }
  }

  device.queue.submit([encoder.finish()]);

  // Async readback — alternates between two buffers, one frame behind
  if (ts) {
    const readBuf = ts.readBuffers[ts.readIndex];
    ts.readIndex = (ts.readIndex + 1) % 2;
    if (readBuf.mapState === 'unmapped') {
      readBuf.mapAsync(GPUMapMode.READ).then(() => {
        const times = new BigInt64Array(readBuf.getMappedRange());
        const gpuMs = Number(times[1] - times[0]) / 1e6;
        readBuf.unmap();
        if (ts.onGpuTime) ts.onGpuTime(gpuMs);
      });
    }
  }
}
