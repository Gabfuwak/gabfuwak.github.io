import { type Camera, getMVP } from './camera';
import { type Mesh} from './mesh_gen'
import rasterShaderCode from '../shaders/assignment.wgsl?raw';
import raytraceShaderCode from '../shaders/raytracing.wgsl?raw';

export interface Scene {
  // Shared geometry (created with VERTEX | STORAGE so both pipelines can use it)
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  colorBuffer: GPUBuffer;
  normalBuffer: GPUBuffer;

  // Shared uniforms (MVP, lights)
  uniformBuffer: GPUBuffer;

  // Rasterizer pipeline objects
  rasterPipeline: GPURenderPipeline;
  rasterBindGroup: GPUBindGroup;

  // Ray tracer pipeline objects
  rayPipeline: GPURenderPipeline;
  rayBindGroup: GPUBindGroup;   // geometry bound as storage here
  quadVertexBuffer: GPUBuffer;  // just the full-screen quad

  // CPU-side state for animation
  lights: { pos: Float32Array, color: Float32Array }[];
  camera: Camera;
  mvp: Float32Array;
}

function createRasterPipeline(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
  uniformBuffer: GPUBuffer,
  vertexBufferLayout: GPUVertexBufferLayout,
  vertexColorBufferLayout: GPUVertexBufferLayout,
  vertexNormalsBufferLayout: GPUVertexBufferLayout,
): { pipeline: GPURenderPipeline, bindGroup: GPUBindGroup } {

  const rasterShaderModule = device.createShaderModule({
    label: "Shader",
    code: rasterShaderCode
  });

  const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      }],
    });

  const rasterBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer },
    }],
  });

  const rasterPipeline = device.createRenderPipeline({
    label: "Rasterizer pipeline",
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    vertex: {
      module: rasterShaderModule,
      entryPoint: "vertexMain",
      buffers: [vertexBufferLayout, vertexColorBufferLayout, vertexNormalsBufferLayout]
    },
    fragment: {
      module: rasterShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat
      }]
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    }
  });

  return { pipeline: rasterPipeline, bindGroup: rasterBindGroup };
}

function createRayTracePipeline(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
  uniformBuffer: GPUBuffer,
  vertexBuffer: GPUBuffer,
  indexBuffer: GPUBuffer,
  colorBuffer: GPUBuffer,
  normalBuffer: GPUBuffer,
): { pipeline: GPURenderPipeline, bindGroup: GPUBindGroup, quadVertexBuffer: GPUBuffer} {


  const quadVertices = new Float32Array([
    -1, -1,  1, -1,  -1, 1,
     1, -1,  1,  1,  -1, 1,
  ]);

  const quadVertexBuffer = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadVertexBuffer, 0, quadVertices);


  const raytraceShaderModule = device.createShaderModule({
    label: "Shader",
    code: raytraceShaderCode
  });

  const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "storage", access: "read-only"},
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "storage", access: "read-only"},
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "storage", access: "read-only" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "storage", access: "read-only"},
        },
      ] as GPUBindGroupLayoutEntry[],
    });

  const rayBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: vertexBuffer } },
      { binding: 2, resource: { buffer: indexBuffer } },
      { binding: 3, resource: { buffer: colorBuffer } },
      { binding: 4, resource: { buffer: normalBuffer } },
    ],
  });

  const quadBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
      format: "float32x2",
      offset: 0,
      shaderLocation: 0,
    }],
  };

  const raytracePipeline = device.createRenderPipeline({
    label: "Raytraced pipeline",
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    vertex: {
      module: raytraceShaderModule,
      entryPoint: "vertexMain",
      buffers: [quadBufferLayout]
    },
    fragment: {
      module: raytraceShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat
      }]
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: false,
      depthCompare: "always",
    },
  });

  return { pipeline: raytracePipeline, bindGroup: rayBindGroup, quadVertexBuffer: quadVertexBuffer};
}

export function initScene(device: GPUDevice, camera: Camera, mesh: Mesh, lightPos: Float32Array) : Scene {

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

  const mvpMatrix = getMVP(camera);

  const vertexPositions = mesh.positions;
  const indexData = mesh.indices;
  const vertexColors = mesh.colors;
  const vertexNormals = mesh.normals;

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

  const colorBuffer = device.createBuffer({
    label: "Vertex colors",
    size: vertexColors.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  const normalsBuffer = device.createBuffer({
    label: "Vertex normals",
    size: vertexNormals.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });

  device.queue.writeBuffer(vertexBuffer, 0, vertexPositions);
  device.queue.writeBuffer(indexBuffer, 0, indexData);
  device.queue.writeBuffer(colorBuffer, 0, vertexColors);
  device.queue.writeBuffer(normalsBuffer, 0, vertexNormals);

  // Combine MVP and lightPos into a single uniform buffer
  const uniformData = new Float32Array(16 + 4); // 16 floats for mat4x4, 4 for vec3 (padded)
  uniformData.set(mvpMatrix, 0);  // First 16 floats
  uniformData.set(lightPos, 16);   // Next 3 floats (vec3 needs 16-byte alignment)

  const uniformBuffer = device.createBuffer({
    size: 80, // 64 bytes (mat4x4) + 16 bytes (vec3 with padding)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 12,
    attributes: [{
      format: "float32x3" as GPUVertexFormat,
      offset: 0,
      shaderLocation: 0,
    }],
  };

  const vertexColorBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 12,
    attributes: [{
      format: "float32x3" as GPUVertexFormat,
      offset: 0,
      shaderLocation: 1,
    }],
  };

  const vertexNormalsBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 12,
    attributes: [{
      format: "float32x3" as GPUVertexFormat,
      offset: 0,
      shaderLocation: 2,
    }],
  };

  const raster = createRasterPipeline(device,
                                      canvasFormat,
                                      uniformBuffer,
                                      vertexBufferLayout,
                                      vertexColorBufferLayout,
                                      vertexNormalsBufferLayout
                                     );

  const rayTrace = createRayTracePipeline(device,
                                        canvasFormat,
                                        uniformBuffer,
                                        vertexBuffer,
                                        indexBuffer,
                                        colorBuffer,
                                        normalsBuffer
                                       );

  return {
    vertexBuffer: vertexBuffer,
    indexBuffer: indexBuffer,
    colorBuffer: colorBuffer,
    normalBuffer: normalsBuffer,
    uniformBuffer: uniformBuffer,
    rasterPipeline: raster.pipeline,
    rasterBindGroup: raster.bindGroup,
    rayPipeline: rayTrace.pipeline,
    rayBindGroup: rayTrace.bindGroup,
    quadVertexBuffer: rayTrace.quadVertexBuffer,
    indexCount : indexData.length,
    lights: [{pos: lightPos, color: new Float32Array([1.0, 1.0, 1.0])}],
    camera: camera,
    mvp: mvpMatrix,
  };
}
