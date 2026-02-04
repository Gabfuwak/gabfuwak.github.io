import { type Camera, getMVP, getCameraBasis } from './camera';
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

  // Rasterizer uniforms (MVP, lights)
  uniformBuffer: GPUBuffer;

  // Ray tracer uniforms (camera basis, fov, lights)
  rayUniformBuffer: GPUBuffer;

  // Rasterizer pipeline objects
  rasterPipeline: GPURenderPipeline;
  rasterBindGroup: GPUBindGroup;

  // Ray tracer pipeline objects
  rayPipeline: GPURenderPipeline;
  rayBindGroup: GPUBindGroup;   // geometry bound as storage here
  quadVertexBuffer: GPUBuffer;  // just the full-screen quad

  // CPU-side state for animation
  lights: Float32Array;
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
  rayUniformBuffer: GPUBuffer,
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
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ] as GPUBindGroupLayoutEntry[],
    });

  const rayBindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: rayUniformBuffer } },
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

export function initScene(device: GPUDevice, camera: Camera, mesh: Mesh, lights: Float32Array) : Scene {

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

  // Rasterizer uniform buffer: MVP + lights
  const MAX_LIGHTS = 4;
  const uniformData = new Float32Array(16 + 4 + MAX_LIGHTS * 3);
  uniformData.set(mvpMatrix, 0);
  uniformData[16] = lights.length / 3; // numLights
  uniformData.set(lights, 20);

  console.log('lights.length:', lights.length, 'nbLights:', lights.length / 3);
  console.log('uniformData length:', uniformData.length, 'bytes:', uniformData.length * 4);

  const uniformBuffer = device.createBuffer({
    size: 64 + 16 + 12 * MAX_LIGHTS,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Raytracer uniform buffer: camera basis + fov + aspect + lightPos
  const basis = getCameraBasis(camera);
  const fovFactor = Math.tan(camera.fov / 2);

  const rayUniformData = new Float32Array(20 + lights.length); // 80 bytes / 4 = 20 floats
  rayUniformData.set([...camera.position, fovFactor], 0);     // camera_pos + fov_factor
  rayUniformData.set([...basis.forward, camera.aspect], 4);   // camera_forward + aspect_ratio
  rayUniformData.set([...basis.right, 0], 8);                 // camera_right + padding
  rayUniformData.set([...basis.up, 0], 12);                   // camera_up + padding
  rayUniformData[16] = lights.length / 3;                     // number of lights
  rayUniformData.set(lights, 20);                             // lights data

  const rayUniformBuffer = device.createBuffer({
    size: 64 + 16 + (12 * 4), // camera_pos(16) + forward(16) + right(16) + up(16) + lightNb(4) + padding(12) + 4 lights (12 * 4) = 128 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(rayUniformBuffer, 0, rayUniformData);

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
                                        rayUniformBuffer,
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
    rayUniformBuffer: rayUniformBuffer,
    rasterPipeline: raster.pipeline,
    rasterBindGroup: raster.bindGroup,
    rayPipeline: rayTrace.pipeline,
    rayBindGroup: rayTrace.bindGroup,
    quadVertexBuffer: rayTrace.quadVertexBuffer,
    indexCount : indexData.length,
    lights: lights,
    camera: camera,
    mvp: mvpMatrix,
  };
}
