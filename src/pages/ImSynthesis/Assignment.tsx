import { useEffect, useState, useRef } from 'react';
import { createCornellBox } from '../../utils/mesh_gen';
import { initWebGPU, initCamera, getCameraBasis, extractSceneData, getMVP} from '../../utils/webgpu';
import { type Scene, type Light} from '../../utils/scene';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';
import rasterShaderCode from '../../shaders/assignment.wgsl?raw';
import raytraceShaderCode from '../../shaders/raytracing.wgsl?raw';

export default function Playground() {
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [showPerformanceWarning, setShowPerformanceWarning] = useState(false);

  const [isAnimating, setIsAnimating] = useState(false);

  const [useRaytracer, setUseRaytracer] = useState(false);

  const isAnimatingRef = useRef(false);
  const useRayTracerRef = useRef(false);

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

  useEffect(() => {
    isAnimatingRef.current = isAnimating;
    useRayTracerRef.current = useRaytracer;
  }, [isAnimating, useRaytracer]);

  useEffect(() => {




    const canvas = document.querySelector("canvas");
    if (!canvas) return;

    let cancelled = false;
    let fpsCheckTimeout: number;



    (async () => {
      try {
        // Check WebGPU support
        if (!navigator.gpu) {
          setWebgpuSupported(false);
          return;
        }

        // @ts-ignore - keeping adapter for reference
        const { device, context, _adapter } = await initWebGPU(canvas);

        const camera = initCamera(canvas,
                                  [278, 273, -800], // position
                                  [278, 273, -799], // target (center of Cornell box)
                                  [0, 1, 0], // up
                                  2 * Math.atan(0.025 / (2 * 0.035)), // fov from sensor height 0.025 and focal length 0.035
                                  0.1, // near
                                  2000, // far
                                 );
        const lights : Light[]= [
          {
            position: new Float32Array([278.0, 530.0, 280.0]),
            color: new Float32Array([1, 1, 1, 1]),
          },
          {
            position: new Float32Array([400, 530, 150]),
            color: new Float32Array([1, 1, 1, 1]),
          }
        ];
            

        const scene_objects = [{
          mesh: createCornellBox(),
          transform: new Float32Array ([1, 0, 0, 0,  // no transform
                                        0, 1, 0, 0,
                                        0, 0, 1, 0,
                                        0, 0, 0, 1])
        }];

        const scene : Scene = {
          objects: scene_objects,
          lights: lights,
          camera: camera,
          
        };

        const merged = extractSceneData(scene);

        const vertexPositions = merged.positions;
        const indexData = merged.indices;
        const vertexColors = merged.colors;
        const vertexNormals = merged.normals;


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

        const normalBuffer = device.createBuffer({
          label: "Vertex normals",
          size: vertexNormals.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
        });

        const rastUniformBuffer = device.createBuffer({
          size: (16 + 4 + 4 * 4) * 4, // MVP + light count + 4 lights (padded vec4)
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });


        const rayUniformBuffer = device.createBuffer({
          size: 64 + 16 + (16 * 4), // camera_pos(16) + forward(16) + right(16) + up(16) + lightNb(4) + padding(12) + 4 lights (16 * 4) = 144 bytes
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });


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

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        const rast_pipeline_info = createRasterPipeline(device,
                                            canvasFormat,
                                            rastUniformBuffer,
                                            vertexBufferLayout,
                                            vertexColorBufferLayout,
                                            vertexNormalsBufferLayout
                                           );

        const ray_pipeline_info = createRayTracePipeline(device,
                                              canvasFormat,
                                              rayUniformBuffer,
                                              vertexBuffer,
                                              indexBuffer,
                                              colorBuffer,
                                              normalBuffer
                                             );



        device.queue.writeBuffer(vertexBuffer, 0, vertexPositions.buffer, vertexPositions.byteOffset, vertexPositions.byteLength);
        device.queue.writeBuffer(indexBuffer, 0, indexData.buffer , indexData.byteOffset, indexData.byteLength);
        device.queue.writeBuffer(colorBuffer, 0, vertexColors.buffer, vertexColors.byteOffset, vertexColors.byteLength);
        device.queue.writeBuffer(normalBuffer, 0, vertexNormals.buffer, vertexNormals.byteOffset, vertexNormals.byteLength);

        // initial uniform write to have a render without the animation
        const MAX_LIGHTS = 4;
        const initialUniformData = new Float32Array(16 + 4 + MAX_LIGHTS * 4);
        initialUniformData.set(getMVP(scene.camera), 0);
        initialUniformData[16] = scene.lights.length;
        for (let i = 0; i < scene.lights.length; i++) {
          initialUniformData.set(scene.lights[i].position, 20 + i * 4);
        }
        device.queue.writeBuffer(rastUniformBuffer, 0, initialUniformData);

        const basis = getCameraBasis(scene.camera);
        const fovFactor = Math.tan(scene.camera.fov / 2);
        const initialRayUniformData = new Float32Array(20 + MAX_LIGHTS * 4);
        initialRayUniformData.set([...scene.camera.position, fovFactor], 0);
        initialRayUniformData.set([...basis.forward, scene.camera.aspect], 4);
        initialRayUniformData.set([...basis.right, 0], 8);
        initialRayUniformData.set([...basis.up, 0], 12);
        initialRayUniformData[16] = scene.lights.length;
        for (let i = 0; i < scene.lights.length; i++) {
          initialRayUniformData.set(scene.lights[i].position, 20 + i * 4);
        }
        device.queue.writeBuffer(rayUniformBuffer, 0, initialRayUniformData);

        const depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const startTime = performance.now();

        // Helper to render one frame
        const renderFrame = (timestamp: number) => {
          if (cancelled) return;

          if (isAnimatingRef.current) {
            const elapsed = (timestamp - startTime) / 1000; // time in seconds
            // Orbit the light around the center of the box (278, 273)
            const radius = 250;
            const x = 278 + radius * Math.sin(elapsed);
            const z = 280 + radius * Math.cos(elapsed);

            const x2 = 278 + radius * Math.sin(elapsed*1.5);
            const z2 = 280 + radius * Math.cos(elapsed*1.5);

            scene.lights[0].position.set([x,  300, z,]);
            scene.lights[1].position.set([x2, 400, z2]);

            // Update rasterizer uniform buffer with updated light position
            const MAX_LIGHTS = 4;
            const uniformData = new Float32Array(16 + 4 + MAX_LIGHTS * 4);
            uniformData.set(getMVP(scene.camera), 0);
            uniformData[16] = scene.lights.length;
            for (let i = 0; i < scene.lights.length; i++) {
              uniformData.set(scene.lights[i].position, 20 + i * 4);
            }
            device.queue.writeBuffer(rastUniformBuffer, 0, uniformData);

            // Update raytracer uniform buffer with updated light position
            const basis = getCameraBasis(scene.camera);
            const fovFactor = Math.tan(scene.camera.fov / 2);
            const rayUniformData = new Float32Array(20 + MAX_LIGHTS * 4);
            rayUniformData.set([...scene.camera.position, fovFactor], 0);
            rayUniformData.set([...basis.forward, scene.camera.aspect], 4);
            rayUniformData.set([...basis.right, 0], 8);
            rayUniformData.set([...basis.up, 0], 12);
            rayUniformData[16] = scene.lights.length;
            for (let i = 0; i < scene.lights.length; i++) {
              rayUniformData.set(scene.lights[i].position, 20 + i * 4);
            }
            device.queue.writeBuffer(rayUniformBuffer, 0, rayUniformData);
          }


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

          if (useRayTracerRef.current) {
            pass.setPipeline(ray_pipeline_info.pipeline);
            pass.setBindGroup(0, ray_pipeline_info.bindGroup);
            pass.setVertexBuffer(0, ray_pipeline_info.quadVertexBuffer);
            pass.draw(6); // 6 vertices, two triangles
          } else {
            pass.setPipeline(rast_pipeline_info.pipeline);
            pass.setBindGroup(0, rast_pipeline_info.bindGroup);
            pass.setVertexBuffer(0, vertexBuffer);
            pass.setVertexBuffer(1, colorBuffer);
            pass.setVertexBuffer(2, normalBuffer);
            pass.setIndexBuffer(indexBuffer, "uint32");
            pass.drawIndexed(indexData.length);
          }

          pass.end();

          const commandBuffer = encoder.finish();
          device.queue.submit([commandBuffer]);
        };

        // Render initial frame immediately
        renderFrame(0);

        // Measure FPS for performance warning
        const measureFPS = () => {
          let frames = 0;
          const startTime = performance.now();
          const duration = 2000; // Measure for 2 seconds

          const renderLoop = (timestamp : number) => {
            if (cancelled) return;

            renderFrame(timestamp);

            frames++;
            const elapsed = performance.now() - startTime;

            if (elapsed < duration && !cancelled) {
              requestAnimationFrame(renderLoop);
            } else {
              const fps = (frames / elapsed) * 1000;
              console.log(`Average FPS: ${fps.toFixed(1)}`);
              // Show warning if FPS is below 10
              if (fps < 10) {
                setShowPerformanceWarning(true);
              }
              requestAnimationFrame(renderLoop);
            }
          };

          requestAnimationFrame(renderLoop);
        };

        // Start FPS measurement after a brief delay
        fpsCheckTimeout = window.setTimeout(measureFPS, 500);

      } catch (error) {
        console.error("WebGPU initialization failed:", error);
        setWebgpuSupported(false);
      }
    })();

    return () => {
      cancelled = true;
      if (fpsCheckTimeout) {
        clearTimeout(fpsCheckTimeout);
      }
    };
  }, []);

  return (
    <div>
      <h1>WebGPU playground</h1>

      <p>
      Congrats on finding this page, you went though the source code!
      </p>

      <p>
      This is what I'm currently working on. It might be a future experiment or a class assignment, who nose...
      </p>

      {webgpuSupported ? (
        <>
          {showPerformanceWarning && <VulkanWarning />}
          <canvas width="512" height="512" style={{ background: 'black', display: 'block', margin: '0 auto' }}></canvas>
        </>
      ) : (
        <WebGPUWarning />
      )}

      <input
        type="checkbox"
        id="animatingCheckbox"
        checked={isAnimating}
        onChange={(e) => setIsAnimating(e.target.checked)}
      />
      <label htmlFor="animatingCheckbox">Light animation</label>

      <input
        type="checkbox"
        id="raytracingCheckbox"
        checked={useRaytracer}
        onChange={(e) => setUseRaytracer(e.target.checked)}
      />
      <label htmlFor="raytracingCheckbox">Raytraced</label>



    </div>
  );
}
