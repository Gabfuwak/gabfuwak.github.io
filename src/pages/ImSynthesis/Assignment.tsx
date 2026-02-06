import { useEffect, useState, useRef, useMemo } from 'react';
import { createSphere, create_quad } from '../../utils/mesh_gen';
import { initWebGPU, initCamera, getCameraBasis, extractSceneData, getMVP} from '../../utils/webgpu';
import { type Scene, type Light, type Material} from '../../utils/scene';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';
import rasterShaderCode from '../../shaders/assignment.wgsl?raw';
import raytraceShaderCode from '../../shaders/raytracing.wgsl?raw';
import { rgbToOklab, oklabToRgb } from '../../utils/colorSpaceUtils';

export default function Playground() {
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [showPerformanceWarning, setShowPerformanceWarning] = useState(false);

  const [isAnimating, setIsAnimating] = useState(false);

  const [useRaytracer, setUseRaytracer] = useState(false);

  const [sceneReady, setSceneReady] = useState(false);
  const [selectedObject, setSelectedObject] = useState<number>(-1);
  const [oklabL, setOklabL] = useState(0.6);
  const [oklabA, setOklabA] = useState(0.15);
  const [oklabB, setOklabB] = useState(0.08);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isAnimatingRef = useRef(false);
  const useRayTracerRef = useRef(false);
  const sceneRef = useRef<Scene | null>(null);
  const triggerRenderRef = useRef<(() => void) | null>(null);

  function createRasterPipeline(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
    uniformBuffer: GPUBuffer,
    vertexBufferLayout: GPUVertexBufferLayout,
    objectIdBufferLayout: GPUVertexBufferLayout,
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
        buffers: [vertexBufferLayout, objectIdBufferLayout, vertexNormalsBufferLayout]
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
    objectIdBuffer: GPUBuffer,
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
        { binding: 3, resource: { buffer: objectIdBuffer } },
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

  const previewColor = useMemo(() => {
    const [r, g, b] = oklabToRgb(oklabL, oklabA, oklabB);
    const toSRGB = (c: number) => {
      const linear = Math.max(0, Math.min(1, c));
      return linear <= 0.0031308
        ? linear * 12.92
        : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
    };
    return `rgb(${Math.round(toSRGB(r) * 255)}, ${Math.round(toSRGB(g) * 255)}, ${Math.round(toSRGB(b) * 255)})`;
  }, [oklabL, oklabA, oklabB]);

  useEffect(() => {
    if (sceneRef.current && selectedObject >= 0 && selectedObject < sceneRef.current.objects.length) {
      const [r, g, b] = oklabToRgb(oklabL, oklabA, oklabB);
      sceneRef.current.objects[selectedObject].material.diffuseAlbedo.set([r, g, b]);
      if (triggerRenderRef.current) {
        triggerRenderRef.current();
      }
    }
  }, [oklabL, oklabA, oklabB, selectedObject]);

  useEffect(() => {




    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    let fpsCheckTimeout: number;
    let animFrameId = 0;
    let pollId = 0;
    let gpuDevice: GPUDevice | null = null;



    (async () => {
      try {
        // Check WebGPU support
        if (!navigator.gpu) {
          setWebgpuSupported(false);
          return;
        }

        // @ts-ignore - keeping adapter for reference
        const { device, context, _adapter } = await initWebGPU(canvas);
        gpuDevice = device;

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
            

        // Define materials
        const whiteMaterial: Material = { diffuseAlbedo: new Float32Array([1.0, 1.0, 1.0]) };
        const redMaterial: Material = { diffuseAlbedo: new Float32Array([0.65, 0.05, 0.05]) };
        const greenMaterial: Material = { diffuseAlbedo: new Float32Array([0.12, 0.45, 0.15]) };

        const identityTransform = new Float32Array([1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1]);

        const scene_objects = [
          // Cornell Box - Floor
          {
            mesh: create_quad(
              [552.8, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 559.2],
              [549.6, 0.0, 559.2],
              [1.0, 1.0, 1.0]
            ),
            material: whiteMaterial,
            transform: identityTransform
          },
          // Cornell Box - Ceiling
          {
            mesh: create_quad(
              [556.0, 548.8, 0.0],
              [556.0, 548.8, 559.2],
              [0.0, 548.8, 559.2],
              [0.0, 548.8, 0.0],
              [1.0, 1.0, 1.0]
            ),
            material: whiteMaterial,
            transform: identityTransform
          },
          // Cornell Box - Light (area on ceiling)
          {
            mesh: create_quad(
              [343.0, 548.8, 227.0],
              [343.0, 548.8, 332.0],
              [213.0, 548.8, 332.0],
              [213.0, 548.8, 227.0],
              [1.0, 1.0, 1.0]
            ),
            material: whiteMaterial,
            transform: identityTransform
          },
          // Cornell Box - Back wall
          {
            mesh: create_quad(
              [549.6, 0.0, 559.2],
              [0.0, 0.0, 559.2],
              [0.0, 548.8, 559.2],
              [556.0, 548.8, 559.2],
              [1.0, 1.0, 1.0]
            ),
            material: whiteMaterial,
            transform: identityTransform
          },
          // Cornell Box - Right wall (green)
          {
            mesh: create_quad(
              [0.0, 0.0, 559.2],
              [0.0, 0.0, 0.0],
              [0.0, 548.8, 0.0],
              [0.0, 548.8, 559.2],
              [0.12, 0.45, 0.15]
            ),
            material: greenMaterial,
            transform: identityTransform
          },
          // Cornell Box - Left wall (red)
          {
            mesh: create_quad(
              [552.8, 0.0, 0.0],
              [549.6, 0.0, 559.2],
              [556.0, 548.8, 559.2],
              [556.0, 548.8, 0.0],
              [0.65, 0.05, 0.05]
            ),
            material: redMaterial,
            transform: identityTransform
          },
          // Bottom-left sphere (red) - equilateral triangle with side = 1.1 * diameter = 132
          {
            mesh: createSphere(60, 32, 32, [0.8, 0.2, 0.2]),
            material: { diffuseAlbedo: new Float32Array([0.8, 0.2, 0.2]) },
            transform: new Float32Array([1, 0, 0, 0,
                                          0, 1, 0, 0,
                                          0, 0, 1, 0,
                                          212, 227, 280, 1]),
            label: "Ball 1"
          },
          // Bottom-right sphere (yellow)
          {
            mesh: createSphere(60, 32, 32, [0.8, 0.8, 0.2]),
            material: { diffuseAlbedo: new Float32Array([0.8, 0.8, 0.2]) },
            transform: new Float32Array([1, 0, 0, 0,
                                          0, 1, 0, 0,
                                          0, 0, 1, 0,
                                          344, 227, 280, 1]),
            label: "Ball 2"
          },
          // Top sphere (blue) - apex of equilateral triangle
          {
            mesh: createSphere(60, 32, 32, [0.2, 0.4, 0.8]),
            material: { diffuseAlbedo: new Float32Array([0.2, 0.4, 0.8]) },
            transform: new Float32Array([1, 0, 0, 0,
                                          0, 1, 0, 0,
                                          0, 0, 1, 0,
                                          278, 341, 280, 1]),
            label: "Ball 3"
          }
        ];

        const scene : Scene = {
          objects: scene_objects,
          lights: lights,
          camera: camera,

        };

        sceneRef.current = scene;

        // Initialize color picker with first labeled object
        const firstLabeledIdx = scene.objects.findIndex(obj => obj.label);
        if (firstLabeledIdx !== -1) {
          const mat = scene.objects[firstLabeledIdx].material.diffuseAlbedo;
          const [l, a, b] = rgbToOklab(mat[0], mat[1], mat[2]);
          setSelectedObject(firstLabeledIdx);
          setOklabL(l);
          setOklabA(a);
          setOklabB(b);
        }
        setSceneReady(true);

        const merged = extractSceneData(scene);

        const vertexPositions = merged.positions;
        const indexData = merged.indices;
        const objectIds = merged.objectIds;
        const vertexNormals = merged.normals;
        const materials = merged.materials;


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

        const MAX_LIGHTS = 4;
        const MAX_MATERIALS = 16;
        const UNIFORM_LENGTH = 16 + 4 + MAX_LIGHTS * 4 + 4 + MAX_MATERIALS * 4;

        // Shared: pack lights + materials into a Float32Array starting at offset 16
        const packLightsAndMaterials = (out: Float32Array, scene: Scene, materials: Material[]) => {
          out[16] = scene.lights.length;
          for (let i = 0; i < scene.lights.length; i++) {
            out.set(scene.lights[i].position, 20 + i * 4);
          }
          const matOffset = 20 + MAX_LIGHTS * 4;
          out[matOffset] = materials.length;
          for (let i = 0; i < materials.length; i++) {
            out.set(materials[i].diffuseAlbedo, matOffset + 4 + i * 4);
          }
        };

        const packRasterUniforms = (scene: Scene, materials: Material[]): Float32Array => {
          const data = new Float32Array(UNIFORM_LENGTH);
          data.set(getMVP(scene.camera), 0);
          packLightsAndMaterials(data, scene, materials);
          return data;
        };

        const packRayUniforms = (scene: Scene, materials: Material[]): Float32Array => {
          const data = new Float32Array(UNIFORM_LENGTH);
          const basis = getCameraBasis(scene.camera);
          const fovFactor = Math.tan(scene.camera.fov / 2);
          data.set([...scene.camera.position, fovFactor], 0);
          data.set([...basis.forward, scene.camera.aspect], 4);
          data.set([...basis.right, 0], 8);
          data.set([...basis.up, 0], 12);
          packLightsAndMaterials(data, scene, materials);
          return data;
        };

        const rastUniformBuffer = device.createBuffer({
          size: UNIFORM_LENGTH * 4,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const rayUniformBuffer = device.createBuffer({
          size: UNIFORM_LENGTH * 4,
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

        const objectIdBufferLayout: GPUVertexBufferLayout = {
          arrayStride: 4,
          attributes: [{
            format: "uint32" as GPUVertexFormat,
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
                                            objectIdBufferLayout,
                                            vertexNormalsBufferLayout
                                           );

        const ray_pipeline_info = createRayTracePipeline(device,
                                              canvasFormat,
                                              rayUniformBuffer,
                                              vertexBuffer,
                                              indexBuffer,
                                              objectIdBuffer,
                                              normalBuffer
                                             );



        device.queue.writeBuffer(vertexBuffer, 0, vertexPositions.buffer, vertexPositions.byteOffset, vertexPositions.byteLength);
        device.queue.writeBuffer(indexBuffer, 0, indexData.buffer , indexData.byteOffset, indexData.byteLength);
        device.queue.writeBuffer(objectIdBuffer, 0, objectIds.buffer, objectIds.byteOffset, objectIds.byteLength);
        device.queue.writeBuffer(normalBuffer, 0, vertexNormals.buffer, vertexNormals.byteOffset, vertexNormals.byteLength);

        // Initial uniform write so we get a frame before animation starts
        device.queue.writeBuffer(rastUniformBuffer, 0, packRasterUniforms(scene, materials));
        device.queue.writeBuffer(rayUniformBuffer, 0, packRayUniforms(scene, materials));

        const depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const startTime = performance.now();

        const updateUniforms = () => {
          device.queue.writeBuffer(rastUniformBuffer, 0, packRasterUniforms(scene, materials));
          device.queue.writeBuffer(rayUniformBuffer, 0, packRayUniforms(scene, materials));
        };

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
          }

          updateUniforms();


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
            pass.setVertexBuffer(1, objectIdBuffer);
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

        // Expose trigger function for color updates
        triggerRenderRef.current = () => {
          if (!cancelled) {
            renderFrame(performance.now());
          }
        };

        // Animation loop — only runs while isAnimating is true
        const animationLoop = (timestamp: number) => {
          if (cancelled) return;
          renderFrame(timestamp);
          if (isAnimatingRef.current) {
            animFrameId = requestAnimationFrame(animationLoop);
          }
        };

        // Start/stop the loop when isAnimating changes.
        // We poll via rAF so we notice the ref change promptly.
        const pollAnimation = () => {
          if (cancelled) return;
          if (isAnimatingRef.current && animFrameId === 0) {
            animFrameId = requestAnimationFrame(animationLoop);
          }
          requestAnimationFrame(pollAnimation);
        };
        pollId = requestAnimationFrame(pollAnimation);

        // Recurring FPS measurement: measure for 2s, pause 3s, repeat
        const measureFPS = () => {
          if (cancelled) return;
          let frames = 0;
          const measureStart = performance.now();
          const duration = 2000;

          const measureLoop = (timestamp: number) => {
            if (cancelled) return;
            renderFrame(timestamp);
            frames++;
            if (performance.now() - measureStart < duration) {
              requestAnimationFrame(measureLoop);
            } else {
              const fps = (frames / duration) * 1000;
              console.log(`Average FPS: ${fps.toFixed(1)}`);
              if (fps < 10) {
                setShowPerformanceWarning(true);
              } else {
                setShowPerformanceWarning(false);
              }
              // Schedule next measurement cycle
              fpsCheckTimeout = window.setTimeout(measureFPS, 3000);
            }
          };

          requestAnimationFrame(measureLoop);
        };

        fpsCheckTimeout = window.setTimeout(measureFPS, 500);

      } catch (error) {
        console.error("WebGPU initialization failed:", error);
        setWebgpuSupported(false);
      }
    })();

    return () => {
      cancelled = true;
      clearTimeout(fpsCheckTimeout);
      cancelAnimationFrame(animFrameId);
      cancelAnimationFrame(pollId);
      gpuDevice?.destroy();
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
          <canvas ref={canvasRef} width="512" height="512" style={{ background: 'black', display: 'block', margin: '0 auto' }}></canvas>

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

          <br /><br />

          {sceneReady && (
            <div style={{ maxWidth: '400px' }}>
              <h3>OKLab Color Picker</h3>

              <div style={{ marginBottom: '10px' }}>
                <label htmlFor="objectSelect">Object: </label>
                <select
                  id="objectSelect"
                  value={selectedObject}
                  onChange={(e) => {
                    const idx = parseInt(e.target.value);
                    setSelectedObject(idx);
                    if (sceneRef.current && idx >= 0 && idx < sceneRef.current.objects.length) {
                      const mat = sceneRef.current.objects[idx].material.diffuseAlbedo;
                      const [l, a, b] = rgbToOklab(mat[0], mat[1], mat[2]);
                      setOklabL(l);
                      setOklabA(a);
                      setOklabB(b);
                    }
                  }}
                >
                  {sceneRef.current?.objects.map((obj, idx) => (
                    obj.label ? <option key={idx} value={idx}>{obj.label}</option> : null
                  ))}
                </select>
              </div>

              <div style={{
                width: '100px',
                height: '100px',
                border: '2px solid #333',
                marginBottom: '15px',
                backgroundColor: previewColor,
              }} />

              <div style={{ marginBottom: '10px' }}>
                <label htmlFor="oklabL">Lightness: {oklabL.toFixed(2)}</label><br />
                <input
                  type="range"
                  id="oklabL"
                  min="0"
                  max="1"
                  step="0.01"
                  value={oklabL}
                  onChange={(e) => setOklabL(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '10px' }}>
                <label htmlFor="oklabA">a (green ← → red): {oklabA.toFixed(2)}</label><br />
                <input
                  type="range"
                  id="oklabA"
                  min="-0.4"
                  max="0.4"
                  step="0.01"
                  value={oklabA}
                  onChange={(e) => setOklabA(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '10px' }}>
                <label htmlFor="oklabB">b (blue ← → yellow): {oklabB.toFixed(2)}</label><br />
                <input
                  type="range"
                  id="oklabB"
                  min="-0.4"
                  max="0.4"
                  step="0.01"
                  value={oklabB}
                  onChange={(e) => setOklabB(parseFloat(e.target.value))}
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
