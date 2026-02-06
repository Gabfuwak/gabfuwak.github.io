import { useEffect, useState, useRef, useMemo } from 'react';
import { createSphere, create_quad } from '../../utils/mesh_gen';
import { initWebGPU, initCamera, getCameraBasis, extractSceneData, getMVP} from '../../utils/webgpu';
import { type Scene, type Light, type Material} from '../../utils/scene';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';
import rasterShaderCode from '../../shaders/assignment.wgsl?raw';
import raytraceShaderCode from '../../shaders/raytracing.wgsl?raw';
import { rgbToOklab, oklabToRgb } from '../../utils/colorSpaceUtils';

// ---------------------------------------------------------------------------
// Scene definition
// ---------------------------------------------------------------------------

function buildScene(canvas: HTMLCanvasElement): Scene {
  const camera = initCamera(canvas,
    [278, 273, -800],
    [278, 273, -799],
    [0, 1, 0],
    2 * Math.atan(0.025 / (2 * 0.035)),
    0.1,
    2000,
  );

  const lights: Light[] = [
    { position: new Float32Array([278.0, 530.0, 280.0]), color: new Float32Array([1, 1, 1, 1]) },
    { position: new Float32Array([400, 530, 150]),        color: new Float32Array([1, 1, 1, 1]) },
  ];

  const whiteMaterial: Material = { diffuseAlbedo: new Float32Array([1.0, 1.0, 1.0]) };
  const redMaterial:   Material = { diffuseAlbedo: new Float32Array([0.65, 0.05, 0.05]) };
  const greenMaterial: Material = { diffuseAlbedo: new Float32Array([0.12, 0.45, 0.15]) };

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
      material: whiteMaterial, transform: identityTransform,
    },
    // Cornell Box - Ceiling
    {
      mesh: create_quad(
        [556.0, 548.8, 0.0], [556.0, 548.8, 559.2],
        [0.0, 548.8, 559.2], [0.0, 548.8, 0.0],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform,
    },
    // Cornell Box - Light (area on ceiling)
    {
      mesh: create_quad(
        [343.0, 548.8, 227.0], [343.0, 548.8, 332.0],
        [213.0, 548.8, 332.0], [213.0, 548.8, 227.0],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform,
    },
    // Cornell Box - Back wall
    {
      mesh: create_quad(
        [549.6, 0.0, 559.2], [0.0, 0.0, 559.2],
        [0.0, 548.8, 559.2], [556.0, 548.8, 559.2],
        [1.0, 1.0, 1.0],
      ),
      material: whiteMaterial, transform: identityTransform,
    },
    // Cornell Box - Right wall (green)
    {
      mesh: create_quad(
        [0.0, 0.0, 559.2], [0.0, 0.0, 0.0],
        [0.0, 548.8, 0.0], [0.0, 548.8, 559.2],
        [0.12, 0.45, 0.15],
      ),
      material: greenMaterial, transform: identityTransform,
    },
    // Cornell Box - Left wall (red)
    {
      mesh: create_quad(
        [552.8, 0.0, 0.0], [549.6, 0.0, 559.2],
        [556.0, 548.8, 559.2], [556.0, 548.8, 0.0],
        [0.65, 0.05, 0.05],
      ),
      material: redMaterial, transform: identityTransform,
    },
    // Bottom-left sphere (red)
    {
      mesh: createSphere(60, 10, 10, [0.8, 0.2, 0.2]),
      material: { diffuseAlbedo: new Float32Array([0.8, 0.2, 0.2]) },
      transform: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 212,227,280,1]),
      label: "Ball 1",
    },
    // Bottom-right sphere (yellow)
    {
      mesh: createSphere(60, 10, 10, [0.8, 0.8, 0.2]),
      material: { diffuseAlbedo: new Float32Array([0.8, 0.8, 0.2]) },
      transform: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 344,227,280,1]),
      label: "Ball 2",
    },
    // Top sphere (blue)
    {
      mesh: createSphere(60, 10, 10, [0.2, 0.4, 0.8]),
      material: { diffuseAlbedo: new Float32Array([0.2, 0.4, 0.8]) },
      transform: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 278,341,280,1]),
      label: "Ball 3",
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
  updateMaterial(idx: number, rgb: [number, number, number]): void;
  startFPSMonitor(onResult: (fps: number) => void): void;
  stopFPSMonitor(): void;
  destroy(): void;
}

async function createEngine(canvas: HTMLCanvasElement, scene: Scene): Promise<Engine> {
  // @ts-ignore - keeping adapter for reference
  const { device, context, _adapter } = await initWebGPU(canvas);

  const merged = extractSceneData(scene);
  const { positions: vertexPositions, indices: indexData, objectIds, normals: vertexNormals, materials } = merged;

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

  device.queue.writeBuffer(vertexBuffer, 0, vertexPositions.buffer, vertexPositions.byteOffset, vertexPositions.byteLength);
  device.queue.writeBuffer(indexBuffer, 0, indexData.buffer, indexData.byteOffset, indexData.byteLength);
  device.queue.writeBuffer(objectIdBuffer, 0, objectIds.buffer, objectIds.byteOffset, objectIds.byteLength);
  device.queue.writeBuffer(normalBuffer, 0, vertexNormals.buffer, vertexNormals.byteOffset, vertexNormals.byteLength);

  // ----- Uniforms -----

  const MAX_LIGHTS = 4;
  const MAX_MATERIALS = 16;
  const UNIFORM_LENGTH = 16 + 4 + MAX_LIGHTS * 4 + 4 + MAX_MATERIALS * 4;

  const packLightsAndMaterials = (out: Float32Array) => {
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

  const packRasterUniforms = (): Float32Array<ArrayBuffer> => {
    const data = new Float32Array(UNIFORM_LENGTH);
    data.set(getMVP(scene.camera), 0);
    packLightsAndMaterials(data);
    return data;
  };

  const packRayUniforms = (): Float32Array<ArrayBuffer> => {
    const data = new Float32Array(UNIFORM_LENGTH);
    const basis = getCameraBasis(scene.camera);
    const fovFactor = Math.tan(scene.camera.fov / 2);
    data.set([...scene.camera.position, fovFactor], 0);
    data.set([...basis.forward, scene.camera.aspect], 4);
    data.set([...basis.right, 0], 8);
    data.set([...basis.up, 0], 12);
    packLightsAndMaterials(data);
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

  device.queue.writeBuffer(rastUniformBuffer, 0, packRasterUniforms());
  device.queue.writeBuffer(rayUniformBuffer, 0, packRayUniforms());

  // ----- Pipelines -----

  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 12,
    attributes: [{ format: "float32x3" as GPUVertexFormat, offset: 0, shaderLocation: 0 }],
  };
  const objectIdBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 4,
    attributes: [{ format: "uint32" as GPUVertexFormat, offset: 0, shaderLocation: 1 }],
  };
  const vertexNormalsBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 12,
    attributes: [{ format: "float32x3" as GPUVertexFormat, offset: 0, shaderLocation: 2 }],
  };

  const createRasterPipeline = () => {
    const shaderModule = device.createShaderModule({ label: "Shader", code: rasterShaderCode });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      }],
    });
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: rastUniformBuffer } }],
    });
    const pipeline = device.createRenderPipeline({
      label: "Rasterizer pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout, objectIdBufferLayout, vertexNormalsBufferLayout],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    return { pipeline, bindGroup };
  };

  const createRayTracePipeline = () => {
    const quadVertices = new Float32Array([
      -1, -1,  1, -1,  -1, 1,
       1, -1,  1,  1,  -1, 1,
    ]);
    const quadVertexBuffer = device.createBuffer({
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(quadVertexBuffer, 0, quadVertices);

    const shaderModule = device.createShaderModule({ label: "Shader", code: raytraceShaderCode });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ] as GPUBindGroupLayoutEntry[],
    });
    const bindGroup = device.createBindGroup({
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
      attributes: [{ format: "float32x2", offset: 0, shaderLocation: 0 }],
    };
    const pipeline = device.createRenderPipeline({
      label: "Raytraced pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: { module: shaderModule, entryPoint: "vertexMain", buffers: [quadBufferLayout] },
      fragment: { module: shaderModule, entryPoint: "fragmentMain", targets: [{ format: canvasFormat }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
    });
    return { pipeline, bindGroup, quadVertexBuffer };
  };

  const rast = createRasterPipeline();
  const ray = createRayTracePipeline();

  // ----- Depth texture -----

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // ----- Render state -----

  let destroyed = false;
  let useRaytracer = false;
  let animating = false;
  let animFrameId = 0;
  let frameCount = 0;
  const startTime = performance.now();

  const updateUniforms = () => {
    device.queue.writeBuffer(rastUniformBuffer, 0, packRasterUniforms());
    device.queue.writeBuffer(rayUniformBuffer, 0, packRayUniforms());
  };

  const renderFrame = (timestamp: number) => {
    if (destroyed) return;

    if (animating) {
      const elapsed = (timestamp - startTime) / 1000;
      const radius = 250;
      scene.lights[0].position.set([278 + radius * Math.sin(elapsed), 300, 280 + radius * Math.cos(elapsed)]);
      scene.lights[1].position.set([278 + radius * Math.sin(elapsed * 1.5), 400, 280 + radius * Math.cos(elapsed * 1.5)]);
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

    if (useRaytracer) {
      pass.setPipeline(ray.pipeline);
      pass.setBindGroup(0, ray.bindGroup);
      pass.setVertexBuffer(0, ray.quadVertexBuffer);
      pass.draw(6);
    } else {
      pass.setPipeline(rast.pipeline);
      pass.setBindGroup(0, rast.bindGroup);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setVertexBuffer(1, objectIdBuffer);
      pass.setVertexBuffer(2, normalBuffer);
      pass.setIndexBuffer(indexBuffer, "uint32");
      pass.drawIndexed(indexData.length);
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
    frameCount++;
  };

  // ----- Animation loop -----

  const animationLoop = (timestamp: number) => {
    if (destroyed) return;
    renderFrame(timestamp);
    if (animating) {
      animFrameId = requestAnimationFrame(animationLoop);
    }
  };

  // ----- FPS monitor -----
  // Passively counts frames rendered by any source (animation, material updates, etc.).
  // Only drives its own rAF loop when nothing else is rendering.

  let fpsTimeout = 0;
  let fpsMeasuring = false;
  let fpsAnimFrameId = 0;

  const startFPSMonitor = (onResult: (fps: number) => void) => {
    stopFPSMonitor();
    fpsMeasuring = true;

    const measure = () => {
      if (destroyed || !fpsMeasuring) return;
      const startCount = frameCount;
      const measureStart = performance.now();
      const duration = 2000;

      // If nothing else is driving renders, pump frames ourselves.
      const pumpLoop = (timestamp: number) => {
        if (destroyed || !fpsMeasuring) return;
        if (!animating) renderFrame(timestamp);
        if (performance.now() - measureStart < duration) {
          fpsAnimFrameId = requestAnimationFrame(pumpLoop);
        }
      };
      fpsAnimFrameId = requestAnimationFrame(pumpLoop);

      fpsTimeout = window.setTimeout(() => {
        if (destroyed || !fpsMeasuring) return;
        const elapsed = (performance.now() - measureStart) / 1000;
        const fps = (frameCount - startCount) / elapsed;
        console.log(`Average FPS: ${fps.toFixed(1)}`);
        onResult(fps);
        fpsTimeout = window.setTimeout(measure, 3000);
      }, duration);
    };

    fpsTimeout = window.setTimeout(measure, 500);
  };

  const stopFPSMonitor = () => {
    fpsMeasuring = false;
    clearTimeout(fpsTimeout);
    cancelAnimationFrame(fpsAnimFrameId);
    fpsAnimFrameId = 0;
  };

  // ----- Public API -----

  return {
    scene,

    render() {
      if (!destroyed) renderFrame(performance.now());
    },

    startAnimation() {
      animating = true;
      animFrameId = requestAnimationFrame(animationLoop);
    },

    stopAnimation() {
      animating = false;
      cancelAnimationFrame(animFrameId);
      animFrameId = 0;
    },

    setUseRaytracer(val: boolean) {
      useRaytracer = val;
    },

    updateMaterial(idx: number, rgb: [number, number, number]) {
      if (idx >= 0 && idx < scene.objects.length) {
        scene.objects[idx].material.diffuseAlbedo.set(rgb);
        if (!destroyed) renderFrame(performance.now());
      }
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
// React component
// ---------------------------------------------------------------------------

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
  const engineRef = useRef<Engine | null>(null);

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
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!navigator.gpu) {
      setWebgpuSupported(false);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const scene = buildScene(canvas);
        const engine = await createEngine(canvas, scene);
        if (cancelled) { engine.destroy(); return; }

        engineRef.current = engine;

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

        engine.startFPSMonitor(fps => setShowPerformanceWarning(fps < 10));
        engine.render();
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
  }, []);

  return (
    <div>
      <h1>WebGPU playground</h1>

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
            onChange={(e) => {
              const val = e.target.checked;
              setIsAnimating(val);
              if (val) engineRef.current?.startAnimation();
              else { engineRef.current?.stopAnimation(); engineRef.current?.render(); }
            }}
          />
          <label htmlFor="animatingCheckbox">Light animation</label>

          <input
            type="checkbox"
            id="raytracingCheckbox"
            checked={useRaytracer}
            onChange={(e) => {
              const val = e.target.checked;
              setUseRaytracer(val);
              engineRef.current?.setUseRaytracer(val);
              engineRef.current?.render();
            }}
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
                    const scene = engineRef.current?.scene;
                    if (scene && idx >= 0 && idx < scene.objects.length) {
                      const mat = scene.objects[idx].material.diffuseAlbedo;
                      const [l, a, b] = rgbToOklab(mat[0], mat[1], mat[2]);
                      setOklabL(l);
                      setOklabA(a);
                      setOklabB(b);
                    }
                  }}
                >
                  {engineRef.current?.scene.objects.map((obj, idx) => (
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
                  onChange={(e) => {
                    const l = parseFloat(e.target.value);
                    setOklabL(l);
                    const [r, g, b] = oklabToRgb(l, oklabA, oklabB);
                    engineRef.current?.updateMaterial(selectedObject, [r, g, b]);
                  }}
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
                  onChange={(e) => {
                    const a = parseFloat(e.target.value);
                    setOklabA(a);
                    const [r, g, b] = oklabToRgb(oklabL, a, oklabB);
                    engineRef.current?.updateMaterial(selectedObject, [r, g, b]);
                  }}
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
                  onChange={(e) => {
                    const bVal = parseFloat(e.target.value);
                    setOklabB(bVal);
                    const [r, g, b] = oklabToRgb(oklabL, oklabA, bVal);
                    engineRef.current?.updateMaterial(selectedObject, [r, g, b]);
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
