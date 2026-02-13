import { useEffect, useState, useRef, useMemo } from 'react';
import { load_mesh, create_quad, get_mat } from '../../utils/mesh_gen';
import dragonObj from '../../assets/dragon_2348.obj?raw';
import { initWebGPU, initCamera, getCameraBasis, extractSceneData, getMVP} from '../../utils/webgpu';
import { type Scene, type Light, type Material} from '../../utils/scene';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';
import shaderCode from '../../shaders/assignment.wgsl?raw';
import noiseShaderCode from '../../shaders/noise.wgsl?raw';
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

  const boxCenter = [278, 274, 280];

  const normalize = (x: number, y: number, z: number): Float32Array => {
    const len = Math.sqrt(x * x + y * y + z * z);
    return new Float32Array([x / len, y / len, z / len]);
  };

  const lights: Light[] = [
    // Keylight
    {
        position: new Float32Array([71, 412, -140]),
        color: new Float32Array([1.0, 1.0, 1.0]),
        intensity: 1.5,
        direction: normalize(
          boxCenter[0] - 71,
          boxCenter[1] - 412,
          boxCenter[2] - (-140)
        ),
        angle: 360.0
    },
    // Fill light
    {
        position: new Float32Array([485, 137, -140]),
        color: new Float32Array([0.0, 0.2, 0.8]),
        intensity: 2.5,
        direction: normalize(
          boxCenter[0] - 485,
          boxCenter[1] - 137,
          boxCenter[2] - (-140)
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
      fresnel: new Float32Array([0.05, 0.05, 0.05]), // plastic
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
      roughness: 0,
      metalness: 0,
      fresnel: new Float32Array([0.05, 0.05, 0.05]),
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
    // Stanford dragon
    {
      mesh: load_mesh(dragonObj, [0.8, 0.2, 0.2]),
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
  updateMaterial(idx: number, rgb: [number, number, number], roughness?: number, metalness?: number): void;
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

  device.queue.writeBuffer(vertexBuffer, 0, vertexPositions.buffer, vertexPositions.byteOffset, vertexPositions.byteLength);
  device.queue.writeBuffer(indexBuffer, 0, indexData.buffer, indexData.byteOffset, indexData.byteLength);
  device.queue.writeBuffer(objectIdBuffer, 0, objectIds.buffer, objectIds.byteOffset, objectIds.byteLength);
  device.queue.writeBuffer(normalBuffer, 0, vertexNormals.buffer, vertexNormals.byteOffset, vertexNormals.byteLength);
  device.queue.writeBuffer(uvBuffer, 0, vertexUVs.buffer, vertexUVs.byteOffset, vertexUVs.byteLength);

  // ----- Uniforms -----

  const MAX_LIGHTS = 4;
  const MAX_MATERIALS = 16;
  const MATERIAL_SIZE = 8; // vec3 baseColor + roughness + vec3 fresnel + metalness = 8 floats
  const MVP_SIZE = 16;
  const SHARED_HEADER = 4; // camera_pos (vec3) + nbLights (f32)
  const LIGHTS_SIZE = MAX_LIGHTS * 12;
  const MATERIALS_HEADER = 4; // nbMaterials + 3 padding floats
  const RAY_CAMERA_SIZE = 12; // forward+fov, right+aspect, up+pad
  const RAY_OFFSET = MVP_SIZE + SHARED_HEADER + LIGHTS_SIZE + MATERIALS_HEADER + MAX_MATERIALS * MATERIAL_SIZE;
  const UNIFORM_LENGTH = RAY_OFFSET + RAY_CAMERA_SIZE;


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
    for (let i = 0; i < materials.length; i++) {
      const baseIdx = matOffset + 4 + i * MATERIAL_SIZE;
      out.set(materials[i].diffuseAlbedo, baseIdx);
      out[baseIdx + 3] = materials[i].roughness ?? 0;
      out.set(materials[i].fresnel, baseIdx + 4);
      out[baseIdx + 7] = materials[i].metalness ?? 0;
    }
  };

  const packUniforms = (): Float32Array<ArrayBuffer> => {
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
    data.set([...basis.up, 0], RAY_OFFSET + 8);
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
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

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

  // ----- Render state -----

  let destroyed = false;
  let useRaytracer = false;
  let animating = false;
  let animFrameId = 0;
  let frameCount = 0;
  const startTime = performance.now();

  const updateUniforms = () => {
    device.queue.writeBuffer(uniformBuffer, 0, packUniforms());
  };

  const renderFrame = (timestamp: number) => {
    if (destroyed) return;

    if (animating) {
      const elapsed = (timestamp - startTime) / 1000;
      const angle = elapsed * 1.25; // matches solution: angle = time_ms / 800
      const lx = 278 + 220 * Math.cos(angle);
      const lz = 280 + 220 * Math.sin(angle);
      const ly = 274 + 180 * Math.sin(angle);
      scene.lights[3].position.set([lx, ly, lz]);
      const dx = 278 - lx, dy = 274 - ly, dz = 280 - lz;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      scene.lights[3].direction.set([dx / len, dy / len, dz / len]);
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
      pass.setPipeline(rayPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6);
    } else {
      pass.setPipeline(rastPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(indexData.length);
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

    updateMaterial(idx: number, rgb: [number, number, number], roughness?: number, metalness?: number) {
      if (idx >= 0 && idx < scene.objects.length) {
        scene.objects[idx].material.diffuseAlbedo.set(rgb);
        if (roughness !== undefined) scene.objects[idx].material.roughness = roughness;
        if (metalness !== undefined) scene.objects[idx].material.metalness = metalness;
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
  const [roughness, setRoughness] = useState(0.5);
  const [metalness, setMetalness] = useState(0);

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
          const mat = scene.objects[firstLabeledIdx].material;
          const [l, a, b] = rgbToOklab(mat.diffuseAlbedo[0], mat.diffuseAlbedo[1], mat.diffuseAlbedo[2]);
          setSelectedObject(firstLabeledIdx);
          setOklabL(l);
          setOklabA(a);
          setOklabB(b);
          setRoughness(mat.roughness ?? 0.5);
          setMetalness(mat.metalness ?? 0);
        }
        setSceneReady(true);

        engine.startFPSMonitor(fps => setShowPerformanceWarning(fps < 30));
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
          <canvas ref={canvasRef} width="1024" height="1024" style={{ background: 'black', display: 'block', margin: '0 auto' }}></canvas>

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
                      const mat = scene.objects[idx].material;
                      const [l, a, b] = rgbToOklab(mat.diffuseAlbedo[0], mat.diffuseAlbedo[1], mat.diffuseAlbedo[2]);
                      setOklabL(l);
                      setOklabA(a);
                      setOklabB(b);
                      setRoughness(mat.roughness ?? 0.5);
                      setMetalness(mat.metalness ?? 0);
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
                    engineRef.current?.updateMaterial(selectedObject, [r, g, b], roughness, metalness);
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
                    engineRef.current?.updateMaterial(selectedObject, [r, g, b], roughness, metalness);
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
                    engineRef.current?.updateMaterial(selectedObject, [r, g, b], roughness, metalness);
                  }}
                  style={{ width: '100%' }}
                />
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
                    const [red, green, blue] = oklabToRgb(oklabL, oklabA, oklabB);
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
                    const [red, green, blue] = oklabToRgb(oklabL, oklabA, oklabB);
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
