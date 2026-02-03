import { useEffect, useState, useRef } from 'react';
import { createCornellBox } from '../../utils/mesh_gen';
import { initWebGPU, initCamera, initScene, getCameraBasis } from '../../utils/webgpu';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';

export default function Playground() {
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [showPerformanceWarning, setShowPerformanceWarning] = useState(false);

  const [isAnimating, setIsAnimating] = useState(false);

  const [useRaytracer, setUseRaytracer] = useState(false);

  const isAnimatingRef = useRef(false);
  const useRayTracerRef = useRef(false);

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
        const lightPos = new Float32Array([278, 530, 280]);

        const cornellbox = createCornellBox();

        const scene = initScene(device, camera, cornellbox, lightPos)


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
            lightPos.set([x, 300, z]);

            // Update rasterizer uniform buffer with updated light position
            const uniformData = new Float32Array(16 + 4);
            uniformData.set(scene.mvp, 0);
            uniformData.set(lightPos, 16);
            device.queue.writeBuffer(scene.uniformBuffer, 0, uniformData);

            // Update raytracer uniform buffer with updated light position
            const basis = getCameraBasis(scene.camera);
            const fovFactor = Math.tan(scene.camera.fov / 2);
            const rayUniformData = new Float32Array(20);
            rayUniformData.set([...scene.camera.position, fovFactor], 0);
            rayUniformData.set([...basis.forward, scene.camera.aspect], 4);
            rayUniformData.set([...basis.right, 0], 8);
            rayUniformData.set([...basis.up, 0], 12);
            rayUniformData.set([...lightPos, 0], 16);
            device.queue.writeBuffer(scene.rayUniformBuffer, 0, rayUniformData);
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
            pass.setPipeline(scene.rayPipeline);
            pass.setBindGroup(0, scene.rayBindGroup);
            pass.setVertexBuffer(0, scene.quadVertexBuffer);
            pass.draw(6); // 6 vertices, two triangles
          } else {
            pass.setPipeline(scene.rasterPipeline);
            pass.setBindGroup(0, scene.rasterBindGroup);
            pass.setVertexBuffer(0, scene.vertexBuffer);
            pass.setVertexBuffer(1, scene.colorBuffer);
            pass.setVertexBuffer(2, scene.normalBuffer);
            pass.setIndexBuffer(scene.indexBuffer, "uint32");
            pass.drawIndexed(scene.indexCount);
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
