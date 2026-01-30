import { useEffect, useState } from 'react';
import { createBox } from '../../utils/mesh_gen';
import { initWebGPU, initCamera, getMVP } from '../../utils/webgpu';
import shaderCode from '../../shaders/triangle.wgsl?raw';
import WebGPUWarning from '../../components/WebGPUWarning';
import VulkanWarning from '../../components/VulkanWarning';

export default function Assignment() {
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [showPerformanceWarning, setShowPerformanceWarning] = useState(false);

  useEffect(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;

    let fpsCheckTimeout: number;

    (async () => {
      try {
        // Check WebGPU support
        if (!navigator.gpu) {
          setWebgpuSupported(false);
          return;
        }

        console.time("WebGPU init");
        // @ts-ignore - keeping adapter for reference
        const { device, context, _adapter } = await initWebGPU(canvas);
        console.timeEnd("WebGPU init");

        const camera = initCamera(canvas);
        const mvpMatrix = getMVP(camera);

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        const box = createBox(1.0, 1.0, 1.0)

        const vertexPositions = box.positions;

        const indexData = box.indices;


        const vertexColors = new Float32Array ([
          // R, G, B
          1.0, 1.0, 1.0,
          1.0, 1.0, 0.0,
          1.0, 0.0, 1.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 1.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0,
          0.0, 0.0, 0.0,
        ]);

        const vertexBuffer = device.createBuffer({
          label: "Vertices",
          size: vertexPositions.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        const indexBuffer = device.createBuffer({
          label: "Vertex indices",
          size: indexData.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });


        const vertexColorBuffer = device.createBuffer({
          label: "Vertex colors",
          size: vertexColors.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(vertexBuffer, 0, vertexPositions);
        device.queue.writeBuffer(indexBuffer, 0, indexData);
        device.queue.writeBuffer(vertexColorBuffer, 0, vertexColors);


        const mvpBuffer = device.createBuffer({
          size: 64, // 16 floats made of 4 bytes each
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // @ts-ignore - Float32Array type compatibility issue
        device.queue.writeBuffer(mvpBuffer, 0, mvpMatrix);

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

        const cellShaderModule = device.createShaderModule({
          label: "Cell shader",
          code: shaderCode
        });


        const cellPipeline = device.createRenderPipeline({
          label: "Cell pipeline",
          layout: "auto",
          vertex: {
            module: cellShaderModule,
            entryPoint: "vertexMain",
            buffers: [vertexBufferLayout, vertexColorBufferLayout]
          },
          fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{
              format: canvasFormat
            }]
          }
        });

        const bindGroup = device.createBindGroup({
          layout: cellPipeline.getBindGroupLayout(0),
          entries: [{
            binding: 0,
            resource: { buffer: mvpBuffer },
          }],
        });

        const encoder = device.createCommandEncoder();

        const pass = encoder.beginRenderPass({
          colorAttachments: [{
             view: context.getCurrentTexture().createView(),
             loadOp: "clear",
             clearValue: { r: 0, g: 0, b: 0.4, a: 1 },
             storeOp: "store",
          }]
        });



        pass.setPipeline(cellPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.setVertexBuffer(0, vertexBuffer);
        pass.setVertexBuffer(1, vertexColorBuffer);
        pass.setIndexBuffer(indexBuffer, "uint32");
        pass.drawIndexed(indexData.length, 1);

        pass.end();

        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        // Measure FPS for performance warning
        const measureFPS = () => {
          let frames = 0;
          const startTime = performance.now();
          const duration = 2000; // Measure for 2 seconds

          const renderLoop = () => {
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
              colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0.4, a: 1 },
                storeOp: "store",
              }]
            });

            pass.setPipeline(cellPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.setVertexBuffer(0, vertexBuffer);
            pass.setVertexBuffer(1, vertexColorBuffer);
            pass.setIndexBuffer(indexBuffer, "uint32");
            pass.drawIndexed(indexData.length, 1);
            pass.end();

            const commandBuffer = encoder.finish();
            device.queue.submit([commandBuffer]);

            frames++;
            const elapsed = performance.now() - startTime;

            if (elapsed < duration) {
              requestAnimationFrame(renderLoop);
            } else {
              const fps = (frames / elapsed) * 1000;
              console.log(`Average FPS: ${fps.toFixed(1)}`);
              // Show warning if FPS is below 10
              if (fps < 10) {
                setShowPerformanceWarning(true);
              }
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
      if (fpsCheckTimeout) {
        clearTimeout(fpsCheckTimeout);
      }
    };
  }, []);

  return (
    <div>
      <h1>WebGPU Triangle</h1>
      <p><em>January 28, 2026</em></p>

      {webgpuSupported ? (
        <>
          {showPerformanceWarning && <VulkanWarning />}
          <canvas width="512" height="512" style={{ background: 'black', display: 'block', margin: '0 auto' }}></canvas>
        </>
      ) : (
        <WebGPUWarning />
      )}



    </div>
  );
}
