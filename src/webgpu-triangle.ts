import { initWebGPU, initCamera, getMVP } from './utils/webgpu';
import shaderCode from './shaders/triangle.wgsl?raw';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const warningContainer = document.getElementById('warning-container')!;

function showWebGPUWarning() {
  warningContainer.innerHTML = `
    <div class="webgpu-warning">
      <p><strong>⚠ WebGPU is not supported or not enabled in your browser</strong></p>
      <p><strong>Chrome:</strong></p>
      <ul>
        <li>Go to <code>chrome://flags</code></li>
        <li>Search for "Unsafe WebGPU"</li>
        <li>Enable the flag and restart Chrome</li>
        <li>Check that Vulkan is enabled at <code>chrome://gpu</code></li>
      </ul>
      <p><strong>Firefox:</strong></p>
      <ul>
        <li>Go to <code>about:config</code></li>
        <li>Search for <code>dom.webgpu.enabled</code></li>
        <li>Set it to true and restart Firefox</li>
        <li><em>Note: Firefox on Linux does not support Vulkan, so WebGPU will be much slower</em></li>
      </ul>
    </div>`;
  canvas.style.display = 'none';
}

function showVulkanWarning() {
  warningContainer.innerHTML = `
    <div style="text-align:center;margin:0 auto 1rem;max-width:600px;">
      <p id="vulkan-toggle" style="color:red;font-size:0.9rem;cursor:pointer;border:1px solid red;background:#ffe6e6;padding:0.5rem;margin:0;">
        Running too slow? Click here.
      </p>
    </div>`;
  document.getElementById('vulkan-toggle')!.addEventListener('click', () => {
    warningContainer.innerHTML = `
      <div style="text-align:center;margin:0 auto 1rem;max-width:600px;">
        <div style="border:1px solid red;background:#ffe6e6;padding:1rem;text-align:left;">
          <p><strong>⚠ Low performance detected</strong></p>
          <p>Your setup may be too slow for complex renderings.</p>
          <p><strong>Chrome:</strong></p>
          <ul>
            <li>Go to <code>chrome://gpu</code> to check your graphics feature status</li>
            <li>Make sure hardware acceleration is enabled in Chrome settings</li>
            <li>Update your graphics drivers if Vulkan/D3D12 is disabled</li>
          </ul>
          <p><strong>Firefox on Linux:</strong></p>
          <ul>
            <li>Firefox does not support Vulkan on Linux. Too bad :(</li>
            <li>Consider using Chrome for better WebGPU performance</li>
          </ul>
          <button onclick="this.closest('div').parentElement.remove()">Close</button>
        </div>
      </div>`;
  });
}

async function main() {
  if (!navigator.gpu) {
    showWebGPUWarning();
    return;
  }

  let fpsCheckTimeout: number;

  try {
    // @ts-ignore
    const { device, context, _adapter } = await initWebGPU(canvas);
    const camera = initCamera(canvas);
    const mvpMatrix = getMVP(camera);
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    const vertexPositions = new Float32Array([
      -1.0, -1.0, 0.0,
       1.0, -1.0, 0.0,
       0.0,  1.0, 0.0,
    ]);
    const indexData = new Uint32Array([0, 1, 2]);
    const vertexColors = new Float32Array([
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
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
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // @ts-ignore
    device.queue.writeBuffer(mvpBuffer, 0, mvpMatrix);

    const cellShaderModule = device.createShaderModule({ label: "Cell shader", code: shaderCode });

    const cellPipeline = device.createRenderPipeline({
      label: "Cell pipeline",
      layout: "auto",
      vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [
          { arrayStride: 12, attributes: [{ format: "float32x3" as GPUVertexFormat, offset: 0, shaderLocation: 0 }] },
          { arrayStride: 12, attributes: [{ format: "float32x3" as GPUVertexFormat, offset: 0, shaderLocation: 1 }] },
        ],
      },
      fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{ format: canvasFormat }],
      },
    });

    const bindGroup = device.createBindGroup({
      layout: cellPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: mvpBuffer } }],
    });

    const draw = () => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0.4, a: 1 },
          storeOp: "store",
        }],
      });
      pass.setPipeline(cellPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.setVertexBuffer(1, vertexColorBuffer);
      pass.setIndexBuffer(indexBuffer, "uint32");
      pass.drawIndexed(indexData.length, 1);
      pass.end();
      device.queue.submit([encoder.finish()]);
    };

    draw();

    fpsCheckTimeout = window.setTimeout(() => {
      let frames = 0;
      const startTime = performance.now();
      const duration = 2000;

      const renderLoop = () => {
        draw();
        frames++;
        if (performance.now() - startTime < duration) {
          requestAnimationFrame(renderLoop);
        } else {
          const fps = (frames / (performance.now() - startTime)) * 1000;
          if (fps < 10) showVulkanWarning();
        }
      };
      requestAnimationFrame(renderLoop);
    }, 500);

  } catch (err) {
    console.error("WebGPU initialization failed:", err);
    showWebGPUWarning();
    clearTimeout(fpsCheckTimeout!);
  }
}

main();
