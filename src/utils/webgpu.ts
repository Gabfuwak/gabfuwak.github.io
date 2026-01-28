export async function initWebGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("Failed to get WebGPU context.");
  }
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });


  return { device, context, _adapter: adapter };
}



export function mat4Identity(): Float32Array {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}

export function mat4Translate(x: number, y: number, z: number): Float32Array {
    const m = mat4Identity();
    m[12] = x;
    m[13] = y;
    m[14] = z;
    return m;
}

export function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            out[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                    a[1 * 4 + row] * b[col * 4 + 1] +
                    a[2 * 4 + row] * b[col * 4 + 2] +
                    a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    return out;
}

export function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovy * 0.5);
    const nf = 1.0 / (near - far);

    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far * nf, -1,
        0, 0, (near * far) * nf, 0,
    ]);
}


export function initCamera (
  canvas: HTMLCanvasElement,
  cameraPos: [number, number, number] = [0, 0, -4],
  fov: number = Math.PI / 4,
  near: number = 0.1,
  far: number = 100
): Float32Array {
  const model = mat4Translate(0.0, 0.0, 0.0);
  const view  = mat4Translate(cameraPos[0], cameraPos[1], cameraPos[2]); // camera position (looking at 0,0,0)
  const aspect = canvas.width / canvas.height;
  const proj  = mat4Perspective(
      fov,
      aspect,
      near,
      far
  );
  // MVP = Projection * View * Model
  const mv  = mat4Multiply(view, model);
  const mvpMatrix = mat4Multiply(proj, mv);
  return mvpMatrix;

}
