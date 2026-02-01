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



// Look-at matrix calculation
export function lookAt(
  out: Float32Array,
  eye: [number, number, number],
  target: [number, number, number],
  up: [number, number, number]
): void {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];

  let len = Math.hypot(zx, zy, zz);
  zx /= len; zy /= len; zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;

  len = Math.hypot(xx, xy, xz);
  xx /= len; xy /= len; xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
}

// Pan camera based on direction and distance
export function pan(camera: Camera, dx: number, dy: number): void {
  // Calculate forward direction
  const forwardX = camera.target[0] - camera.position[0];
  const forwardY = camera.target[1] - camera.position[1];
  const forwardZ = camera.target[2] - camera.position[2];

  // Normalize forward
  const forwardLen = Math.hypot(forwardX, forwardY, forwardZ);
  const fwdX = forwardX / forwardLen;
  const fwdY = forwardY / forwardLen;
  const fwdZ = forwardZ / forwardLen;

  // Calculate right vector (cross product of forward and up)
  const rightX = fwdY * camera.up[2] - fwdZ * camera.up[1];
  const rightY = fwdZ * camera.up[0] - fwdX * camera.up[2];
  const rightZ = fwdX * camera.up[1] - fwdY * camera.up[0];

  // Normalize right
  const rightLen = Math.hypot(rightX, rightY, rightZ);
  const rghtX = rightX / rightLen;
  const rghtY = rightY / rightLen;
  const rghtZ = rightZ / rightLen;

  // Move camera position and target
  camera.position[0] += rghtX * dx + camera.up[0] * dy;
  camera.position[1] += rghtY * dx + camera.up[1] * dy;
  camera.position[2] += rghtZ * dx + camera.up[2] * dy;

  camera.target[0] += rghtX * dx + camera.up[0] * dy;
  camera.target[1] += rghtY * dx + camera.up[1] * dy;
  camera.target[2] += rghtZ * dx + camera.up[2] * dy;
}

export function initCamera(
  canvas: HTMLCanvasElement,
  cameraPos: [number, number, number] = [0, 0, -4],
  target: [number, number, number] = [0, 0, 0],
  up: [number, number, number] = [0, 1, 0],
  fov: number = Math.PI / 4,
  near: number = 0.1,
  far: number = 100
): Camera {
  const aspect = canvas.width / canvas.height;
  return {
    position: cameraPos,
    target: target,
    up: up,
    fov,
    aspect,
    near,
    far,
  };
}

export function getMVP(camera: Camera): Float32Array {
  const model = mat4Translate(0.0, 0.0, 0.0);
  const view = new Float32Array(16);
  lookAt(view, camera.position, camera.target, camera.up);
  const proj = mat4Perspective(
    camera.fov,
    camera.aspect,
    camera.near,
    camera.far
  );
  // MVP = Projection * View * Model
  const mv = mat4Multiply(view, model);
  const mvpMatrix = mat4Multiply(proj, mv);
  return mvpMatrix;
}

// Simple pinhole camera interface
export interface Camera {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov: number;
  aspect: number;
  near: number;
  far: number;
}



