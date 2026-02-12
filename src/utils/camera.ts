import { mat4Multiply, mat4Translate, mat4Perspective } from './mat4';

// Simple pinhole camera interface
export interface Camera {
  position: [number, number, number];
  target:   [number, number, number];
  up:       [number, number, number];
  fov:    number;
  aspect: number;
  near:   number;
  far:    number;
}

export function initCamera(
  canvas: HTMLCanvasElement,
  cameraPos: [number, number, number] = [0, 0, -4],
  target:    [number, number, number] = [0, 0,  0],
  up:        [number, number, number] = [0, 1,  0],
  fov:  number = Math.PI / 4,
  near: number = 0.1,
  far:  number = 100
): Camera {
  return { position: cameraPos, target, up, fov, aspect: canvas.width / canvas.height, near, far };
}

// Look-at view matrix (writes into out)
export function lookAt(
  out: Float32Array,
  eye:    [number, number, number],
  target: [number, number, number],
  up:     [number, number, number]
): void {
  let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
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

  out[0] = xx; out[1] = yx; out[2]  = zx; out[3]  = 0;
  out[4] = xy; out[5] = yy; out[6]  = zy; out[7]  = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
}

export function getMVP(camera: Camera): Float32Array {
  const model = mat4Translate(0, 0, 0);
  const view  = new Float32Array(16);
  lookAt(view, camera.position, camera.target, camera.up);
  const proj = mat4Perspective(camera.fov, camera.aspect, camera.near, camera.far);
  return mat4Multiply(proj, mat4Multiply(view, model));
}

export function getCameraBasis(camera: Camera): { forward: Float32Array, right: Float32Array, up: Float32Array } {
  const fx = camera.target[0] - camera.position[0];
  const fy = camera.target[1] - camera.position[1];
  const fz = camera.target[2] - camera.position[2];
  const flen = Math.hypot(fx, fy, fz);
  const forward = new Float32Array([fx / flen, fy / flen, fz / flen]);

  const rx = forward[1] * camera.up[2] - forward[2] * camera.up[1];
  const ry = forward[2] * camera.up[0] - forward[0] * camera.up[2];
  const rz = forward[0] * camera.up[1] - forward[1] * camera.up[0];
  const rlen = Math.hypot(rx, ry, rz);
  const right = new Float32Array([rx / rlen, ry / rlen, rz / rlen]);

  const ux = right[1] * forward[2] - right[2] * forward[1];
  const uy = right[2] * forward[0] - right[0] * forward[2];
  const uz = right[0] * forward[1] - right[1] * forward[0];
  const ulen = Math.hypot(ux, uy, uz);
  const up = new Float32Array([ux / ulen, uy / ulen, uz / ulen]);

  return { forward, right, up };
}

// Pan camera by (dx, dy) in camera-local space
export function pan(camera: Camera, dx: number, dy: number): void {
  const forwardX = camera.target[0] - camera.position[0];
  const forwardY = camera.target[1] - camera.position[1];
  const forwardZ = camera.target[2] - camera.position[2];
  const flen = Math.hypot(forwardX, forwardY, forwardZ);
  const fwdX = forwardX / flen, fwdY = forwardY / flen, fwdZ = forwardZ / flen;

  const rightX = fwdY * camera.up[2] - fwdZ * camera.up[1];
  const rightY = fwdZ * camera.up[0] - fwdX * camera.up[2];
  const rightZ = fwdX * camera.up[1] - fwdY * camera.up[0];
  const rlen = Math.hypot(rightX, rightY, rightZ);
  const rghtX = rightX / rlen, rghtY = rightY / rlen, rghtZ = rightZ / rlen;

  camera.position[0] += rghtX * dx + camera.up[0] * dy;
  camera.position[1] += rghtY * dx + camera.up[1] * dy;
  camera.position[2] += rghtZ * dx + camera.up[2] * dy;
  camera.target[0]   += rghtX * dx + camera.up[0] * dy;
  camera.target[1]   += rghtY * dx + camera.up[1] * dy;
  camera.target[2]   += rghtZ * dx + camera.up[2] * dy;
}
