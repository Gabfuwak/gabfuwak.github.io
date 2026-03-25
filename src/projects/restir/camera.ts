// === camera.ts ===
// Camera state + controls. No MVP — ray gen uses basis vectors directly.

export interface Camera {
  position: [number, number, number];
  target:   [number, number, number];
  up:       [number, number, number];
  fov:      number;
  aspect:   number;
}

export interface CameraBasis {
  forward: Float32Array;
  right:   Float32Array;
  up:      Float32Array;
}

export function initCamera(canvas: HTMLCanvasElement): Camera {
  return {
    position: [278, 273, -800],
    target:   [278, 273, -799],
    up:       [0, 1, 0],
    fov:      Math.PI / 3,
    aspect:   canvas.width / canvas.height,
  };
}

// Returns { forward, right, up } as Float32Arrays — used for uniform packing
export function getCameraBasis(camera: Camera): CameraBasis {
  const fx = camera.target[0] - camera.position[0];
  const fy = camera.target[1] - camera.position[1];
  const fz = camera.target[2] - camera.position[2];
  const flen = Math.hypot(fx, fy, fz);
  const forward = new Float32Array([fx/flen, fy/flen, fz/flen]);

  const rx = forward[1]*camera.up[2] - forward[2]*camera.up[1];
  const ry = forward[2]*camera.up[0] - forward[0]*camera.up[2];
  const rz = forward[0]*camera.up[1] - forward[1]*camera.up[0];
  const rlen = Math.hypot(rx, ry, rz);
  const right = new Float32Array([rx/rlen, ry/rlen, rz/rlen]);

  const ux = right[1]*forward[2] - right[2]*forward[1];
  const uy = right[2]*forward[0] - right[0]*forward[2];
  const uz = right[0]*forward[1] - right[1]*forward[0];
  const ulen = Math.hypot(ux, uy, uz);
  const up = new Float32Array([ux/ulen, uy/ulen, uz/ulen]);

  return { forward, right, up };
}

export function pan(camera: Camera, dx: number, dy: number): void {
  const fx = camera.target[0]-camera.position[0];
  const fy = camera.target[1]-camera.position[1];
  const fz = camera.target[2]-camera.position[2];
  const flen = Math.hypot(fx, fy, fz);
  const fwdX = fx/flen, fwdY = fy/flen, fwdZ = fz/flen;

  const rx = fwdY*camera.up[2] - fwdZ*camera.up[1];
  const ry = fwdZ*camera.up[0] - fwdX*camera.up[2];
  const rz = fwdX*camera.up[1] - fwdY*camera.up[0];
  const rlen = Math.hypot(rx, ry, rz);
  const rghtX = rx/rlen, rghtY = ry/rlen, rghtZ = rz/rlen;

  camera.position[0] += rghtX*dx + camera.up[0]*dy;
  camera.position[1] += rghtY*dx + camera.up[1]*dy;
  camera.position[2] += rghtZ*dx + camera.up[2]*dy;
  camera.target[0]   += rghtX*dx + camera.up[0]*dy;
  camera.target[1]   += rghtY*dx + camera.up[1]*dy;
  camera.target[2]   += rghtZ*dx + camera.up[2]*dy;
}

export function moveForward(camera: Camera, distance: number): void {
  const { forward } = getCameraBasis(camera);
  for (let i = 0; i < 3; i++) {
    camera.position[i] += forward[i] * distance;
    camera.target[i]   += forward[i] * distance;
  }
}

export function rotateYaw(camera: Camera, angle: number): void {
  const { forward } = getCameraBasis(camera);
  const dist = Math.hypot(
    camera.target[0]-camera.position[0],
    camera.target[1]-camera.position[1],
    camera.target[2]-camera.position[2],
  );
  const c = Math.cos(angle), s = Math.sin(angle);
  const fX = forward[0]*c + forward[2]*s;
  const fZ = -forward[0]*s + forward[2]*c;
  camera.target[0] = camera.position[0] + fX * dist;
  camera.target[1] = camera.position[1] + forward[1] * dist;
  camera.target[2] = camera.position[2] + fZ * dist;
}

export function rotatePitch(camera: Camera, angle: number): void {
  const { forward, up } = getCameraBasis(camera);
  const dist = Math.hypot(
    camera.target[0]-camera.position[0],
    camera.target[1]-camera.position[1],
    camera.target[2]-camera.position[2],
  );
  const c = Math.cos(angle), s = Math.sin(angle);
  const fX = forward[0]*c + up[0]*s;
  const fY = forward[1]*c + up[1]*s;
  const fZ = forward[2]*c + up[2]*s;
  const len = Math.hypot(fX, fY, fZ);
  camera.target[0] = camera.position[0] + (fX/len) * dist;
  camera.target[1] = camera.position[1] + (fY/len) * dist;
  camera.target[2] = camera.position[2] + (fZ/len) * dist;
}
