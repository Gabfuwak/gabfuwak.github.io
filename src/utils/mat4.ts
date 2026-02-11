// Column-major 4x4 matrix utilities.
// All functions return a new Float32Array(16).

export function mat4Identity(): Float32Array {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
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

export function mat4Translate(x: number, y: number, z: number): Float32Array {
    const m = mat4Identity();
    m[12] = x; m[13] = y; m[14] = z;
    return m;
}

export function mat4Scale(s: number): Float32Array {
    const m = mat4Identity();
    m[0] = s; m[5] = s; m[10] = s;
    return m;
}

export function mat4RotateX(radians: number): Float32Array {
    const c = Math.cos(radians), s = Math.sin(radians);
    const m = mat4Identity();
    m[5] = c; m[9]  = -s;
    m[6] = s; m[10] =  c;
    return m;
}

export function mat4RotateY(radians: number): Float32Array {
    const c = Math.cos(radians), s = Math.sin(radians);
    const m = mat4Identity();
    m[0] =  c; m[8]  = s;
    m[2] = -s; m[10] = c;
    return m;
}

export function mat4RotateZ(radians: number): Float32Array {
    const c = Math.cos(radians), s = Math.sin(radians);
    const m = mat4Identity();
    m[0] = c; m[4] = -s;
    m[1] = s; m[5] =  c;
    return m;
}

export function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fovy * 0.5);
    const nf = 1.0 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0,  0,
        0,          f, 0,  0,
        0,          0, far * nf, -1,
        0,          0, (near * far) * nf, 0,
    ]);
}
