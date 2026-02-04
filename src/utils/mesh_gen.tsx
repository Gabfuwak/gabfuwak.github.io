export interface Mesh {
  positions: Float32Array<ArrayBuffer>;
  normals: Float32Array<ArrayBuffer>;
  uvs: Float32Array<ArrayBuffer>;
  colors: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
}

export function createSphere(
  radius: number,
  latitudeRes: number,
  longitudeRes: number,
  color: [number, number, number, number] = [1.0, 1.0, 1.0, 1.0]
): Mesh {
        const positions: number[] = [];
        const normals: number[] = [];
        const uvs: number[] = [];
        const indices: number[] = [];

        const colors: number[] = [];

        for (let lat = 0; lat <= latitudeRes; lat++) {
          const theta = lat * Math.PI / latitudeRes;
          const sinTheta = Math.sin(theta);
          const cosTheta = Math.cos(theta);

          for (let lon = 0; lon <= longitudeRes; lon++) {
            const phi = lon * 2 * Math.PI / longitudeRes;
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const x = cosPhi * sinTheta;
            const y = cosTheta;
            const z = sinPhi * sinTheta;

            positions.push(radius * x, radius * y, radius * z);
            normals.push(x, y, z);
            uvs.push(lon / longitudeRes, 1 - lat / latitudeRes);
            colors.push(...color);
          }
        }
      
        for (let lat = 0; lat < latitudeRes; lat++) {
          for (let lon = 0; lon < longitudeRes; lon++) {
            const first = lat * (longitudeRes + 1) + lon;
            const second = first + longitudeRes + 1;

            indices.push(
              first, second, first + 1,
              second, second + 1, first + 1
            );
          }
        }

        const mesh: Mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
        return mesh;
      }


export function createBox(
  width: number,
  height: number,
  length: number,
  color: [number, number, number, number] = [1.0, 1.0, 1.0, 1.0]
): Mesh {
        const w = width/2.0;
        const h = height/2.0;
        const l = length/2.0;
        const positions = [
          -w, -h,  l,
           w, -h,  l,
           w,  h,  l,
          -w,  h,  l,
          -w, -h, -l,
           w, -h, -l,
           w,  h, -l,
          -w,  h, -l,
        ];
        const den = Math.sqrt (w*w + h*h + l*l);
        const wn = w/den;
        const hn = h/den;
        const ln = l/den;
        const normals = [
          -wn, -hn,  ln,
           wn, -hn,  ln,
           wn,  hn,  ln,
          -wn,  hn,  ln,
          -wn, -hn, -ln,
           wn, -hn, -ln,
           wn,  hn, -ln,
          -wn,  hn, -ln,
        ];
        const uvs = [
          0.375, 0.750, 
          0.625, 0.750, 
          0.625, 1.0, 
          0.375, 1.0, 
          0.375, 0.250, 
          0.625, 0.250,  
          0.625, 0.5, 
          0.375, 0.5, 
        ];
        const indices = [
          0, 1, 2,
          0, 2, 3,
          1, 5, 6,
          1, 6, 2,
          5, 4, 7,
          5, 7, 6,
          4, 0, 3,
          4, 3, 7,
          3, 2, 6,
          3, 6, 7,
          4, 5, 1,
          4, 1, 0,
        ];
        const colors = new Array(8).fill(color).flat();
        const mesh: Mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
        return mesh;
      }

// Apply a 4x4 matrix to a mesh, returning a new transformed mesh
export function transformMesh(mesh: Mesh, m: Float32Array): Mesh {
  const newPos = new Float32Array(mesh.positions.length);
  const newNorm = new Float32Array(mesh.normals.length);
  
  // Transform positions (w=1)
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i];
    const y = mesh.positions[i+1];
    const z = mesh.positions[i+2];
    
    newPos[i]   = m[0]*x + m[4]*y + m[8]*z + m[12];
    newPos[i+1] = m[1]*x + m[5]*y + m[9]*z + m[13];
    newPos[i+2] = m[2]*x + m[6]*y + m[10]*z + m[14];
  }
  
  // Transform normals (upper 3x3 only, then normalize)
  // Note: This assumes no non-uniform scale. If you have scale, 
  // use inverse-transpose of upper 3x3 instead.
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const nx = mesh.normals[i];
    const ny = mesh.normals[i+1];
    const nz = mesh.normals[i+2];
    
    let x = m[0]*nx + m[4]*ny + m[8]*nz;
    let y = m[1]*nx + m[5]*ny + m[9]*nz;
    let z = m[2]*nx + m[6]*ny + m[10]*nz;
    
    const len = Math.sqrt(x*x + y*y + z*z);
    if (len > 0.0001) {
      x /= len; y /= len; z /= len;
    }
    
    newNorm[i] = x;
    newNorm[i+1] = y;
    newNorm[i+2] = z;
  }
  
  return {
    positions: newPos,
    normals: newNorm,
    colors: mesh.colors, // Colors don't transform
    uvs: mesh.uvs,
    indices: mesh.indices
  };
} 

export function merge_meshes(meshes: Mesh[]): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];

  let vertexOffset = 0;

  for (const mesh of meshes) {
    positions.push(...mesh.positions);
    normals.push(...mesh.normals);
    uvs.push(...mesh.uvs);
    colors.push(...mesh.colors);

    for (let i = 0; i < mesh.indices.length; i++) {
      indices.push(mesh.indices[i] + vertexOffset);
    }

    vertexOffset += mesh.positions.length / 3;
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    colors: new Float32Array(colors),
    indices: new Uint32Array(indices),
  };
}

export function create_quad(
  a: number[],
  b: number[],
  c: number[],
  d: number[],
  color: [number, number, number]
): Mesh {
          const cross = (a: Float32Array | number[], b: Float32Array | number[]): Float32Array => {
            const dst = new Float32Array(3);

            const t0 = a[1] * b[2] - a[2] * b[1];
            const t1 = a[2] * b[0] - a[0] * b[2];
            const t2 = a[0] * b[1] - a[1] * b[0];

            dst[0] = t0;
            dst[1] = t1;
            dst[2] = t2;

            return dst;
          }

          const subtract = (a: number[], b: number[]): Float32Array => {
            const dst = new Float32Array(3);

            dst[0] = a[0] - b[0];
            dst[1] = a[1] - b[1];
            dst[2] = a[2] - b[2];

            return dst;
          }

          const normalize = (v: Float32Array): Float32Array => {
            const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            return new Float32Array([v[0] / len, v[1] / len, v[2] / len]);
          }

          const positions = [
              a,
              b,
              c,
              d
          ].flat();

          const normal: number[] = Array.from(normalize(cross(subtract(a, b), subtract(a, d))));

          const normals = new Array(4).fill(normal).flat();

          const indices: number[] = [
              0, 1, 2,
              0, 2, 3,
          ]

          const uvs: number[] = [0,0, 1,0, 1,1, 0,1];

          const colors = [
              color,
              color,
              color,
              color
          ].flat()

        const mesh: Mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
        return mesh;

      }

export function createCornellBox(): Mesh {
  const white: [number, number, number] = [1.0, 1.0, 1.0];
  const red: [number, number, number] = [0.65, 0.05, 0.05];
  const green: [number, number, number] = [0.12, 0.45, 0.15];
  const light: [number, number, number] = [1.0, 1.0, 1.0];

  // Floor
  const floor = create_quad(
    [552.8, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 559.2],
    [549.6, 0.0, 559.2],
    white
  );

  // Ceiling
  const ceiling = create_quad(
    [556.0, 548.8, 0.0],
    [556.0, 548.8, 559.2],
    [0.0, 548.8, 559.2],
    [0.0, 548.8, 0.0],
    white
  );

  // Light
  const lightQuad = create_quad(
    [343.0, 548.8, 227.0],
    [343.0, 548.8, 332.0],
    [213.0, 548.8, 332.0],
    [213.0, 548.8, 227.0],
    light
  );

  // Back wall
  const backWall = create_quad(
    [549.6, 0.0, 559.2],
    [0.0, 0.0, 559.2],
    [0.0, 548.8, 559.2],
    [556.0, 548.8, 559.2],
    white
  );

  // Right wall (green)
  const rightWall = create_quad(
    [0.0, 0.0, 559.2],
    [0.0, 0.0, 0.0],
    [0.0, 548.8, 0.0],
    [0.0, 548.8, 559.2],
    green
  );

  // Left wall (red)
  const leftWall = create_quad(
    [552.8, 0.0, 0.0],
    [549.6, 0.0, 559.2],
    [556.0, 548.8, 559.2],
    [556.0, 548.8, 0.0],
    red
  );

  // Short block (5 faces)
  const shortBlockTop = create_quad(
    [130.0, 165.0, 65.0],
    [82.0, 165.0, 225.0],
    [240.0, 165.0, 272.0],
    [290.0, 165.0, 114.0],
    white
  );

  const shortBlockFront = create_quad(
    [290.0, 0.0, 114.0],
    [290.0, 165.0, 114.0],
    [240.0, 165.0, 272.0],
    [240.0, 0.0, 272.0],
    white
  );

  const shortBlockRight = create_quad(
    [130.0, 0.0, 65.0],
    [130.0, 165.0, 65.0],
    [290.0, 165.0, 114.0],
    [290.0, 0.0, 114.0],
    white
  );

  const shortBlockBack = create_quad(
    [82.0, 0.0, 225.0],
    [82.0, 165.0, 225.0],
    [130.0, 165.0, 65.0],
    [130.0, 0.0, 65.0],
    white
  );

  const shortBlockLeft = create_quad(
    [240.0, 0.0, 272.0],
    [240.0, 165.0, 272.0],
    [82.0, 165.0, 225.0],
    [82.0, 0.0, 225.0],
    white
  );

  // Tall block (5 faces)
  const tallBlockTop = create_quad(
    [423.0, 330.0, 247.0],
    [265.0, 330.0, 296.0],
    [314.0, 330.0, 456.0],
    [472.0, 330.0, 406.0],
    white
  );

  const tallBlockFront = create_quad(
    [423.0, 0.0, 247.0],
    [423.0, 330.0, 247.0],
    [472.0, 330.0, 406.0],
    [472.0, 0.0, 406.0],
    white
  );

  const tallBlockRight = create_quad(
    [472.0, 0.0, 406.0],
    [472.0, 330.0, 406.0],
    [314.0, 330.0, 456.0],
    [314.0, 0.0, 456.0],
    white
  );

  const tallBlockBack = create_quad(
    [314.0, 0.0, 456.0],
    [314.0, 330.0, 456.0],
    [265.0, 330.0, 296.0],
    [265.0, 0.0, 296.0],
    white
  );

  const tallBlockLeft = create_quad(
    [265.0, 0.0, 296.0],
    [265.0, 330.0, 296.0],
    [423.0, 330.0, 247.0],
    [423.0, 0.0, 247.0],
    white
  );

  return merge_meshes([
    floor,
    ceiling,
    lightQuad,
    backWall,
    rightWall,
    leftWall,
    shortBlockTop,
    shortBlockFront,
    shortBlockRight,
    shortBlockBack,
    shortBlockLeft,
    tallBlockTop,
    tallBlockFront,
    tallBlockRight,
    tallBlockBack,
    tallBlockLeft,
  ]);
}
