// === scene.ts ===
// Geometry helpers + Cornell box scene definition

export interface Material {
  id:            number;
  diffuseAlbedo: number[];
  roughness:     number;
  metalness:     number;
  fresnel:       number[];
  emission:      number;
}

export interface MeshData {
  positions: Float32Array;
  normals:   Float32Array;
  uvs:       Float32Array;
  colors:    Float32Array;
  indices:   Uint32Array;
}

export interface SceneObject {
  mesh:      MeshData;
  material:  Material;
  transform: Float32Array;
  label?:    string;
}

export interface Light {
  position:  [number, number, number];
  color:     [number, number, number];
  intensity: number;
  direction: [number, number, number];
  angle:     number;
}

export interface Scene {
  objects: SceneObject[];
  lights:  Light[];
}

export interface Instance {
  meshIndex:  number;
  transform:  Float32Array;
  materialId: number;
  label:      string;
}

export interface SceneData {
  meshes:    MeshData[];
  instances: Instance[];
  materials: Material[];
}

export interface EmissiveList {
  buffer:     ArrayBuffer;
  count:      number;
  totalPower: number;
}

// --- Vec3 helpers ---

function _cross(a: number[], b: number[]): number[] {
  return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
}

function _sub(a: number[], b: number[]): number[] {
  return [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
}

function _normalize(v: number[]): number[] {
  const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  return [v[0]/len, v[1]/len, v[2]/len];
}

// --- Mat4 helpers (column-major) ---

function _mat4Identity(): Float32Array {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}
function _mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++)
    for (let row = 0; row < 4; row++)
      out[col*4+row] = a[0*4+row]*b[col*4+0] + a[1*4+row]*b[col*4+1] + a[2*4+row]*b[col*4+2] + a[3*4+row]*b[col*4+3];
  return out;
}
function _mat4Translate(x: number, y: number, z: number): Float32Array { const m = _mat4Identity(); m[12]=x; m[13]=y; m[14]=z; return m; }
function _mat4Scale(s: number): Float32Array            { const m = _mat4Identity(); m[0]=s; m[5]=s; m[10]=s; return m; }
function _mat4RotateX(r: number): Float32Array { const c=Math.cos(r), s=Math.sin(r), m=_mat4Identity(); m[5]=c; m[9]=-s; m[6]=s; m[10]=c; return m; }
function _mat4RotateY(r: number): Float32Array { const c=Math.cos(r), s=Math.sin(r), m=_mat4Identity(); m[0]=c; m[8]=s; m[2]=-s; m[10]=c; return m; }
function _mat4RotateZ(r: number): Float32Array { const c=Math.cos(r), s=Math.sin(r), m=_mat4Identity(); m[0]=c; m[4]=-s; m[1]=s; m[5]=c; return m; }

// Inverse of a TRS matrix (uniform scale assumed).
// inv(T*R*S) upper-3x3 = (R*S)^T / s^2, translation = -upper3x3_inv * t
export function _mat4InverseTRS(m: Float32Array): Float32Array {
  const s2 = m[0]*m[0] + m[1]*m[1] + m[2]*m[2]; // |col0|^2 = scale^2
  const inv = new Float32Array(16);
  inv[0]  = m[0]/s2;  inv[4]  = m[1]/s2;  inv[8]  = m[2]/s2;
  inv[1]  = m[4]/s2;  inv[5]  = m[5]/s2;  inv[9]  = m[6]/s2;
  inv[2]  = m[8]/s2;  inv[6]  = m[9]/s2;  inv[10] = m[10]/s2;
  inv[12] = -(inv[0]*m[12] + inv[4]*m[13] + inv[8]*m[14]);
  inv[13] = -(inv[1]*m[12] + inv[5]*m[13] + inv[9]*m[14]);
  inv[14] = -(inv[2]*m[12] + inv[6]*m[13] + inv[10]*m[14]);
  inv[15] = 1;
  return inv;
}

// Build a TRS matrix from { translation:[x,y,z], rotation:[rx,ry,rz], scale:s }
function get_mat({ translation: [tx,ty,tz], rotation: [rx,ry,rz], scale: s }: { translation: [number,number,number]; rotation: [number,number,number]; scale: number }): Float32Array {
  return _mat4Multiply(_mat4Translate(tx,ty,tz),
    _mat4Multiply(_mat4RotateY(ry),
      _mat4Multiply(_mat4RotateX(rx),
        _mat4Multiply(_mat4RotateZ(rz), _mat4Scale(s)))));
}

// --- Mesh primitives ---

// UV sphere centered at origin
function create_sphere(radius: number, latRes: number, lonRes: number, color: number[]): MeshData {
  const positions: number[] = [], normals: number[] = [], uvs: number[] = [], colors: number[] = [], indices: number[] = [];
  for (let lat = 0; lat <= latRes; lat++) {
    const theta = lat * Math.PI / latRes;
    const sinT = Math.sin(theta), cosT = Math.cos(theta);
    for (let lon = 0; lon <= lonRes; lon++) {
      const phi = lon * 2 * Math.PI / lonRes;
      const x = Math.cos(phi) * sinT, y = cosT, z = Math.sin(phi) * sinT;
      positions.push(radius * x, radius * y, radius * z);
      normals.push(x, y, z);
      uvs.push(lon / lonRes, 1 - lat / latRes);
      colors.push(color[0], color[1], color[2]);
    }
  }
  for (let lat = 0; lat < latRes; lat++) {
    for (let lon = 0; lon < lonRes; lon++) {
      const a = lat * (lonRes + 1) + lon;
      const b = a + lonRes + 1;
      indices.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  return {
    positions: new Float32Array(positions), normals: new Float32Array(normals),
    uvs: new Float32Array(uvs), colors: new Float32Array(colors), indices: new Uint32Array(indices),
  };
}

// 4 corners -> 2-triangle mesh with auto-computed flat normal
// a, b, c, d: [x,y,z] arrays, color: [r,g,b]
function create_quad(a: number[], b: number[], c: number[], d: number[], color: number[]): MeshData {
  const normal = _normalize(_cross(_sub(a, b), _sub(a, d)));

  return {
    positions: new Float32Array([...a, ...b, ...c, ...d]),
    normals:   new Float32Array([...normal, ...normal, ...normal, ...normal]),
    uvs:       new Float32Array([0,0, 1,0, 1,1, 0,1]),
    colors:    new Float32Array([...color, ...color, ...color, ...color]),
    indices:   new Uint32Array([0,1,2, 0,2,3]),
  };
}

function merge_meshes(meshes: MeshData[]): MeshData {
  const positions: number[] = [], normals: number[] = [], uvs: number[] = [], colors: number[] = [], indices: number[] = [];
  let vertexOffset = 0;

  for (const mesh of meshes) {
    for (let i = 0; i < mesh.positions.length; i++) positions.push(mesh.positions[i]);
    for (let i = 0; i < mesh.normals.length; i++)   normals.push(mesh.normals[i]);
    for (let i = 0; i < mesh.uvs.length; i++)        uvs.push(mesh.uvs[i]);
    for (let i = 0; i < mesh.colors.length; i++)     colors.push(mesh.colors[i]);
    for (let i = 0; i < mesh.indices.length; i++)    indices.push(mesh.indices[i] + vertexOffset);
    vertexOffset += mesh.positions.length / 3;
  }

  return {
    positions: new Float32Array(positions),
    normals:   new Float32Array(normals),
    uvs:       new Float32Array(uvs),
    colors:    new Float32Array(colors),
    indices:   new Uint32Array(indices),
  };
}

// Apply a column-major mat4 to a mesh.
// Positions: w=1, normals: upper 3x3 + renormalize.
function transformMesh(mesh: MeshData, m: Float32Array): MeshData {
  const newPos  = new Float32Array(mesh.positions.length);
  const newNorm = new Float32Array(mesh.normals.length);

  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i], y = mesh.positions[i+1], z = mesh.positions[i+2];
    newPos[i]   = m[0]*x + m[4]*y + m[8]*z  + m[12];
    newPos[i+1] = m[1]*x + m[5]*y + m[9]*z  + m[13];
    newPos[i+2] = m[2]*x + m[6]*y + m[10]*z + m[14];
  }

  for (let i = 0; i < mesh.normals.length; i += 3) {
    const nx = mesh.normals[i], ny = mesh.normals[i+1], nz = mesh.normals[i+2];
    let x = m[0]*nx + m[4]*ny + m[8]*nz;
    let y = m[1]*nx + m[5]*ny + m[9]*nz;
    let z = m[2]*nx + m[6]*ny + m[10]*nz;
    const len = Math.sqrt(x*x + y*y + z*z);
    if (len > 0.0001) { x /= len; y /= len; z /= len; }
    newNorm[i] = x; newNorm[i+1] = y; newNorm[i+2] = z;
  }

  return { positions: newPos, normals: newNorm, uvs: mesh.uvs, colors: mesh.colors, indices: mesh.indices };
}

// Parse a Wavefront OBJ string into a Mesh.
// Handles v/vt/vn, fan triangulation, flat normal fallback.
function load_mesh(obj_text: string, color: number[] = [1, 1, 1]): MeshData {
  const pos_raw: number[] = [], norm_raw: number[] = [], uv_raw: number[] = [];
  const positions: number[] = [], normals: number[] = [], uvs: number[] = [], colors: number[] = [];

  for (const line of obj_text.split('\n')) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'v')  { pos_raw.push(+parts[1], +parts[2], +parts[3]); }
    else if (parts[0] === 'vn') { norm_raw.push(+parts[1], +parts[2], +parts[3]); }
    else if (parts[0] === 'vt') { uv_raw.push(+parts[1], +parts[2]); }
    else if (parts[0] === 'f') {
      const verts = parts.slice(1);
      for (let i = 1; i < verts.length - 1; i++) {
        for (const token of [verts[0], verts[i], verts[i + 1]]) {
          const [p, t, n] = token.split('/');
          const pi = (parseInt(p) - 1) * 3;
          positions.push(pos_raw[pi], pos_raw[pi+1], pos_raw[pi+2]);
          if (n) { const ni = (parseInt(n) - 1) * 3; normals.push(norm_raw[ni], norm_raw[ni+1], norm_raw[ni+2]); }
          else   { normals.push(0, 0, 0); }
          if (t) { const ti = (parseInt(t) - 1) * 2; uvs.push(uv_raw[ti], uv_raw[ti+1]); }
          else   { uvs.push(0, 0); }
          colors.push(color[0], color[1], color[2]);
        }
      }
    }
  }

  // Compute flat normals when OBJ has none
  if (norm_raw.length === 0) {
    for (let i = 0; i < positions.length; i += 9) {
      const ax=positions[i+3]-positions[i],   ay=positions[i+4]-positions[i+1], az=positions[i+5]-positions[i+2];
      const bx=positions[i+6]-positions[i],   by=positions[i+7]-positions[i+1], bz=positions[i+8]-positions[i+2];
      const nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
      const len = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
      for (let v = 0; v < 3; v++) { normals[i+v*3]=nx/len; normals[i+v*3+1]=ny/len; normals[i+v*3+2]=nz/len; }
    }
  }

  const vertex_count = positions.length / 3;
  const indices = new Uint32Array(vertex_count);
  for (let i = 0; i < vertex_count; i++) indices[i] = i;

  return {
    positions: new Float32Array(positions),
    normals:   new Float32Array(normals),
    uvs:       new Float32Array(uvs),
    colors:    new Float32Array(colors),
    indices,
  };
}

// --- Scene definition ---

// Scene configs: camera position per scene
export const SCENE_CAMERAS: Record<string, { position: [number, number, number]; target: [number, number, number] }> = {
  cornell: { position: [278, 273, -800], target: [278, 273, -799] },
  sponza:  { position: [0, 400, 0],      target: [1, 400, 0] },
};

export async function buildScene(sceneName: string, opts: Record<string, unknown> = {}): Promise<Scene> {
  if (sceneName === "sponza") return buildSponzaScene(opts);
  return buildCornellScene(opts);
}

// Seeded PRNG so light positions are deterministic across rebuilds
function mulberry32(seed: number): () => number {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

async function buildSponzaScene(opts: Record<string, unknown> = {}): Promise<Scene> {
  const lightCount = (opts.lightCount as number) ?? 100;
  const sponzaObj = await fetch("/assets/sponza.obj").then(r => r.text());
  const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);

  const materials = {
    stone: { id:0, diffuseAlbedo:[0.76,0.70,0.58], roughness:0.7, metalness:0, fresnel:[0.04,0.04,0.04], emission:0 },
    light: { id:1, diffuseAlbedo:[1.0,0.85,0.6],   roughness:0,   metalness:0, fresnel:[0.04,0.04,0.04], emission:30.0 },
  };

  // Merge all light spheres into one mesh (avoids TLAS bloat)
  const baseSphere = create_sphere(15, 8, 8, [1.0, 0.85, 0.6]);
  const rng = mulberry32(42);
  const lightSpheres: MeshData[] = [];
  for (let i = 0; i < lightCount; i++) {
    const x = (rng() * 2 - 1) * 1500;
    const y = 50 + rng() * 1150;
    const z = (rng() * 2 - 1) * 900;
    lightSpheres.push(transformMesh(baseSphere, _mat4Translate(x, y, z)));
  }
  const mergedLights = merge_meshes(lightSpheres);

  const objects: SceneObject[] = [
    { mesh: load_mesh(sponzaObj, [0.76, 0.70, 0.58]),
      material: materials.stone, transform: IDENTITY, label: "Sponza" },
    { mesh: mergedLights,
      material: materials.light, transform: IDENTITY, label: "Lights" },
  ];

  return { objects, lights: [] };
}

async function buildCornellScene(opts: Record<string, unknown> = {}): Promise<Scene> {
  const dragonTris = (opts.dragonRes as string) || "2348";
  const dragonObj = await fetch(`/assets/dragon_${dragonTris}.obj`).then(r => r.text());

  const IDENTITY = new Float32Array([
    1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
  ]);

  const materials = {
    white:  { id:0, diffuseAlbedo:[1,1,1],          roughness:0,   metalness:0,   fresnel:[0.05,0.05,0.05], emission:0 },
    red:    { id:1, diffuseAlbedo:[0.65,0.05,0.05], roughness:0,   metalness:0,   fresnel:[0.05,0.05,0.05], emission:0 },
    green:  { id:2, diffuseAlbedo:[0.12,0.45,0.15], roughness:0,   metalness:0,   fresnel:[0.05,0.05,0.05], emission:0 },
    light:  { id:3, diffuseAlbedo:[1,1,1],          roughness:0,   metalness:0,   fresnel:[0.05,0.05,0.05], emission:30.0 },
    dragon: { id:4, diffuseAlbedo:[1.0,0.71,0.29],  roughness:0.3, metalness:1.0, fresnel:[1.0,0.71,0.29],  emission:0 },
  };

  const w = [1,1,1], r = [0.65,0.05,0.05], g = [0.12,0.45,0.15];

  const objects: SceneObject[] = [
    // Cornell box walls — one object per material
    { mesh: create_quad([552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2], w),
      material: materials.white, transform: IDENTITY, label: "Floor" },
    { mesh: create_quad([556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0], w),
      material: materials.white, transform: IDENTITY, label: "Ceiling" },
    { mesh: create_quad([549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2], w),
      material: materials.white, transform: IDENTITY, label: "Back wall" },
    { mesh: create_quad([0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2], g),
      material: materials.green, transform: IDENTITY, label: "Right wall" },
    { mesh: create_quad([552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0], r),
      material: materials.red, transform: IDENTITY, label: "Left wall" },
    { mesh: create_quad([556,548.8,0],[0,548.8,0],[0,0,0],[552.8,0,0], w),
      material: materials.white, transform: IDENTITY, label: "Front wall" },

    // Light quad (slightly below ceiling to avoid z-fighting)
    { mesh: create_quad([343,548,227],[343,548,332],[213,548,332],[213,548,227], w),
      material: materials.light, transform: IDENTITY, label: "Light" },

    // Dragon
    { mesh: load_mesh(dragonObj),
      material: materials.dragon,
      transform: get_mat({ translation:[279,115,269], rotation:[0, Math.PI/4, 0], scale:2 }),
      label: "Dragon" },
  ];

  const lights: Light[] = [];

  return { objects, lights };
}

// --- Emissive triangle list for NEE light sampling ---
// Called after BVH build — uses unified allIndices/allPositions (post-reorder).
// Returns { buffer: ArrayBuffer, count, totalPower }.
// GPU layout per entry: [triIndex(u32), instanceIndex(u32), area(f32), cdf(f32)]
// meshIndexOffsets[meshIndex] = triangle offset into unified index buffer for that mesh.

export function buildEmissiveTriangleList(
  sceneData: SceneData,
  allIndices: Uint32Array,
  allPositions: Float32Array,
  meshIndexOffsets: number[]
): EmissiveList {
  const entries: { triIndex: number; instanceIndex: number; area: number; power: number; cdf: number }[] = [];

  for (let ii = 0; ii < sceneData.instances.length; ii++) {
    const inst = sceneData.instances[ii];
    const mat = sceneData.materials[inst.materialId];
    if (!mat.emission || mat.emission <= 0) continue;

    const mesh = sceneData.meshes[inst.meshIndex];
    const triCount = mesh.indices.length / 3;
    const iOff = meshIndexOffsets[inst.meshIndex];

    for (let ti = 0; ti < triCount; ti++) {
      const triIdx = iOff + ti;
      const i0 = allIndices[triIdx * 3];
      const i1 = allIndices[triIdx * 3 + 1];
      const i2 = allIndices[triIdx * 3 + 2];

      const v0 = [allPositions[i0*3], allPositions[i0*3+1], allPositions[i0*3+2]];
      const v1 = [allPositions[i1*3], allPositions[i1*3+1], allPositions[i1*3+2]];
      const v2 = [allPositions[i2*3], allPositions[i2*3+1], allPositions[i2*3+2]];

      const e1 = _sub(v1, v0);
      const e2 = _sub(v2, v0);
      const cx = _cross(e1, e2);
      const area = 0.5 * Math.sqrt(cx[0]*cx[0] + cx[1]*cx[1] + cx[2]*cx[2]);

      const area_m2 = area / 1e6; // convert mm² → m² so flux is comparable to point lights (4π × intensity)
      const power = mat.emission * area_m2;
      entries.push({ triIndex: triIdx, instanceIndex: ii, area: area_m2, power, cdf: 0 });
    }
  }

  // Build CDF (prefix sum of power)
  let cumulative = 0;
  for (const e of entries) {
    cumulative += e.power;
    e.cdf = cumulative;
  }

  // Pack into GPU buffer
  const buf = new ArrayBuffer(entries.length * 16);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  for (let i = 0; i < entries.length; i++) {
    u32[i * 4]     = entries[i].triIndex;
    u32[i * 4 + 1] = entries[i].instanceIndex;
    f32[i * 4 + 2] = entries[i].area;
    f32[i * 4 + 3] = entries[i].cdf;
  }

  console.log(`Emissive triangles: ${entries.length}, total power: ${cumulative.toFixed(2)}`);
  return { buffer: buf, count: entries.length, totalPower: cumulative };
}

// --- Scene extraction ---
// Deduplicates meshes by reference so instances sharing geometry get one BLAS.
// Returns { meshes, instances, materials }
//   meshes[]:    { positions, normals, uvs, indices }
//   instances[]: { meshIndex, transform, materialId, label }
//   materials[]: dense palette indexed by material id

export function extractSceneData(scene: Scene): SceneData {
  const materialMap = new Map<number, Material>();
  for (const obj of scene.objects) materialMap.set(obj.material.id, obj.material);

  const maxId = materialMap.size === 0 ? -1 : Math.max(...materialMap.keys());
  const materials: Material[] = [];
  for (let i = 0; i <= maxId; i++) {
    materials.push(materialMap.get(i) ?? {
      id: i, diffuseAlbedo: [0,0,0], roughness: 0, metalness: 0, fresnel: [0,0,0], emission: 0,
    });
  }

  // Deduplicate meshes by object reference
  const meshMap = new Map<MeshData, number>(); // mesh ref -> meshIndex
  const meshes: MeshData[] = [];
  const instances: Instance[] = [];

  for (const obj of scene.objects) {
    let meshIndex = meshMap.get(obj.mesh);
    if (meshIndex === undefined) {
      meshIndex = meshes.length;
      meshMap.set(obj.mesh, meshIndex);
      meshes.push({
        positions: obj.mesh.positions,
        normals:   obj.mesh.normals,
        uvs:       obj.mesh.uvs,
        colors:    obj.mesh.colors,
        indices:   obj.mesh.indices,
      });
    }
    instances.push({
      meshIndex,
      transform:  obj.transform,
      materialId: obj.material.id,
      label:      obj.label ?? "",
    });
  }

  return { meshes, instances, materials };
}

// Re-export _mat4Translate for use in main.ts
export { _mat4Translate };
