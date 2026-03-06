import { type SceneObject, type Scene, extractSceneData } from "./scene";
import { type Mesh } from "./mesh_gen";

export interface AABB{
  minCorner: Float32Array, // vec3f
  maxCorner: Float32Array, // vec3f
}

export interface WireframeMesh {
  positions: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
}

type BVHLeaf = {
  kind: "leaf";
  boundingBox: AABB;
  triangles: Uint32Array;
};

type BVHInterior = {
  kind: "interior";
  boundingBox: AABB;
  left: BVHNode;
  right: BVHNode;
};

export type BVHNode = BVHLeaf | BVHInterior;

export type Axis = 0 | 1 | 2;

export function dominantAxis(aabb: AABB): Axis {
  const dx = aabb.maxCorner[0] - aabb.minCorner[0];
  const dy = aabb.maxCorner[1] - aabb.minCorner[1];
  const dz = aabb.maxCorner[2] - aabb.minCorner[2];
  if (dx >= dy && dx >= dz)
    return 0;
  if (dy >= dz)
    return 1;

  return 2;
}

export function createAABBWireframe(aabbs: AABB[]): WireframeMesh {
  const positions: number[] = [];
  const indices: number[] = [];

  for (const aabb of aabbs) {
    const base = positions.length / 3;
    const mn = aabb.minCorner;
    const mx = aabb.maxCorner;

    positions.push(
      mn[0], mn[1], mn[2],  // 0
      mx[0], mn[1], mn[2],  // 1
      mn[0], mx[1], mn[2],  // 2
      mx[0], mx[1], mn[2],  // 3
      mn[0], mn[1], mx[2],  // 4
      mx[0], mn[1], mx[2],  // 5
      mn[0], mx[1], mx[2],  // 6
      mx[0], mx[1], mx[2],  // 7
    );

    const edges = [
      0,1, 1,3, 3,2, 2,0,  // bottom face
      4,5, 5,7, 7,6, 6,4,  // top face
      0,4, 1,5, 2,6, 3,7,  // vertical edges
    ];
    indices.push(...edges.map(i => base + i));
  }

  return {
    positions: new Float32Array(positions),
    indices: new Uint32Array(indices),
  };
}

export function getObjectAABB(object: SceneObject) : AABB{
  const { mesh, transform } = object;
  const minCorner = new Float32Array([ Infinity,  Infinity,  Infinity]);
  const maxCorner = new Float32Array([-Infinity, -Infinity, -Infinity]);

  for (let i = 0; i < mesh.positions.length; i += 3) {
    const x = mesh.positions[i];
    const y = mesh.positions[i + 1];
    const z = mesh.positions[i + 2];

    const wx = transform[0]*x + transform[4]*y + transform[8]*z  + transform[12];
    const wy = transform[1]*x + transform[5]*y + transform[9]*z  + transform[13];
    const wz = transform[2]*x + transform[6]*y + transform[10]*z + transform[14];

    if (wx < minCorner[0]) minCorner[0] = wx;
    if (wy < minCorner[1]) minCorner[1] = wy;
    if (wz < minCorner[2]) minCorner[2] = wz;

    if (wx > maxCorner[0]) maxCorner[0] = wx;
    if (wy > maxCorner[1]) maxCorner[1] = wy;
    if (wz > maxCorner[2]) maxCorner[2] = wz;
  }

  return { minCorner, maxCorner };
}

function growAABB(x: number, y: number, z: number, box: AABB): void{
  if (x < box.minCorner[0]) box.minCorner[0] = x;
  if (y < box.minCorner[1]) box.minCorner[1] = y;
  if (z < box.minCorner[2]) box.minCorner[2] = z;

  if (x > box.maxCorner[0]) box.maxCorner[0] = x;
  if (y > box.maxCorner[1]) box.maxCorner[1] = y;
  if (z > box.maxCorner[2]) box.maxCorner[2] = z;
}

function mergeAABB(box1: AABB, box2: AABB): void{
  const minX = box2.minCorner[0];
  const minY = box2.minCorner[1];
  const minZ = box2.minCorner[2];
  const maxX = box2.maxCorner[0];
  const maxY = box2.maxCorner[1];
  const maxZ = box2.maxCorner[2];

  growAABB(minX, minY, minZ, box1);
  growAABB(maxX, maxY, maxZ, box1);

}


function getTriangleSetAABB(mesh: Mesh, triangleSet: Uint32Array): AABB {
  const minCorner = new Float32Array([ Infinity,  Infinity,  Infinity]);
  const maxCorner = new Float32Array([-Infinity, -Infinity, -Infinity]);

  for (let i = 0; i < triangleSet.length; i++) {
    const t = triangleSet[i];
    for (let v = 0; v < 3; v++) {
      const vi = mesh.indices[t * 3 + v];
      const x = mesh.positions[vi * 3];
      const y = mesh.positions[vi * 3 + 1];
      const z = mesh.positions[vi * 3 + 2];

      growAABB(x, y, z, {minCorner, maxCorner});
    }
  }

  return { minCorner, maxCorner };
}

function boxSAHCost(box: AABB, tri_nb: number){
  let size_x = (box.maxCorner[0] - box.minCorner[0]); 
  let size_y = (box.maxCorner[1] - box.minCorner[1]);
  let size_z = (box.maxCorner[2] - box.minCorner[2])
  let half_surface_area = size_x * (size_y + size_z) + size_y * size_z;

  return half_surface_area * tri_nb;
}

function partitionSAHAxis(mesh: Mesh, triangleSet: Uint32Array, axis: Axis): { left: Uint32Array, right: Uint32Array, cost: number } {
  const n = triangleSet.length;

  let range_min = 1e30;
  let range_max = -1e30;

  const centers = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const tri = triangleSet[i];
    const i0 = mesh.indices[tri*3];
    const i1 = mesh.indices[tri*3 + 1];
    const i2 = mesh.indices[tri*3 + 2];
    const c = (mesh.positions[i0*3 + axis] + mesh.positions[i1*3 + axis] + mesh.positions[i2*3 + axis]) / 3;
    centers[i] = c;
    if(c < range_min) range_min = c;
    if(c > range_max) range_max = c;
  }



  const nb_buckets = 12;

  // fallback for degenerate case: all centroids identical, can't bucket-split
  if (range_max === range_min) {
    const mid = Math.floor(n / 2);
    return { left: triangleSet.slice(0, mid), right: triangleSet.slice(mid), cost: 1e30 };
  }

  const bucket_counts = new Uint32Array(nb_buckets);

  const bucketAABBs: AABB[] = Array.from({ length: nb_buckets }, () => ({
    minCorner: new Float32Array([Infinity, Infinity, Infinity]),
    maxCorner: new Float32Array([-Infinity, -Infinity, -Infinity]),
  }));

  // populate buckets
  for (let i = 0; i < n; i++) {
    const bucketIdx = Math.min(Math.max(Math.floor((centers[i] - range_min) / (range_max - range_min) * nb_buckets), 0), nb_buckets - 1);
    bucket_counts[bucketIdx]++;

    const tri = triangleSet[i];
    for (let v = 0; v < 3; v++) {
      const vi = mesh.indices[tri * 3 + v];
      const x = mesh.positions[vi * 3];
      const y = mesh.positions[vi * 3 + 1];
      const z = mesh.positions[vi * 3 + 2];
      growAABB(x, y, z, bucketAABBs[bucketIdx]);
    }
  }

  const suffixAABBs: AABB[] = Array.from({ length: nb_buckets }, () => ({
    minCorner: new Float32Array([Infinity, Infinity, Infinity]),
    maxCorner: new Float32Array([-Infinity, -Infinity, -Infinity]),
  }));
  const suffixCounts = new Uint32Array(nb_buckets);

  suffixAABBs[nb_buckets - 1] = {
    minCorner: new Float32Array(bucketAABBs[nb_buckets - 1].minCorner),
    maxCorner: new Float32Array(bucketAABBs[nb_buckets - 1].maxCorner),
  };
  suffixCounts[nb_buckets - 1] = bucket_counts[nb_buckets - 1];

  for (let b = nb_buckets - 2; b >= 0; b--) {
    suffixAABBs[b] = {
      minCorner: new Float32Array(suffixAABBs[b + 1].minCorner),
      maxCorner: new Float32Array(suffixAABBs[b + 1].maxCorner),
    };
    if (bucket_counts[b] > 0) mergeAABB(suffixAABBs[b], bucketAABBs[b]);
    suffixCounts[b] = suffixCounts[b + 1] + bucket_counts[b];
  }

  let best_split = 1; // bucket index
  let best_cost = 1e30;

  {
    let leftAABB = {
      minCorner: new Float32Array(bucketAABBs[0].minCorner),
      maxCorner: new Float32Array(bucketAABBs[0].maxCorner),
    };
    let leftCount = bucket_counts[0];

    for (let b = 1; b < nb_buckets; b++) {
      const cost = boxSAHCost(leftAABB, leftCount) + boxSAHCost(suffixAABBs[b], suffixCounts[b]);
      if (cost < best_cost) {
        best_cost = cost;
        best_split = b;
      }
      if (bucket_counts[b] > 0) mergeAABB(leftAABB, bucketAABBs[b]);
      leftCount += bucket_counts[b];
    }
  }

  // partition triangleSet into left/right based on best_split bucket
  let leftCount = 0;
  for (let i = 0; i < n; i++) {
    const b = Math.min(Math.floor((centers[i] - range_min) / (range_max - range_min) * nb_buckets), nb_buckets - 1);
    if (b < best_split) leftCount++;
  }

  const left = new Uint32Array(leftCount);
  const right = new Uint32Array(n - leftCount);
  let li = 0, ri = 0;
  for (let i = 0; i < n; i++) {
    const b = Math.min(Math.floor((centers[i] - range_min) / (range_max - range_min) * nb_buckets), nb_buckets - 1);
    if (b < best_split) left[li++] = triangleSet[i];
    else right[ri++] = triangleSet[i];
  }

  return { left, right, cost: best_cost };
}

function partitionSAH(mesh: Mesh, triangleSet: Uint32Array) {
  let best = partitionSAHAxis(mesh, triangleSet, 0);
  for (const axis of [1, 2] as Axis[]) {
    const candidate = partitionSAHAxis(mesh, triangleSet, axis);
    if (candidate.cost < best.cost) best = candidate;
  }
  return { partition: [best.left, best.right], cost: best.cost };
}

function getTriangleSetBVH(mesh: Mesh, triangleSet: Uint32Array, depth: number): BVHNode {
  const box: AABB = getTriangleSetAABB(mesh, triangleSet);
  if (triangleSet.length <= 1 || depth >= 32) {
    return {
      kind: "leaf",
      boundingBox: box,
      triangles: new Uint32Array(triangleSet),
    };
  }

  const partition_data = partitionSAH(mesh, triangleSet);
  const [leftSet, rightSet] = partition_data.partition;

  // degenerate split: one side is empty, can't recurse
  if (leftSet.length === 0 || rightSet.length === 0) {
    return {
      kind: "leaf",
      boundingBox: box,
      triangles: new Uint32Array(triangleSet),
    };
  }

  // if the partition + traversal cost (estimated with boxcost with one triangle) is more expensive than the current box, end it there
  if (partition_data.cost + boxSAHCost(box, 1) > boxSAHCost(box, triangleSet.length)) {
    return {
      kind: "leaf",
      boundingBox: box,
      triangles: new Uint32Array(triangleSet),
    };
  }


  return {
    kind: "interior",
    boundingBox: box,
    left: getTriangleSetBVH(mesh, leftSet, depth + 1),
    right: getTriangleSetBVH(mesh, rightSet, depth + 1),
  };
}

export function getObjectBVH(object: SceneObject): BVHNode {
  const t = object.transform;
  const src = object.mesh.positions;
  const worldPositions = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i], y = src[i+1], z = src[i+2];
    worldPositions[i]   = t[0]*x + t[4]*y + t[8]*z  + t[12];
    worldPositions[i+1] = t[1]*x + t[5]*y + t[9]*z  + t[13];
    worldPositions[i+2] = t[2]*x + t[6]*y + t[10]*z + t[14];
  }
  const worldMesh = { ...object.mesh, positions: worldPositions };

  const nb_triangles = object.mesh.indices.length / 3;
  const triangleSet = new Uint32Array(Array.from({ length: nb_triangles }, (_, i) => i));

  return getTriangleSetBVH(worldMesh, triangleSet, 0);
}

// ---------------------------------------------------------------------------
// BVH flattening
// ---------------------------------------------------------------------------

export interface FlatNode {
  minCorner: Float32Array;  // AABB min corner
  maxCorner: Float32Array;  // AABB max corner
  isLeaf: boolean;
  leftChild: number;        // index into flat array, -1 if leaf
  rightChild: number;       // index into flat array, -1 if leaf
  triangleIndex: number;    // start index into primitives array, -1 if interior
  nbTris: number;
}

function flattenNode(node: BVHNode, index: number, out: FlatNode[], primitives: number[]): number {
  if (node.kind === "leaf") {
    const primStart = primitives.length;
    for (const tri of node.triangles) primitives.push(tri);
    out[index] = {
      minCorner: node.boundingBox.minCorner,
      maxCorner: node.boundingBox.maxCorner,
      isLeaf: true,
      leftChild: -1,
      rightChild: -1,
      triangleIndex: primStart,
      nbTris: node.triangles.length,
    };
    return index + 1;
  }

  // Write interior node now, rightChild unknown — filled in after recursing left
  out[index] = {
    minCorner: node.boundingBox.minCorner,
    maxCorner: node.boundingBox.maxCorner,
    isLeaf: false,
    leftChild: index + 1,
    rightChild: -1,
    triangleIndex: -1,
    nbTris: -1
  };

  const rightChildIdx = flattenNode(node.left, index + 1, out, primitives);
  out[index].rightChild = rightChildIdx;
  return flattenNode(node.right, rightChildIdx, out, primitives);
}

export function flattenBVH(root: BVHNode): { nodes: FlatNode[]; primitives: Uint32Array } {
  const out: FlatNode[] = [];
  const prims: number[] = [];
  flattenNode(root, 0, out, prims);
  return { nodes: out, primitives: new Uint32Array(prims) };
}

// ---------------------------------------------------------------------------
// Scene-level BVH
// ---------------------------------------------------------------------------

export function buildSceneBVH(scene: Scene): BVHNode {
  const merged = extractSceneData(scene);
  const mesh: Mesh = {
    positions: new Float32Array(merged.positions),
    normals:   new Float32Array(merged.normals),
    uvs:       new Float32Array(merged.uvs),
    colors:    new Float32Array(0),
    indices:   new Uint32Array(merged.indices),
  };
  const nb_triangles = mesh.indices.length / 3;
  const triangleSet = new Uint32Array(Array.from({ length: nb_triangles }, (_, i) => i));
  return getTriangleSetBVH(mesh, triangleSet, 0);
}
