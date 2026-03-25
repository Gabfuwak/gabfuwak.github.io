// === bvh.ts ===
// SAH BVH build — writes directly into a flat number array, no intermediate tree.
// GPU node layout (8 values, stride 32 bytes):
//   [0..2] minCorner (f32)   [3] nbTris if leaf, 0 if interior (u32)
//   [4..6] maxCorner (f32)   [7] triangleIndex if leaf, rightChild if interior (u32)

export interface BVHMesh {
  positions: Float32Array;
  indices:   Uint32Array;
}

export interface AABB {
  minCorner: Float32Array;
  maxCorner: Float32Array;
}

export interface BVH {
  data:       number[];
  nodeCount:  number;
  primitives: Uint32Array;
}

export interface TLASResult {
  data:      number[];
  nodeCount: number;
}

// --- AABB helpers ---

function growAABB(x: number, y: number, z: number, box: AABB): void {
  if (x < box.minCorner[0]) box.minCorner[0] = x;
  if (y < box.minCorner[1]) box.minCorner[1] = y;
  if (z < box.minCorner[2]) box.minCorner[2] = z;
  if (x > box.maxCorner[0]) box.maxCorner[0] = x;
  if (y > box.maxCorner[1]) box.maxCorner[1] = y;
  if (z > box.maxCorner[2]) box.maxCorner[2] = z;
}

function getTriangleSetAABB(mesh: BVHMesh, triangleSet: Uint32Array | number[]): AABB {
  const minCorner = new Float32Array([Infinity, Infinity, Infinity]);
  const maxCorner = new Float32Array([-Infinity, -Infinity, -Infinity]);

  for (let i = 0; i < triangleSet.length; i++) {
    const t = triangleSet[i];
    for (let v = 0; v < 3; v++) {
      const vi = mesh.indices[t * 3 + v];
      const x = mesh.positions[vi * 3];
      const y = mesh.positions[vi * 3 + 1];
      const z = mesh.positions[vi * 3 + 2];
      growAABB(x, y, z, { minCorner, maxCorner });
    }
  }

  return { minCorner, maxCorner };
}

function boxSAHCost(box: AABB, triCount: number): number {
  const sx = box.maxCorner[0] - box.minCorner[0];
  const sy = box.maxCorner[1] - box.minCorner[1];
  const sz = box.maxCorner[2] - box.minCorner[2];
  return (sx * (sy + sz) + sy * sz) * triCount;
}

// --- SAH partition (12 buckets, one axis) ---

interface PartitionResult {
  left:  Uint32Array;
  right: Uint32Array;
  cost:  number;
}

function partitionSAHAxis(mesh: BVHMesh, triangleSet: Uint32Array, axis: number): PartitionResult {
  const n = triangleSet.length;

  let rangeMin = 1e30;
  let rangeMax = -1e30;

  const centers = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const tri = triangleSet[i];
    const i0 = mesh.indices[tri * 3];
    const i1 = mesh.indices[tri * 3 + 1];
    const i2 = mesh.indices[tri * 3 + 2];
    const c = (mesh.positions[i0 * 3 + axis] + mesh.positions[i1 * 3 + axis] + mesh.positions[i2 * 3 + axis]) / 3;
    centers[i] = c;
    if (c < rangeMin) rangeMin = c;
    if (c > rangeMax) rangeMax = c;
  }

  const nbBuckets = 12;

  // Degenerate case: all centroids identical
  if (rangeMax === rangeMin) {
    const mid = Math.floor(n / 2);
    return { left: triangleSet.slice(0, mid), right: triangleSet.slice(mid), cost: 1e30 };
  }

  const bucketCounts = new Uint32Array(nbBuckets);
  const bucketMins = new Float32Array(nbBuckets * 3).fill(Infinity);
  const bucketMaxs = new Float32Array(nbBuckets * 3).fill(-Infinity);

  const scale = nbBuckets / (rangeMax - rangeMin);
  const bucketIndices = new Uint8Array(n);

  // Populate buckets
  for (let i = 0; i < n; i++) {
    const bi = Math.min(Math.max(Math.floor((centers[i] - rangeMin) * scale), 0), nbBuckets - 1);
    bucketIndices[i] = bi;
    bucketCounts[bi]++;

    const tri = triangleSet[i];
    const b3 = bi * 3;
    for (let v = 0; v < 3; v++) {
      const vi = mesh.indices[tri * 3 + v];
      const x = mesh.positions[vi * 3];
      const y = mesh.positions[vi * 3 + 1];
      const z = mesh.positions[vi * 3 + 2];
      if (x < bucketMins[b3])   bucketMins[b3]   = x;
      if (y < bucketMins[b3+1]) bucketMins[b3+1] = y;
      if (z < bucketMins[b3+2]) bucketMins[b3+2] = z;
      if (x > bucketMaxs[b3])   bucketMaxs[b3]   = x;
      if (y > bucketMaxs[b3+1]) bucketMaxs[b3+1] = y;
      if (z > bucketMaxs[b3+2]) bucketMaxs[b3+2] = z;
    }
  }

  // Suffix sweep (right-to-left cumulative AABB + count)
  const suffixMins = new Float32Array(nbBuckets * 3).fill(Infinity);
  const suffixMaxs = new Float32Array(nbBuckets * 3).fill(-Infinity);
  const suffixCounts = new Uint32Array(nbBuckets);

  const last3 = (nbBuckets - 1) * 3;
  suffixMins[last3]   = bucketMins[last3];   suffixMins[last3+1] = bucketMins[last3+1]; suffixMins[last3+2] = bucketMins[last3+2];
  suffixMaxs[last3]   = bucketMaxs[last3];   suffixMaxs[last3+1] = bucketMaxs[last3+1]; suffixMaxs[last3+2] = bucketMaxs[last3+2];
  suffixCounts[nbBuckets - 1] = bucketCounts[nbBuckets - 1];

  for (let b = nbBuckets - 2; b >= 0; b--) {
    const b3 = b * 3, next3 = b3 + 3;
    suffixMins[b3]   = suffixMins[next3];   suffixMins[b3+1] = suffixMins[next3+1]; suffixMins[b3+2] = suffixMins[next3+2];
    suffixMaxs[b3]   = suffixMaxs[next3];   suffixMaxs[b3+1] = suffixMaxs[next3+1]; suffixMaxs[b3+2] = suffixMaxs[next3+2];
    if (bucketCounts[b] > 0) {
      if (bucketMins[b3]   < suffixMins[b3])   suffixMins[b3]   = bucketMins[b3];
      if (bucketMins[b3+1] < suffixMins[b3+1]) suffixMins[b3+1] = bucketMins[b3+1];
      if (bucketMins[b3+2] < suffixMins[b3+2]) suffixMins[b3+2] = bucketMins[b3+2];
      if (bucketMaxs[b3]   > suffixMaxs[b3])   suffixMaxs[b3]   = bucketMaxs[b3];
      if (bucketMaxs[b3+1] > suffixMaxs[b3+1]) suffixMaxs[b3+1] = bucketMaxs[b3+1];
      if (bucketMaxs[b3+2] > suffixMaxs[b3+2]) suffixMaxs[b3+2] = bucketMaxs[b3+2];
    }
    suffixCounts[b] = suffixCounts[b + 1] + bucketCounts[b];
  }

  // Prefix sweep to find best split
  let bestSplit = 1;
  let bestCost = 1e30;

  {
    let lMinX = bucketMins[0], lMinY = bucketMins[1], lMinZ = bucketMins[2];
    let lMaxX = bucketMaxs[0], lMaxY = bucketMaxs[1], lMaxZ = bucketMaxs[2];
    let leftCount = bucketCounts[0];

    for (let b = 1; b < nbBuckets; b++) {
      const s3 = b * 3;
      const lsx = lMaxX - lMinX, lsy = lMaxY - lMinY, lsz = lMaxZ - lMinZ;
      const rsx = suffixMaxs[s3] - suffixMins[s3], rsy = suffixMaxs[s3+1] - suffixMins[s3+1], rsz = suffixMaxs[s3+2] - suffixMins[s3+2];
      const cost = (lsx*(lsy+lsz) + lsy*lsz) * leftCount + (rsx*(rsy+rsz) + rsy*rsz) * suffixCounts[b];
      if (cost < bestCost) { bestCost = cost; bestSplit = b; }
      if (bucketCounts[b] > 0) {
        if (bucketMins[s3]   < lMinX) lMinX = bucketMins[s3];
        if (bucketMins[s3+1] < lMinY) lMinY = bucketMins[s3+1];
        if (bucketMins[s3+2] < lMinZ) lMinZ = bucketMins[s3+2];
        if (bucketMaxs[s3]   > lMaxX) lMaxX = bucketMaxs[s3];
        if (bucketMaxs[s3+1] > lMaxY) lMaxY = bucketMaxs[s3+1];
        if (bucketMaxs[s3+2] > lMaxZ) lMaxZ = bucketMaxs[s3+2];
      }
      leftCount += bucketCounts[b];
    }
  }

  // Partition triangleSet by best bucket split
  let leftCount = 0;
  for (let i = 0; i < n; i++) {
    if (bucketIndices[i] < bestSplit) leftCount++;
  }

  const left = new Uint32Array(leftCount);
  const right = new Uint32Array(n - leftCount);
  let li = 0, ri = 0;
  for (let i = 0; i < n; i++) {
    if (bucketIndices[i] < bestSplit) left[li++] = triangleSet[i];
    else right[ri++] = triangleSet[i];
  }

  return { left, right, cost: bestCost };
}

// --- SAH partition (try all 3 axes, pick cheapest) ---

function partitionSAH(mesh: BVHMesh, triangleSet: Uint32Array): { partition: [Uint32Array, Uint32Array]; cost: number } {
  let best = partitionSAHAxis(mesh, triangleSet, 0);
  for (const axis of [1, 2]) {
    const candidate = partitionSAHAxis(mesh, triangleSet, axis);
    if (candidate.cost < best.cost) best = candidate;
  }
  return { partition: [best.left, best.right], cost: best.cost };
}

// --- Entry point: build BVH from a mesh, writing directly into a flat number array ---
// mesh: { positions: Float32Array, indices: Uint32Array }
// Returns: { data: number[], nodeCount: number, primitives: Uint32Array }

export function buildBVH(mesh: BVHMesh): BVH {
  const STRIDE = 8;
  const data: number[] = [];
  const primitives: number[] = [];
  let nodeCount = 0;

  function recurse(triangleSet: Uint32Array, depth: number): void {
    const idx = nodeCount++;
    const b = idx * STRIDE;

    const box = getTriangleSetAABB(mesh, triangleSet);

    function writeLeaf(): void {
      const primStart = primitives.length;
      for (const t of triangleSet) primitives.push(t);
      data[b]   = box.minCorner[0]; data[b+1] = box.minCorner[1]; data[b+2] = box.minCorner[2];
      data[b+3] = triangleSet.length;
      data[b+4] = box.maxCorner[0]; data[b+5] = box.maxCorner[1]; data[b+6] = box.maxCorner[2];
      data[b+7] = primStart;
    }

    if (triangleSet.length <= 1 || depth >= 32) { writeLeaf(); return; }

    const { partition, cost } = partitionSAH(mesh, triangleSet);
    const [left, right] = partition;
    if (left.length === 0 || right.length === 0) { writeLeaf(); return; }
    if (cost + boxSAHCost(box, 1) > boxSAHCost(box, triangleSet.length)) { writeLeaf(); return; }

    // Interior: write AABB + nbTris=0, backpatch rightChild after left subtree
    data[b]   = box.minCorner[0]; data[b+1] = box.minCorner[1]; data[b+2] = box.minCorner[2];
    data[b+3] = 0;
    data[b+4] = box.maxCorner[0]; data[b+5] = box.maxCorner[1]; data[b+6] = box.maxCorner[2];

    recurse(left, depth + 1);
    data[b+7] = nodeCount; // right child index (left subtree is done)
    recurse(right, depth + 1);
  }

  const nbTri = mesh.indices.length / 3;
  const all = new Uint32Array(nbTri);
  for (let i = 0; i < nbTri; i++) all[i] = i;
  recurse(all, 0);

  return { data, nodeCount, primitives: new Uint32Array(primitives) };
}

// Build TLAS over pre-computed world-space AABBs (one per instance).
// aabbs: [{ minCorner: Float32Array[3], maxCorner: Float32Array[3] }]
// Returns { data, nodeCount } — same flat format, but leaves store instance index at [b+7].
export function buildTLAS(aabbs: AABB[]): TLASResult {
  const STRIDE = 8;
  const data: number[] = [];
  let nodeCount = 0;

  function recurse(instanceSet: number[]): void {
    const idx = nodeCount++;
    const b = idx * STRIDE;

    // Merge AABBs of all instances in this set
    const min = [1e30, 1e30, 1e30], max = [-1e30, -1e30, -1e30];
    for (const i of instanceSet) {
      for (let k = 0; k < 3; k++) {
        if (aabbs[i].minCorner[k] < min[k]) min[k] = aabbs[i].minCorner[k];
        if (aabbs[i].maxCorner[k] > max[k]) max[k] = aabbs[i].maxCorner[k];
      }
    }
    data[b]   = min[0]; data[b+1] = min[1]; data[b+2] = min[2];
    data[b+4] = max[0]; data[b+5] = max[1]; data[b+6] = max[2];

    if (instanceSet.length === 1) {
      data[b+3] = 1;                // leaf
      data[b+7] = instanceSet[0];  // instance index
      return;
    }

    // Median split along longest axis
    const axis = [max[0]-min[0], max[1]-min[1], max[2]-min[2]].indexOf(
      Math.max(max[0]-min[0], max[1]-min[1], max[2]-min[2])
    );
    const sorted = instanceSet.slice().sort((a, bIdx) => {
      const ca = (aabbs[a].minCorner[axis] + aabbs[a].maxCorner[axis]) * 0.5;
      const cb = (aabbs[bIdx].minCorner[axis] + aabbs[bIdx].maxCorner[axis]) * 0.5;
      return ca - cb;
    });
    const mid = Math.floor(sorted.length / 2);

    data[b+3] = 0; // interior
    recurse(sorted.slice(0, mid));
    data[b+7] = nodeCount; // right child index (after left subtree is done)
    recurse(sorted.slice(mid));
  }

  recurse(Array.from({ length: aabbs.length }, (_, i) => i));
  return { data, nodeCount };
}
