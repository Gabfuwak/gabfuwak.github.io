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
  triangleIndex: number;
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

      if (x < minCorner[0]) minCorner[0] = x;
      if (y < minCorner[1]) minCorner[1] = y;
      if (z < minCorner[2]) minCorner[2] = z;

      if (x > maxCorner[0]) maxCorner[0] = x;
      if (y > maxCorner[1]) maxCorner[1] = y;
      if (z > maxCorner[2]) maxCorner[2] = z;
    }
  }

  return { minCorner, maxCorner };
}

function partitionAlongMedian(mesh: Mesh, triangleSet: Uint32Array, axis: Axis) {
  const triangle_centers = new Float32Array(Array.from(
    triangleSet, (idx: number, _) => {
      const i0 = mesh.indices[idx*3];
      const i1 = mesh.indices[idx*3 + 1];
      const i2 = mesh.indices[idx*3 + 2];

      const p0 = mesh.positions[i0*3 + axis];
      const p1 = mesh.positions[i1*3 + axis];
      const p2 = mesh.positions[i2*3 + axis];

      const center = (p0 + p1 + p2) / 3;
      return center;
    }
  ));

  const pairs = Array.from(triangleSet, (tri, i) => ({ tri, center: triangle_centers[i] }));
  pairs.sort((a, b) => a.center - b.center);

  const triangleSetLeft = [];
  const triangleSetRight = [];

  for (let i = 0; i < pairs.length; i++) {
    if (i < pairs.length / 2) {
      triangleSetLeft.push(pairs[i].tri);
    } else {
      triangleSetRight.push(pairs[i].tri);
    }
  }

  return [new Uint32Array(triangleSetLeft), new Uint32Array(triangleSetRight)];
}

function getTriangleSetBVH(mesh: Mesh, triangleSet: Uint32Array): BVHNode {
  const box: AABB = getTriangleSetAABB(mesh, triangleSet);
  if (triangleSet.length <= 1) {
    return {
      kind: "leaf",
      boundingBox: box,
      triangleIndex: triangleSet.length === 1 ? triangleSet[0] : -1,
    };
  }

  const axis = dominantAxis(box);
  const [leftSet, rightSet] = partitionAlongMedian(mesh, triangleSet, axis);
  return {
    kind: "interior",
    boundingBox: box,
    left: getTriangleSetBVH(mesh, leftSet),
    right: getTriangleSetBVH(mesh, rightSet),
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

  return getTriangleSetBVH(worldMesh, triangleSet);
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
  triangleIndex: number;    // triangle index in mesh.indices, -1 if interior
}

function flattenNode(node: BVHNode, index: number, out: FlatNode[]): number {
  if (node.kind === "leaf") {
    out[index] = {
      minCorner: node.boundingBox.minCorner,
      maxCorner: node.boundingBox.maxCorner,
      isLeaf: true,
      leftChild: -1,
      rightChild: -1,
      triangleIndex: node.triangleIndex,
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
  };

  const rightChildIdx = flattenNode(node.left, index + 1, out);
  out[index].rightChild = rightChildIdx;
  return flattenNode(node.right, rightChildIdx, out);
}

export function flattenBVH(root: BVHNode): FlatNode[] {
  const out: FlatNode[] = [];
  flattenNode(root, 0, out);
  return out;
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
  return getTriangleSetBVH(mesh, triangleSet);
}
