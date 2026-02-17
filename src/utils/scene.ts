import { type Camera } from './camera';
import { type Mesh, merge_meshes, transformMesh } from './mesh_gen'


export interface Scene{
  objects: SceneObject[],
  lights: Light[],
  camera: Camera,
}

export interface SceneObject{
  mesh: Mesh,
  material: Material,
  transform: Float32Array, // mat4x4
  label?: string,
}

export interface Light{
  position: Float32Array, // vec3f
  color: Float32Array, // vec3f
  intensity: number,
  direction: Float32Array, // vec3f
  angle: number, //f32 light opening in degrees. 360 => point light
}

export interface Material{
  id: number,

  diffuseAlbedo: Float32Array, // vec3f
  roughness: number, //f32
  fresnel: Float32Array, // vec3f
  metalness: number, //f32
}

export interface MergedScene {
  // Baked world-space geometry
  positions: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
  uvs: Float32Array;

  // Metadata for materials/object identity
  objectIds: Uint32Array;

  // Materials palette, indexed by material id
  materials: Material[];

  // Bookkeeping (kept CPU-side)
  ranges: Array<Range>;
}


export interface Range {
    objectId: number;
    indexOffset: number;
    indexCount: number;
}


export function extractSceneData(scene: Scene) : MergedScene {
  const materialMap = new Map<number, Material>();
  const raw_meshes: Mesh[] = [];
  const objectIds: number[] = [];
  const ranges: Range[] = [];
  let object_id_counter = 0;
  let offset_counter = 0;

  for (const scene_object of scene.objects) {
    const mat = scene_object.material;
    materialMap.set(mat.id, mat);

    const transformed = transformMesh(scene_object.mesh, scene_object.transform);
    raw_meshes.push(transformed);

    const nb_vtx = scene_object.mesh.positions.length / 3;
    for (let i = 0; i < nb_vtx; i++) objectIds.push(mat.id);

    ranges.push({
      objectId: object_id_counter,
      indexOffset: offset_counter,
      indexCount: scene_object.mesh.indices.length,
    });

    offset_counter += scene_object.mesh.indices.length;
    object_id_counter++;
  }

  // Build a dense array indexed by material id (fill gaps with a black default)
  const maxId = materialMap.size === 0 ? -1 : Math.max(...materialMap.keys());
  const materials: Material[] = [];
  for (let i = 0; i <= maxId; i++) {
    materials.push(materialMap.get(i) ?? {
      id: i,
      diffuseAlbedo: new Float32Array(3),
      roughness: 0,
      fresnel: new Float32Array(3),
      metalness: 0,
    });
  }

  const merged_mesh = merge_meshes(raw_meshes);
  return {
    positions: merged_mesh.positions,
    indices: merged_mesh.indices,
    normals: merged_mesh.normals,
    uvs: merged_mesh.uvs,
    objectIds: new Uint32Array(objectIds),
    materials,
    ranges,
  };
}
