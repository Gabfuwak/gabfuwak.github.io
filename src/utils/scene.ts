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
}

export interface Material{
  diffuseAlbedo: Float32Array, // vec3f
}

export interface MergedScene {
  // Baked world-space geometry
  positions: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;

  // Metadata for materials/object identity
  objectIds: Uint32Array;

  // Materials array (one per object)
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
  let raw_meshes = [];
  let objectIds = [];
  let materials: Material[] = [];
  let object_id_counter = 0;
  let offset_counter = 0;
  let ranges: Array<Range> = [];
  for(let scene_object of scene.objects){
    const transformed = transformMesh(scene_object.mesh, scene_object.transform);
    raw_meshes.push(transformed);

    let nb_vtx = scene_object.mesh.positions.length / 3;
    objectIds.push( ... new Array(nb_vtx).fill(object_id_counter));
    materials.push(scene_object.material);
    ranges.push({
        objectId: object_id_counter,
        indexOffset: offset_counter,
        indexCount: scene_object.mesh.indices.length
    });

    offset_counter += scene_object.mesh.indices.length;
    object_id_counter++;
  }
  let merged_mesh = merge_meshes(raw_meshes);
  return{
    positions: merged_mesh.positions,
    indices: merged_mesh.indices,
    normals: merged_mesh.normals,
    objectIds: new Uint32Array(objectIds),
    materials: materials,
    ranges: ranges,
  };
}
