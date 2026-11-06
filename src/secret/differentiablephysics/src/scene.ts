import type { Vec3 } from "./camera";

export interface PointLight {
  position: Vec3;
  intensity: number;
  color: Vec3;
}

export interface Material {
  diffuseAlbedo: Vec3;
  roughness: number;
  fresnel: Vec3;
  metalness: number;
}

export interface Particle {
  center: Vec3;
  predicted: Vec3;
  velocity: Vec3;
  inverseMass: number; // Zero fixes the particle in place.
  radius: number;
  materialId: number;
}

// Faces reference particles, so rendering follows their simulated positions.
export interface Triangle {
  a: number;
  b: number;
  c: number;
  materialId: number;
}

export interface RenderOptions {
  showParticles: boolean;
  showTriangles: boolean;
  showFloor: boolean;
}

export const triangles: Triangle[] = [
  { a: 0, b: 2, c: 1, materialId: 0 },
  { a: 0, b: 1, c: 3, materialId: 0 },
  { a: 1, b: 2, c: 3, materialId: 0 },
  { a: 2, b: 0, c: 3, materialId: 0 },
];

// Shared by the collision constraint and the rendered horizontal plane.
export const floor = {
  height: 0,
  materialId: 1,
  restitution: 0.5, // Fraction of incoming normal speed returned by an impact.
  friction: 0.4, // Coulomb coefficient for tangential velocity reduction.
};

export const materials: Material[] = [
  { diffuseAlbedo: [0.2, 0.2, 0.9], roughness: 1.0, metalness: 0.0, fresnel: [0.2, 0.2, 0.9] },
  { diffuseAlbedo: [0.65, 0.65, 0.65], roughness: 1.0, metalness: 0.0, fresnel: [0.04, 0.04, 0.04] },
];

export const lights: PointLight[] = [
  { position: [0.05, 0.5, -0.3], intensity: 1.0, color: [1, 1, 1] },
];
