import type { Particle } from "./scene";
import { floor } from "./scene";
import type { Vec3 } from "./camera";

const solverIterations = 10;

export interface DistanceConstraint {
  a: number;
  b: number;
  restLength: number;
  iterationStiffness: number;
}

export function createParticles(): Particle[] {
  const edgeLength = 0.12; // Metres.
  const baseAltitude = Math.sqrt(3)*edgeLength/2;
  const positions: Vec3[] = [
    [0.29 - edgeLength/2, 0.29, -baseAltitude/3],
    [0.29 + edgeLength/2, 0.29, -baseAltitude/3],
    [0.29, 0.29, 2*baseAltitude/3],
    [0.29, 0.29 - Math.sqrt(2/3)*edgeLength, 0],
  ];

  return positions.map((center): Particle => ({
    center,
    predicted: [...center],
    velocity: [0.0, 0.0, 0.0],
    inverseMass: 1, // All vertices are free for the floor-contact experiment.
    radius: 0.01,
    materialId: 0,
  }));
}

export function createDistanceConstraints(particles: Particle[]): DistanceConstraint[] {
  // All six edges; capture rest lengths before the simulation starts.
  const edges: [number, number][] = [
    [0, 1], [0, 2], [0, 3],
    [1, 2], [1, 3], [2, 3],
  ];
  // FIXME: add this in material
  const stiffness = 0.8;
  // 3.3: correct stiffness for iteration count.
  const iterationStiffness = 1 - Math.pow(1 - stiffness, 1/solverIterations);
  return edges.map(([a, b]) => {
    const p1 = particles[a].center;
    const p2 = particles[b].center;
    return {
      a, b,
      restLength: Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]),
      iterationStiffness,
    };
  });
}

function projectDistance(particle1: Particle, particle2: Particle, restLength: number, stiffness: number) {
  const weightSum = particle1.inverseMass + particle2.inverseMass;
  if (weightSum === 0) {
    return; // Neither endpoint can move.
  }
  const p1 = particle1.predicted;
  const p2 = particle2.predicted;
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  const dz = p1[2] - p2[2];
  const distance = Math.hypot(dx, dy, dz);
  if (distance < 1e-10) {
    return; // The distance gradient is undefined for coincident points.
  }

  const C = distance - restLength;
  const scale = stiffness*C/(weightSum*distance);
  const scale1 = particle1.inverseMass*scale;
  const scale2 = particle2.inverseMass*scale;
  p1[0] -= scale1*dx;
  p1[1] -= scale1*dy;
  p1[2] -= scale1*dz;
  p2[0] += scale2*dx;
  p2[1] += scale2*dy;
  p2[2] += scale2*dz;
}

function projectFloor(predictedY: number, radius: number, floorHeight: number): number {
  const C = predictedY - floorHeight - radius;
  if (C < 0) {
    return floorHeight + radius;
  }
  return predictedY;
}

// PBD velocity update (paper, step 16): positions handle penetration;
// velocities handle bounce and sliding against this stationary floor.
function applyFloorVelocityResponse(particle: Particle, incomingNormalVelocity: number) {
  if (particle.inverseMass === 0 || particle.center[1] > floor.height + particle.radius) {
    return;
  }

  const bounceThreshold = 0.2; // m/s: do not turn resting gravity steps into tiny bounces.
  const bounceVelocity = incomingNormalVelocity < -bounceThreshold
    ? -floor.restitution*incomingNormalVelocity
    : 0;
  particle.velocity[1] = Math.max(particle.velocity[1], bounceVelocity);

  // Limit friction by the normal impulse per unit mass; never reverse sliding.
  const normalVelocityChange = particle.velocity[1] - incomingNormalVelocity;
  const tangentSpeed = Math.hypot(particle.velocity[0], particle.velocity[2]);
  if (normalVelocityChange > 0 && tangentSpeed > 0) {
    const scale = Math.max(0, 1 - floor.friction*normalVelocityChange/tangentSpeed);
    particle.velocity[0] *= scale;
    particle.velocity[2] *= scale;
  }
}

export function updateParticles(particles: Particle[], constraints: DistanceConstraint[], dt: number){
  let gravity : Vec3 = [0.0, -9.81, 0.0];
  const floorHeight = floor.height;

  // Predict future positions
  for (let particle of particles){
    if (particle.inverseMass === 0) {
      particle.predicted[0] = particle.center[0];
      particle.predicted[1] = particle.center[1];
      particle.predicted[2] = particle.center[2];
      continue;
    }
    particle.velocity[1] += gravity[1]*dt;
    particle.predicted[0] = particle.center[0] + particle.velocity[0]*dt;
    particle.predicted[1] = particle.center[1] + particle.velocity[1]*dt;
    particle.predicted[2] = particle.center[2] + particle.velocity[2]*dt;
  }

  // Gauss-Seidel correction
  for (let iteration = 0; iteration < solverIterations; iteration++){
    for (const constraint of constraints){
      projectDistance(
        particles[constraint.a],
        particles[constraint.b],
        constraint.restLength,
        constraint.iterationStiffness,
      );
    }
    for (let particle of particles){
      if (particle.inverseMass === 0) {
        continue;
      }
      particle.predicted[1] = projectFloor(particle.predicted[1], particle.radius, floorHeight);
    }
  }

  for (let particle of particles){
    // Keep the free-prediction velocity: projection can shorten the travelled distance.
    const incomingNormalVelocity = particle.velocity[1];
    // Reconstruct velocity before accepting the corrected position.
    particle.velocity[0] = (particle.predicted[0] - particle.center[0])/dt;
    particle.velocity[1] = (particle.predicted[1] - particle.center[1])/dt;
    particle.velocity[2] = (particle.predicted[2] - particle.center[2])/dt;
    particle.center[0] = particle.predicted[0];
    particle.center[1] = particle.predicted[1];
    particle.center[2] = particle.predicted[2];
    applyFloorVelocityResponse(particle, incomingNormalVelocity);
  }
}

