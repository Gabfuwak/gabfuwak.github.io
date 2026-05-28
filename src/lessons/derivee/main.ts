import { createPlot } from './plot';
import { createHLimitViz } from './h-limit';

// Step 0 — affine intro
const gAffine = document.getElementById('graph-affine');
if (gAffine) {
  createPlot(gAffine, {
    fn: (x) => 0.5 * x + 1,
    xDomain: [0, 4],
  });
}

// Step 2 — parabola intro (flat, non-orthonormal)
const gParabole = document.getElementById('graph-parabole');
if (gParabole) {
  createPlot(gParabole, {
    fn: (x) => x * x,
    xDomain: [0, 4],
    yDomain: [0, 10],
    orthonormal: false,
    height: 280,
  });
}

// Step 6 — Q3: secant converging to tangent at x=1 as h → 0
const gHLimit = document.getElementById('graph-h-limit');
if (gHLimit) createHLimitViz(gHLimit);

// Step 4 — affine with draggable secant
// The secant always coincides with the curve: pente stays 0,50 no matter where
// the handles are placed.
const gSecAffine = document.getElementById('graph-secante-affine');
if (gSecAffine) {
  createPlot(gSecAffine, {
    fn: (x) => 0.5 * x + 1,
    xDomain: [0, 4],
    orthonormal: false,
    height: 280,
    secant: { x1: 0.5, x2: 3.5, draggable: true },
  });
}

// Step 4 — parabola with draggable secant
// Initial position matches Q2 first calculation (t1=0, t2=3 → pente = 3).
// Student can drag to see pente change — e.g. move x1 to 1 to get pente = 4.
const gSecParabole = document.getElementById('graph-secante-parabole');
if (gSecParabole) {
  createPlot(gSecParabole, {
    fn: (x) => x * x,
    xDomain: [0, 4],
    yDomain: [0, 10],
    orthonormal: false,
    height: 280,
    secant: { x1: 0, x2: 3, draggable: true },
  });
}
