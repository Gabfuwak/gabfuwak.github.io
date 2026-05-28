// Visualization for Q3: secant from x=1 to x=1+h, with h controlled by a
// logarithmic slider from 2 down to ~0.001. Shows the secant rotating toward
// the tangent as h → 0.

const NS = 'http://www.w3.org/2000/svg';
const svgEl = <K extends keyof SVGElementTagNameMap>(name: K): SVGElementTagNameMap[K] =>
  document.createElementNS(NS, name);

const COLOR_CURVE  = '#0066cc';
const COLOR_FIXED  = '#0066cc';
const COLOR_SECANT = '#e67e22';
const COLOR_AXIS   = '#888';
const COLOR_TICK   = '#aaa';
const COLOR_LABEL  = '#444';
const COLOR_GRID   = '#eee';

function formatNum(v: number, decimals = 2): string {
  return v.toFixed(decimals).replace('.', ',');
}

function formatH(h: number): string {
  if (h >= 1)    return formatNum(h, 1);
  if (h >= 0.1)  return formatNum(h, 2);
  if (h >= 0.01) return formatNum(h, 3);
  return formatNum(h, 4);
}

function hFromSlider(t: number): number {
  // Logarithmic: t=0 → h=2, t=100 → h=0.001
  return Math.pow(10,
    Math.log10(2) * (1 - t / 100) + Math.log10(0.001) * (t / 100)
  );
}

export function createHLimitViz(container: HTMLElement) {
  const W = 360, H = 280;
  const ML = 40, MR = 20, MT = 20, MB = 35;
  const PW = W - ML - MR, PH = H - MT - MB;
  const xMin = 0, xMax = 4, yMin = 0, yMax = 10;

  const xToSvg = (x: number) => ML + ((x - xMin) / (xMax - xMin)) * PW;
  const yToSvg = (y: number) => MT + PH - ((y - yMin) / (yMax - yMin)) * PH;

  // ── SVG ────────────────────────────────────────────────
  const svg = svgEl('svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.cssText = 'display:block;width:100%;max-width:' + W + 'px;height:auto;user-select:none;margin:0 auto';

  // Clip path
  const uid = 'hl' + Math.random().toString(36).slice(2, 8);
  const defs = svgEl('defs');
  const clip = svgEl('clipPath');
  clip.setAttribute('id', uid);
  const cr = svgEl('rect');
  cr.setAttribute('x', String(ML)); cr.setAttribute('y', String(MT));
  cr.setAttribute('width', String(PW)); cr.setAttribute('height', String(PH));
  clip.appendChild(cr); defs.appendChild(clip); svg.appendChild(defs);

  // Grid
  const xTicks = [0, 1, 2, 3, 4];
  const yTicks = [0, 2, 4, 6, 8, 10];

  for (const t of xTicks) {
    const sx = xToSvg(t);
    const g = svgEl('line');
    g.setAttribute('x1', String(sx)); g.setAttribute('y1', String(MT));
    g.setAttribute('x2', String(sx)); g.setAttribute('y2', String(MT + PH));
    g.setAttribute('stroke', COLOR_GRID); g.setAttribute('stroke-width', '1');
    svg.appendChild(g);
  }
  for (const t of yTicks) {
    const sy = yToSvg(t);
    const g = svgEl('line');
    g.setAttribute('x1', String(ML)); g.setAttribute('y1', String(sy));
    g.setAttribute('x2', String(ML + PW)); g.setAttribute('y2', String(sy));
    g.setAttribute('stroke', COLOR_GRID); g.setAttribute('stroke-width', '1');
    svg.appendChild(g);
  }

  // Axes
  for (const [x1,y1,x2,y2] of [
    [ML, MT+PH, ML+PW, MT+PH],
    [ML, MT,    ML,    MT+PH],
  ] as [number,number,number,number][]) {
    const a = svgEl('line');
    a.setAttribute('x1',String(x1)); a.setAttribute('y1',String(y1));
    a.setAttribute('x2',String(x2)); a.setAttribute('y2',String(y2));
    a.setAttribute('stroke', COLOR_AXIS); a.setAttribute('stroke-width','1');
    svg.appendChild(a);
  }

  // Ticks + labels
  for (const t of xTicks) {
    const sx = xToSvg(t);
    const tk = svgEl('line');
    tk.setAttribute('x1',String(sx)); tk.setAttribute('y1',String(MT+PH));
    tk.setAttribute('x2',String(sx)); tk.setAttribute('y2',String(MT+PH+4));
    tk.setAttribute('stroke', COLOR_TICK); svg.appendChild(tk);
    const lb = svgEl('text');
    lb.setAttribute('x', String(sx)); lb.setAttribute('y', String(MT+PH+18));
    lb.setAttribute('text-anchor','middle'); lb.setAttribute('font-size','11');
    lb.setAttribute('font-family','system-ui,sans-serif'); lb.setAttribute('fill', COLOR_LABEL);
    lb.textContent = String(t); svg.appendChild(lb);
  }
  for (const t of yTicks) {
    const sy = yToSvg(t);
    const tk = svgEl('line');
    tk.setAttribute('x1',String(ML-4)); tk.setAttribute('y1',String(sy));
    tk.setAttribute('x2',String(ML));   tk.setAttribute('y2',String(sy));
    tk.setAttribute('stroke', COLOR_TICK); svg.appendChild(tk);
    const lb = svgEl('text');
    lb.setAttribute('x', String(ML-7)); lb.setAttribute('y', String(sy+4));
    lb.setAttribute('text-anchor','end'); lb.setAttribute('font-size','11');
    lb.setAttribute('font-family','system-ui,sans-serif'); lb.setAttribute('fill', COLOR_LABEL);
    lb.textContent = String(t); svg.appendChild(lb);
  }

  // Curve x²
  const pts: string[] = [];
  for (let i = 0; i <= 240; i++) {
    const x = xMin + (xMax - xMin) * i / 240;
    pts.push(`${xToSvg(x).toFixed(2)},${yToSvg(x * x).toFixed(2)}`);
  }
  const curve = svgEl('polyline');
  curve.setAttribute('points', pts.join(' '));
  curve.setAttribute('fill', 'none');
  curve.setAttribute('stroke', COLOR_CURVE);
  curve.setAttribute('stroke-width', '2');
  curve.setAttribute('stroke-linejoin', 'round');
  curve.setAttribute('clip-path', `url(#${uid})`);
  svg.appendChild(curve);

  // Secant line
  const secantLine = svgEl('line');
  secantLine.setAttribute('stroke', COLOR_SECANT);
  secantLine.setAttribute('stroke-width', '2');
  secantLine.setAttribute('clip-path', `url(#${uid})`);
  svg.appendChild(secantLine);

  // Point B (moving)
  const ptB = svgEl('circle');
  ptB.setAttribute('r', '5');
  ptB.setAttribute('fill', '#fff');
  ptB.setAttribute('stroke', COLOR_SECANT);
  ptB.setAttribute('stroke-width', '2');
  svg.appendChild(ptB);

  // Point A (fixed at x=1)
  const ptA = svgEl('circle');
  ptA.setAttribute('cx', String(xToSvg(1)));
  ptA.setAttribute('cy', String(yToSvg(1)));
  ptA.setAttribute('r', '5');
  ptA.setAttribute('fill', COLOR_FIXED);
  svg.appendChild(ptA);

  // Slope label (top-right)
  const slopeBg = svgEl('rect');
  slopeBg.setAttribute('x', String(W - MR - 120)); slopeBg.setAttribute('y', String(MT));
  slopeBg.setAttribute('width', '120'); slopeBg.setAttribute('height', '22');
  slopeBg.setAttribute('fill', '#fff'); slopeBg.setAttribute('opacity', '0.85');
  svg.appendChild(slopeBg);

  const slopeLabel = svgEl('text');
  slopeLabel.setAttribute('x', String(W - MR));
  slopeLabel.setAttribute('y', String(MT + 15));
  slopeLabel.setAttribute('text-anchor', 'end');
  slopeLabel.setAttribute('font-family', 'system-ui,sans-serif');
  slopeLabel.setAttribute('font-size', '13');
  slopeLabel.setAttribute('fill', COLOR_SECANT);
  slopeLabel.setAttribute('font-weight', '600');
  svg.appendChild(slopeLabel);

  container.appendChild(svg);

  // ── HTML controls ──────────────────────────────────────
  const controls = document.createElement('div');
  controls.className = 'h-controls';

  const hDisplay = document.createElement('div');
  hDisplay.className = 'h-display';

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '100';
  slider.value = '0';
  slider.className = 'h-slider-input';

  controls.appendChild(hDisplay);
  controls.appendChild(slider);
  container.appendChild(controls);

  // ── Update function ────────────────────────────────────
  function update(h: number) {
    const x2 = 1 + h;
    const y2 = x2 * x2;
    const ax = xToSvg(1),   ay = yToSvg(1);
    const bx = xToSvg(x2),  by = yToSvg(y2);

    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    const ext = 28;
    const ux = dx / len, uy = dy / len;

    secantLine.setAttribute('x1', String(ax - ux * ext));
    secantLine.setAttribute('y1', String(ay - uy * ext));
    secantLine.setAttribute('x2', String(bx + ux * ext));
    secantLine.setAttribute('y2', String(by + uy * ext));

    ptB.setAttribute('cx', String(bx));
    ptB.setAttribute('cy', String(by));

    const slope = (y2 - 1) / h;
    slopeLabel.textContent = `taux = ${formatNum(slope, 3)}`;
    hDisplay.textContent = `h = ${formatH(h)}`;
  }

  slider.addEventListener('input', () => {
    update(hFromSlider(parseInt(slider.value)));
  });

  update(hFromSlider(0));
}
