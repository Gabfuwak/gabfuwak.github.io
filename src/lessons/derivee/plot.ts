// Tiny SVG plot helper for the derivative lesson.
// Draws axes + grid + a function curve, optionally with a draggable secant.
// Orthonormal by default (1 unit x = 1 unit y).

const NS = 'http://www.w3.org/2000/svg';

function el<K extends keyof SVGElementTagNameMap>(name: K): SVGElementTagNameMap[K] {
  return document.createElementNS(NS, name);
}

export interface SecantOpts {
  x1: number;
  x2: number;
  draggable?: boolean;
}

export interface PlotOpts {
  fn: (x: number) => number;
  xDomain: [number, number];
  yDomain?: [number, number];
  width?: number;
  height?: number;        // only used when orthonormal=false
  orthonormal?: boolean;  // default true
  grid?: boolean;         // default true
  secant?: SecantOpts;
}

const COLOR_CURVE  = '#0066cc';
const COLOR_AXIS   = '#888';
const COLOR_TICK   = '#aaa';
const COLOR_LABEL  = '#444';
const COLOR_GRID   = '#eee';
const COLOR_SECANT = '#e67e22';

function formatTick(v: number): string {
  if (Math.abs(v) < 1e-10) return '0';
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2).replace(/\.?0+$/, '').replace('.', ',');
}

function niceStep(range: number, target = 5): number {
  if (range <= 0) return 1;
  const rough = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  if      (norm < 1.5) return 1 * mag;
  else if (norm < 3)   return 2 * mag;
  else if (norm < 7)   return 5 * mag;
  else                 return 10 * mag;
}

function ticksWithStep(min: number, max: number, step: number): number[] {
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let t = start; t <= max + 1e-9; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return ticks;
}

export function createPlot(container: HTMLElement, opts: PlotOpts) {
  const orthonormal = opts.orthonormal ?? true;
  const showGrid    = opts.grid ?? true;

  const W  = opts.width  ?? 360;
  const ML = 40, MR = 20, MT = 20, MB = 35;
  const PW = W - ML - MR;

  const [xMin, xMax] = opts.xDomain;

  // Determine y domain. If the user provided one, use it as-is.
  // Otherwise: sample the function, anchor the domain to 0 if data is one-sided,
  // and (after picking yStep below) snap to step boundaries with a bit of headroom.
  let yMin: number, yMax: number;
  let dMin = NaN, dMax = NaN;
  if (opts.yDomain) {
    [yMin, yMax] = opts.yDomain;
  } else {
    const N = 200;
    dMin = Infinity; dMax = -Infinity;
    for (let i = 0; i <= N; i++) {
      const x = xMin + (xMax - xMin) * i / N;
      const y = opts.fn(x);
      if (y < dMin) dMin = y;
      if (y > dMax) dMax = y;
    }
    if (dMin >= 0) dMin = 0;
    if (dMax <= 0) dMax = 0;
    yMin = dMin;
    yMax = dMax;
  }

  // ── Tick steps ──────────────────────────────────────────
  // Per-axis nice step, then unified when orthonormal AND ranges are comparable
  // (avoids one axis having 0.5 ticks while the other has 1). The unification is
  // skipped if it would leave fewer than 3 ticks on either axis.
  let xStep = niceStep(xMax - xMin);
  let yStep = niceStep(yMax - yMin || 1);
  if (orthonormal) {
    const ratio = Math.max(xStep, yStep) / Math.min(xStep, yStep);
    if (ratio <= 4) {
      const common = Math.max(xStep, yStep);
      const xN = Math.floor((xMax - xMin) / common) + 1;
      const yN = Math.floor((yMax - yMin) / common) + 1;
      if (xN >= 3 && yN >= 3) {
        xStep = common;
        yStep = common;
      }
    }
  }

  // Snap the auto-computed yDomain onto step boundaries, with headroom if
  // the curve hits an edge exactly (so it never grazes the plot border).
  if (!opts.yDomain) {
    yMin = Math.floor(dMin / yStep) * yStep;
    yMax = Math.ceil(dMax / yStep) * yStep;
    if (Math.abs(yMax - dMax) < yStep * 1e-6 && dMax > 0) yMax += yStep;
    if (Math.abs(yMin - dMin) < yStep * 1e-6 && dMin < 0) yMin -= yStep;
    if (yMax - yMin < yStep * 0.5) yMax = yMin + yStep;
  }

  // Plot area height
  let PH: number, H: number;
  if (orthonormal) {
    PH = PW * (yMax - yMin) / (xMax - xMin);
    H  = PH + MT + MB;
  } else {
    H  = opts.height ?? 280;
    PH = H - MT - MB;
  }

  const xToSvg = (x: number) => ML + ((x - xMin) / (xMax - xMin)) * PW;
  const yToSvg = (y: number) => MT + PH - ((y - yMin) / (yMax - yMin)) * PH;
  const svgToX = (sx: number) => xMin + ((sx - ML) / PW) * (xMax - xMin);

  const svg = el('svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.style.display = 'block';
  svg.style.margin = '1rem auto';
  svg.style.userSelect = 'none';
  svg.style.width = '100%';
  svg.style.maxWidth = `${W}px`;
  svg.style.height = 'auto';

  // Clip path so the curve never spills out of the plot area
  const uid = 'p' + Math.random().toString(36).slice(2, 8);
  const defs = el('defs');
  const clip = el('clipPath');
  clip.setAttribute('id', uid);
  const clipRect = el('rect');
  clipRect.setAttribute('x', String(ML));
  clipRect.setAttribute('y', String(MT));
  clipRect.setAttribute('width', String(PW));
  clipRect.setAttribute('height', String(PH));
  clip.appendChild(clipRect);
  defs.appendChild(clip);
  svg.appendChild(defs);

  const xTicks = ticksWithStep(xMin, xMax, xStep);
  const yTicks = ticksWithStep(yMin, yMax, yStep);

  // ── Grid (behind everything) ────────────────────────────
  if (showGrid) {
    for (const t of xTicks) {
      const sx = xToSvg(t);
      const line = el('line');
      line.setAttribute('x1', String(sx));
      line.setAttribute('y1', String(MT));
      line.setAttribute('x2', String(sx));
      line.setAttribute('y2', String(MT + PH));
      line.setAttribute('stroke', COLOR_GRID);
      line.setAttribute('stroke-width', '1');
      svg.appendChild(line);
    }
    for (const t of yTicks) {
      const sy = yToSvg(t);
      const line = el('line');
      line.setAttribute('x1', String(ML));
      line.setAttribute('y1', String(sy));
      line.setAttribute('x2', String(ML + PW));
      line.setAttribute('y2', String(sy));
      line.setAttribute('stroke', COLOR_GRID);
      line.setAttribute('stroke-width', '1');
      svg.appendChild(line);
    }
  }

  // ── Axes ────────────────────────────────────────────────
  const xAxis = el('line');
  xAxis.setAttribute('x1', String(ML));
  xAxis.setAttribute('y1', String(MT + PH));
  xAxis.setAttribute('x2', String(ML + PW));
  xAxis.setAttribute('y2', String(MT + PH));
  xAxis.setAttribute('stroke', COLOR_AXIS);
  xAxis.setAttribute('stroke-width', '1');
  svg.appendChild(xAxis);

  const yAxis = el('line');
  yAxis.setAttribute('x1', String(ML));
  yAxis.setAttribute('y1', String(MT));
  yAxis.setAttribute('x2', String(ML));
  yAxis.setAttribute('y2', String(MT + PH));
  yAxis.setAttribute('stroke', COLOR_AXIS);
  yAxis.setAttribute('stroke-width', '1');
  svg.appendChild(yAxis);

  for (const t of xTicks) {
    const sx = xToSvg(t);
    const tick = el('line');
    tick.setAttribute('x1', String(sx));
    tick.setAttribute('y1', String(MT + PH));
    tick.setAttribute('x2', String(sx));
    tick.setAttribute('y2', String(MT + PH + 4));
    tick.setAttribute('stroke', COLOR_TICK);
    svg.appendChild(tick);

    const label = el('text');
    label.setAttribute('x', String(sx));
    label.setAttribute('y', String(MT + PH + 18));
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-family', 'system-ui, sans-serif');
    label.setAttribute('font-size', '11');
    label.setAttribute('fill', COLOR_LABEL);
    label.textContent = formatTick(t);
    svg.appendChild(label);
  }

  for (const t of yTicks) {
    const sy = yToSvg(t);
    const tick = el('line');
    tick.setAttribute('x1', String(ML - 4));
    tick.setAttribute('y1', String(sy));
    tick.setAttribute('x2', String(ML));
    tick.setAttribute('y2', String(sy));
    tick.setAttribute('stroke', COLOR_TICK);
    svg.appendChild(tick);

    const label = el('text');
    label.setAttribute('x', String(ML - 7));
    label.setAttribute('y', String(sy + 4));
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('font-family', 'system-ui, sans-serif');
    label.setAttribute('font-size', '11');
    label.setAttribute('fill', COLOR_LABEL);
    label.textContent = formatTick(t);
    svg.appendChild(label);
  }

  // ── Curve ───────────────────────────────────────────────
  const N = 240;
  const pts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const x = xMin + (xMax - xMin) * i / N;
    const y = opts.fn(x);
    pts.push(`${xToSvg(x).toFixed(2)},${yToSvg(y).toFixed(2)}`);
  }
  const curve = el('polyline');
  curve.setAttribute('points', pts.join(' '));
  curve.setAttribute('fill', 'none');
  curve.setAttribute('stroke', COLOR_CURVE);
  curve.setAttribute('stroke-width', '2');
  curve.setAttribute('stroke-linejoin', 'round');
  curve.setAttribute('clip-path', `url(#${uid})`);
  svg.appendChild(curve);

  // ── Secant (optional) ───────────────────────────────────
  if (opts.secant) {
    let x1 = opts.secant.x1;
    let x2 = opts.secant.x2;

    const slopeBg = el('rect');
    slopeBg.setAttribute('x', String(W - MR - 110));
    slopeBg.setAttribute('y', String(MT));
    slopeBg.setAttribute('width', '110');
    slopeBg.setAttribute('height', '22');
    slopeBg.setAttribute('fill', '#fff');
    slopeBg.setAttribute('opacity', '0.85');
    svg.appendChild(slopeBg);

    const slopeText = el('text');
    slopeText.setAttribute('x', String(W - MR));
    slopeText.setAttribute('y', String(MT + 15));
    slopeText.setAttribute('text-anchor', 'end');
    slopeText.setAttribute('font-family', 'system-ui, sans-serif');
    slopeText.setAttribute('font-size', '13');
    slopeText.setAttribute('fill', COLOR_SECANT);
    slopeText.setAttribute('font-weight', '600');
    svg.appendChild(slopeText);

    const line = el('line');
    line.setAttribute('stroke', COLOR_SECANT);
    line.setAttribute('stroke-width', '2');
    line.setAttribute('clip-path', `url(#${uid})`);
    svg.appendChild(line);

    const handles: SVGCircleElement[] = [];
    for (let i = 0; i < 2; i++) {
      const h = el('circle');
      h.setAttribute('r', '7');
      h.setAttribute('fill', '#fff');
      h.setAttribute('stroke', COLOR_SECANT);
      h.setAttribute('stroke-width', '2');
      if (opts.secant.draggable) {
        h.style.cursor = 'ew-resize';
        h.style.touchAction = 'none';
      }
      svg.appendChild(h);
      handles.push(h);
    }

    function update() {
      const y1 = opts.fn(x1);
      const y2 = opts.fn(x2);
      const sx1 = xToSvg(x1), sy1 = yToSvg(y1);
      const sx2 = xToSvg(x2), sy2 = yToSvg(y2);

      const dx = sx2 - sx1, dy = sy2 - sy1;
      const len = Math.hypot(dx, dy) || 1;
      const ext = 24;
      const ux = dx / len, uy = dy / len;
      line.setAttribute('x1', String(sx1 - ux * ext));
      line.setAttribute('y1', String(sy1 - uy * ext));
      line.setAttribute('x2', String(sx2 + ux * ext));
      line.setAttribute('y2', String(sy2 + uy * ext));

      handles[0].setAttribute('cx', String(sx1));
      handles[0].setAttribute('cy', String(sy1));
      handles[1].setAttribute('cx', String(sx2));
      handles[1].setAttribute('cy', String(sy2));

      const slope = (y2 - y1) / (x2 - x1);
      slopeText.textContent = `pente = ${slope.toFixed(2).replace('.', ',')}`;
    }
    update();

    if (opts.secant.draggable) {
      let dragging: 0 | 1 | null = null;

      const onMove = (e: PointerEvent) => {
        if (dragging === null) return;
        const rect = svg.getBoundingClientRect();
        const svgX = (e.clientX - rect.left) * (W / rect.width);
        let xData = svgToX(svgX);
        xData = Math.max(xMin, Math.min(xMax, xData));
        const minGap = (xMax - xMin) * 0.02;
        if (dragging === 0) {
          if (xData > x2 - minGap) xData = x2 - minGap;
          x1 = xData;
        } else {
          if (xData < x1 + minGap) xData = x1 + minGap;
          x2 = xData;
        }
        update();
      };
      const onUp = () => {
        dragging = null;
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      handles.forEach((h, i) => {
        h.addEventListener('pointerdown', (e) => {
          dragging = i as 0 | 1;
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
          e.preventDefault();
        });
      });
    }
  }

  container.appendChild(svg);
  return svg;
}
