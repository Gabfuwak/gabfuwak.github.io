// BRDF hemisphere sampling visualization (2D slice)
// Uses the exact same GGX formulas as common.wgsl

export interface BrdfVizOpts {
  width?:             number;
  height?:            number;
  roughness?:         number;
  metalness?:         number;
  baseColor?:         [number, number, number];
  fresnel?:           [number, number, number];
  showLobe?:          boolean;
  showSamples?:       boolean;
  numSamples?:        number;
  sampling?:          'uniform' | 'cosine' | 'brdf';
  showSlider?:        boolean;
  showLight?:         boolean;
  showLightRay?:      boolean;
  swatchLightAngle?:  number | null;
  areaLight?:         boolean;
  areaLightWidth?:    number;
  showSwatch?:        boolean;
  showNeeRays?:       boolean;
  crossOutAreaHits?:  boolean;
  draggableLight?:    boolean;
  misWeights?:        boolean;
  misWeightMode?:     'length' | 'opacity' | 'thickness';
  neeRayCount?:       number;
  incomingAngle?:     number;
  lightAngle?:        number;
  arrowLen?:          number;
}

export function createBrdfViz(container: HTMLElement, opts: BrdfVizOpts = {}): SVGSVGElement {
  const W = opts.width || 520;
  const H = opts.height || 300;
  const roughness = opts.roughness ?? 0.4;
  const metalness = opts.metalness ?? 1.0;
  const baseColor = opts.baseColor ?? [0.8, 0.8, 0.8] as [number, number, number];
  const fresnel = opts.fresnel ?? [0.04, 0.04, 0.04] as [number, number, number];
  const showLobe = opts.showLobe ?? true;
  const showSamples = opts.showSamples ?? false;
  let numSamples = opts.numSamples || 9;
  const sampling = opts.sampling || 'uniform';
  const showSlider = opts.showSlider ?? false;
  const showLight = opts.showLight ?? true;
  const showLightRay = opts.showLightRay ?? true;
  const swatchLightAngle = opts.swatchLightAngle != null ? opts.swatchLightAngle * Math.PI / 180 : null;
  const showAreaLight = opts.areaLight ?? false;
  const areaLightWidth = opts.areaLightWidth ?? 80;
  const showSwatch = opts.showSwatch ?? true;
  const showNeeRays = opts.showNeeRays ?? false;
  const crossOutAreaHits = opts.crossOutAreaHits ?? false;
  const draggableLight = opts.draggableLight ?? true;
  const misWeights = opts.misWeights ?? false;
  const misWeightMode = opts.misWeightMode ?? 'length';
  const neeRayCount = opts.neeRayCount ?? 3;

  const hx = W / 2, hy = H * 0.78;
  const surfaceY = hy;
  const arrowLen = opts.arrowLen ?? 120;
  const lobeScale = 110;
  const inRayLen = 150;
  const sunDist = 130;
  const sunR = 12;

  let cameraAngle = -(opts.incomingAngle ?? 55) * Math.PI / 180;
  let lightAngle = (opts.lightAngle ?? 30) * Math.PI / 180;

  const ns = document.createElementNS.bind(document, 'http://www.w3.org/2000/svg');

  const clipH = Math.ceil(hy + 20);
  const svg = ns('svg') as SVGSVGElement;
  svg.setAttribute('width', String(W));
  svg.setAttribute('height', String(clipH));
  svg.setAttribute('viewBox', `0 0 ${W} ${clipH}`);
  svg.style.display = 'block';
  svg.style.margin = '16px auto';
  svg.style.cursor = 'default';
  svg.style.userSelect = 'none';

  const defs = ns('defs') as SVGDefsElement;
  svg.appendChild(defs);

  const uid = 'viz' + Math.random().toString(36).slice(2, 6);

  // arrowhead markers
  for (const [suffix, color, opacity] of [['cam', '#333', 1.0], ['light', '#e67e22', 0.8]] as [string, string, number][]) {
    const marker = ns('marker') as SVGMarkerElement;
    marker.setAttribute('id', `${uid}-ah-${suffix}`);
    marker.setAttribute('markerWidth', '8');
    marker.setAttribute('markerHeight', '6');
    marker.setAttribute('refX', '7');
    marker.setAttribute('refY', '3');
    marker.setAttribute('orient', 'auto');
    const poly = ns('polygon') as SVGPolygonElement;
    poly.setAttribute('points', '0 0, 8 3, 0 6');
    poly.setAttribute('fill', color);
    poly.setAttribute('opacity', String(opacity));
    marker.appendChild(poly);
    defs.appendChild(marker);
  }

  // NEE arrow marker (gold)
  const neeMarker = ns('marker') as SVGMarkerElement;
  neeMarker.setAttribute('id', `${uid}-ah-nee`);
  neeMarker.setAttribute('markerWidth', '8');
  neeMarker.setAttribute('markerHeight', '6');
  neeMarker.setAttribute('refX', '7');
  neeMarker.setAttribute('refY', '3');
  neeMarker.setAttribute('orient', 'auto');
  const neePoly = ns('polygon') as SVGPolygonElement;
  neePoly.setAttribute('points', '0 0, 8 3, 0 6');
  neePoly.setAttribute('fill', '#f1c40f');
  neeMarker.appendChild(neePoly);
  defs.appendChild(neeMarker);

  // sample arrow marker
  const sampleMarker = ns('marker') as SVGMarkerElement;
  sampleMarker.setAttribute('id', `${uid}-ah-sample`);
  sampleMarker.setAttribute('markerWidth', '8');
  sampleMarker.setAttribute('markerHeight', '6');
  sampleMarker.setAttribute('refX', '7');
  sampleMarker.setAttribute('refY', '3');
  sampleMarker.setAttribute('orient', 'auto');
  const samplePoly = ns('polygon') as SVGPolygonElement;
  samplePoly.setAttribute('points', '0 0, 8 3, 0 6');
  samplePoly.setAttribute('fill', 'rgba(60,60,60,0.55)');
  sampleMarker.appendChild(samplePoly);
  defs.appendChild(sampleMarker);

  // --- static elements ---

  // surface (orange hatched)
  const surfGrp = ns('g') as SVGGElement;
  const surfLine = ns('line') as SVGLineElement;
  Object.entries({x1: 20, y1: surfaceY, x2: W - 20, y2: surfaceY, stroke: '#e67e22', 'stroke-width': 3}).forEach(([k,v]) => surfLine.setAttribute(k, String(v)));
  surfGrp.appendChild(surfLine);
  for (let x = 30; x < W - 30; x += 14) {
    const h = ns('line') as SVGLineElement;
    Object.entries({x1: x, y1: surfaceY + 2, x2: x - 10, y2: surfaceY + 14, stroke: '#e67e22', 'stroke-width': 1.5}).forEach(([k,v]) => h.setAttribute(k, String(v)));
    surfGrp.appendChild(h);
  }
  svg.appendChild(surfGrp);

  // normal (thin dashed, centered)
  const norm = ns('line') as SVGLineElement;
  Object.entries({x1: hx, y1: surfaceY, x2: hx, y2: surfaceY - 140, stroke: '#999', 'stroke-width': 1, 'stroke-dasharray': '4 3'}).forEach(([k,v]) => norm.setAttribute(k, String(v)));
  svg.appendChild(norm);
  const nLabel = ns('text') as SVGTextElement;
  nLabel.setAttribute('x', String(hx + 5)); nLabel.setAttribute('y', String(surfaceY - 142));
  nLabel.setAttribute('font-size', '12'); nLabel.setAttribute('fill', '#999');
  nLabel.setAttribute('font-family', 'serif'); nLabel.setAttribute('font-style', 'italic');
  nLabel.textContent = 'n';
  svg.appendChild(nLabel);

  // --- area light (ceiling quad, fixed) ---
  if (showAreaLight) {
    const alW = areaLightWidth, alH = 12;
    const alX = hx - alW / 2;
    const alY = surfaceY - 185;

    const alLabel = ns('text') as SVGTextElement;
    alLabel.setAttribute('x', String(alX + alW / 2));
    alLabel.setAttribute('y', String(alY - 6));
    alLabel.setAttribute('text-anchor', 'middle');
    alLabel.setAttribute('font-size', '11');
    alLabel.setAttribute('font-family', 'monospace');
    alLabel.setAttribute('fill', '#c87800');
    alLabel.textContent = 'area light';

    const alRect = ns('rect') as SVGRectElement;
    Object.entries({ x: alX, y: alY, width: alW, height: alH, rx: 2, fill: '#ffe066', stroke: '#f39c12', 'stroke-width': 2 }).forEach(([k, v]) => alRect.setAttribute(k, String(v)));

    svg.appendChild(alLabel);
    svg.appendChild(alRect);
    for (let i = 0; i <= 4; i++) {
      const gx = alX + (i / 4) * alW;
      const glow = ns('line') as SVGLineElement;
      Object.entries({ x1: gx, y1: alY + alH, x2: gx, y2: alY + alH + 10, stroke: '#ffe066', 'stroke-width': 1.5, opacity: '0.8' }).forEach(([k, v]) => glow.setAttribute(k, String(v)));
      svg.appendChild(glow);
    }
  }

  // --- dynamic group (rebuilt on drag) ---
  const dynGroup = ns('g') as SVGGElement;
  svg.appendChild(dynGroup);

  // --- color response swatch (right side) ---
  const swatchSize = 40;
  const swatchX = W - 60, swatchY = 30;
  let swatchRect: SVGRectElement | undefined;
  let swatchLabel: SVGTextElement | undefined;
  if (showSwatch) {
    swatchRect = ns('rect') as SVGRectElement;
    Object.entries({x: swatchX, y: swatchY, width: swatchSize, height: swatchSize, rx: 4, stroke: '#ccc', 'stroke-width': 1}).forEach(([k,v]) => swatchRect!.setAttribute(k, String(v)));
    svg.appendChild(swatchRect);
    swatchLabel = ns('text') as SVGTextElement;
    swatchLabel.setAttribute('x', String(swatchX + swatchSize / 2)); swatchLabel.setAttribute('y', String(swatchY + swatchSize + 14));
    swatchLabel.setAttribute('text-anchor', 'middle'); swatchLabel.setAttribute('font-size', '11');
    swatchLabel.setAttribute('fill', '#666'); swatchLabel.setAttribute('font-family', 'monospace');
    svg.appendChild(swatchLabel);
  }

  // --- BRDF math (matches common.wgsl) ---
  function ggxD(ndoth: number): number {
    const alpha = roughness * roughness;
    const alpha_sq = alpha * alpha;
    const d = ndoth * ndoth * (alpha_sq - 1) + 1;
    return alpha_sq / Math.max(1e-7, Math.PI * d * d);
  }
  function g1Schlick(ndotv: number): number {
    const alpha = roughness * roughness;
    const k = alpha * Math.sqrt(2 / Math.PI);
    return ndotv / Math.max(0.0001, ndotv * (1 - k) + k);
  }
  function schlickFVec3(hdotv: number): [number, number, number] {
    const t = Math.max(0, 1 - hdotv);
    const t5 = t ** 5;
    return fresnel.map((f0, i) => {
      const f0m = f0 * (1 - metalness) + baseColor[i] * metalness;
      return f0m + (1 - f0m) * t5;
    }) as [number, number, number];
  }
  function schlickFScalar(hdotv: number): number {
    const f = schlickFVec3(hdotv);
    return (f[0] + f[1] + f[2]) / 3;
  }

  function evalBrdfScalar(wi: [number, number], theta_o: number): number {
    const wo: [number, number] = [Math.sin(theta_o), Math.cos(theta_o)];
    const ndotwo = Math.max(0, wo[1]);
    const ndotwi = Math.max(0, wi[1]);
    if (ndotwo <= 0 || ndotwi <= 0) return 0;
    let hv: [number, number] = [wi[0] + wo[0], wi[1] + wo[1]];
    const hlen = Math.sqrt(hv[0] * hv[0] + hv[1] * hv[1]);
    if (hlen < 1e-8) return 0;
    hv = [hv[0] / hlen, hv[1] / hlen];
    const ndoth = Math.max(0, hv[1]);
    const hdotwi = Math.max(0, hv[0] * wi[0] + hv[1] * wi[1]);
    if (ndoth <= 0 || hdotwi <= 0) return 0;
    const D = ggxD(ndoth);
    const G = g1Schlick(ndotwi) * g1Schlick(ndotwo);
    const F = schlickFScalar(hdotwi);
    const spec = (D * F * G) / Math.max(0.0001, 4 * ndotwi * ndotwo);
    const lum = (baseColor[0] + baseColor[1] + baseColor[2]) / 3;
    const kD = (1 - metalness) * (1 - F);
    const diffuse = kD * lum / Math.PI;
    return spec + diffuse;
  }

  // BRDF response color for a specific (wi, wo) pair — matches evaluateRadiance in common.wgsl
  function responseColor(wi: [number, number], wo: [number, number]): [number, number, number] {
    const ndotl = Math.max(0, wo[1]);
    const ndotv = Math.max(0, wi[1]);
    if (ndotl <= 0 || ndotv <= 0) return [0, 0, 0];
    let hv: [number, number] = [wi[0] + wo[0], wi[1] + wo[1]];
    const hlen = Math.sqrt(hv[0] * hv[0] + hv[1] * hv[1]);
    if (hlen < 1e-8) return [0, 0, 0];
    hv = [hv[0] / hlen, hv[1] / hlen];
    const ndoth = Math.max(0, hv[1]);
    const hdotv = Math.max(0, hv[0] * wi[0] + hv[1] * wi[1]);
    if (ndoth <= 0 || hdotv <= 0) return [0, 0, 0];
    const D = ggxD(ndoth);
    const G = g1Schlick(ndotv) * g1Schlick(ndotl);
    const F = schlickFVec3(hdotv);
    const denom = Math.max(0.0001, 4 * ndotv * ndotl);
    const clamp = (v: number) => Math.min(255, Math.round(v * 255));
    return [0, 1, 2].map(i => {
      const spec = D * F[i] * G / denom * ndotl;
      const diff = baseColor[i] / Math.PI * ndotl;
      const kD = (1 - metalness) * (1 - F[i]);
      return clamp((kD * diff + spec) * 1.75);
    }) as [number, number, number];
  }

  // --- PDF helpers for MIS balance heuristic ---
  function brdfPdf2D(theta: number, wi: [number, number]): number {
    const ndotl = Math.max(0, Math.cos(theta));
    if (ndotl <= 0) return 0;
    const wo: [number, number] = [Math.sin(theta), Math.cos(theta)];
    let hv: [number, number] = [wi[0] + wo[0], wi[1] + wo[1]];
    const hlen = Math.sqrt(hv[0] * hv[0] + hv[1] * hv[1]);
    if (hlen < 1e-8) return 0;
    hv = [hv[0] / hlen, hv[1] / hlen];
    const ndoth = Math.max(0, hv[1]);
    const vdoth = Math.max(0, wi[0] * hv[0] + wi[1] * hv[1]);
    const D = ggxD(ndoth);
    const pDiffuse = 0.5 * (1 - metalness);
    return pDiffuse * ndotl / Math.PI + (1 - pDiffuse) * D * ndoth / Math.max(0.0001, 4 * vdoth);
  }

  function neePdf2D(theta: number): number {
    if (!showAreaLight || areaLightWidth <= 0) return 0;
    const cosTheta = Math.cos(theta);
    if (cosTheta <= 0) return 0;
    const lightDist = 185 / cosTheta;
    const xAt = hx + Math.sin(theta) * lightDist;
    const alX0 = hx - areaLightWidth / 2, alX1 = hx + areaLightWidth / 2;
    if (xAt < alX0 || xAt > alX1) return 0;
    const angleLeft = Math.atan2(alX0 - hx, 185);
    const angleRight = Math.atan2(alX1 - hx, 185);
    const angularWidth = angleRight - angleLeft;
    if (angularWidth <= 0) return 0;
    return 1.0 / angularWidth;
  }

  function applyMisWeight(w: number, baseLen: number, baseOpacity: number, baseWidth: number): { len: number; opacity: number; width: number } {
    let len = baseLen, opacity = baseOpacity, width = baseWidth;
    if (misWeightMode === 'length') len = baseLen * Math.max(0.1, w);
    else if (misWeightMode === 'opacity') opacity = 0.1 + 0.8 * w;
    else if (misWeightMode === 'thickness') width = 0.5 + 2.5 * w;
    return { len, opacity, width };
  }

  function rebuild(): void {
    while (dynGroup.firstChild) dynGroup.removeChild(dynGroup.firstChild);

    const wi: [number, number] = [-Math.sin(cameraAngle), Math.cos(cameraAngle)];
    const wo: [number, number] = [Math.sin(lightAngle), Math.cos(lightAngle)];

    // --- camera ray (arrowhead at hit point) ---
    const camDx = Math.sin(cameraAngle), camDy = Math.cos(cameraAngle);
    const camLine = ns('line') as SVGLineElement;
    Object.entries({
      x1: hx - camDx * inRayLen, y1: hy - camDy * inRayLen,
      x2: hx, y2: hy,
      stroke: '#333', 'stroke-width': 2.5, 'marker-end': `url(#${uid}-ah-cam)`
    }).forEach(([k,v]) => camLine.setAttribute(k, String(v)));
    dynGroup.appendChild(camLine);

    // --- light ray (from sun to hit point) ---
    if (showLight) {
      const sunX = hx + Math.sin(lightAngle) * sunDist;
      const sunY = hy - Math.cos(lightAngle) * sunDist;

      if (showLightRay) {
        const lightLine = ns('line') as SVGLineElement;
        Object.entries({
          x1: sunX, y1: sunY,
          x2: hx, y2: hy,
          stroke: '#e67e22', 'stroke-width': 2, 'stroke-dasharray': '6 3',
          'marker-end': `url(#${uid}-ah-light)`
        }).forEach(([k,v]) => lightLine.setAttribute(k, String(v)));
        dynGroup.appendChild(lightLine);
      }

      const sunIcon = ns('circle') as SVGCircleElement;
      Object.entries({cx: sunX, cy: sunY, r: sunR, fill: '#f39c12', stroke: '#e67e22', 'stroke-width': 2, cursor: draggableLight ? 'grab' : 'default'}).forEach(([k,v]) => sunIcon.setAttribute(k, String(v)));
      dynGroup.appendChild(sunIcon);
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        const ray = ns('line') as SVGLineElement;
        Object.entries({
          x1: sunX + Math.cos(a) * (sunR + 2), y1: sunY + Math.sin(a) * (sunR + 2),
          x2: sunX + Math.cos(a) * (sunR + 7), y2: sunY + Math.sin(a) * (sunR + 7),
          stroke: '#f39c12', 'stroke-width': 1.5, 'stroke-linecap': 'round'
        }).forEach(([k,v]) => ray.setAttribute(k, String(v)));
        dynGroup.appendChild(ray);
      }
    }

    // --- BRDF lobe ---
    if (showLobe) {
      const steps = 200;
      const pts: string[] = [];
      for (let i = 0; i <= steps; i++) {
        const theta = -Math.PI / 2 + Math.PI * i / steps;
        const val = Math.log2(1 + 20 * evalBrdfScalar(wi, theta)) / Math.log2(21);
        const dx = Math.sin(theta), dy = -Math.cos(theta);
        pts.push(`${hx + dx * val * lobeScale},${hy + dy * val * lobeScale}`);
      }
      const lobe = ns('polyline') as SVGPolylineElement;
      lobe.setAttribute('points', pts.join(' '));
      Object.entries({fill: 'none', stroke: '#27ae60', 'stroke-width': 2.5, 'stroke-dasharray': '8 4'}).forEach(([k,v]) => lobe.setAttribute(k, String(v)));
      dynGroup.appendChild(lobe);
    }

    // --- sample arrows (optional, for path tracer vizs) ---
    if (showSamples) {
      let sampleAngles: number[] = [];
      if (sampling === 'uniform') {
        for (let i = 0; i < numSamples; i++) {
          const t = (i + 0.5) / numSamples;
          sampleAngles.push(-Math.PI / 2 + t * Math.PI);
        }
      } else if (sampling === 'cosine') {
        for (let i = 0; i < numSamples; i++) {
          const t = (i + 0.5) / numSamples;
          sampleAngles.push(Math.asin(2 * t - 1));
        }
      } else if (sampling === 'brdf') {
        const N = 512;
        const vals = new Float64Array(N);
        let sum = 0;
        for (let j = 0; j < N; j++) {
          const theta = -Math.PI / 2 + Math.PI * j / (N - 1);
          vals[j] = evalBrdfScalar(wi, theta);
          sum += vals[j];
        }
        if (sum > 0) {
          const cdf = new Float64Array(N);
          cdf[0] = vals[0] / sum;
          for (let j = 1; j < N; j++) cdf[j] = cdf[j - 1] + vals[j] / sum;
          for (let i = 0; i < numSamples; i++) {
            const target = (i + 0.5) / numSamples;
            let lo = 0, hi = N - 1;
            while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < target) lo = mid + 1; else hi = mid; }
            sampleAngles.push(-Math.PI / 2 + Math.PI * lo / (N - 1));
          }
        }
      }

      const alX0 = hx - areaLightWidth / 2, alX1 = hx + areaLightWidth / 2;

      for (const theta of sampleAngles) {
        const dx = Math.sin(theta), dy = -Math.cos(theta);
        const hitsLight = crossOutAreaHits && showAreaLight && Math.cos(theta) > 0
          && (() => { const xAt = hx + Math.sin(theta) * 185 / Math.cos(theta); return xAt >= alX0 && xAt <= alX1; })();

        let aLen = arrowLen, aOpacity = hitsLight ? 0.2 : 0.55, aWidth = 1.5;
        if (misWeights) {
          const pBrdf = brdfPdf2D(theta, wi);
          const pNee = neePdf2D(theta);
          const w = (pBrdf + pNee > 0) ? pBrdf / (pBrdf + pNee) : 1;
          ({ len: aLen, opacity: aOpacity, width: aWidth } = applyMisWeight(w, arrowLen, 0.55, 1.5));
        }

        const line = ns('line') as SVGLineElement;
        Object.entries({
          x1: hx, y1: hy,
          x2: hx + dx * aLen, y2: hy + dy * aLen,
          stroke: '#333', 'stroke-width': aWidth, 'stroke-dasharray': '6 3',
          opacity: aOpacity, 'marker-end': `url(#${uid}-ah-sample)`
        }).forEach(([k,v]) => line.setAttribute(k, String(v)));
        dynGroup.appendChild(line);

        if (hitsLight && !misWeights) {
          const ex = hx + dx * arrowLen, ey = hy + dy * arrowLen;
          const r = 7;
          for (const [x1o, y1o, x2o, y2o] of [[-r,-r,r,r],[-r,r,r,-r]] as [number,number,number,number][]) {
            const xLine = ns('line') as SVGLineElement;
            Object.entries({
              x1: ex+x1o, y1: ey+y1o, x2: ex+x2o, y2: ey+y2o,
              stroke: '#e74c3c', 'stroke-width': 2.5, 'stroke-linecap': 'round'
            }).forEach(([k,v]) => xLine.setAttribute(k, String(v)));
            dynGroup.appendChild(xLine);
          }
        }
      }
    }

    // --- NEE shadow rays (yellow arrows toward lights) ---
    if (showNeeRays) {
      const alY = hy - 185;
      if (showAreaLight) {
        const nRays = neeRayCount;
        for (let i = 0; i < nRays; i++) {
          const tx = hx - areaLightWidth / 2 + (i + 0.5) / nRays * areaLightWidth;
          const ty = alY + 6;
          const neeDx = tx - hx, neeDy = hy - ty;
          const neeTheta = Math.atan2(neeDx, neeDy);
          const fullLen = Math.sqrt(neeDx * neeDx + neeDy * neeDy);

          let aLen = fullLen, aOpacity = 1.0, aWidth = 2;
          if (misWeights) {
            const pNee = neePdf2D(neeTheta);
            const pBrdf = brdfPdf2D(neeTheta, wi);
            const w = (pNee + pBrdf > 0) ? pNee / (pNee + pBrdf) : 1;
            ({ len: aLen, opacity: aOpacity, width: aWidth } = applyMisWeight(w, fullLen, 1.0, 2));
          }

          const dirLen = Math.sqrt(neeDx * neeDx + neeDy * neeDy);
          const ndx = neeDx / dirLen, ndy = -neeDy / dirLen;
          const neeArrow = ns('line') as SVGLineElement;
          Object.entries({
            x1: hx, y1: hy, x2: hx + ndx * aLen, y2: hy + ndy * aLen,
            stroke: '#f1c40f', 'stroke-width': aWidth, opacity: aOpacity,
            'marker-end': `url(#${uid}-ah-nee)`
          }).forEach(([k,v]) => neeArrow.setAttribute(k, String(v)));
          dynGroup.appendChild(neeArrow);
        }
      }
      if (showLight) {
        const sunX = hx + Math.sin(lightAngle) * sunDist;
        const sunY = hy - Math.cos(lightAngle) * sunDist;
        const neeArrow = ns('line') as SVGLineElement;
        Object.entries({
          x1: hx, y1: hy, x2: sunX, y2: sunY,
          stroke: '#f1c40f', 'stroke-width': 2,
          'marker-end': `url(#${uid}-ah-nee)`
        }).forEach(([k,v]) => neeArrow.setAttribute(k, String(v)));
        dynGroup.appendChild(neeArrow);
      }
    }

    // --- update color swatch ---
    if (showSwatch && swatchRect && swatchLabel) {
      const swatchWo = swatchLightAngle != null
        ? [Math.sin(swatchLightAngle), Math.cos(swatchLightAngle)] as [number, number]
        : wo;
      const rgb = responseColor(wi, swatchWo);
      swatchRect.setAttribute('fill', `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
      swatchLabel.textContent = `${rgb[0]},${rgb[1]},${rgb[2]}`;
    }
  }

  // --- drag interaction ---
  let dragTarget: 'camera' | 'light' | null = null;

  function svgCoords(e: PointerEvent): [number, number] {
    const rect = svg.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (W / rect.width),
      (e.clientY - rect.top) * (H / rect.height),
    ];
  }

  function coordsToAngle(sx: number, sy: number): number {
    const dx = sx - hx, dy = hy - sy;
    return Math.atan2(dx, dy);
  }

  svg.addEventListener('pointerdown', (e: PointerEvent) => {
    const [sx, sy] = svgCoords(e);
    const sunX = hx + Math.sin(lightAngle) * sunDist;
    const sunY = hy - Math.cos(lightAngle) * sunDist;
    const dSun = Math.sqrt((sx - sunX) ** 2 + (sy - sunY) ** 2);

    const camEndX = hx - Math.sin(cameraAngle) * inRayLen;
    const camEndY = hy - Math.cos(cameraAngle) * inRayLen;
    const dCam = Math.sqrt((sx - camEndX) ** 2 + (sy - camEndY) ** 2);

    if (draggableLight && dSun < sunR + 15) {
      dragTarget = 'light';
    } else if (dCam < 25) {
      dragTarget = 'camera';
    } else {
      dragTarget = (draggableLight && dSun < dCam) ? 'light' : 'camera';
    }

    svg.setPointerCapture(e.pointerId);
    if (dragTarget === 'light') lightAngle = coordsToAngle(sx, sy);
    else cameraAngle = -coordsToAngle(sx, sy);
    rebuild();
  });

  svg.addEventListener('pointermove', (e: PointerEvent) => {
    if (!dragTarget) return;
    const [sx, sy] = svgCoords(e);
    if (dragTarget === 'light') lightAngle = coordsToAngle(sx, sy);
    else cameraAngle = -coordsToAngle(sx, sy);
    rebuild();
  });

  svg.addEventListener('pointerup', (e: PointerEvent) => {
    dragTarget = null;
    svg.releasePointerCapture(e.pointerId);
  });

  rebuild();
  container.appendChild(svg);

  if (showSlider) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; justify-content:center; gap:8px; margin-top:4px; font:13px monospace; color:#555;';
    const label = document.createElement('span');
    label.textContent = 'Samples: ' + numSamples;
    const slider = document.createElement('input');
    Object.assign(slider, { type: 'range', min: 3, max: 40, value: numSamples, step: 1 });
    slider.style.width = '180px';
    slider.addEventListener('input', () => {
      numSamples = parseInt(slider.value);
      label.textContent = 'Samples: ' + numSamples;
      rebuild();
    });
    row.appendChild(label);
    row.appendChild(slider);
    container.appendChild(row);
  }

  return svg;
}

// CDF bar visualization for NEE light sampling
export interface CdfEntry {
  flux: number;
  type: 'emissive' | 'point';
}

export interface CdfVizOpts {
  entries?:   CdfEntry[];
  barWidth?:  number;
}

export function createCdfViz(container: HTMLElement, opts: CdfVizOpts = {}): void {
  const entries = opts.entries || [];
  const totalFlux = entries.reduce((s, e) => s + e.flux, 0);
  const barW = opts.barWidth || 420;
  const barH = 36;
  const padTop = 24;
  const padBot = 28;
  const svgH = padTop + barH + padBot;

  const ns = document.createElementNS.bind(document, 'http://www.w3.org/2000/svg');

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'position:relative; width:fit-content; margin:16px auto;';

  const svg = ns('svg') as SVGSVGElement;
  svg.setAttribute('width', String(barW));
  svg.setAttribute('height', String(svgH));
  svg.setAttribute('viewBox', `0 0 ${barW} ${svgH}`);
  svg.style.display = 'block';

  // --- group labels ---
  const groups: Record<string, { x0: number; x1: number }> = {};
  let x = 0;
  for (const e of entries) {
    const w = (e.flux / totalFlux) * barW;
    if (!groups[e.type]) groups[e.type] = { x0: x, x1: x + w };
    else groups[e.type].x1 = x + w;
    x += w;
  }
  const labelStyle: Record<string, string> = { 'font-size': '12', 'font-family': 'monospace', 'text-anchor': 'middle' };
  if (groups.emissive) {
    const lbl = ns('text') as SVGTextElement;
    lbl.setAttribute('x', String((groups.emissive.x0 + groups.emissive.x1) / 2));
    lbl.setAttribute('y', String(padTop - 8));
    lbl.setAttribute('fill', '#c87800');
    Object.entries(labelStyle).forEach(([k, v]) => lbl.setAttribute(k, v));
    lbl.textContent = 'Emissive triangles';
    svg.appendChild(lbl);
  }
  if (groups.point) {
    const lbl = ns('text') as SVGTextElement;
    lbl.setAttribute('x', String((groups.point.x0 + groups.point.x1) / 2));
    lbl.setAttribute('y', String(padTop - 8));
    lbl.setAttribute('fill', '#2980b9');
    Object.entries(labelStyle).forEach(([k, v]) => lbl.setAttribute(k, v));
    lbl.textContent = 'Point lights';
    svg.appendChild(lbl);
  }

  // --- bar segments ---
  const segments: { x: number; w: number; entry: CdfEntry; index: number; highlight: SVGRectElement }[] = [];
  x = 0;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const w = (e.flux / totalFlux) * barW;
    const fill = e.type === 'emissive' ? '#ffe066' : '#b3d9f2';
    const stroke = e.type === 'emissive' ? '#f39c12' : '#4a9eda';

    const rect = ns('rect') as SVGRectElement;
    Object.entries({ x, y: padTop, width: w, height: barH, fill, stroke, 'stroke-width': 2 })
      .forEach(([k, v]) => rect.setAttribute(k, String(v)));
    svg.appendChild(rect);

    if (w > 28) {
      const txt = ns('text') as SVGTextElement;
      txt.setAttribute('x', String(x + w / 2));
      txt.setAttribute('y', String(padTop + barH / 2 + 4));
      txt.setAttribute('text-anchor', 'middle');
      txt.setAttribute('font-size', '11');
      txt.setAttribute('font-family', 'monospace');
      txt.setAttribute('fill', '#333');
      txt.textContent = e.flux.toFixed(2);
      svg.appendChild(txt);
    }

    const hl = ns('rect') as SVGRectElement;
    Object.entries({ x: x + 1, y: padTop + 1, width: w - 2, height: barH - 2, fill: 'none', stroke: '#e74c3c', 'stroke-width': 3, rx: 2, visibility: 'hidden' })
      .forEach(([k, v]) => hl.setAttribute(k, String(v)));
    svg.appendChild(hl);

    segments.push({ x, w, entry: e, index: i, highlight: hl });
    x += w;
  }

  // --- sample marker (arrow pointing down into bar) ---
  const marker = ns('polygon') as SVGPolygonElement;
  marker.setAttribute('fill', '#e74c3c');
  marker.setAttribute('visibility', 'hidden');
  svg.appendChild(marker);

  // --- result text ---
  const resultText = ns('text') as SVGTextElement;
  resultText.setAttribute('x', String(barW / 2));
  resultText.setAttribute('y', String(padTop + barH + padBot - 4));
  resultText.setAttribute('text-anchor', 'middle');
  resultText.setAttribute('font-size', '12');
  resultText.setAttribute('font-family', 'monospace');
  resultText.setAttribute('fill', '#555');
  svg.appendChild(resultText);

  wrapper.appendChild(svg);

  // --- sample button ---
  const btn = document.createElement('button');
  btn.textContent = 'Sample';
  btn.style.cssText = `position:absolute; left:${barW + 16}px; top:${padTop}px; height:${barH}px; font: 13px monospace; padding: 0 14px; cursor: pointer; border: 2px solid #999; border-radius: 4px; background: #f5f5f5;`;
  wrapper.appendChild(btn);

  const nEmissive = entries.filter(e => e.type === 'emissive').length;
  btn.addEventListener('click', () => {
    const r = Math.random() * totalFlux;
    let cum = 0;
    for (const seg of segments) {
      cum += seg.entry.flux;
      if (r < cum) {
        for (const s of segments) s.highlight.setAttribute('visibility', 'hidden');
        seg.highlight.setAttribute('visibility', 'visible');
        const drawX = (r / totalFlux) * barW;
        const my = padTop - 2;
        marker.setAttribute('points', `${drawX-5},${my-8} ${drawX+5},${my-8} ${drawX},${my}`);
        marker.setAttribute('visibility', 'visible');
        if (seg.entry.type === 'emissive') {
          resultText.textContent = '\u2192 sampled emissive triangle ' + seg.index;
        } else {
          resultText.textContent = '\u2192 sampled point light ' + (seg.index - nEmissive);
        }
        break;
      }
    }
  });

  container.appendChild(wrapper);
}

// Reservoir sampling visualization for RIS
export function createReservoirViz(container: HTMLElement): void {
  const N = 64, gridCols = 8, M = 4, mCols = 2;
  const cellSize = 28, mCellSize = 40, finalSize = 46;
  const stepDelay = 500;

  let radiance: number[] = [];
  function regenerate(): void {
    radiance = [];
    for (let i = 0; i < N; i++) radiance.push(0.05 + Math.random() * 0.95);
  }
  regenerate();

  function radianceColor(v: number): string {
    const r = Math.round(v * 255);
    const g = Math.round(v * 240);
    const b = Math.round(v * 40);
    return `rgb(${r},${g},${b})`;
  }

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'display:flex; align-items:center; justify-content:center; gap:16px; margin:20px auto;';

  // --- N grid ---
  const nPanel = document.createElement('div');
  nPanel.style.textAlign = 'center';
  const nLabel = document.createElement('div');
  nLabel.style.cssText = 'color:#e67e22; font:italic 13px monospace; margin-bottom:6px;';
  nLabel.textContent = 'N = 64 lights';
  nPanel.appendChild(nLabel);

  const nGrid = document.createElement('div');
  nGrid.style.cssText = `display:grid; grid-template-columns:repeat(${gridCols},${cellSize}px); gap:1px; border:2.5px solid #e67e22; border-radius:6px; padding:2px; background:#fff;`;

  const nCells: HTMLDivElement[] = [];
  for (let i = 0; i < N; i++) {
    const cell = document.createElement('div');
    cell.style.cssText = `width:${cellSize}px; height:${cellSize}px; background:${radianceColor(radiance[i])}; border-radius:2px; transition:outline 0.15s;`;
    nGrid.appendChild(cell);
    nCells.push(cell);
  }
  nPanel.appendChild(nGrid);
  wrapper.appendChild(nPanel);

  // --- Arrow 1 ---
  const arrow1 = document.createElement('div');
  arrow1.style.cssText = 'text-align:center; font:12px monospace;';
  arrow1.innerHTML = '<div style="color:#4a9eda; font-style:italic; line-height:1.3;">Flux CDF<br>sample</div><div style="font-size:26px; color:#333; margin-top:2px;">\u2192</div>';
  wrapper.appendChild(arrow1);

  // --- M grid ---
  const mPanel = document.createElement('div');
  mPanel.style.textAlign = 'center';
  const mLabel = document.createElement('div');
  mLabel.style.cssText = 'color:#4a9eda; font:italic 13px monospace; margin-bottom:6px;';
  mLabel.textContent = 'M = 4 candidates';
  mPanel.appendChild(mLabel);

  const mGrid = document.createElement('div');
  mGrid.style.cssText = `display:inline-grid; grid-template-columns:repeat(${mCols},${mCellSize}px); gap:2px; border:2.5px solid #4a9eda; border-radius:8px; padding:3px; background:#fff; box-sizing:content-box;`;

  const mCells: HTMLDivElement[] = [];
  for (let i = 0; i < M; i++) {
    const cell = document.createElement('div');
    cell.style.cssText = `width:${mCellSize}px; height:${mCellSize}px; background:#fff; border:1px solid #ddd; border-radius:3px; transition:all 0.2s;`;
    mGrid.appendChild(cell);
    mCells.push(cell);
  }
  mPanel.appendChild(mGrid);
  wrapper.appendChild(mPanel);

  // --- Arrow 2 ---
  const arrow2 = document.createElement('div');
  arrow2.style.cssText = 'text-align:center; font:12px monospace;';
  arrow2.innerHTML = '<div style="color:#27ae60; font-style:italic; line-height:1.3;">Reservoir<br>selection</div><div style="font-size:26px; color:#333; margin-top:2px;">\u2192</div>';
  wrapper.appendChild(arrow2);

  // --- Final cell ---
  const fPanel = document.createElement('div');
  fPanel.style.cssText = 'display:flex; flex-direction:column; align-items:center;';
  const fLabel = document.createElement('div');
  fLabel.style.cssText = 'color:#27ae60; font:italic 13px monospace; margin-bottom:6px;';
  fLabel.textContent = 'Final candidate';
  fPanel.appendChild(fLabel);

  const finalCell = document.createElement('div');
  finalCell.style.cssText = `width:${finalSize}px; height:${finalSize}px; border:3px solid #27ae60; border-radius:8px; background:#fff; transition:background 0.3s;`;
  fPanel.appendChild(finalCell);
  wrapper.appendChild(fPanel);

  container.appendChild(wrapper);

  // --- State readout ---
  const stateDiv = document.createElement('div');
  stateDiv.style.cssText = 'text-align:center; font:12px monospace; color:#666; margin:6px auto; min-height:20px; max-width:600px;';
  container.appendChild(stateDiv);

  // --- Sample button ---
  const btn = document.createElement('button');
  btn.textContent = 'Sample';
  btn.style.cssText = 'display:block; margin:8px auto; font:13px monospace; padding:5px 18px; cursor:pointer; border:2px solid #999; border-radius:4px; background:#f5f5f5;';
  container.appendChild(btn);

  let animating = false;

  function reset(): void {
    nCells.forEach((c, i) => { c.style.outline = 'none'; c.style.background = radianceColor(radiance[i]); });
    mCells.forEach(c => { c.style.background = '#fff'; c.style.outline = 'none'; });
    finalCell.style.background = '#fff';
    stateDiv.innerHTML = '';
  }

  btn.addEventListener('click', () => {
    if (animating) return;
    animating = true;
    regenerate();
    reset();
    nCells.forEach((c, i) => c.style.background = radianceColor(radiance[i]));

    const totalFlux = radiance.reduce((s, v) => s + v, 0);

    const candidates: number[] = [];
    for (let k = 0; k < M; k++) {
      let r = Math.random() * totalFlux, cum = 0;
      for (let j = 0; j < N; j++) {
        cum += radiance[j];
        if (r < cum) { candidates.push(j); break; }
      }
    }

    const targetVal: number[] = [];
    for (let i = 0; i < N; i++) targetVal.push(0.05 + Math.random() * 0.95);

    let step = 0;

    function showCandidate(): void {
      if (step >= M) { setTimeout(() => reservoirStep(0, 0, -1), stepDelay); return; }
      const idx = candidates[step];
      nCells[idx].style.outline = '3px solid #4a9eda';
      nCells[idx].style.outlineOffset = '-1px';
      mCells[step].style.background = radianceColor(radiance[idx]);
      stateDiv.textContent = `Drawing candidate ${step + 1}/${M}: light #${idx}`;
      step++;
      setTimeout(showCandidate, stepDelay);
    }

    function reservoirStep(i: number, wSum: number, selected: number): void {
      mCells.forEach(c => c.style.outline = 'none');

      if (i >= M) {
        stateDiv.innerHTML = `<span style="color:#27ae60">\u2713 Final: light #${selected}</span>  (w_sum = ${wSum.toFixed(1)})`;
        animating = false;
        return;
      }

      const idx = candidates[i];
      const q = radiance[idx] / totalFlux;
      const p_hat = targetVal[idx];
      const w = p_hat / q;
      const newWSum = wSum + w;
      const prob = (selected < 0) ? 1.0 : w / newWSum;
      const accept = (selected < 0) || Math.random() < prob;
      const newSelected = accept ? idx : selected;

      mCells[i].style.outline = '3px solid #e74c3c';
      mCells[i].style.outlineOffset = '-1px';

      finalCell.style.background = radianceColor(radiance[newSelected]);

      const sym = accept ? '<span style="color:#27ae60">\u2713</span>' : '<span style="color:#e74c3c">\u2717</span>';
      stateDiv.innerHTML = `w = p\u0302/q = ${p_hat.toFixed(2)}/${q.toFixed(3)} = ${w.toFixed(1)}  P(accept) = ${prob.toFixed(2)} ${sym}`;

      setTimeout(() => reservoirStep(i + 1, newWSum, newSelected), stepDelay * 1.5);
    }

    setTimeout(showCandidate, 200);
  });
}
