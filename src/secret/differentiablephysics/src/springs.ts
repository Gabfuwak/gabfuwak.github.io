import katex from 'katex';
import 'katex/dist/katex.min.css';
import './springs.css';

for (const element of document.querySelectorAll<HTMLElement>('[data-math]')) {
  katex.render(element.dataset.math!, element, {
    displayMode: element.classList.contains('equation'),
  });
}

type State = { x: number; velocity: number };
const referenceStep = 1 / 120;
const duration = 15;
const symplectic = document.querySelector<HTMLInputElement>('#symplectic')!;
symplectic.checked = false;

function step(state: State, k: number, h: number) {
  // Both methods use acceleration at the old position, with a 1 kg mass.
  const acceleration = -k * state.x;
  if (symplectic.checked) {
    state.velocity += acceleration * h;
    state.x += state.velocity * h;
  } else {
    state.x += state.velocity * h;
    state.velocity += acceleration * h;
  }
}

setupLiveSpring();

const plot = document.querySelector<SVGSVGElement>('.extension-plot')!;
const plotStiffness = document.querySelector<HTMLInputElement>('#plot-stiffness')!;
const timestep = document.querySelector<HTMLInputElement>('#timestep')!;
const timestepValue = document.querySelector<HTMLOutputElement>('#timestep-value')!;
const singleStepPlot = document.querySelector<SVGSVGElement>('.single-step-plot')!;
const singleStepValues = document.querySelector<HTMLOutputElement>('#single-step-values')!;
let samplePhase = 0; // Fraction of the exact oscillation's period.
plotStiffness.addEventListener('input', drawComparison);
timestep.addEventListener('input', drawComparison);
symplectic.addEventListener('change', () => {
  const equation = document.querySelector<HTMLElement>('#position-equation')!;
  equation.dataset.math = String.raw`x_{\text{new}} = x_{\text{old}} + v_{\text{${symplectic.checked ? 'new' : 'old'}}}\,h`;
  katex.render(equation.dataset.math, equation, { displayMode: true });
  document.querySelector('.step-label')!.textContent =
    `One ${symplectic.checked ? 'symplectic' : 'explicit'} Euler step`;
  drawComparison();
});
singleStepPlot.addEventListener('pointerdown', event => {
  if (event.button !== 0 || !plotStiffness.checkValidity()) return;
  const point = new DOMPoint(event.clientX, event.clientY)
    .matrixTransform(singleStepPlot.getScreenCTM()!.inverse());
  const period = 2 * Math.PI / Math.sqrt(plotStiffness.valueAsNumber);
  const h = 10 ** timestep.valueAsNumber;
  samplePhase = Math.max(0, Math.min(1, (point.x - 75) / 540 * (period + h) / period));
  drawSingleStep(plotStiffness.valueAsNumber, h);
});
singleStepPlot.addEventListener('keydown', event => {
  if (!plotStiffness.checkValidity()) return;
  if (event.key === 'ArrowLeft') samplePhase = Math.max(0, samplePhase - 0.01);
  else if (event.key === 'ArrowRight') samplePhase = Math.min(1, samplePhase + 0.01);
  else if (event.key === 'Home') samplePhase = 0;
  else if (event.key === 'End') samplePhase = 1;
  else return;
  event.preventDefault();
  drawSingleStep(plotStiffness.valueAsNumber, 10 ** timestep.valueAsNumber);
});
drawComparison();

function simulate(k: number, h: number) {
  const state: State = { x: 0.35, velocity: 0 };
  const samples = [{ time: 0, x: state.x }];
  for (let i = 1; i <= Math.ceil(duration / h); i++) {
    const previousX = state.x;
    step(state, k, h);
    // Clip the final plotted segment at 15 s; integration still uses full h.
    const time = Math.min(i * h, duration);
    const x = i * h > duration
      ? previousX + (state.x - previousX) * (duration - (i - 1) * h) / h
      : state.x;
    samples.push({ time, x });
  }
  return samples;
}

function drawComparison() {
  // Slider position is log10(h), spanning 0.001 to 0.7 seconds.
  const h = 10 ** timestep.valueAsNumber;
  timestepValue.value = `${h.toPrecision(3)} s`;
  timestep.setAttribute('aria-valuetext', `${h.toPrecision(3)} seconds`);
  if (!plotStiffness.checkValidity()) return;
  const k = plotStiffness.valueAsNumber;
  const reference = simulate(k, referenceStep);
  const selected = simulate(k, h);
  let extent = 0.1;
  for (const samples of [reference, selected]) {
    for (const sample of samples) extent = Math.max(extent, Math.abs(sample.x));
  }
  extent *= 2.2;
  for (const [selector, samples] of [
    ['.plot-reference', reference], ['.plot-line', selected],
  ] as const) {
    plot.querySelector(selector)!.setAttribute('points', samples.map(sample =>
      `${75 + 540 * sample.time / duration},${135 - 90 * sample.x / extent}`,
    ).join(' '));
  }
  plot.querySelector('.plot-high')!.textContent = String(Number(extent.toPrecision(3)));
  plot.querySelector('.plot-low')!.textContent = String(Number((-extent).toPrecision(3)));
  plot.setAttribute('aria-label', `Extension over 15 seconds, stiffness ${k} newtons per meter, comparing timestep ${h} seconds with 1/120 second. Both start at 0.35 meters extension with zero velocity.`);
  drawSingleStep(k, h);
}

function drawSingleStep(k: number, h: number) {
  const amplitude = 0.35;
  const omega = Math.sqrt(k); // sqrt(k/m), with m = 1 kg
  const period = 2 * Math.PI / omega;
  const startTime = samplePhase * period;
  const startX = amplitude * Math.cos(omega * startTime);
  const predicted = {
    x: startX,
    velocity: -amplitude * omega * Math.sin(omega * startTime),
  };
  step(predicted, k, h);
  const exactNextX = amplitude * Math.cos(omega * (startTime + h));
  const extent = Math.max(amplitude * 1.2, Math.abs(predicted.x) * 1.1);
  // Reserve space for a full step even when starting at the end of the period.
  const endTime = period + h;
  const px = (time: number) => 75 + 540 * time / endTime;
  const py = (x: number) => 135 - 90 * x / extent;
  const points = [];
  for (let i = 0; i <= 240; i++) {
    const time = period * i / 240;
    points.push(`${px(time)},${py(amplitude * Math.cos(omega * time))}`);
  }
  singleStepPlot.querySelector('.exact-curve')!.setAttribute('points', points.join(' '));
  singleStepPlot.querySelector('.step-line')!.setAttribute(
    'd', `M ${px(startTime)} ${py(startX)} L ${px(startTime + h)} ${py(predicted.x)}`,
  );
  singleStepPlot.querySelector('.step-error')!.setAttribute(
    'd', `M ${px(startTime + h)} ${py(exactNextX)} V ${py(predicted.x)}`,
  );
  for (const [selector, time, x] of [
    ['.start-point', startTime, startX],
    ['.exact-point', startTime + h, exactNextX],
    ['.step-point', startTime + h, predicted.x],
  ] as const) {
    const point = singleStepPlot.querySelector(selector)!;
    point.setAttribute('cx', String(px(time)));
    point.setAttribute('cy', String(py(x)));
  }
  singleStepPlot.querySelector('.single-high')!.textContent = String(Number(extent.toPrecision(3)));
  singleStepPlot.querySelector('.single-low')!.textContent = String(Number((-extent).toPrecision(3)));
  for (const [selector, time] of [['.single-mid-time', period / 2], ['.single-period', period]] as const) {
    const label = singleStepPlot.querySelector(selector)!;
    label.setAttribute('x', String(px(time)));
    label.textContent = time.toFixed(2);
  }
  const error = Math.abs(predicted.x - exactNextX);
  singleStepValues.value = `Start: ${startTime.toFixed(3)} s · h: ${h.toPrecision(3)} s · Error: ${error.toPrecision(3)} m`;
  singleStepPlot.setAttribute('aria-valuemax', String(period));
  singleStepPlot.setAttribute('aria-valuenow', String(startTime));
  singleStepPlot.setAttribute('aria-valuetext', `${startTime.toFixed(3)} seconds; next-step position error ${error.toPrecision(3)} meters`);
}

function setupLiveSpring() {
  const figure = document.querySelector<HTMLElement>('#live-spring')!;
  const svg = figure.querySelector<SVGSVGElement>('.spring')!;
  const coil = figure.querySelector<SVGPathElement>('.coil')!;
  const particle = figure.querySelector<SVGCircleElement>('.particle')!;
  const arrow = figure.querySelector<SVGPathElement>('.force-arrow')!;
  const stiffness = figure.querySelector<HTMLInputElement>('.stiffness')!;
  const extension = figure.querySelector<HTMLOutputElement>('.extension')!;
  const force = figure.querySelector<HTMLOutputElement>('.force')!;
  const bounds = figure.querySelector<HTMLOutputElement>('.bounds')!;
  const restY = 230; // 1 m below the support, at 200 drawing units per meter
  const scale = 200;
  const state: State = { x: 0, velocity: 0 };
  let k = Number(stiffness.value);
  let dragging: number | null = null;
  let grabOffset = 0;
  let previousTime = 0;
  let accumulatedTime = 0;

  function draw() {
    const y = restY + state.x * scale;
    let path = 'M 240 30 V 60';
    for (let i = 1; i < 20; i++) {
      path += ` L ${i % 2 ? 225 : 255} ${60 + (y - 85) * i / 20}`;
    }
    coil.setAttribute('d', `${path} L 240 ${y - 25} V ${y}`);
    particle.setAttribute('cy', String(y));
    particle.setAttribute('aria-valuenow', state.x.toFixed(2));
    particle.setAttribute('aria-valuetext', `${state.x.toFixed(2)} meters extension`);
    const f = -k * state.x;
    arrow.setAttribute('d', `M 285 ${y} V ${y + f * 8}`);
    arrow.style.display = Math.abs(f) < 0.05 ? 'none' : '';
    extension.value = `${(Math.abs(state.x) < 0.005 ? 0 : state.x).toFixed(2)} m`;
    force.value = `${(Math.abs(f) < 0.005 ? 0 : f).toFixed(2)} N`;
    bounds.hidden = y >= 15 && y <= 385;
  }

  function pointerY(event: PointerEvent) {
    return new DOMPoint(event.clientX, event.clientY)
      .matrixTransform(svg.getScreenCTM()!.inverse()).y;
  }

  function moveTo(value: number) {
    state.x = Math.max(-0.65, Math.min(0.65, value));
    state.velocity = 0;
    accumulatedTime = 0;
    draw();
  }

  particle.addEventListener('pointerdown', event => {
    if (dragging !== null || event.button !== 0) return;
    dragging = event.pointerId;
    grabOffset = pointerY(event) - (restY + state.x * scale);
    state.velocity = 0;
    particle.setPointerCapture(event.pointerId);
  });
  particle.addEventListener('pointermove', event => {
    if (dragging === event.pointerId) {
      moveTo((pointerY(event) - grabOffset - restY) / scale);
    }
  });
  particle.addEventListener('lostpointercapture', () => {
    dragging = null;
    state.velocity = 0;
    accumulatedTime = 0;
    previousTime = performance.now();
  });
  particle.addEventListener('keydown', event => {
    if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
    event.preventDefault();
    moveTo(state.x + (event.key === 'ArrowDown' ? 0.05 : -0.05));
  });
  stiffness.addEventListener('input', () => {
    if (!stiffness.checkValidity()) return;
    k = stiffness.valueAsNumber;
    state.velocity = 0;
    draw();
  });

  function animate(now: number) {
    if (previousTime && !document.hidden && dragging === null) {
      accumulatedTime += Math.min((now - previousTime) / 1000, 0.05);
      while (accumulatedTime >= referenceStep) {
        step(state, k, referenceStep);
        accumulatedTime -= referenceStep;
      }
      draw();
    }
    previousTime = now;
    requestAnimationFrame(animate);
  }
  document.addEventListener('visibilitychange', () => {
    previousTime = 0;
    accumulatedTime = 0;
  });
  draw();
  requestAnimationFrame(animate);
}
