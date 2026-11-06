import type { RenderOptions } from "./scene";

export function initDisplayControls(): RenderOptions {
  const panel = document.querySelector<HTMLElement>("#display-controls");
  const toggle = document.querySelector<HTMLButtonElement>("#display-controls-toggle");
  const body = document.querySelector<HTMLElement>("#display-controls-body");
  const particles = document.querySelector<HTMLInputElement>("#show-particles");
  const triangles = document.querySelector<HTMLInputElement>("#show-triangles");
  const floor = document.querySelector<HTMLInputElement>("#show-floor");

  if (!panel || !toggle || !body || !particles || !triangles || !floor) {
    throw new Error("Display controls are missing from the page");
  }

  const options: RenderOptions = {
    showParticles: particles.checked,
    showTriangles: triangles.checked,
    showFloor: floor.checked,
  };

  particles.addEventListener("change", () => {
    options.showParticles = particles.checked;
  });
  triangles.addEventListener("change", () => {
    options.showTriangles = triangles.checked;
  });
  floor.addEventListener("change", () => {
    options.showFloor = floor.checked;
  });

  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(expanded));
    body.hidden = !expanded;
  });

  panel.addEventListener("keydown", (event) => {
    event.stopPropagation();
  });

  return options;
}
