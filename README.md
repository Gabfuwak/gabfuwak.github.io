# Portfolio Website

Single-page personal portfolio featuring Efficient path tracing, AR robot interaction, Multi view vision, and Expressive locomotion. Efficient path tracing has a dragon render; the other three projects have animated GIFs.

The homepage uses the public identity Gabfuwak with the descriptor “Computer graphics, robotics & simulation.” Repository links remain public; this is a pseudonymous presentation, not an anonymity guarantee.

Built with HTML + TypeScript + Vite.

## Styling

The main pages share `src/style.css`: a warm Solarized Light palette, EB Garamond (loaded from Google Fonts, with local serif fallbacks), and thin rules instead of decorative panels. The ReSTIR report uses its own stylesheet and is independent of this theme.

The homepage uses a two-column project grid that becomes a single column on narrow screens. Project navigation points to `/#projects`; the former separate Projects listing has been removed. Existing experiment and demo pages remain available.

The Path tracing card links to `/projects/restir/`, and Multi view vision links to its illustrated report at `/projects/multi-view-vision/`. Both images and titles are clickable, with keyboard focus styling. The multiview report links to the code repository. AR robot interaction and Expressive locomotion are currently unlinked.

The Multi view vision animation is served from `src/public/media/multi-view-vision.gif`, extracted unchanged from the final slide of the project presentation. Its CSS crop trims foreground while retaining the stats panel and ray intersections.

The standalone multiview report lives in `src/projects/multi-view-vision/index.html` and uses the shared theme. Seven slide illustrations are stored in `src/public/projects/multi-view-vision/`; the simulation GIF is optional inside a native disclosure panel. The report explains camera rays, adaptive refinement, and tracking without assuming access to the slides, and separates the simulation demonstration from validated real-world performance. No WebGPU or page-specific JavaScript is required.

The multiview report uses the plain title “Multi-view vision” and credits the collaboration as “With André Dal Bosco,” without a school affiliation. An AI-assistance disclosure below the byline applies to the write-up.

The Expressive locomotion animation is served from `src/public/media/expressive-locomotion.gif`, extracted unchanged from slide 13 (the penultimate slide) of `CSC_52081_EP project.pptx`. Its CSS framing enlarges the walker while keeping the full gait in view. The final slide's bug animation is not used.

The Efficient path tracing render is served from `src/public/media/path-traced-dragon.png`, copied unchanged from the supplied PNG. Its CSS crop centers the dragon vertically and retains the colored walls rather than the ceiling.

The AR robot interaction preview uses `src/public/media/microduck.gif`. It retains the 1120 × 700 crop at (500, 380), removes the final 0.7 seconds of the video, and loops for approximately 4.87 seconds at 640 × 400 and 20 FPS. The original video outside the repository is unchanged.

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```
