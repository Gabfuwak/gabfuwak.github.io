import { useEffect, useRef, useState, useCallback } from 'react';
import { initWebGPU } from '../utils/webgpu';
import defaultFragment from '../shaders/notfound.wgsl?raw';

import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightSpecialChars } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { syntaxHighlighting, HighlightStyle, bracketMatching, foldGutter } from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { wgsl } from '@iizukak/codemirror-lang-wgsl';

// --- Catppuccin Latte theme for CodeMirror ---

const latteTheme = EditorView.theme({
  '&': {
    backgroundColor: '#eff1f5',
    color: '#4c4f69',
    fontSize: '0.85rem',
    borderRadius: '2px',
    border: '1px solid #ccd0da',
  },
  '.cm-content': { caretColor: '#dc8a78' },
  '.cm-cursor, .cm-dropCursor': { borderLeftColor: '#dc8a78' },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection': {
    backgroundColor: '#ccd0da',
  },
  '.cm-gutters': {
    backgroundColor: '#e6e9ef',
    color: '#9ca0b0',
    border: 'none',
  },
  '.cm-activeLineGutter': { backgroundColor: '#dce0e8' },
  '.cm-activeLine': { backgroundColor: '#dce0e820' },
  '.cm-matchingBracket': { backgroundColor: '#ccd0da', color: '#4c4f69' },
});

const latteHighlight = HighlightStyle.define([
  { tag: tags.keyword, color: '#8839ef' },
  { tag: tags.controlKeyword, color: '#8839ef' },
  { tag: tags.moduleKeyword, color: '#8839ef' },
  { tag: tags.typeName, color: '#df8e1d' },
  { tag: tags.number, color: '#fe640b' },
  { tag: tags.float, color: '#fe640b' },
  { tag: tags.integer, color: '#fe640b' },
  { tag: tags.string, color: '#40a02b' },
  { tag: tags.comment, color: '#8c8fa1', fontStyle: 'italic' },
  { tag: tags.lineComment, color: '#8c8fa1', fontStyle: 'italic' },
  { tag: tags.blockComment, color: '#8c8fa1', fontStyle: 'italic' },
  { tag: tags.function(tags.variableName), color: '#1e66f5' },
  { tag: tags.definition(tags.variableName), color: '#4c4f69' },
  { tag: tags.variableName, color: '#4c4f69' },
  { tag: tags.propertyName, color: '#1e66f5' },
  { tag: tags.operator, color: '#04a5e5' },
  { tag: tags.punctuation, color: '#6c6f85' },
  { tag: tags.bracket, color: '#6c6f85' },
  { tag: tags.attributeName, color: '#179299' },
  { tag: tags.bool, color: '#fe640b' },
]);

// --- WGSL preamble (hidden from editor, prepended on compile) ---

const PREAMBLE = `struct Uniforms {
  time: f32,
  _pad: f32,
  resolution: vec2f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vertexMain(@location(0) pos: vec2f) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = pos * 0.5 + 0.5;
  return out;
}

struct FragIn {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

`;

const PREAMBLE_LINES = PREAMBLE.split('\n').length;

export default function NotFound() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const editorParentRef = useRef<HTMLDivElement>(null);
  const editorViewRef = useRef<EditorView | null>(null);
  const engineRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    quadBuffer: GPUBuffer;
    uniformBuffer: GPUBuffer;
    bindGroupLayout: GPUBindGroupLayout;
    bindGroup: GPUBindGroup;
    pipeline: GPURenderPipeline | null;
    canvasFormat: GPUTextureFormat;
    animFrameId: number;
    startTime: number;
    destroyed: boolean;
    startLoop: () => void;
  } | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const hasGPU = typeof navigator !== 'undefined' && !!navigator.gpu;

  // Compile shader and (re)create pipeline
  const compile = useCallback((source: string) => {
    const engine = engineRef.current;
    if (!engine) return;

    const fullSource = PREAMBLE + source;

    try {
      const shaderModule = engine.device.createShaderModule({ code: fullSource });

      shaderModule.getCompilationInfo().then(info => {
        const errors = info.messages.filter(m => m.type === 'error');
        if (errors.length > 0) {
          setError(errors.map(e => `Line ${e.lineNum - PREAMBLE_LINES}: ${e.message}`).join('\n'));
          return;
        }

        setError(null);

        try {
          engine.pipeline = engine.device.createRenderPipeline({
            layout: engine.device.createPipelineLayout({ bindGroupLayouts: [engine.bindGroupLayout] }),
            vertex: {
              module: shaderModule,
              entryPoint: 'vertexMain',
              buffers: [{
                arrayStride: 8,
                attributes: [{ format: 'float32x2' as GPUVertexFormat, offset: 0, shaderLocation: 0 }],
              }],
            },
            fragment: {
              module: shaderModule,
              entryPoint: 'fragmentMain',
              targets: [{ format: engine.canvasFormat }],
            },
            primitive: { topology: 'triangle-list' },
          });
          engine.startLoop();
        } catch (e: any) {
          setError(e.message ?? 'Pipeline creation failed');
        }
      });
    } catch (e: any) {
      setError(e.message ?? 'Unknown compilation error');
    }
  }, []);

  // Initialize WebGPU
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !navigator.gpu) return;

    let destroyed = false;

    (async () => {
      try {
        const { device, context } = await initWebGPU(canvas);
        if (destroyed) { device.destroy(); return; }

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        const quadVertices = new Float32Array([
          -1, -1,  1, -1,  -1, 1,
           1, -1,  1,  1,  -1, 1,
        ]);
        const quadBuffer = device.createBuffer({
          size: quadVertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(quadBuffer, 0, quadVertices);

        const uniformBuffer = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = device.createBindGroupLayout({
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' },
          }],
        });
        const bindGroup = device.createBindGroup({
          layout: bindGroupLayout,
          entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        });

        const engine = {
          device, context, quadBuffer, uniformBuffer,
          bindGroupLayout, bindGroup, canvasFormat,
          pipeline: null as GPURenderPipeline | null,
          animFrameId: 0,
          startTime: performance.now(),
          destroyed: false,
          startLoop: () => {},
        };

        const frame = () => {
          if (engine.destroyed || !engine.pipeline) return;
          const t = (performance.now() - engine.startTime) / 1000;
          device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([t, 0, canvas.width, canvas.height]));

          const encoder = device.createCommandEncoder();
          const pass = encoder.beginRenderPass({
            colorAttachments: [{
              view: context.getCurrentTexture().createView(),
              loadOp: 'clear',
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              storeOp: 'store',
            }],
          });
          pass.setPipeline(engine.pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.setVertexBuffer(0, quadBuffer);
          pass.draw(6);
          pass.end();
          device.queue.submit([encoder.finish()]);
          engine.animFrameId = requestAnimationFrame(frame);
        };

        engine.startLoop = () => {
          cancelAnimationFrame(engine.animFrameId);
          engine.animFrameId = requestAnimationFrame(frame);
        };

        engineRef.current = engine;
        setReady(true);
      } catch {
        // no WebGPU
      }
    })();

    return () => {
      destroyed = true;
      if (engineRef.current) {
        engineRef.current.destroyed = true;
        cancelAnimationFrame(engineRef.current.animFrameId);
        engineRef.current.device.destroy();
        engineRef.current = null;
      }
    };
  }, []);

  // Compile default shader once ready
  useEffect(() => {
    if (ready) compile(defaultFragment);
  }, [ready, compile]);

  // Initialize CodeMirror
  useEffect(() => {
    if (!ready || !editorParentRef.current) return;

    const timerRef = { current: 0 };

    const view = new EditorView({
      state: EditorState.create({
        doc: defaultFragment,
        extensions: [
          lineNumbers(),
          highlightActiveLine(),
          highlightSpecialChars(),
          history(),
          bracketMatching(),
          foldGutter(),
          keymap.of([...defaultKeymap, ...historyKeymap]),
          wgsl(),
          latteTheme,
          syntaxHighlighting(latteHighlight),
          EditorView.updateListener.of(update => {
            if (update.docChanged) {
              clearTimeout(timerRef.current);
              timerRef.current = window.setTimeout(() => {
                compile(update.state.doc.toString());
              }, 400);
            }
          }),
        ],
      }),
      parent: editorParentRef.current,
    });

    editorViewRef.current = view;

    return () => {
      clearTimeout(timerRef.current);
      view.destroy();
      editorViewRef.current = null;
    };
  }, [ready, compile]);

  if (!hasGPU) {
    return (
      <div>
        <h1>404 - Page Not Found</h1>
        <p>The page you're looking for doesn't exist.</p>
      </div>
    );
  }

  return (
    <div>
      <h1>404</h1>
      <p>You seem lost. Here, have fun and write a shader.</p>

      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        style={{
          width: '100%',
          height: '400px',
          display: 'block',
          background: 'black',
          marginBottom: '1rem',
        }}
      />

      {error && (
        <pre style={{
          color: '#d20f39',
          background: '#e6e9ef',
          padding: '0.75rem',
          fontSize: '0.85rem',
          marginBottom: '0.5rem',
          whiteSpace: 'pre-wrap',
          borderRadius: '2px',
          border: '1px solid #ccd0da',
        }}>{error}</pre>
      )}

      <div ref={editorParentRef} />
    </div>
  );
}
