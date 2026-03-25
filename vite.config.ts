import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  root: 'src',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/index.html'),
        projects: resolve(__dirname, 'src/projects.html'),
        experiments: resolve(__dirname, 'src/experiments.html'),
        webgpuTriangle: resolve(__dirname, 'src/experiments/webgpu-triangle.html'),
        notFound: resolve(__dirname, 'src/404.html'),
        playground: resolve(__dirname, 'src/playground.html'),
        secretAssignment: resolve(__dirname, 'src/secret/assignment.html'),
        secretPlayground: resolve(__dirname, 'src/secret/playground.html'),
      restir: resolve(__dirname, 'src/projects/restir/index.html'),
      },
    },
  },
})
