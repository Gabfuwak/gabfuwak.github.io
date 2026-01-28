export type Tag = 'AI' | 'Computer graphics' | 'Robotics';

export interface ExperimentMeta {
  id: string;
  title: string;
  date: string;
  description: string;
  path: string;
  tags: Tag[];
}

export const experiments: ExperimentMeta[] = [
  {
    id: 'webgpu-triangle',
    title: 'WebGPU Triangle',
    date: 'January 28, 2026',
    description: 'A hello world for WebGPU and for this website!',
    path: '/experiments/webgpu-triangle',
    tags: ['Computer graphics'],
  },
];

export const allTags: Tag[] = ['AI', 'Computer graphics', 'Robotics'];
