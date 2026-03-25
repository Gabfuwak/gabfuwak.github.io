type Tag = 'AI' | 'Computer graphics' | 'Robotics';

interface ExperimentMeta {
  id: string;
  title: string;
  date: string;
  description: string;
  path: string;
  tags: Tag[];
}

const experiments: ExperimentMeta[] = [
  {
    id: 'webgpu-triangle',
    title: 'WebGPU Triangle',
    date: 'January 28, 2026',
    description: 'A hello world for WebGPU and for this website!',
    path: '/experiments/webgpu-triangle.html',
    tags: ['Computer graphics'],
  },
];

const allTags: Tag[] = ['AI', 'Computer graphics', 'Robotics'];
let selectedTag: Tag | null = null;

function render() {
  const filtered = selectedTag
    ? experiments.filter(e => e.tags.includes(selectedTag!))
    : experiments;

  const list = document.getElementById('experiment-list')!;
  list.innerHTML = filtered.map(exp => `
    <li>
      <a href="${exp.path}">${exp.title}</a>
      ${exp.tags.map(t => `<span class="tag">${t}</span>`).join('')}
      <p>${exp.date} - ${exp.description}</p>
    </li>
  `).join('');

  document.querySelectorAll<HTMLButtonElement>('#tag-filters button').forEach(btn => {
    btn.style.fontWeight = btn.dataset.tag === (selectedTag ?? 'All') ? 'bold' : 'normal';
  });
}

const filters = document.getElementById('tag-filters')!;

const allBtn = document.createElement('button');
allBtn.textContent = 'All';
allBtn.dataset.tag = 'All';
allBtn.addEventListener('click', () => { selectedTag = null; render(); });
filters.appendChild(allBtn);

for (const tag of allTags) {
  const btn = document.createElement('button');
  btn.textContent = tag;
  btn.dataset.tag = tag;
  btn.style.marginLeft = '0.5rem';
  btn.addEventListener('click', () => { selectedTag = tag; render(); });
  filters.appendChild(btn);
}

render();
