import { useState } from 'react';
import { Link } from 'react-router-dom';
import { experiments, allTags } from '../data/experiments';
import type { Tag } from '../data/experiments';

export default function Experiments() {
  const [selectedTag, setSelectedTag] = useState<Tag | null>(null);

  const filteredExperiments = selectedTag
    ? experiments.filter(exp => exp.tags.includes(selectedTag))
    : experiments;

  return (
    <div>
      <h1>Experiments</h1>

      <div style={{ marginBottom: '2rem' }}>
        <button
          onClick={() => setSelectedTag(null)}
          style={{ fontWeight: selectedTag === null ? 'bold' : 'normal' }}
        >
          All
        </button>
        {allTags.map(tag => (
          <button
            key={tag}
            onClick={() => setSelectedTag(tag)}
            style={{
              marginLeft: '0.5rem',
              fontWeight: selectedTag === tag ? 'bold' : 'normal'
            }}
          >
            {tag}
          </button>
        ))}
      </div>

      <ul>
        {filteredExperiments.map(exp => (
          <li key={exp.id}>
            <Link to={exp.path}>{exp.title}</Link>
            {exp.tags.map(tag => (
              <span key={tag} className="tag">{tag}</span>
            ))}
            <p>{exp.date} - {exp.description}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
