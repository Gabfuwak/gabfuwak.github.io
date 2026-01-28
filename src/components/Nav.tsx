import { Link } from 'react-router-dom';

export default function Nav() {
  return (
    <nav>
      <Link to="/">Home</Link>
      {' | '}
      <Link to="/projects">Projects</Link>
      {' | '}
      <Link to="/experiments">Experiments</Link>
      <hr />
    </nav>
  );
}
