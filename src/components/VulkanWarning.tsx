import { useState } from 'react';

export default function VulkanWarning() {
  const [isOpen, setIsOpen] = useState(false);

  const containerStyle = {
    textAlign: 'center' as const,
    margin: '0 auto 1rem auto',
    maxWidth: '600px',
  };

  if (!isOpen) {
    return (
      <div style={containerStyle}>
        <p
          style={{
            color: 'red',
            fontSize: '0.9rem',
            cursor: 'pointer',
            border: '1px solid red',
            background: '#ffe6e6',
            padding: '0.5rem',
            margin: 0,
          }}
          onClick={() => setIsOpen(true)}
        >
          Running too slow? Click here.
        </p>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <div style={{
        border: '1px solid red',
        background: '#ffe6e6',
        padding: '1rem',
        textAlign: 'left' as const,
      }}>
        <p><strong>⚠ Low performance detected</strong></p>
        <p>Your setup may be too slow for complex renderings.</p>
        <p><strong>Chrome:</strong></p>
        <ul>
          <li>Go to <code>chrome://gpu</code> to check your graphics feature status</li>
          <li>Make sure hardware acceleration is enabled in Chrome settings</li>
          <li>Update your graphics drivers if Vulkan/D3D12 is disabled</li>
        </ul>
        <p><strong>Firefox on Linux:</strong></p>
        <ul>
          <li>Firefox does not support Vulkan on Linux. Too bad :(</li>
          <li>Consider using Chrome for better WebGPU performance</li>
        </ul>
        <button onClick={() => setIsOpen(false)}>Close</button>
      </div>
    </div>
  );
}
