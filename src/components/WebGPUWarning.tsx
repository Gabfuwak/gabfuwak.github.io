export default function WebGPUWarning() {
  return (
    <div className="webgpu-warning">
      <p><strong>⚠ WebGPU is not supported or not enabled in your browser</strong></p>
      <p><strong>How to enable WebGPU:</strong></p>
      <p><strong>Chrome:</strong></p>
      <ul>
        <li>Go to <code>chrome://flags</code></li>
        <li>Search for "Unsafe WebGPU"</li>
        <li>Enable the flag and restart Chrome</li>
        <li>Check that Vulkan is enabled at <code>chrome://gpu</code> (if not enabled, it will work but be slower and may not handle complex renderings)</li>
      </ul>
      <p><strong>Firefox:</strong></p>
      <ul>
        <li>Go to <code>about:config</code></li>
        <li>Search for <code>dom.webgpu.enabled</code></li>
        <li>Set it to true and restart Firefox</li>
        <li><em>Note: Firefox on Linux does not support Vulkan, so WebGPU will be much slower</em></li>
      </ul>
    </div>
  );
}
