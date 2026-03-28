export async function loadFragments({ root, fragments }) {
  if (!root) throw new Error('Fragment root element is required');
  const htmlParts = [];
  for (const fragment of fragments) {
    const response = await fetch(fragment, { cache: 'no-cache' });
    if (!response.ok) {
      throw new Error(`Failed to load fragment: ${fragment} (${response.status})`);
    }
    htmlParts.push(await response.text());
  }
  root.innerHTML = htmlParts.join('\n');
}

export function loadClassicScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = false;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.body.appendChild(script);
  });
}
