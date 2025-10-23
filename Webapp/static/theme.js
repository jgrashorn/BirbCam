(function () {
  const key = 'birbcam-theme';
  const root = document.documentElement;

  function apply(mode) {
    if (mode === 'auto') root.setAttribute('data-theme', 'auto');
    else root.setAttribute('data-theme', mode);
  }

  const saved = localStorage.getItem(key) || 'auto';
  apply(saved);

  const btn = document.getElementById('themeToggle');
  if (!btn) return;

  btn.addEventListener('click', () => {
    const current = localStorage.getItem(key) || 'auto';
    const next = current === 'auto' ? 'dark' : current === 'dark' ? 'light' : 'auto';
    localStorage.setItem(key, next);
    apply(next);
  });
})();