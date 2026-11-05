// DOM helpers, status line and log.
export const $ = (id) => document.getElementById(id);

export function el(tag, props = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'value') e.value = v;
    else if (k in e && typeof v !== 'string') e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) e.append(c);
  return e;
}

export const fmtTime = (s) => { s = Math.max(0, s || 0); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; };
const fmtBytes = (b) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);

export function log(msg, cls = '') {
  const box = $('log');
  box.append(el('div', { class: cls }, `[${new Date().toLocaleTimeString()}] ${msg}`));
  box.scrollTop = box.scrollHeight;
}
export function showError(msg) { $('error').textContent = msg || ''; if (msg) log(msg, 'error'); }
export function setStage(text) { $('stage').textContent = text; }
export function setProgress(v, text = '') {
  $('progress').classList.toggle('indeterminate', v == null);
  $('progress').firstElementChild.style.width = v == null ? '' : `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
  $('progressText').textContent = text;
}
export function friendlyError(e) {
  const m = (e && e.message) || String(e);
  if (/Failed to fetch|NetworkError|load failed/i.test(m)) return `A download failed; check the connection and try again. (${m})`;
  if (/out of memory|OOM|allocation/i.test(m)) return `Out of memory; use a smaller Whisper model or WebAssembly. (${m})`;
  return m;
}
export function download(name, data, mime) {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mime });
  const a = el('a', { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
}

// Model download progress reported by the workers (files over 2 MB).
const downloads = new Map();
export function onDownload(m) {
  if (!m.file) return;
  const key = `${m.name}/${m.file}`;
  if (m.status === 'progress' || m.status === 'progress_total') downloads.set(key, { loaded: m.loaded, total: m.total });
  else if (m.status === 'done') downloads.delete(key);
  const box = $('downloads'); box.textContent = '';
  for (const [k, d] of downloads) {
    if (!d.total || d.total < 2e6) continue;
    box.append(el('div', { class: 'dl' }, [el('span', {}, `${k.split('/').slice(-2).join('/')} (${fmtBytes(d.total)})`), el('span', {}, `${Math.round((d.loaded / d.total) * 100)}%`)]));
  }
}
