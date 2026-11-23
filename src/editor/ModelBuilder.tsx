import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { isTauri } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { backendError, editor } from './api';
import { Canvas } from './Canvas';
import { BlockSelection } from './BlockSelection';
import { Inspector } from './Inspector';
import { addBlock, graphPath, extractBlock, addBoundary, addNode, autoLayout, boundaryKey, connect, connectionError, current, edgeKey, history, move, newModel,
  nodeKey, owner, removeItem, renameBoundary, type Diagnostic, type Document, type Graph, type Operation, type Point, type Wire } from './model';
import { useValidation } from './useValidation';
import './editor.css';

function target(d: Diagnostic, g: Graph) {
  if (d.location.graph !== g.id) return null;
  if (d.location.node) return nodeKey(d.location.node);
  const m = d.location.path.match(/^\/graphs\/\d+\/(nodes|edges|inputs|outputs)\/(\d+)/);
  if (!m) return null;
  const i = Number(m[2]);
  if (m[1] === 'nodes') return g.nodes[i] && nodeKey(g.nodes[i].id);
  if (m[1] === 'edges') return g.edges[i] && owner(g.edges[i].to);
  const field = m[1] as 'inputs' | 'outputs';
  return g[field][i] && boundaryKey(field === 'inputs' ? 'input' : 'output', g[field][i].id);
}
function Source({ files, path, close }: { files: Record<string, string>; path: string; close: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [file, setFile] = useState('model.py');
  useEffect(() => { ref.current?.showModal(); }, []);
  return <dialog ref={ref} onClose={close} className="source-dialog" aria-labelledby="source-title">
    <div className="source-top"><h2 id="source-title">Generated Python</h2><button onClick={() => ref.current?.close()}>Close</button></div>
    <p className="object-id">{path}</p>
    <label>Source file <select value={file} onChange={e => setFile(e.target.value)}>
      {Object.keys(files).map(f => <option key={f}>{f}</option>)}</select></label>
    <pre tabIndex={0}><code>{files[file]}</code></pre>
  </dialog>;
}
export function ModelBuilder({ active }: { active: boolean }) {
  const initial = useMemo(newModel, []);
  const [state, dispatch] = useReducer(history, { doc: initial, past: [], future: [] });
  const { doc } = state;
  const [trail, setTrail] = useState<string[]>([]);
  const [group, setGroup] = useState<string[] | null>(null);
  const crumbs = [doc.model.entry];
  for (const graphId of trail) {
    if (!current(doc, graphId)) break;
    crumbs.push(graphId);
  }
  const g = current(doc, crumbs.at(-1)!);
  function navigate(path: string[]) {
    setTrail(path); setGroup(null); select(null); setFocus(null);
  }
  function openBlock(id: string) {
    const node = g.nodes.find(n => nodeKey(n.id) === id);
    const block = node?.kind === 'block' && doc.blocks?.find(b => b.id === node.block);
    if (block && !crumbs.includes(block.graph)) navigate([...crumbs.slice(1), block.graph]);
  }
  const [saved, setSaved] = useState(initial), [path, setPath] = useState<string | null>(null);
  const [registry, setRegistry] = useState<Operation[]>([]), [registryError, setRegistryError] = useState(''), [retry, setRetry] = useState(0);
  const [query, setQuery] = useState(''), [selected, select] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ key: string; count: number } | null>(null);
  const [reset, setReset] = useState(0), [busy, setBusy] = useState(''), [message, setMessage] = useState('');
  const [extra, setExtra] = useState<Diagnostic[]>([]);
  const [output, setOutput] = useState<{ path: string; sources: Record<string, string> } | null>(null), [inspect, setInspect] = useState(false);
  const [invalidFields, setInvalidFields] = useState<Set<string>>(() => new Set());
  const invalid = useCallback((key: string, bad: boolean) => setInvalidFields(old => {
    if (old.has(key) === bad) return old;
    const next = new Set(old); if (bad) next.add(key); else next.delete(key); return next;
  }), []);
  const dirty = saved !== doc || !!invalidFields.size, dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;
  const locked = useRef(false);
  useEffect(() => {
    let attached = true;
    editor.registry().then(value => { if (attached) { setRegistry(value); setRegistryError(''); } },
      e => { if (attached) setRegistryError(backendError(e).message); });
    return () => { attached = false; };
  }, [retry]);
  useEffect(() => {
    if (!isTauri()) return;
    const win = getCurrentWindow();
    const stop = win.onCloseRequested(async e => {
      if (locked.current) { e.preventDefault(); return; }
      if (dirtyRef.current) {
        e.preventDefault();
        if (await editor.discard()) await win.destroy();
      }
    });
    return () => { void stop.then(unlisten => unlisten()); };
  }, []);
  const validation = useValidation(doc, path, !!registry.length);
  const diagnostics = extra.length ? extra : validation.diagnostics;
  const flagged = useMemo(() => new Set(diagnostics.filter(d => d.severity === 'error').map(d => target(d, g)).filter((k): k is string => !!k)), [diagnostics, g]);
  const selection = selected && (g.nodes.some(n => nodeKey(n.id) === selected) ||
    g.inputs.some(p => boundaryKey('input', p.id) === selected) || g.outputs.some(p => boundaryKey('output', p.id) === selected) ||
    g.edges.some(e => edgeKey(e) === selected)) ? selected : null;
  const edit = (next: Document) => { if (locked.current) return; dispatch({ type: 'edit', doc: next }); setExtra([]); setMessage(''); setOutput(null); };
  function jump(key: string) { select(key); setFocus(old => ({ key, count: (old?.count ?? 0) + 1 })); }
  function add(id: string, point?: Point) {
    if (locked.current) return;
    const i = g.nodes.length + g.inputs.length + g.outputs.length;
    const pos = point ?? { x: 40 + (i % 4) * 210, y: 60 + Math.floor(i / 4) * 150 };
    const op = registry.find(item => item.id === id);
    let result: [Document, string];
    if (id === 'input' || id === 'output') result = addBoundary(doc, id, pos, g.id);
    else if (id.startsWith('block:')) {
      try { result = addBlock(doc, id.slice(6), pos, g.id); }
      catch (e) { setMessage(backendError(e).message); return; }
    }
    else if (op) result = addNode(doc, op, pos, g.id); else return;
    edit(result[0]);
    if (point) select(result[1]);
    else jump(result[1]);
  }
  function link(w: Wire, replacing?: string) {
    const problem = connectionError(g, w, replacing);
    if (problem) setMessage(problem); else {
      edit(connect(doc, w, replacing, g.id));
      if (replacing && selected === replacing) select(edgeKey(w));
    }
  }
  function remove(key: string) { edit(removeItem(doc, key, g.id)); if (selected === key) select(null); }
  function timeTravel(type: 'undo' | 'redo') {
    if (locked.current) return;
    setGroup(null); dispatch({ type }); setExtra([]); setOutput(null); setMessage('');
  }
  function replace(next: Document, file: string | null) {
    setGroup(null); setTrail([]); dispatch({ type: 'reset', doc: next }); setSaved(next); setPath(file); select(null); setFocus(null);
    setReset(n => n + 1); setExtra([]); setOutput(null); setInvalidFields(new Set());
  }
  async function action(kind: 'New' | 'Open' | 'Save' | 'Save as' | 'Compile') {
    if (locked.current) return;
    locked.current = true; setBusy(kind); setMessage(''); setExtra([]);
    try {
      if (kind === 'New' || kind === 'Open') {
        const chosen = kind === 'Open' ? await editor.openPath() : null;
        if (kind === 'Open' && !chosen) return;
        if (dirty && !(await editor.discard())) return;
        if (chosen) replace((await editor.load(chosen)).document, chosen);
        else replace(newModel(), null);
      } else if (kind === 'Compile') {
        const folder = await editor.exportPath(path);
        if (folder) { setOutput(await editor.compile(doc, folder, path)); setMessage('Exported to ' + folder); }
      } else {
        const file = kind === 'Save' && path ? path : await editor.savePath(path ?? doc.model.id + '.yaml');
        if (file) { await editor.save(doc, file); setSaved(doc); setPath(file); setMessage('Saved ' + file); }
      }
    } catch (e) {
      const err = backendError(e); setMessage(err.message); setExtra(err.diagnostics);
    } finally { locked.current = false; setBusy(''); }
  }
  useEffect(() => {
    if (!active) return;
    const shortcut = (e: KeyboardEvent) => {
      if (busy || inspect) return;
      const typing = e.target instanceof HTMLElement && !!e.target.closest('input, select, textarea, [contenteditable=true]');
      const key = e.key.toLowerCase(), mod = e.ctrlKey || e.metaKey;
      if (mod && ['s','o','n'].includes(key)) {
        e.preventDefault();
        if (key !== 's' || !invalidFields.size) void action(key === 's' ? e.shiftKey ? 'Save as' : 'Save' : key === 'o' ? 'Open' : 'New');
      } else if (!typing && mod && (key === 'z' || key === 'y')) {
        e.preventDefault(); timeTravel(key === 'y' || e.shiftKey ? 'redo' : 'undo');
      } else if (!typing && selection && ['delete','backspace'].includes(key)) { e.preventDefault(); remove(selection); }
    };
    window.addEventListener('keydown', shortcut);
    return () => window.removeEventListener('keydown', shortcut);
  });
  const offered = registry.filter(op => (op.name + ' ' + op.kind).toLowerCase().includes(query.toLowerCase()));
  return <div className="builder">
    <div className="builder-heading"><div><p className="workspace-label">Model Builder</p><h1>{doc.model.name}</h1></div>
      <div className="file-state" title={path ?? undefined}><span className={dirty ? 'dirty' : ''}>{dirty ? 'Unsaved changes' : path ? 'Saved' : 'New model'}</span>
        {path && <small>{path.split('/').at(-1)}</small>}</div></div>
    <div className="editor-tools" role="toolbar" aria-label="Model actions">
      {(['New','Open','Save','Save as'] as const).map(kind => <button key={kind}
        disabled={!!busy || (kind.startsWith('Save') && !!invalidFields.size)} onClick={() => void action(kind)}>{kind}</button>)}
      <span className="tool-separator" />
      <button disabled={!!busy || !state.past.length} onClick={() => timeTravel('undo')}>Undo</button>
      <button disabled={!!busy || !state.future.length} onClick={() => timeTravel('redo')}>Redo</button>
      <span className="tool-space" /><button disabled={!registry.length || !!busy || !!invalidFields.size} onClick={() => { setExtra([]); validation.validate(); }}>Validate</button>
      <button className="compile-btn" disabled={!registry.length || !!busy || !!invalidFields.size} onClick={() => void action('Compile')}>{busy === 'Compile' ? 'Compiling…' : 'Compile'}</button>
    </div>
    {(message || busy && busy !== 'Compile') && <p className="editor-message" role="status">{busy ? busy + '…' : message}</p>}
    <nav className="graph-crumbs" aria-label="Graph breadcrumbs">{crumbs.map((id,i) =>
      <button key={id} aria-current={i === crumbs.length-1 ? 'page' : undefined} onClick={() => navigate(crumbs.slice(1,i+1))}>
        {i ? 'Definition / ' : 'Model / '}{id}</button>)}</nav>
    <div className="editor-workspace" inert={!!busy}>
      <section className="library editor-panel" aria-label="Node library"><h2>Node library</h2>
        <input type="search" aria-label="Search operations" placeholder="Search operations" value={query} onChange={e => setQuery(e.target.value)} />
        <h4>Graph ports</h4><div className="boundary-actions"><button onClick={() => add('input')}>+ Input</button><button onClick={() => add('output')}>+ Output</button></div>
        {registryError ? <p className="panel-note" role="alert">{registryError} <button onClick={() => setRetry(n => n + 1)}>Retry</button></p>
          : !registry.length && <p>Loading registry…</p>}
        <h4>Reusable blocks</h4>
        <BlockSelection key={reset + g.id} graph={g} initial={selection} ids={group} change={setGroup} create={(ids,name) => {
          try { const [next,key] = extractBlock(doc,g.id,ids,name); edit(next); jump(key); return true; }
          catch (e) { setMessage(backendError(e).message); return false; }
        }} />
        {doc.blocks?.filter(b => b.id.toLowerCase().includes(query.toLowerCase()) && current(doc,b.graph)).map(b =>
          <button className="library-op" key={b.id} onClick={() => add('block:' + b.id)}>Add instance / {b.id}</button>)}
        {['module','operator','python'].map(kind => { const entries = offered.filter(op => op.kind === kind);
          return !!entries.length && <div key={kind}><h4>{kind === 'module' ? 'Modules' : kind === 'python' ? 'Python modules' : 'Operators'}</h4>
            {entries.map(op => <button className="library-op" draggable key={op.id} title={op.description} aria-label={'Add ' + op.name}
              onDragStart={e => { e.dataTransfer.setData('application/mlker-operation', op.id); e.dataTransfer.effectAllowed = 'copy'; }}
              onClick={() => add(op.id)}>{op.name}</button>)}</div>;
        })}
        {!!registry.length && !offered.length && <p className="panel-note">No matching operations.</p>}
      </section>
      <ReactFlowProvider key={reset + g.id}><Canvas group={group}
        toggleGroup={id => setGroup(previous => previous?.includes(id) ? previous.filter(n => n !== id) : [...previous ?? [], id])}
        openBlock={openBlock} graph={g} selection={selection} select={select} focus={focus}
        issues={flagged} error={setMessage} onDropOp={add} onWire={link} onLayout={tiles => edit(autoLayout(doc, tiles, g.id))} onMove={(id, pos) => edit(move(doc, id, pos, g.id))} /></ReactFlowProvider>
      <Inspector graphId={g.id} openBlock={openBlock} key={reset + g.id + '/' + selection} doc={doc} selection={selection} registry={registry} edit={edit} link={link}
        invalid={invalid} remove={remove} rename={id => { if (!selection) return;
          const [side, old] = selection.split(':') as ['input' | 'output', string];
          try { edit(renameBoundary(doc, side, old, id, g.id)); select(boundaryKey(side, id)); } catch (e) { setMessage(backendError(e).message); }
        }} />
    </div>
    <section className="editor-diagnostics" aria-label="Validation diagnostics"><div className="diagnostic-heading"><h2>Diagnostics</h2>
      <span role="status">{invalidFields.size ? 'Invalid property value' : validation.status}</span>
      {output && <button onClick={() => setInspect(true)}>Inspect generated source</button>}</div>
      {!!diagnostics.length && <ul>{diagnostics.map((d, i) => {
        const graph = doc.graphs.find(item => item.id === d.location.graph);
        const key = graph && target(d, graph);
        return <li key={i}><button disabled={!graph} onClick={() => { if (graph && graph !== g) navigate(graphPath(doc, doc.model.entry, graph.id)?.slice(1) ?? [graph.id]); if (key) jump(key); }}><b>{d.severity}</b> {d.message}
          <small>{[d.location.graph,d.location.node ?? d.location.path].filter(Boolean).join(' / ')}</small></button></li>;
      })}</ul>}
    </section>
    {inspect && output && <Source files={output.sources} path={output.path} close={() => setInspect(false)} />}
  </div>;
}
