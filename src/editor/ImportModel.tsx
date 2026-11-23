import { useEffect, useRef, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { backendError, editor } from './api';
import { Canvas } from './Canvas';
import { current, type Document } from './model';

type Catalog = Awaited<ReturnType<typeof editor.browse>>;
const noop = () => {};
export function ImportModel({close, accept}: {close: () => void; accept: (doc: Document) => Promise<void>}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [source, setSource] = useState('pytorch'), [identity,setIdentity] = useState('');
  const [config,setConfig] = useState('{}'), [inputs,setInputs] = useState('[]');
  const [strategy,setStrategy] = useState('fx'), [trusted,setTrusted] = useState(false);
  const [pretrained,setPretrained] = useState(false), [verify,setVerify] = useState(true);
  const [weights,setWeights] = useState(''), [checkpoint,setCheckpoint] = useState('');
  const [query,setQuery] = useState(''), [filter,setFilter] = useState(false), [offset,setOffset] = useState(0);
  const [catalog,setCatalog] = useState<Catalog | null>(null), [busy,setBusy] = useState(''), [error,setError] = useState('');
  const [preview,setPreview] = useState<Document | null>(null);
  useEffect(() => { dialog.current?.showModal?.(); }, []);
  async function browse(start: number) {
    setBusy('Searching'); setError('');
    try { setCatalog(await editor.browse(query,filter,start)); setOffset(start); }
    catch (e) { setError(backendError(e).message); } finally { setBusy(''); }
  }
  async function load() {
    setError(''); setBusy('Importing');
    try {
      const response = await editor.importModel({source, identity, strategy, trusted:source === 'pytorch' && trusted,
        config:JSON.parse(config), inputs:JSON.parse(inputs), pretrained:source === 'timm' && pretrained,
        verify, checkpoint:checkpoint || null, weights_dir:weights || null});
      setPreview(response.document);
    } catch (e) { setError(backendError(e).message); } finally { setBusy(''); }
  }
  const report = preview?.model.provenance;
  return <dialog ref={dialog} className="source-dialog import-dialog" aria-labelledby="import-title"
    onCancel={e => { e.preventDefault(); if (!busy) close(); }}>
    <div className="source-top"><h2 id="import-title">Import model</h2><button disabled={!!busy} onClick={close}>Close</button></div>
    {error && <p role="alert">{error}</p>}
    {busy && <p role="status">{busy}…</p>}
    {preview && report ? <>
      <h3>{String(report.quality)}</h3><p>{String(report.source_identity)} · {String(report.strategy)} · Eval comparison {String((report.verification as {status:string}).status)}</p>
      {typeof (report.verification as {detail?:string}).detail === 'string' && <p>{(report.verification as {detail:string}).detail}</p>}
      <div className="import-preview"><ReactFlowProvider><Canvas readOnly graph={current(preview)} selection={null} issues={new Set()} focus={null}
        select={noop} onMove={noop} onLayout={noop} onDropOp={noop} onWire={noop} error={noop} /></ReactFlowProvider></div>
      <ul>{(report.warnings as string[]).map((warning,i) => <li key={i}>{warning}</li>)}</ul>
      <details><summary>Converted and opaque nodes</summary><ul>{current(preview).nodes.map(n =>
        <li key={n.id}>{n.id} · {n.registry ?? 'opaque'} · {n.name}</li>)}</ul></details>
      <details><summary>Source module hierarchy</summary><ul>{(report.hierarchy as {path:string;class:string}[]).map(m =>
        <li key={m.path}>{m.path || '(root)'} · {m.class}</li>)}</ul></details>
      <p>Weights · {String((report.weights as {format:string}).format)} {String((report.weights as {path?:string}).path ?? '')}</p>
      <button onClick={() => setPreview(null)}>Back</button>
      <button onClick={() => { void accept(preview).catch(e => setError(backendError(e).message)); }}>Open in editor</button>
    </> : <fieldset disabled={!!busy}>
      <label>Import source<select value={source} onChange={e => {
        setSource(e.target.value); setIdentity(''); setTrusted(false); setPretrained(false); setError('');
      }}><option value="pytorch">PyTorch constructor</option><option value="timm">timm registry</option></select></label>
      {source === 'timm' && <>
        <label>Search architecture names<input value={query} onChange={e => {setQuery(e.target.value); setCatalog(null);}} /></label>
        <label><input type="checkbox" checked={filter} onChange={e => {setFilter(e.target.checked); setCatalog(null);}} /> Has pretrained weights</label>
        <button onClick={() => void browse(0)}>Search installed registry</button>
        {catalog && <><p>{catalog.total} matches</p><div className="import-catalog">{catalog.models.map(m =>
          <button key={m.name} onClick={() => {setIdentity(m.name); setPretrained(false);
            if (m.input_size) setInputs(JSON.stringify([{shape:[1,...m.input_size]}]));}}>
            {m.name} · {m.input_size?.join(' × ')} · {m.num_classes ?? '?'} classes</button>)}</div>
          {offset > 0 && <button onClick={() => void browse(offset-50)}>Previous</button>}
          {offset+50 < catalog.total && <button onClick={() => void browse(offset+50)}>Next</button>}</>}
      </>}
      <label>{source === 'pytorch' ? 'Constructor (package.module:Class or /path/file.py:Class)' : 'Registry model name'}
        <input value={identity} onChange={e => { setIdentity(e.target.value); setTrusted(false); setPretrained(false); }} /></label>
      <label>Constructor options (JSON object)<textarea value={config} onChange={e => {setConfig(e.target.value); setTrusted(false);}} /></label>
      <label>Example tensors (JSON array)<textarea placeholder='[{"shape":[1,3,224,224],"dtype":"float32"}]'
        value={inputs} onChange={e => setInputs(e.target.value)} /></label>
      <label>Strategy<select value={strategy} onChange={e => setStrategy(e.target.value)}>
        <option value="fx">Attempt editable FX conversion</option><option value="composite">Inspect as a composite</option></select></label>
      <label>Optional tensor state_dict file<input value={checkpoint} onChange={e => setCheckpoint(e.target.value)} /></label>
      <label>Optional directory for a new weights snapshot<input value={weights} onChange={e => setWeights(e.target.value)} /></label>
      {strategy === 'fx' && <label><input type="checkbox" checked={verify} onChange={e => setVerify(e.target.checked)} /> Compare eval outputs using example inputs</label>}
      {source === 'pytorch' ? <label><input type="checkbox" checked={trusted} onChange={e => setTrusted(e.target.checked)} /> I trust this constructor to execute with my user permissions</label>
        : <label><input type="checkbox" checked={pretrained} onChange={e => setPretrained(e.target.checked)} /> Load pretrained weights (allows checkpoint download)</label>}
      <button disabled={!identity || (source === 'pytorch' && !trusted)} onClick={() => void load()}>Preview import</button>
    </fieldset>}
  </dialog>;
}
