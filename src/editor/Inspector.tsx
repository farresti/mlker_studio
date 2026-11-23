import { useEffect, useId, useRef, useState } from 'react';
import { current, edgeKey, editNode, editPort, nodeKey, owner, parseParam, parseShape, portId, shapeText,
  type Document, type Endpoint, type Operation, type Value, type Wire } from './model';

type Invalid = (key: string, invalid: boolean) => void;
function Field({ label, value, commit, invalid, hint }: {
  label: string; value: string; commit: (s: string) => void; invalid: Invalid; hint?: string;
}) {
  const [draft, setDraft] = useState(value), [problem, setProblem] = useState('');
  const id = useId();
  useEffect(() => { setDraft(value); setProblem(''); invalid(id, false); }, [value, invalid, id]);
  useEffect(() => () => invalid(id, false), [id, invalid]);
  return <label className="editor-field">{label}
    <input value={draft} placeholder={hint} aria-invalid={!!problem} aria-describedby={problem ? id : undefined}
      onChange={e => { const text = e.target.value; setDraft(text);
        try { commit(text); setProblem(''); invalid(id, false); }
        catch (err) { setProblem(err instanceof SyntaxError ? 'Enter a complete value.' : String((err as Error).message)); invalid(id, true); }
      }} />
    {problem && <small id={id} role="alert">{problem}</small>}
  </label>;
}
function display(value: Value | undefined) {
  if (value === undefined) return '';
  if (typeof value === 'object' && !Array.isArray(value)) return '$' + value.$param;
  return typeof value === 'string' ? value : JSON.stringify(value);
}

export function Inspector({ doc, selection, registry, edit, link, remove, invalid, rename }: {
  doc: Document; selection: string | null; registry: Operation[]; edit: (doc: Document) => void;
  link: (wire: Wire) => void; remove: (id: string) => void; invalid: Invalid; rename: (name: string) => void;
}) {
  const g = current(doc);
  const node = g.nodes.find(n => nodeKey(n.id) === selection);
  const side = selection?.startsWith('input:') ? 'inputs' : selection?.startsWith('output:') ? 'outputs' : null;
  const port = side ? g[side].find(p => p.id === selection!.split(':')[1]) : null;
  const op = registry.find(o => o.id === node?.registry);
  const selectedEdge = g.edges.find(e => edgeKey(e) === selection);
  const destinations: Endpoint[] = node ? node.inputs.map(p => ({ node: node.id, port: p.id }))
    : side === 'outputs' && port ? [{ output: port.id }] : [];
  const sources: Endpoint[] = [...g.inputs.map(p => ({ input: p.id })), ...g.nodes.flatMap(n => n.outputs.map(p => ({ node: n.id, port: p.id })))];
  const title = node?.name || op?.name || node?.id || port?.name || port?.id || (selectedEdge ? 'Connection' : 'Model');
  const labelEndpoint = (e: Endpoint) => 'node' in e ? e.node + ' / ' + e.port : 'input' in e ? 'Input / ' + e.input : 'Output / ' + e.output;
  const idField = useRef<HTMLInputElement>(null);
  return <section className="inspector editor-panel" aria-label="Properties">
    <h2>Properties</h2><h3>{title}</h3>
    {!selection && <>
      <Field label="Model name" value={doc.model.name} invalid={invalid} commit={name => {
        if (!name.trim()) throw Error('A model name is required.');
        edit({ ...doc, model: { ...doc.model, name } });
      }} />
      <p className="panel-note">Select a node or boundary to edit its properties and connections.</p>
    </>}
    {node && <>
      <p className="object-id">{node.id}</p>
      <Field label="Display name" value={node.name ?? ''} invalid={invalid} commit={name => {
        const gNode = { ...node }; if (name) gNode.name = name; else delete gNode.name;
        const g2 = { ...g, nodes: g.nodes.map(n => n.id === node.id ? gNode : n) };
        edit({ ...doc, graphs: doc.graphs.map(x => x.id === g.id ? g2 : x) });
      }} />
      {op ? <><p className="panel-note">{op.description}</p>
        {Object.entries(op.params).map(([key, arg]) => <Field key={key} label={key} invalid={invalid}
          hint={display(arg.initial)} value={display(node.params?.[key] ?? arg.default ?? undefined)}
          commit={text => { const params = { ...node.params };
            if (!text && arg.default !== null) delete params[key]; else params[key] = parseParam(text, arg, g);
            edit(editNode(doc, node.id, { params })); }} />)}
      </> : <p className="panel-note">This node retains its existing contract. Its implementation is read-only.</p>}
    </>}
    {side && port && <>
      <form className="port-id-form" onSubmit={e => { e.preventDefault(); if (idField.current) rename(idField.current.value); }}>
        <label className="editor-field">Port ID<input ref={idField} defaultValue={port.id} required pattern="[A-Za-z_][A-Za-z0-9_.\-]*" /></label>
        <button>Rename</button>
      </form>
      <Field label="Data type" value={port.tensor.dtype ?? ''} hint="float32" invalid={invalid} commit={dtype => {
        if (dtype && !/^[A-Za-z_][A-Za-z0-9_.-]*$/.test(dtype)) throw Error('Use a dtype identifier.');
        edit(editPort(doc, side, port.id, { tensor: { ...port.tensor, dtype: dtype || null } }));
      }} />
      <Field label="Tensor shape" value={shapeText(port.tensor.shape)} hint='["B", 1, 8, 8]' invalid={invalid}
        commit={text => edit(editPort(doc, side, port.id, { tensor: { ...port.tensor, shape: parseShape(text) } }))} />
      <Field label="Layout" value={port.tensor.layout ?? ''} hint="NCHW" invalid={invalid}
        commit={text => edit(editPort(doc, side, port.id, { tensor: { ...port.tensor, layout: text || null } }))} />
    </>}
    {!!destinations.length && <div className="wiring"><h4>Connections</h4>
      {destinations.map(to => {
        const existing = g.edges.find(e => owner(e.to) === owner(to) && portId(e.to) === portId(to));
        return <label className="editor-field" key={portId(to)}>Source for {portId(to)}
          <select value={existing ? JSON.stringify(existing.from) : ''} onChange={e => {
            if (!e.target.value && existing) remove(edgeKey(existing));
            else if (e.target.value) link({ from: JSON.parse(e.target.value), to });
          }}>
            <option value="">Disconnected</option>
            {sources.filter(s => owner(s) !== owner(to)).map(s => <option key={JSON.stringify(s)} value={JSON.stringify(s)}
              disabled={!!existing && (owner(s) !== owner(existing.from) || portId(s) !== portId(existing.from))}>{labelEndpoint(s)}</option>)}
          </select>
        </label>;
      })}</div>}
    {selectedEdge && <p className="panel-note">{labelEndpoint(selectedEdge.from)} → {labelEndpoint(selectedEdge.to)}</p>}
    {selection && <button className="delete-item" onClick={() => remove(selection)}>Delete {selectedEdge ? 'connection' : port ? 'port' : 'node'}</button>}
  </section>;
}
