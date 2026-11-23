import { useEffect, useId, useRef, useState } from 'react';
import { exposeParameter, current, edgeKey, editNode, editPort, nodeKey, owner, parseParam, parseShape, portId, shapeText,
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
function isReference(value: Value | undefined) {
  return value !== undefined && typeof value === 'object' && '$param' in value;
}
function display(value: Value | undefined) {
  if (value === undefined) return '';
  if (typeof value === 'object' && !Array.isArray(value)) return '$' + value.$param;
  return typeof value === 'string' ? value : JSON.stringify(value);
}

export function Inspector({ graphId, openBlock, doc, selection, registry, edit, link, remove, invalid, rename }: {
  graphId: string; openBlock: (key: string) => void; doc: Document; selection: string | null; registry: Operation[]; edit: (doc: Document) => void;
  link: (wire: Wire, replacing?: string) => void; remove: (id: string) => void; invalid: Invalid; rename: (name: string) => void;
}) {
  const g = current(doc, graphId);
  const node = g.nodes.find(n => nodeKey(n.id) === selection);
  const side = selection?.startsWith('input:') ? 'inputs' : selection?.startsWith('output:') ? 'outputs' : null;
  const port = side ? g[side].find(p => p.id === selection!.split(':')[1]) : null;
  const block = node?.kind === 'block' && doc.blocks?.find(b => b.id === node.block);
  const definition = block && current(doc, block.graph);
  const registered = node?.kind === 'python' && registry.find(o => o.kind === 'python' &&
    o.implementation && Object.entries(o.implementation).every(([key,value]) =>
      node.implementation?.[key as keyof typeof node.implementation] === value));
  const op = definition ? { description: 'Instance of ' + block.id, name: block.id,
    params: Object.fromEntries((definition.parameters ?? []).map(p => [p.id,
      {type:p.type, default:p.default, initial:p.default, minimum:null, maximum:null}])) }
    : registered || registry.find(o => o.id === node?.registry);
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
      {g.id !== doc.model.entry && <h4>Block definition / {g.id}</h4>}
      <Field label="Model name" value={doc.model.name} invalid={invalid} commit={name => {
        if (!name.trim()) throw Error('A model name is required.');
        edit({ ...doc, model: { ...doc.model, name } });
      }} />
      {g.parameters?.map(p => <Field key={p.id} label={'Default / ' + p.id} value={display(p.default)} invalid={invalid}
        commit={text => {
          const value = parseParam(text,{type:p.type,default:p.default,initial:p.default,minimum:null,maximum:null},g);
          if (typeof value === 'object' && '$param' in value) throw Error('Enter a literal default.');
          edit({...doc, graphs:doc.graphs.map(graph => graph !== g ? graph : {...g,
            parameters:g.parameters?.map(param => param.id === p.id ? {...p,default:value} : param)})});
        }} />)}
      <p className="panel-note">Select a node or boundary to edit its properties and connections.</p>
    </>}
    {node && <>
      <p className="object-id">{node.id}</p>
      <Field label="Display name" value={node.name ?? ''} invalid={invalid} commit={name => {
        const gNode = { ...node }; if (name) gNode.name = name; else delete gNode.name;
        const g2 = { ...g, nodes: g.nodes.map(n => n.id === node.id ? gNode : n) };
        edit({ ...doc, graphs: doc.graphs.map(x => x.id === g.id ? g2 : x) });
      }} />
      {definition && <button onClick={() => openBlock(nodeKey(node.id))}>Open block definition</button>}
      {node.kind === 'python' && <fieldset><legend>Python implementation</legend>
        <label className="editor-field">Registered module<select value={registered ? registered.id : ''} onChange={e => {
          const next = registry.find(o => o.kind === 'python' && o.id === e.target.value);
          if (!next?.contracts) return;
          const changed = editNode(doc,node.id,{implementation:next.implementation, ...next.contracts,
            params:Object.fromEntries(Object.entries(next.params).map(([k,v]) => [k,v.initial]))},g.id);
          edit({...changed, graphs:changed.graphs.map(graph => graph.id !== g.id ? graph : {...graph,
            edges:graph.edges.filter(w => (!('node' in w.from) || w.from.node !== node.id || next.outputs.includes(w.from.port)) &&
              (!('node' in w.to) || w.to.node !== node.id || next.inputs.includes(w.to.port)))})});
        }}><option value="">Choose implementation</option>{registry.filter(o => o.kind === 'python').map(o =>
          <option key={o.id} value={o.id}>{o.name}</option>)}</select></label>
        <small>{node.implementation?.package}.{node.implementation?.symbol} {node.implementation?.version}</small></fieldset>}
      {op ? <><p className="panel-note">{op.description}</p>
        {Object.entries(op.params).map(([key, arg]) => <div key={key}><Field label={key} invalid={invalid}
          hint={display(arg.initial)} value={display(node.params?.[key] ?? arg.default ?? undefined)}
          commit={text => { const params = { ...node.params };
            if (!text && arg.default !== null && arg.type !== 'string') delete params[key]; else params[key] = parseParam(text, arg, g);
            edit(editNode(doc, node.id, { params },g.id)); }} />
          {definition && node.params && Object.hasOwn(node.params,key) && <button onClick={() => {
            const params = {...node.params}; delete params[key]; edit(editNode(doc,node.id,{params},g.id));
          }}>Use default for {key}</button>}
          {g.id !== doc.model.entry && !isReference(node.params?.[key]) && <button onClick={() => edit(exposeParameter(doc,g.id,node,key,arg))}>Expose {key}</button>}</div>)}
      </> : <p className="panel-note">This node retains its existing contract. Its implementation is read-only.</p>}
    </>}
    {side && port && <>
      <form className="port-id-form" onSubmit={e => { e.preventDefault(); if (idField.current) rename(idField.current.value); }}>
        <label className="editor-field">Port ID<input ref={idField} defaultValue={port.id} required pattern="[A-Za-z_][A-Za-z0-9_.\-]*" /></label>
        <button>Rename</button>
      </form>
      <Field label="Data type" value={port.tensor.dtype ?? ''} hint="float32" invalid={invalid} commit={dtype => {
        if (dtype && !/^[A-Za-z_][A-Za-z0-9_.-]*$/.test(dtype)) throw Error('Use a dtype identifier.');
        edit(editPort(doc, side, port.id, { tensor: { ...port.tensor, dtype: dtype || null } }, g.id));
      }} />
      <Field label="Tensor shape" value={shapeText(port.tensor.shape)} hint='["B", 1, 8, 8]' invalid={invalid}
        commit={text => edit(editPort(doc, side, port.id, { tensor: { ...port.tensor, shape: parseShape(text) } }, g.id))} />
      <Field label="Layout" value={port.tensor.layout ?? ''} hint="NCHW" invalid={invalid}
        commit={text => edit(editPort(doc, side, port.id, { tensor: { ...port.tensor, layout: text || null } }, g.id))} />
    </>}
    {!!destinations.length && <div className="wiring"><h4>Connections</h4>
      {destinations.map(to => {
        const existing = g.edges.find(e => owner(e.to) === owner(to) && portId(e.to) === portId(to));
        return <label className="editor-field" key={portId(to)}>Source for {portId(to)}
          <select value={existing ? JSON.stringify(existing.from) : ''} onChange={e => {
            if (!e.target.value && existing) remove(edgeKey(existing));
            else if (e.target.value) link({ from: JSON.parse(e.target.value), to }, existing && edgeKey(existing));
          }}>
            <option value="">Disconnected</option>
            {sources.filter(s => owner(s) !== owner(to)).map(s => <option key={JSON.stringify(s)} value={JSON.stringify(s)}>{labelEndpoint(s)}</option>)}
          </select>
        </label>;
      })}</div>}
    {selectedEdge && <p className="panel-note">{labelEndpoint(selectedEdge.from)} → {labelEndpoint(selectedEdge.to)}</p>}
    {selection && <button className="delete-item" onClick={() => remove(selection)}>Delete {selectedEdge ? 'connection' : port ? 'port' : 'node'}</button>}
  </section>;
}
