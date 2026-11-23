import { useState } from 'react';
import type { Graph } from './model';

export function BlockSelection({ graph, initial, ids, change, create }: {
  graph: Graph; initial: string | null; ids: string[] | null; change: (ids: string[] | null) => void;
  create: (ids: string[], name: string) => boolean;
}) {
  const [name, setName] = useState('');
  if (!ids) return <button disabled={!graph.nodes.length} onClick={() => {
    change(initial?.startsWith('node:') ? [initial.slice(5)] : []);
  }}>Create block from nodes</button>;
  return <fieldset><legend>Create block</legend>
    <label className="editor-field">Block ID<input value={name} onChange={e => setName(e.target.value)} /></label>
    <p className="panel-note">Select nodes on the canvas or in this list.</p>
    {graph.nodes.map(n => <label className="block-choice" key={n.id}><input type="checkbox" checked={ids.includes(n.id)}
      onChange={e => change(e.target.checked ? [...ids, n.id] : ids.filter(id => id !== n.id))} />{n.name || n.id}</label>)}
    <button disabled={!ids.length || !name} onClick={() => {
      if (create(ids, name)) { change(null); setName(''); }
    }}>Create</button>
    <button onClick={() => change(null)}>Cancel</button></fieldset>;
}
