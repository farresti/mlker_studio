// The editable tree is the JSON representation of the MLKer YAML document.
// Spread existing records on update so unrecognized fields survive round trips.
export type Value = string | number | boolean | number[] | { $param: string };
export type Dim = number | string | null;
export interface Tensor { dtype?: string | null; shape?: Dim[] | null; layout?: string | null }
export interface Port { id: string; name?: string; tensor: Tensor; required?: boolean }
export interface OpNode { id: string; name?: string; kind: string; registry?: string; block?: string;
  implementation?: Implementation; instance_id?: string; parameter_ids?: Record<string,string>;
  inputs: Port[]; outputs: Port[]; params?: Record<string, Value> }
export type Endpoint = { node: string; port: string } | { input: string } | { output: string };
export interface Wire { from: Endpoint; to: Endpoint }
export interface Point { x: number; y: number }
export interface Graph { id: string; inputs: Port[]; outputs: Port[]; nodes: OpNode[]; edges: Wire[];
  parameters?: { id: string; type: string; default: Value }[]; extensions?: Record<string, unknown> }
export interface Document { format: string; schema_version: number; blocks?: { id: string; graph: string }[];
  model: { id: string; revision: string; name: string; entry: string; inputs: string[]; outputs: string[] }; graphs: Graph[] }
export interface Argument { type: string; minimum: number | null; maximum: number | null; default: Value | null; initial: Value }
export interface Implementation { id?: string; package: string; symbol: string; version?: string }
export interface Operation { id: string; name: string; kind: string; inputs: string[]; outputs: string[];
  params: Record<string, Argument>; description: string; implementation?: Implementation; contracts?: {inputs: Port[]; outputs: Port[]} }
export interface Diagnostic { severity: string; code: string; message: string;
  location: { path: string; graph?: string; node?: string; source?: string } }

export const current = (doc: Document, graphId = doc.model.entry) => doc.graphs.find(g => g.id === graphId)!;
export const nodeKey = (id: string) => 'node:' + id;
export const boundaryKey = (side: 'input' | 'output', id: string) => side + ':' + id;
export const owner = (e: Endpoint) => 'node' in e ? nodeKey(e.node) : 'input' in e ? boundaryKey('input', e.input) : boundaryKey('output', e.output);
export const portId = (e: Endpoint) => 'node' in e ? e.port : 'input' in e ? e.input : e.output;
export const edgeKey = (w: Wire) => JSON.stringify([owner(w.from), portId(w.from), owner(w.to), portId(w.to)]);
export const handle = (source: boolean, id: string) => (source ? 'out:' : 'in:') + id;
export function endpoint(key: string, socket: string | null, source: boolean): Endpoint | null {
  if (!socket?.startsWith(source ? 'out:' : 'in:')) return null;
  const [kind, id] = key.split(':');
  const port = socket.slice(socket.indexOf(':') + 1);
  if (kind === 'node') return { node: id, port };
  if (port !== id) return null;
  if (source && kind === 'input') return { input: id };
  if (!source && kind === 'output') return { output: id };
  return null;
}
export function newModel(): Document {
  return { format: 'mlker.model', schema_version: 2,
    model: { id: 'model_' + crypto.randomUUID(), revision: 'r1', name: 'Untitled model', entry: 'main', inputs: [], outputs: [] },
    graphs: [{ id: 'main', inputs: [], outputs: [], nodes: [], edges: [] }] };
}
function updateGraph(doc: Document, graph: Graph, rename?: { field: 'inputs' | 'outputs'; old: string; id: string }): Document {
  const before = current(doc, graph.id);
  const refs = new Set(doc.blocks?.filter(b => b.graph === graph.id).map(b => b.id));
  const changed = (['inputs', 'outputs'] as const).filter(field => graph[field] !== before[field]);
  return { ...doc, graphs: doc.graphs.map(original => {
    const g = original.id === graph.id ? graph : original;
    const calls = new Set(g.nodes.filter(n => n.kind === 'block' && n.block && refs.has(n.block)).map(n => n.id));
    if (!calls.size || !changed.length) return g;
    const nodes = g.nodes.map(n => {
      if (!calls.has(n.id)) return n;
      const call = { ...n };
      for (const field of changed) call[field] = graph[field].map(p => {
        const oldId = rename?.field === field && rename.id === p.id ? rename.old : p.id;
        const existing = n[field].find(v => v.id === oldId);
        const previous = before[field].find(v => v.id === oldId);
        if (!existing || !previous) return p;
        const tensor = { ...existing.tensor };
        for (const key of ['shape','dtype','layout'] as const) {
          if (p.tensor[key] !== previous.tensor[key]) Object.assign(tensor, { [key]: p.tensor[key] });
        }
        return { ...existing, id: p.id, tensor };
      });
      return call;
    });
    function remap(e: Endpoint, field: 'inputs' | 'outputs') {
      if (!('node' in e) || !calls.has(e.node) || !changed.includes(field)) return e;
      const port = rename?.field === field && rename.old === e.port ? rename.id : e.port;
      return graph[field].some(p => p.id === port) ? { ...e, port } : null;
    }
    return { ...g, nodes, edges: g.edges.flatMap(w => {
      const from = remap(w.from, 'outputs'), to = remap(w.to, 'inputs');
      return from && to ? [{ ...w, from, to }] : [];
    }) };
  }) };
}
function layout(g: Graph) {
  return (g.extensions?.['mlker.editor'] ?? {}) as { positions?: Record<string, Point> };
}
export function position(g: Graph, key: string, index: number): Point {
  const pos = layout(g).positions?.[key];
  return pos && Number.isFinite(pos.x) && Number.isFinite(pos.y) ? pos :
    { x: 40 + (index % 4) * 210, y: 60 + Math.floor(index / 4) * 150 };
}
function placed(graph: Graph, changes: Record<string, Point | null>): Graph {
  const extension = layout(graph);
  const positions = { ...extension.positions };
  for (const [key, point] of Object.entries(changes)) {
    if (point) positions[key] = point; else delete positions[key];
  }
  return { ...graph, extensions: { ...graph.extensions, 'mlker.editor': { ...extension, positions } } };
}
export const move = (doc: Document, key: string, point: Point, graphId = doc.model.entry) => updateGraph(doc, placed(current(doc, graphId), { [key]: point }));
export interface LayoutTile { id: string; measured?: { width?: number; height?: number } }

export function autoLayout(doc: Document, tiles: LayoutTile[], graphId = doc.model.entry): Document {
  const graph = current(doc, graphId);
  const vertices = new Map(tiles.map(t => [t.id, { incoming: 0, next: [] as string[], depth: 0 }]));
  for (const edge of graph.edges) {
    const from = vertices.get(owner(edge.from)), to = vertices.get(owner(edge.to));
    if (from && to) { from.next.push(owner(edge.to)); to.incoming++; }
  }
  // Longest dependency path sets each column. Cyclic drafts share a final column.
  const queue = tiles.filter(t => !vertices.get(t.id)!.incoming).map(t => t.id);
  let maxDepth = 0;
  for (let i = 0; i < queue.length; i++) {
    const vertex = vertices.get(queue[i])!;
    maxDepth = Math.max(maxDepth, vertex.depth);
    for (const id of vertex.next) {
      const next = vertices.get(id)!;
      next.depth = Math.max(next.depth, vertex.depth + 1);
      if (--next.incoming === 0) queue.push(id);
    }
  }
  const columns: LayoutTile[][] = [];
  for (const tile of tiles) {
    const vertex = vertices.get(tile.id)!;
    const depth = vertex.incoming ? maxDepth + 1 : vertex.depth;
    (columns[depth] ??= []).push(tile);
  }
  let x = 40;
  const changes: Record<string, Point> = {};
  for (const column of columns) {
    if (!column) continue;
    let y = 60, width = 0;
    for (const tile of column) {
      changes[tile.id] = { x, y };
      y += (tile.measured?.height ?? 140) + 48;
      width = Math.max(width, tile.measured?.width ?? 210);
    }
    x += width + 88;
  }
  const previous = layout(graph).positions;
  if (tiles.every(({ id }) => previous?.[id]?.x === changes[id].x && previous[id].y === changes[id].y)) return doc;
  return updateGraph(doc, placed(graph, changes));
}

function unique(base: string, ids: string[]) {
  const taken = new Set(ids);
  let id = base;
  for (let i = 2; taken.has(id); i++) id = base + '_' + i;
  return id;
}
export function addNode(doc: Document, op: Operation, point: Point, graphId = doc.model.entry): [Document, string] {
  const g = current(doc, graphId);
  const id = unique(op.implementation ? 'python_node' : op.name.toLowerCase(), g.nodes.map(n => n.id));
  const ports = (ids: string[]) => ids.map(id => ({ id, tensor: {} }));
  const node: OpNode = { id, kind: op.kind, ...(op.implementation ? { implementation: op.implementation } : { registry: op.id }), inputs: op.contracts?.inputs ?? ports(op.inputs), outputs: op.contracts?.outputs ?? ports(op.outputs),
    params: Object.fromEntries(Object.entries(op.params).map(([name, p]) => [name, p.initial])) };
  return [updateGraph(doc, placed({ ...g, nodes: [...g.nodes, node] }, { [nodeKey(id)]: point })), nodeKey(id)];
}
export function addBoundary(doc: Document, side: 'input' | 'output', point: Point, graphId = doc.model.entry): [Document, string] {
  const g = current(doc, graphId), field = side === 'input' ? 'inputs' : 'outputs';
  const id = unique(side, g[field].map(p => p.id));
  const key = boundaryKey(side, id);
  const tensor: Tensor = side === 'input' ? { dtype: 'float32', shape: ['B', 1, 8, 8] } : {};
  const next = updateGraph(doc, placed({ ...g, [field]: [...g[field], { id, tensor }] }, { [key]: point }));
  if (graphId !== doc.model.entry) return [next, key];
  return [{ ...next, model: { ...next.model, [field]: [...doc.model[field], id] } }, key];
}
export function editNode(doc: Document, id: string, patch: Partial<OpNode>, graphId = doc.model.entry) {
  const g = current(doc, graphId);
  return updateGraph(doc, { ...g, nodes: g.nodes.map(n => n.id === id ? { ...n, ...patch } : n) });
}
export function editPort(doc: Document, field: 'inputs' | 'outputs', id: string, patch: Partial<Port>, graphId = doc.model.entry) {
  const g = current(doc, graphId);
  return updateGraph(doc, { ...g, [field]: g[field].map(p => p.id === id ? { ...p, ...patch } : p) });
}
export function removeItem(doc: Document, key: string, graphId = doc.model.entry) {
  const g = current(doc, graphId);
  if (key.startsWith('[')) return updateGraph(doc, { ...g, edges: g.edges.filter(w => edgeKey(w) !== key) });
  const [kind, id] = key.split(':');
  const graph = placed({ ...g, edges: g.edges.filter(w => owner(w.from) !== key && owner(w.to) !== key) }, { [key]: null });
  if (kind === 'node') return updateGraph(doc, { ...graph, nodes: graph.nodes.filter(n => n.id !== id) });
  const field = kind === 'input' ? 'inputs' : 'outputs';
  return { ...updateGraph(doc, { ...graph, [field]: graph[field].filter(p => p.id !== id) }),
    model: graphId !== doc.model.entry ? doc.model : { ...doc.model, [field]: doc.model[field].filter(p => p !== id) } };
}
function lookup(g: Graph, e: Endpoint, source: boolean) {
  if ('node' in e) return g.nodes.find(n => n.id === e.node)?.[source ? 'outputs' : 'inputs'].find(p => p.id === e.port);
  if (source && 'input' in e) return g.inputs.find(p => p.id === e.input);
  if (!source && 'output' in e) return g.outputs.find(p => p.id === e.output);
}
export function connectionError(g: Graph, w: Wire, replacing?: string): string | null {
  const a = lookup(g, w.from, true), b = lookup(g, w.to, false);
  if (!a || !b) return 'Connect a source port to a destination port.';
  const edges = replacing ? g.edges.filter(e => edgeKey(e) !== replacing) : g.edges;
  if (edges.some(e => owner(e.to) === owner(w.to) && portId(e.to) === portId(w.to))) return 'This port already has a connection.';
  for (const field of ['dtype', 'layout'] as const) {
    if (a.tensor[field] && b.tensor[field] && a.tensor[field] !== b.tensor[field]) return 'Incompatible ' + field + '.';
  }
  const x = a.tensor.shape, y = b.tensor.shape;
  if (x && y && (x.length !== y.length || x.some((d, i) => typeof d === 'number' && typeof y[i] === 'number' && d !== y[i]))) return 'Incompatible tensor shapes.';
  const downstream = new Map<string, string[]>();
  for (const edge of edges) {
    const key = owner(edge.from);
    if (!downstream.has(key)) downstream.set(key, []);
    downstream.get(key)!.push(owner(edge.to));
  }
  const pending = [owner(w.to)], seen = new Set<string>();
  while (pending.length) {
    const key = pending.pop()!;
    if (key === owner(w.from)) return 'This connection would create a cycle.';
    if (!seen.has(key)) { seen.add(key); pending.push(...downstream.get(key) ?? []); }
  }
  return null;
}
export function connect(doc: Document, wire: Wire, replacing?: string, graphId = doc.model.entry) {
  const g = current(doc, graphId);
  if (connectionError(g, wire, replacing)) return doc;
  if (replacing) {
    if (edgeKey(wire) === replacing || !g.edges.some(e => edgeKey(e) === replacing)) return doc;
    return updateGraph(doc, { ...g, edges: g.edges.map(e => edgeKey(e) === replacing ? { ...e, ...wire } : e) });
  }
  return updateGraph(doc, { ...g, edges: [...g.edges, wire] });
}
export function parseParam(text: string, arg: Argument, graph: Graph): Value {
  const trimmed = text.trim();
  if (trimmed.startsWith('$') && arg.type !== 'string') {
    const id = trimmed.slice(1);
    if (!graph.parameters?.some(p => p.id === id && p.type === arg.type)) {
      throw Error('Choose a graph parameter of type ' + arg.type + '.');
    }
    return { $param: id };
  }
  const value = arg.type === 'string' ? text : JSON.parse(text);
  const ok = arg.type === 'int' ? Number.isSafeInteger(value) : arg.type === 'float' ? typeof value === 'number' && Number.isFinite(value)
    : arg.type === 'ints' ? Array.isArray(value) && value.every(Number.isSafeInteger)
    : typeof value === (arg.type === 'bool' ? 'boolean' : 'string');
  if (!ok) throw Error('Expected ' + arg.type + '.');
  if (arg.minimum !== null && value < arg.minimum) throw Error('Minimum is ' + arg.minimum + '.');
  if (arg.maximum !== null && value > arg.maximum) throw Error('Maximum is ' + arg.maximum + '.');
  return value;
}
export function shapeText(shape: Tensor['shape']) { return shape ? JSON.stringify(shape) : ''; }
export function parseShape(text: string): Dim[] | null {
  if (!text.trim()) return null;
  const shape = JSON.parse(text);
  if (!Array.isArray(shape) || !shape.every(d => d === null || (Number.isSafeInteger(d) && d >= 0) ||
    (typeof d === 'string' && /^[A-Za-z_][A-Za-z0-9_.-]*$/.test(d)))) throw Error('Use an array of sizes, symbols or null.');
  return shape;
}
export interface History { doc: Document; past: Document[]; future: Document[] }
export type Action = { type: 'edit' | 'reset'; doc: Document } | { type: 'undo' | 'redo' };
export function history(state: History, action: Action): History {
  if (action.type === 'reset') return { doc: action.doc, past: [], future: [] };
  if (action.type === 'edit') return action.doc === state.doc ? state : { doc: action.doc, past: [...state.past.slice(-79), state.doc], future: [] };
  if (action.type === 'undo' && state.past.length) return { doc: state.past.at(-1)!, past: state.past.slice(0, -1), future: [state.doc, ...state.future] };
  if (action.type === 'redo' && state.future.length) return { doc: state.future[0], past: [...state.past, state.doc], future: state.future.slice(1) };
  return state;
}

export function renameBoundary(doc: Document, side: 'input' | 'output', old: string, id: string, graphId = doc.model.entry): Document {
  if (old === id) return doc;
  const g = current(doc, graphId), field = side === 'input' ? 'inputs' : 'outputs';
  if (!/^[A-Za-z_][A-Za-z0-9_.-]*$/.test(id) || g[field].some(p => p.id === id)) throw Error('Choose a unique port ID.');
  const change = (e: Endpoint): Endpoint => {
    if (side === 'input' && 'input' in e && e.input === old) return { input: id };
    if (side === 'output' && 'output' in e && e.output === old) return { output: id };
    return e;
  };
  const graph = { ...g, [field]: g[field].map(p => p.id === old ? { ...p, id } : p),
    edges: g.edges.map(e => ({ ...e, from: change(e.from), to: change(e.to) })) };
  const oldKey = boundaryKey(side, old), pos = layout(g).positions?.[oldKey];
  return { ...updateGraph(doc, pos ? placed(graph, { [oldKey]: null, [boundaryKey(side, id)]: pos }) : graph, {field, old, id}),
    model: graphId !== doc.model.entry ? doc.model : { ...doc.model, [field]: doc.model[field].map(p => p === old ? id : p) } };
}

export function graphPath(doc: Document, start: string, goal: string): string[] | undefined {
  const seen = new Set<string>(), queue = [[start]];
  for (const path of queue) {
    const id = path.at(-1)!;
    if (id === goal) return path;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const n of current(doc,id)?.nodes ?? []) {
      const block = n.kind === 'block' && doc.blocks?.find(b => b.id === n.block);
      if (block) queue.push([...path,block.graph]);
    }
  }
}

export function addBlock(doc: Document, block: string, point: Point, graphId: string): [Document, string] {
  const g = current(doc, graphId);
  const definition = doc.blocks?.find(b => b.id === block);
  const child = definition && current(doc, definition.graph);
  if (!child) throw Error('Missing definition for ' + block);
  if (graphPath(doc, child.id, graphId)) throw Error('Adding this instance would create a recursive block.');
  const id = unique(block, g.nodes.map(n => n.id));
  const call: OpNode = { id, kind: 'block', block, inputs: child.inputs, outputs: child.outputs };
  return [updateGraph(doc, placed({ ...g, nodes: [...g.nodes, call] }, { [nodeKey(id)]: point })), nodeKey(id)];
}

export function extractBlock(doc: Document, graphId: string, ids: string[], block: string): [Document, string] {
  const g = current(doc, graphId), chosen = new Set(ids);
  if (!/^[A-Za-z_][A-Za-z0-9_.-]*$/.test(block) || doc.blocks?.some(b => b.id === block))
    throw Error('Choose a unique block ID.');
  const nodes = g.nodes.filter(n => chosen.has(n.id)), outside = g.nodes.filter(n => !chosen.has(n.id));
  if (!nodes.length) throw Error('Select nodes to include.');
  const identities = (n: OpNode) => [n.instance_id && 'module:' + n.instance_id,
    ...Object.values(n.parameter_ids ?? {}).map(id => 'slot:' + id)].filter(Boolean);
  const shared = new Set(nodes.flatMap(identities));
  if (outside.some(n => identities(n).some(id => shared.has(id)))) throw Error('Include all nodes sharing a module or parameter.');
  const graph = unique(block, doc.graphs.map(x => x.id));
  const call = unique(block, g.nodes.map(n => n.id));
  const refIds = new Set(nodes.flatMap(n => Object.values(n.params ?? {}).flatMap(v =>
    typeof v === 'object' && '$param' in v ? [v.$param] : [])));
  const parameters = g.parameters?.filter(p => refIds.has(p.id)) ?? [];
  const child: Graph = { id: graph, nodes, parameters, edges: [], inputs: [], outputs: [] };
  const parent: Graph = { ...g, nodes: [...outside], edges: [] };
  const cuts = { inputs: new Map<string,string>(), outputs: new Map<string,string>() };
  for (const w of g.edges) {
    const src = 'node' in w.from && chosen.has(w.from.node), dst = 'node' in w.to && chosen.has(w.to.node);
    if (src === dst) { (src ? child : parent).edges.push(w); continue; }
    const field = src ? 'outputs' : 'inputs';
    const key = owner(w.from) + ':' + portId(w.from);
    let port = cuts[field].get(key);
    if (!port) {
      port = unique(portId(w.from), child[field].map(p => p.id));
      cuts[field].set(key, port);
      child[field].push({ id: port, tensor: lookup(g, w.from, true)?.tensor ?? {} });
      if (src) child.edges.push({ from: w.from, to: { output: port } });
      else parent.edges.push({ from: w.from, to: { node: call, port } });
    }
    if (src) parent.edges.push({ ...w, from: { node: call, port } });
    else child.edges.push({ ...w, from: { input: port } });
  }
  parent.nodes.push({ id: call, kind: 'block', block, inputs: child.inputs, outputs: child.outputs,
    params: Object.fromEntries(parameters.map(p => [p.id, { $param: p.id }])) });
  for (const w of parent.edges.filter(w => 'node' in w.from && w.from.node === call)) {
    if (connectionError(parent, w, edgeKey(w))?.includes('cycle')) throw Error('This selection would create a cycle across the block boundary.');
  }
  const positions = Object.fromEntries(nodes.map(n => [nodeKey(n.id), position(g, nodeKey(n.id), g.inputs.length + g.nodes.indexOf(n))]));
  const parentLayout = Object.fromEntries(Object.keys(positions).map(id => [id, null]));
  const next = updateGraph(doc, placed(parent, { ...parentLayout, [nodeKey(call)]: positions[nodeKey(nodes[0].id)] }));
  const points = Object.values(positions);
  const left = Math.min(...points.map(p => p.x)), right = Math.max(...points.map(p => p.x));
  const top = Math.min(...points.map(p => p.y));
  child.inputs.forEach((p,i) => { positions[boundaryKey('input',p.id)] = {x:left - 250, y:top + i * 150}; });
  child.outputs.forEach((p,i) => { positions[boundaryKey('output',p.id)] = {x:right + 250, y:top + i * 150}; });
  return [{ ...next, graphs: [...next.graphs, placed(child, positions)], blocks: [...doc.blocks ?? [], { id: block, graph }] }, nodeKey(call)];
}

export function exposeParameter(doc: Document, graphId: string, node: OpNode, key: string, arg: Argument) {
  const g = current(doc, graphId), value = node.params?.[key] ?? arg.default ?? arg.initial;
  if (typeof value === 'object' && '$param' in value) return doc;
  const id = unique(key, g.parameters?.map(p => p.id) ?? []);
  const next = editNode(doc, node.id, { params: { ...node.params, [key]: { $param: id } } }, graphId);
  return updateGraph(next, { ...current(next, graphId), parameters: [...g.parameters ?? [], { id, type: arg.type, default: value }] });
}
