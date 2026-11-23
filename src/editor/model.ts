// The editable tree is the JSON representation of the MLKer YAML document.
// Spread existing records on update so unrecognized fields survive round trips.
export type Value = string | number | boolean | number[] | { $param: string };
export type Dim = number | string | null;
export interface Tensor { dtype?: string | null; shape?: Dim[] | null; layout?: string | null }
export interface Port { id: string; name?: string; tensor: Tensor; required?: boolean }
export interface OpNode { id: string; name?: string; kind: string; registry?: string; block?: string;
  inputs: Port[]; outputs: Port[]; params?: Record<string, Value> }
export type Endpoint = { node: string; port: string } | { input: string } | { output: string };
export interface Wire { from: Endpoint; to: Endpoint }
export interface Point { x: number; y: number }
export interface Graph { id: string; inputs: Port[]; outputs: Port[]; nodes: OpNode[]; edges: Wire[];
  parameters?: { id: string; type: string }[]; extensions?: Record<string, unknown> }
export interface Document { format: string; schema_version: number; blocks?: { id: string; graph: string }[];
  model: { id: string; revision: string; name: string; entry: string; inputs: string[]; outputs: string[] }; graphs: Graph[] }
export interface Argument { type: string; minimum: number | null; maximum: number | null; default: Value | null; initial: Value }
export interface Operation { id: string; name: string; kind: string; inputs: string[]; outputs: string[];
  params: Record<string, Argument>; description: string }
export interface Diagnostic { severity: string; code: string; message: string;
  location: { path: string; graph?: string; node?: string; source?: string } }

export const current = (doc: Document) => doc.graphs.find(g => g.id === doc.model.entry)!;
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
function updateGraph(doc: Document, graph: Graph): Document {
  return { ...doc, graphs: doc.graphs.map(g => g.id === graph.id ? graph : g) };
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
export const move = (doc: Document, key: string, point: Point) => updateGraph(doc, placed(current(doc), { [key]: point }));
function unique(base: string, ids: string[]) {
  const taken = new Set(ids);
  let id = base;
  for (let i = 2; taken.has(id); i++) id = base + '_' + i;
  return id;
}
export function addNode(doc: Document, op: Operation, point: Point): [Document, string] {
  const g = current(doc);
  const id = unique(op.name.toLowerCase(), g.nodes.map(n => n.id));
  const ports = (ids: string[]) => ids.map(id => ({ id, tensor: {} }));
  const node: OpNode = { id, kind: op.kind, registry: op.id, inputs: ports(op.inputs), outputs: ports(op.outputs),
    params: Object.fromEntries(Object.entries(op.params).map(([name, p]) => [name, p.initial])) };
  return [updateGraph(doc, placed({ ...g, nodes: [...g.nodes, node] }, { [nodeKey(id)]: point })), nodeKey(id)];
}
export function addBoundary(doc: Document, side: 'input' | 'output', point: Point): [Document, string] {
  const g = current(doc), field = side === 'input' ? 'inputs' : 'outputs';
  const id = unique(side, g[field].map(p => p.id));
  const key = boundaryKey(side, id);
  const tensor: Tensor = side === 'input' ? { dtype: 'float32', shape: ['B', 1, 8, 8] } : {};
  const next = updateGraph(doc, placed({ ...g, [field]: [...g[field], { id, tensor }] }, { [key]: point }));
  return [{ ...next, model: { ...next.model, [field]: [...doc.model[field], id] } }, key];
}
export function editNode(doc: Document, id: string, patch: Partial<OpNode>) {
  const g = current(doc);
  return updateGraph(doc, { ...g, nodes: g.nodes.map(n => n.id === id ? { ...n, ...patch } : n) });
}
export function editPort(doc: Document, field: 'inputs' | 'outputs', id: string, patch: Partial<Port>) {
  const g = current(doc);
  return updateGraph(doc, { ...g, [field]: g[field].map(p => p.id === id ? { ...p, ...patch } : p) });
}
export function removeItem(doc: Document, key: string) {
  const g = current(doc);
  if (key.startsWith('[')) return updateGraph(doc, { ...g, edges: g.edges.filter(w => edgeKey(w) !== key) });
  const [kind, id] = key.split(':');
  const graph = placed({ ...g, edges: g.edges.filter(w => owner(w.from) !== key && owner(w.to) !== key) }, { [key]: null });
  if (kind === 'node') return updateGraph(doc, { ...graph, nodes: graph.nodes.filter(n => n.id !== id) });
  const field = kind === 'input' ? 'inputs' : 'outputs';
  return { ...updateGraph(doc, { ...graph, [field]: graph[field].filter(p => p.id !== id) }),
    model: { ...doc.model, [field]: doc.model[field].filter(p => p !== id) } };
}
function lookup(g: Graph, e: Endpoint, source: boolean) {
  if ('node' in e) return g.nodes.find(n => n.id === e.node)?.[source ? 'outputs' : 'inputs'].find(p => p.id === e.port);
  if (source && 'input' in e) return g.inputs.find(p => p.id === e.input);
  if (!source && 'output' in e) return g.outputs.find(p => p.id === e.output);
}
export function connectionError(g: Graph, w: Wire): string | null {
  const a = lookup(g, w.from, true), b = lookup(g, w.to, false);
  if (!a || !b) return 'Connect a source port to a destination port.';
  if (g.edges.some(e => owner(e.to) === owner(w.to) && portId(e.to) === portId(w.to))) return 'This port already has a connection.';
  for (const field of ['dtype', 'layout'] as const) {
    if (a.tensor[field] && b.tensor[field] && a.tensor[field] !== b.tensor[field]) return 'Incompatible ' + field + '.';
  }
  const x = a.tensor.shape, y = b.tensor.shape;
  if (x && y && (x.length !== y.length || x.some((d, i) => typeof d === 'number' && typeof y[i] === 'number' && d !== y[i]))) return 'Incompatible tensor shapes.';
  const downstream = new Map<string, string[]>();
  for (const edge of g.edges) {
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
export function connect(doc: Document, wire: Wire) {
  const g = current(doc);
  if (connectionError(g, wire)) return doc;
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

export function renameBoundary(doc: Document, side: 'input' | 'output', old: string, id: string): Document {
  if (old === id) return doc;
  const g = current(doc), field = side === 'input' ? 'inputs' : 'outputs';
  if (!/^[A-Za-z_][A-Za-z0-9_.-]*$/.test(id) || g[field].some(p => p.id === id)) throw Error('Choose a unique port ID.');
  if (doc.graphs.some(graph => graph.nodes.some(n => n.kind === 'block' &&
      doc.blocks?.some(b => b.id === n.block && b.graph === g.id)))) {
    throw Error('This graph supplies a reusable block contract. Its ports cannot be renamed here.');
  }
  const change = (e: Endpoint): Endpoint => {
    if (side === 'input' && 'input' in e && e.input === old) return { input: id };
    if (side === 'output' && 'output' in e && e.output === old) return { output: id };
    return e;
  };
  const graph = { ...g, [field]: g[field].map(p => p.id === old ? { ...p, id } : p),
    edges: g.edges.map(e => ({ ...e, from: change(e.from), to: change(e.to) })) };
  const oldKey = boundaryKey(side, old), pos = layout(g).positions?.[oldKey];
  return { ...updateGraph(doc, pos ? placed(graph, { [oldKey]: null, [boundaryKey(side, id)]: pos }) : graph),
    model: { ...doc.model, [field]: doc.model[field].map(p => p === old ? id : p) } };
}
