import { describe, expect, it } from 'vitest';
import { flowEdge, project, wire } from './Canvas';
import { addBoundary, addNode, autoLayout, connect, connectionError, current, edgeKey, history, move, newModel, parseParam,
  parseShape, removeItem, renameBoundary, type Argument, type Operation, type Wire } from './model';
import { backendError } from './api';

const relu: Operation = { id: 'core.ReLU', kind: 'operator', name: 'ReLU', inputs: ['x'], outputs: ['out'], params: {}, description: '' };
const point = { x: 80, y: 120 };

function graph() {
  let doc = newModel();
  [doc] = addBoundary(doc, 'input', point);
  [doc] = addNode(doc, relu, point);
  [doc] = addBoundary(doc, 'output', point);
  return doc;
}
const into: Wire = { from: { input: 'input' }, to: { node: 'relu', port: 'x' } };
const out: Wire = { from: { node: 'relu', port: 'out' }, to: { output: 'output' } };

describe('MLKer canvas adapter', () => {
  it('round trips both boundary directions and qualifies handles', () => {
    for (const value of [into, out]) {
      const edge = flowEdge(value);
      expect(wire({ ...edge, sourceHandle: edge.sourceHandle!, targetHandle: edge.targetHandle! })).toEqual(value);
      expect(edge.id).toEqual(edgeKey(value));
    }
    expect(project(current(graph())).map(n => n.id)).toEqual(['input:input', 'node:relu', 'output:output']);
  });
  it('keeps public IDs, wire references, positions and source extension fields in sync', () => {
    const doc = connect(connect(graph(), into), out);
    const withProvenance = { ...doc, importHint: { future: 1 } };
    const renamed = renameBoundary(withProvenance, 'input', 'input', 'pixels');
    const loaded = JSON.parse(JSON.stringify(renamed));
    expect(loaded.importHint).toEqual({ future: 1 });
    expect(loaded.model.inputs).toEqual(['pixels']);
    expect(current(loaded).edges[0].from).toEqual({ input: 'pixels' });
    expect(project(current(loaded))[0].position).toEqual(point);
    expect(current(removeItem(loaded, 'node:relu')).edges).toEqual([]);
  });
  it('rejects occupied targets, incorrect directions, and cycles', () => {
    const doc = connect(graph(), into);
    expect(connectionError(current(doc), into)).toMatch(/already/);
    expect(connectionError(current(doc), { from: out.to, to: into.to })).toMatch(/source/);
    const [next] = addNode(graph(), relu, point);
    const connected = connect(next, { from: out.from, to: { node: 'relu_2', port: 'x' } });
    expect(connectionError(current(connected), { from: { node: 'relu_2', port: 'out' }, to: into.to })).toMatch(/cycle/);
  });
  it('undoes and redoes layout changes without touching architecture IDs', () => {
    const doc = graph();
    const next = move(doc, 'node:relu', { x: 200, y: 50 });
    const edited = history({ doc, past: [], future: [] }, { type: 'edit', doc: next });
    const undone = history(edited, { type: 'undo' });
    expect(undone.doc).toBe(doc);
    expect(history(undone, { type: 'redo' }).doc).toBe(next);
    expect(current(next).nodes).toBe(current(doc).nodes);
  });
});

describe('properties and errors', () => {
  const arg: Argument = { type: 'int', minimum: 1, maximum: 8, default: null, initial: 3 };
  it('checks value types and bounds before committing a change', () => {
    const g = current(graph());
    expect(parseParam('4', arg, g)).toBe(4);
    for (const value of ['0', '9', 'true', '2.5', '""', '']) expect(() => parseParam(value, arg, g)).toThrow();
    expect(parseParam('$width', arg, { ...g, parameters: [{ id: 'width', type: 'int' }] })).toEqual({ $param: 'width' });
    expect(parseShape('["B",1,null]')).toEqual(['B', 1, null]);
    expect(() => parseShape('[true]')).toThrow();
  });
  it('preserves structured backend diagnostics', () => {
    const diagnostics = [{ code: 'compile_shape', severity: 'error', location: { path: '/graphs/0', node: 'relu' }, message: 'Shape mismatch' }];
    expect(backendError({ code: 'compile_error', message: 'Compilation failed.', diagnostics }))
      .toEqual({ message: 'Compilation failed.', diagnostics });
  });
});

it('reconnects either endpoint as one edit and keeps the original wire on invalid drops', () => {
  const [twoInputs] = addBoundary(graph(), 'input', point);
  const [doc] = addBoundary(twoInputs, 'output', point);
  const original = { ...into, annotation: 'keep' };
  const start = connect(connect(doc, original), out);
  const changedSource = { ...into, from: { input: 'input_2' } };
  const first = connect(start, changedSource, edgeKey(into));
  expect(current(first).edges).toEqual([{ ...changedSource, annotation: 'keep' }, out]);
  expect(connect(first, { ...changedSource, to: out.to }, edgeKey(changedSource))).toBe(first);
  const changedTarget = { ...changedSource, to: { output: 'output_2' } };
  const second = connect(first, changedTarget, edgeKey(changedSource));
  expect(current(second).edges[0]).toEqual({ ...changedTarget, annotation: 'keep' });
  const edited = history({ doc: first, past: [], future: [] }, { type: 'edit', doc: second });
  expect(history(edited, { type: 'undo' }).doc).toBe(first);
});

it('auto layouts branches and disconnected nodes using measured bounds and preserves graph semantics', () => {
  const [branched] = addNode(graph(), relu, point);
  const [isolated] = addBoundary(branched, 'input', point);
  const doc = connect(connect(connect(isolated, into), out), { ...into, to: { node: 'relu_2', port: 'x' } });
  const tiles = project(current(doc)).map(n => ({ id: n.id, measured: { width: 320, height: 260 } }));
  const arranged = autoLayout(doc, tiles);
  const positions = new Map(project(current(arranged)).map(n => [n.id, n.position]));
  expect(positions.get('node:relu')!.x).toBeGreaterThan(positions.get('input:input')!.x + 320);
  expect(positions.get('output:output')!.x).toBeGreaterThan(positions.get('node:relu')!.x + 320);
  expect(positions.get('node:relu_2')!.y).toBeGreaterThan(positions.get('node:relu')!.y + 260);
  expect(positions.get('input:input_2')!.y).toBeGreaterThan(positions.get('input:input')!.y + 260);
  expect(current(arranged).edges).toBe(current(doc).edges);
  expect(autoLayout(arranged, tiles)).toBe(arranged);
});
