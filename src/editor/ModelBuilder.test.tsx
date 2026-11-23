// @vitest-environment jsdom

import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { ModelBuilder } from './ModelBuilder';
import { editor } from './api';
import { newModel, type Graph, type Document } from './model';

vi.mock('./Canvas', () => ({ Canvas: ({graph,openBlock}: {graph: Graph; openBlock:(key:string)=>void}) =>
  <div aria-label="Graph canvas">{graph.nodes.map(n => <button key={n.id} onDoubleClick={() => openBlock('node:'+n.id)}>{n.id}</button>)}</div> }));
vi.mock('./api', async (importOriginal) => {
  const original = await importOriginal<typeof import('./api')>();
  return { ...original, editor: {
    registry: vi.fn(), load: vi.fn(), validate: vi.fn(), save: vi.fn(), compile: vi.fn(),
    openPath: vi.fn(), savePath: vi.fn(), exportPath: vi.fn(), discard: vi.fn(),
  } };
});

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
afterEach(() => vi.clearAllMocks());

it('wires a graph, protects unsaved edits, saves and reopens the canonical document', async () => {
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  let stored: Document | undefined;
  const path = '/tmp/model.yaml';
  vi.mocked(editor.registry).mockResolvedValue([{ id: 'core.ReLU', name: 'ReLU', kind: 'operator',
    inputs: ['x'], outputs: ['out'], params: {}, description: 'Apply ReLU' }]);
  vi.mocked(editor.validate).mockResolvedValue({ diagnostics: [] });
  vi.mocked(editor.savePath).mockResolvedValue(path);
  vi.mocked(editor.openPath).mockResolvedValue(path);
  vi.mocked(editor.save).mockImplementation(async (doc, file) => { stored = structuredClone(doc); return { path: file }; });
  vi.mocked(editor.load).mockImplementation(async () => ({ document: stored! }));
  vi.mocked(editor.exportPath).mockResolvedValue('/tmp/project');
  vi.mocked(editor.compile).mockRejectedValue({ message: 'Compilation failed.', diagnostics: [
    { severity: 'error', code: 'compile_shape', message: 'Shape mismatch', location: { path: '/graphs/0/nodes/0', graph: 'main', node: 'relu' } },
  ] });
  const click = async (label: string) => {
    const button = [...container.querySelectorAll('button')].find(b => (b.getAttribute('aria-label') || b.textContent) === label);
    expect(button).toBeDefined();
    await act(async () => { button!.click(); });
  };
  const source = async (endpoint: object) => {
    const field = container.querySelector<HTMLSelectElement>('.wiring select')!;
    await act(async () => {
      field.value = JSON.stringify(endpoint);
      field.dispatchEvent(new Event('change', { bubbles: true }));
    });
  };
  try {
    await act(async () => root.render(<ModelBuilder active />));
    await click('+ Input');
    await click('Add ReLU');
    await source({ input: 'input' });
    await click('+ Output');
    await source({ node: 'relu', port: 'out' });
    expect(container.textContent).toContain('Unsaved changes');
    vi.mocked(editor.discard).mockResolvedValue(false);
    await click('New');
    expect(editor.discard).toHaveBeenCalledOnce();
    expect(container.textContent).toContain('Unsaved changes');
    await click('Save');
    expect(stored?.graphs[0].edges).toHaveLength(2);
    expect(stored?.model.outputs).toEqual(['output']);
    await click('New');
    await click('Open');
    expect(editor.load).toHaveBeenCalledWith(path);
    await click('Compile');
    expect(container.textContent).toContain('Shape mismatch');
    expect(editor.compile).toHaveBeenCalledWith(stored, '/tmp/project', path);
  } finally {
    await act(async () => root.unmount());
    container.remove();
  }
});

it('coalesces validation and discards a stale response', async () => {
  vi.useFakeTimers();
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  const op = { id: 'core.ReLU', name: 'ReLU', kind: 'operator', inputs: ['x'], outputs: ['out'],
    params: {}, description: 'Apply ReLU' };
  vi.mocked(editor.registry).mockResolvedValue([op]);
  let firstReply: (response: Awaited<ReturnType<typeof editor.validate>>) => void = () => {};
  vi.mocked(editor.validate).mockImplementationOnce(() => new Promise(resolve => { firstReply = resolve; }));
  vi.mocked(editor.validate).mockResolvedValue({ diagnostics: [] });
  try {
    await act(async () => root.render(<ModelBuilder active />));
    await act(async () => vi.advanceTimersByTimeAsync(400));
    expect(editor.validate).toHaveBeenCalledTimes(1);
    const add = container.querySelector<HTMLButtonElement>('[aria-label="Add ReLU"]')!;
    await act(async () => add.click());
    await act(async () => vi.advanceTimersByTimeAsync(400));
    await act(async () => add.click());
    await act(async () => vi.advanceTimersByTimeAsync(400));
    expect(editor.validate).toHaveBeenCalledTimes(1);
    await act(async () => firstReply({ diagnostics: [
      { severity: 'error', code: 'stale', message: 'Stale diagnostic', location: { path: '' } },
    ] }));
    expect(editor.validate).toHaveBeenCalledTimes(2);
    expect(vi.mocked(editor.validate).mock.calls[1][0].graphs[0].nodes.map(n => n.id)).toEqual(['relu', 'relu_2']);
    expect(container.textContent).not.toContain('Stale diagnostic');
    expect(container.textContent).toContain('Valid graph');
  } finally {
    await act(async () => root.unmount());
    container.remove();
    vi.useRealTimers();
  }
});

it('navigates nested definitions by double-click and returns via breadcrumbs without editing', async () => {
  const doc = newModel();
  const graph = (id:string, child?:string): Graph => ({id,inputs:[],outputs:[],edges:[],
    nodes:child ? [{id:'call_'+child,kind:'block',block:child,inputs:[],outputs:[]}] : []});
  doc.graphs = [graph('main','outer'),graph('outer','inner'),graph('inner')];
  doc.blocks = ['inner','outer'].map(id => ({id,graph:id}));
  vi.mocked(editor.load).mockResolvedValue({document:doc});
  vi.mocked(editor.openPath).mockResolvedValue('/tmp/nested.yaml');
  vi.mocked(editor.registry).mockResolvedValue([]);
  const container = document.createElement('div'), root = createRoot(container);
  try {
    await act(async () => root.render(<ModelBuilder active />));
    const button = (name:string) => [...container.querySelectorAll('button')].find(b => b.textContent === name)!;
    await act(async () => button('Open').click());
    for (const id of ['outer','inner']) {
      await act(async () => button('call_'+id).dispatchEvent(new MouseEvent('dblclick',{bubbles:true})));
    }
    expect(container.querySelector('[aria-current=page]')?.textContent).toBe('Definition / inner');
    await act(async () => button('Definition / outer').click());
    expect(button('call_inner')).toBeDefined();
    await act(async () => button('Model / main').click());
    expect(button('call_outer')).toBeDefined();
    expect(button('Undo').disabled).toBe(true);
  } finally { await act(async () => root.unmount()); }
});

it('adds a trusted Python node and saves its typed override and registration identity', async () => {
  const impl = {id:'example.scale',package:'mlker_example_layers',symbol:'Scale',version:'1.0'};
  const contracts = {inputs:[{id:'x',tensor:{dtype:'float32'}}],outputs:[{id:'out',tensor:{}}]};
  vi.mocked(editor.registry).mockResolvedValue([{id:impl.id,name:'Scale',description:'Scale a tensor',kind:'python',
    implementation:impl,contracts,inputs:['x'],outputs:['out'],
    params:{factor:{type:'float',default:2,initial:2,minimum:null,maximum:null}}}]);
  vi.mocked(editor.savePath).mockResolvedValue('/tmp/scale.yaml');
  const container = document.createElement('div'), root = createRoot(container);
  try {
    await act(async () => root.render(<ModelBuilder active />));
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="Add Scale"]')!.click());
    expect(container.querySelector('fieldset select')?.closest('fieldset')?.textContent).toContain('Python implementation');
    const field = [...container.querySelectorAll('label')].find(label => label.textContent === 'factor')!.querySelector('input')!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value')!.set!.call(field,'3.5');
      field.dispatchEvent(new Event('input',{bubbles:true}));
    });
    await act(async () => [...container.querySelectorAll('button')].find(b => b.textContent === 'Save')!.click());
    const saved = vi.mocked(editor.save).mock.calls.at(-1)![0];
    expect(saved.graphs[0].nodes[0]).toMatchObject({kind:'python',implementation:impl,...contracts,params:{factor:3.5}});
  } finally { await act(async () => root.unmount()); }
});
