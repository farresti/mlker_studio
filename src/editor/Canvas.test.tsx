// @vitest-environment jsdom

import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Node, NodeChange } from '@xyflow/react';
import { expect, it, vi } from 'vitest';
import { Canvas } from './Canvas';
import type { Graph } from './model';

const flow = vi.hoisted(() => ({
  initialized: false,
  nodes: [] as Node[],
  onNodesChange: (_changes: NodeChange[]) => {},
  instance: { viewportInitialized: true, getZoom: () => 0.5, getNode: vi.fn(), fitView: vi.fn() },
}));

vi.mock('@xyflow/react', () => ({
  ReactFlow: (props: { nodes: Node[]; onNodesChange: (changes: NodeChange[]) => void }) => {
    flow.nodes = props.nodes;
    flow.onNodesChange = props.onNodesChange;
    return null;
  },
  useReactFlow: () => flow.instance,
  useNodesInitialized: () => flow.initialized,
  Background: () => null,
  Controls: () => null,
}));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

it('retains measured bounds during selection, drag and add, and fits a measured focus target', async () => {
  const root = createRoot(document.createElement('div'));
  const onMove = vi.fn();
  const graph: Graph = { id: 'main', inputs: [{ id: 'image', tensor: {} }], nodes: [], outputs: [], edges: [] };
  const added: Graph = { ...graph, nodes: [{ id: 'conv', kind: 'module', inputs: [], outputs: [] }] };
  const focus = { key: 'node:conv', count: 1 };
  const render = (g: Graph, selected: string | null, target: typeof focus | null = null) => act(async () => {
    root.render(<Canvas graph={g} selection={selected} focus={target} onMove={onMove}
      select={() => {}} onWire={() => {}} onLayout={() => {}} onDropOp={() => {}} error={() => {}} issues={new Set()} />);
  });
  try {
    await render(graph, null);
    const dimensions = { width: 152, height: 90 };
    await act(async () => flow.onNodesChange([{ id: 'input:image', type: 'dimensions', dimensions }]));
    await render(graph, 'input:image');
    expect(flow.nodes[0].measured).toEqual(dimensions);
    const position = { x: 300, y: 90 };
    await act(async () => flow.onNodesChange([{ id: 'input:image', type: 'position', position, dragging: true }]));
    expect(flow.nodes[0]).toMatchObject({ measured: dimensions, position });
    expect(onMove).not.toHaveBeenCalled();
    await act(async () => flow.onNodesChange([{ id: 'input:image', type: 'position', position, dragging: false }]));
    expect(onMove).toHaveBeenCalledExactlyOnceWith('input:image', position);
    await render(added, focus.key, focus);
    expect(flow.nodes[0].measured).toEqual(dimensions);
    expect(flow.instance.fitView).not.toHaveBeenCalled();
    await act(async () => flow.onNodesChange([{ id: focus.key, type: 'dimensions', dimensions }]));
    flow.initialized = true;
    flow.instance.getNode.mockReturnValue(flow.nodes[1]);
    await render(added, focus.key, focus);
    expect(flow.instance.fitView).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ nodes: [expect.objectContaining({ id: focus.key, measured: dimensions })], maxZoom: 0.5 }),
    );
  } finally {
    await act(async () => root.unmount());
  }
});
