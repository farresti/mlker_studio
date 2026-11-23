import { memo, useEffect, useMemo, useState } from 'react';
import { Background, Controls, Handle, Position, ReactFlow, useReactFlow, type Connection, type Edge,
  type Node, type NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { boundaryKey, connectionError, edgeKey, endpoint, handle, nodeKey, owner, portId, position,
  type Graph, type Point, type Port, type Wire } from './model';

type Tile = Node<{ title: string; category: string; inputs: Port[]; outputs: Port[]; issue: boolean }, 'tile'>;
export function project(g: Graph): Tile[] {
  const nodes = [
    ...g.inputs.map(p => ({ id: boundaryKey('input', p.id), category: 'Input', title: p.name || p.id, inputs: [], outputs: [p] })),
    ...g.nodes.map(n => ({ id: nodeKey(n.id), category: n.kind, title: n.name || n.registry?.split('.').at(-1) || n.id, inputs: n.inputs, outputs: n.outputs })),
    ...g.outputs.map(p => ({ id: boundaryKey('output', p.id), category: 'Output', title: p.name || p.id, inputs: [p], outputs: [] })),
  ];
  return nodes.map(({ id, ...data }, i) => ({
    id, type: 'tile', position: position(g, id, i), ariaLabel: data.category + ' ' + data.title,
    data: { ...data, issue: false },
  }));
}
export function flowEdge(w: Wire): Edge {
  return { id: edgeKey(w), source: owner(w.from), target: owner(w.to),
    sourceHandle: handle(true, portId(w.from)), targetHandle: handle(false, portId(w.to)) };
}
export function wire(c: Connection): Wire | null {
  const from = endpoint(c.source, c.sourceHandle, true), to = endpoint(c.target, c.targetHandle, false);
  return from && to ? { from, to } : null;
}
function portInfo(p: Port) {
  return [p.tensor.dtype, p.tensor.shape?.map(d => d ?? '?').join(' × ')].filter(Boolean).join(' · ');
}
const GraphTile = memo(function GraphTile({ data, selected }: NodeProps<Tile>) {
  return <div className={'graph-tile ' + (selected ? 'chosen ' : '') + (data.issue ? 'has-issue' : '')}
    data-category={data.category}>
    <div className="tile-heading"><small>{data.category}</small><strong>{data.title}</strong></div>
    {([['inputs', false], ['outputs', true]] as const).map(([side, source]) => data[side].map(p =>
      <div className={'tile-port ' + side} key={side + p.id} title={portInfo(p)}>
        <Handle type={source ? 'source' : 'target'} position={source ? Position.Right : Position.Left}
          id={handle(source, p.id)} aria-label={side + ' ' + p.id} />
        <span>{p.id}</span>{portInfo(p) && <small>{portInfo(p)}</small>}
      </div>))}
  </div>;
});
const nodeTypes = { tile: GraphTile };

export function Canvas({ graph, selection, select, onMove, onWire, onDropOp, issues, focus, error, reset }: {
  graph: Graph; selection: string | null; select: (id: string | null) => void; onMove: (id: string, pos: Point) => void;
  onWire: (w: Wire) => void; onDropOp: (id: string, pos: Point) => void; issues: Set<string>;
  focus: { key: string; count: number } | null; error: (message: string) => void; reset: number;
}) {
  const flow = useReactFlow<Tile>();
  const [live, setLive] = useState<Record<string, Point>>({});
  const base = useMemo(() => project(graph), [graph]);
  const nodes = useMemo(() => base.map(n => ({ ...n, selected: n.id === selection, position: live[n.id] ?? n.position,
    data: { ...n.data, issue: issues.has(n.id) } })), [base, live, selection, issues]);
  const edges = useMemo(() => Array.from(new Map(graph.edges.map(w => [edgeKey(w), w])).values(),
    w => ({ ...flowEdge(w), selected: edgeKey(w) === selection })), [graph.edges, selection]);
  useEffect(() => {
    const timer = setTimeout(() => { if (flow.getNodes().length) void flow.fitView({ padding: 0.2, maxZoom: 1 }); }, 70);
    return () => clearTimeout(timer);
  }, [reset, flow]);
  useEffect(() => {
    if (!focus) return;
    const node = flow.getNode(focus.key);
    if (node) void flow.setCenter(node.position.x + 80, node.position.y + 48, { zoom: 1, duration: 180 });
  }, [focus, flow]);
  return <section className="editor-canvas" aria-label="Graph canvas"
    onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
    onDrop={e => { e.preventDefault(); const id = e.dataTransfer.getData('application/mlker-operation');
      if (id) onDropOp(id, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY })); }}>
    <ReactFlow<Tile> nodes={nodes} edges={edges} nodeTypes={nodeTypes} colorMode="dark" minZoom={0.12} maxZoom={2}
      deleteKeyCode={null} multiSelectionKeyCode={null} selectionKeyCode={null} selectNodesOnDrag={false}
      onNodeClick={(_, n) => select(n.id)} onEdgeClick={(_, e) => select(e.id)} onPaneClick={() => select(null)}
      onEdgesChange={changes => { for (const change of changes) {
        if (change.type === 'select' && change.selected) select(change.id);
      } }}
      onNodesChange={changes => { for (const c of changes) {
        if (c.type === 'position' && c.position) {
          if (c.dragging) setLive(old => ({ ...old, [c.id]: c.position! }));
          else { onMove(c.id, c.position); setLive({}); }
        }
        if (c.type === 'select' && c.selected) select(c.id);
      } }}
      isValidConnection={c => { const w = wire(c as Connection); return !!w && !connectionError(graph, w); }}
      onConnect={c => { const w = wire(c); if (w) onWire(w); }}
      onConnectEnd={(_, state) => { if (state.toNode && !state.isValid) error('Connection rejected. Check direction, tensor types, existing drivers, and cycles.'); }}>
      <Background gap={20} size={1} color="#43514d" /><Controls showInteractive={false} />
    </ReactFlow>
    {!nodes.length && <div className="canvas-empty"><strong>Start your graph</strong><p>Add a model input or choose an operation from the library.</p></div>}
  </section>;
}
