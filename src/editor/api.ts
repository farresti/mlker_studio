import { invoke, isTauri } from '@tauri-apps/api/core';
import { ask, open, save } from '@tauri-apps/plugin-dialog';
import type { Diagnostic, Document, Operation } from './model';

function request<T>(method: string, params: Record<string, unknown> = {}) {
  if (!isTauri()) return Promise.reject(new Error('Open the desktop application to use the graph backend.'));
  return invoke<T>('editor_request', { method, params });
}
const filters = [{ name: 'MLKer model', extensions: ['yaml', 'yml'] }];
export const editor = {
  browse: (query: string, pretrained_only: boolean, offset: number) =>
    request<{models: {name: string; architecture: string; input_size: number[] | null; num_classes: number | null}[]; total: number}>(
      'browse_timm', {query, pretrained_only, offset}),
  importModel: (options: Record<string, unknown>) => request<{document: Document}>('import_model', options),
  async registry() {
    const data = await request<{ version: number; operations: Operation[] }>('registry');
    if (data.version !== 1) throw Error('Unsupported registry version.');
    return data.operations;
  },
  load: (path: string) => request<{ document: Document }>('load', { path }),
  save: (document: Document, path: string) => request<{ path: string }>('save', { document, path, source: path }),
  validate: (document: Document, source: string | null) => request<{ diagnostics: Diagnostic[] }>('validate', { document, source }),
  compile: (document: Document, path: string, source: string | null) =>
    request<{ path: string; sources: Record<string, string> }>('compile', { document, path, source }),
  openPath: () => open({ filters, title: 'Open MLKer model' }),
  savePath: (defaultPath: string) => save({ filters, defaultPath, title: 'Save MLKer model' }),
  exportPath: (modelPath: string | null) => open({ directory: true, title: 'Choose a new or existing model export directory',
    defaultPath: modelPath?.slice(0, modelPath.lastIndexOf('/') + 1), canCreateDirectories: true }),
  discard: () => ask('Discard unsaved changes to this model?', { title: 'Unsaved model', kind: 'warning' }),
};
export function backendError(error: unknown): { message: string; diagnostics: Diagnostic[] } {
  if (error && typeof error === 'object' && 'message' in error) return {
    message: String(error.message),
    diagnostics: 'diagnostics' in error && Array.isArray(error.diagnostics) ? error.diagnostics : [],
  };
  return { message: String(error), diagnostics: [] };
}
