import { useEffect, useRef, useState } from 'react';
import { backendError, editor } from './api';
import type { Diagnostic, Document } from './model';

export function useValidation(doc: Document, path: string | null, enabled: boolean) {
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [status, setStatus] = useState('Waiting for backend');
  const [check, setCheck] = useState(0);
  const generation = useRef(0), running = useRef(false), next = useRef<(() => Promise<void>) | null>(null);
  const manual = useRef(false);
  useEffect(() => {
    const token = ++generation.current;
    next.current = null;
    setDiagnostics([]);
    if (!enabled) return;
    setStatus('Checking…');
    const timer = setTimeout(async () => {
      next.current = async () => {
        try {
          const result = await editor.validate(doc, path);
          if (generation.current !== token) return;
          setDiagnostics(result.diagnostics);
          const count = result.diagnostics.filter(d => d.severity === 'error').length;
          setStatus(count ? count + (count === 1 ? ' error' : ' errors') : 'Valid graph');
        } catch (e) { if (generation.current === token) setStatus(backendError(e).message); }
      };
      if (running.current) return;
      running.current = true;
      try { while (next.current) { const job = next.current; next.current = null; await job(); } }
      finally { running.current = false; }
    }, manual.current ? 0 : 400);
    manual.current = false;
    return () => { clearTimeout(timer); generation.current++; next.current = null; };
  }, [doc, path, enabled, check]);
  return { diagnostics, status, validate: () => { manual.current = true; setCheck(n => n + 1); } };
}
