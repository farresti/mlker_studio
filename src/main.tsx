import { createRoot } from "react-dom/client";
import { useEffect, useState } from "react";
import { getStatus, type AppStatus } from "./status";
import "./style.css";
import { ModelBuilder } from "./editor/ModelBuilder";

const modules = [
  { name: "ONNX Inspector", summary: "Inspect model graphs, operators, and tensor shapes.", motif: "graph" },
  { name: "Experiments", summary: "Explore training metrics and compare experiment runs.", motif: "curves" },
  { name: "Model Builder", summary: "Design reusable neural networks on a visual canvas.", motif: "blocks" },
] as const;
type Page = "Home" | (typeof modules)[number]["name"] | "Settings";

function ModuleArt({ motif }: { motif: string }) {
  return <div className={`art ${motif}`} aria-hidden="true"><i /><i /><i /><i /><i /></div>;
}

function SystemStatus() {
  const [status, setStatus] = useState<AppStatus>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getStatus().then(
      (value) => { if (active) { setStatus(value); setError(""); } },
      (reason: unknown) => {
        if (active) {
          setStatus(undefined);
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      },
    ).finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [attempt]);

  const services = [
    ["Desktop backend", status?.backend],
    ["SQLite database", status?.database],
    ["Python worker", status?.worker],
  ] as const;

  return <section className="system" aria-labelledby="system-title">
    <div className="section-heading">
      <h2 id="system-title">System status</h2>
      <button className="refresh" disabled={busy} onClick={() => {
        setBusy(true);
        setAttempt(value => value + 1);
      }}>{busy ? "Checking…" : "Refresh status"}</button>
    </div>
    <div role="status" aria-live="polite" aria-busy={busy}>
      {error && !busy && <p className="notice">{error}</p>}
      <dl className="service-list">
        {services.map(([name, service]) => <div className="service" key={name}>
          <dt>{name}</dt>
          <dd className="detail">{busy ? "—" : service?.detail ?? "—"}</dd>
          <dd className={`state ${busy || !service ? "" : service.available ? "ready" : "warning"}`}>
            {busy ? "Checking" : !service ? "Unknown" : service.available ? "Ready" : "Unavailable"}
          </dd>
        </div>)}
      </dl>
      {status && <p className="version">MLKer Studio {status.version}</p>}
    </div>
  </section>;
}

function App() {
  const [page, setPage] = useState<Page>("Home");
  const [builderVisited, setBuilderVisited] = useState(false);
  function navigate(next: Page) {
    if (next === "Model Builder") setBuilderVisited(true);
    setPage(next);
  }
  function navItem(name: Page) {
    return <button key={name} aria-current={page === name ? "page" : undefined}
      className="nav-item" onClick={() => navigate(name)}>{name}</button>;
  }
  const selected = modules.find(module => module.name === page);

  return <div className={page === "Model Builder" ? "shell builder-shell" : "shell"}>
    <aside>
      <div className="wordmark">MLKer <span>Studio</span></div>
      <nav aria-label="Workspace">
        {navItem("Home")}
        {modules.map(module => navItem(module.name))}
        <div className="preferences">{navItem("Settings")}</div>
      </nav>
    </aside>
    <main>
      {page !== "Model Builder" && <header>
        <h1>{page === "Home" ? "Workspace overview" : page}</h1>
        <p>{page === "Home" ? "Your local environment for visual machine learning development."
          : selected?.summary ?? "Local application preferences."}</p>
      </header>}
      {page === "Home" ? <>
        <section aria-labelledby="module-title">
          <h2 id="module-title">Modules</h2>
          <div className="module-grid">
            {modules.map(module => <button className="module-card" key={module.name} onClick={() => navigate(module.name)}>
              <ModuleArt motif={module.motif} />
              <strong>{module.name}</strong><span>{module.summary}</span>
            </button>)}
          </div>
        </section>
        <SystemStatus />
      </> : page !== "Model Builder" && <section className="placeholder">
        {selected && <ModuleArt motif={selected.motif} />}
        <h2>{page === "Settings" ? "Preferences" : "Module preview"}</h2>
        <p>{page === "Settings" ? "Settings will be available in a later milestone."
          : `${page} tools will be available in a later milestone.`}</p>
        <button className="refresh" onClick={() => navigate("Home")}>Return to Home</button>
      </section>}
      {builderVisited && <div hidden={page !== "Model Builder"}><ModelBuilder active={page === "Model Builder"} /></div>}
    </main>
  </div>;
}

createRoot(document.getElementById("root")!).render(<App />);
