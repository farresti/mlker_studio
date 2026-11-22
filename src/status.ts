import { invoke, isTauri } from "@tauri-apps/api/core";

export interface Subsystem {
  available: boolean;
  detail: string;
}

export interface AppStatus {
  version: string;
  backend: Subsystem;
  database: Subsystem;
  worker: Subsystem;
}

export async function getStatus(): Promise<AppStatus> {
  if (!isTauri()) {
    throw new Error("Open the desktop app to check local services.");
  }
  return invoke<AppStatus>("app_status");
}
