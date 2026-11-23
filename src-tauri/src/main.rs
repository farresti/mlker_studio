use mlker_studio::{
    database,
    editor::{EditorError, EditorService},
    worker,
};
use serde::Serialize;
use std::ffi::OsString;
use tauri::{path::BaseDirectory, Manager};

#[derive(Serialize)]
struct Subsystem {
    available: bool,
    detail: String,
}

impl From<Result<String, String>> for Subsystem {
    fn from(result: Result<String, String>) -> Self {
        match result {
            Ok(detail) => Self {
                available: true,
                detail,
            },
            Err(detail) => Self {
                available: false,
                detail,
            },
        }
    }
}

#[derive(Serialize)]
struct AppStatus {
    version: String,
    backend: Subsystem,
    database: Subsystem,
    worker: Subsystem,
}

#[tauri::command]
async fn app_status(app: tauri::AppHandle) -> AppStatus {
    let directory = app.path().app_local_data_dir();
    let database_check = tauri::async_runtime::spawn_blocking(move || {
        let directory = directory.map_err(|e| e.to_string())?;
        database::initialize(&directory)
            .map(|v| format!("SQLite · schema {v}"))
            .map_err(|e| format!("{}: {e}", directory.display()))
    });
    let python = std::env::var_os("MLKER_PYTHON").unwrap_or_else(|| OsString::from("python3"));
    let worker = match app
        .path()
        .resolve("python/mlker_worker", BaseDirectory::Resource)
    {
        Ok(package) => worker::ping(&python, &package)
            .await
            .map(|v| format!("Python {v} · protocol v1")),
        Err(error) => Err(error.to_string()),
    };
    let database = database_check.await.unwrap_or_else(|e| Err(e.to_string()));
    AppStatus {
        version: app.package_info().version.to_string(),
        backend: Subsystem {
            available: true,
            detail: "Rust · Tauri 2".into(),
        },
        database: database.into(),
        worker: worker.into(),
    }
}

#[tauri::command]
async fn editor_request(
    app: tauri::AppHandle,
    state: tauri::State<'_, EditorService>,
    method: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, EditorError> {
    let python = std::env::var_os("MLKER_PYTHON").unwrap_or_else(|| OsString::from("python3"));
    let package = app
        .path()
        .resolve("python/mlker_worker", BaseDirectory::Resource)
        .map_err(EditorError::transport)?;
    state.request(&python, &package, &method, params).await
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(EditorService::default())
        .invoke_handler(tauri::generate_handler![app_status, editor_request])
        .run(tauri::generate_context!())
        .expect("could not run MLKer Studio");
}
