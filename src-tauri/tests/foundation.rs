use mlker_studio::{database, worker};
use rusqlite::Connection;
use std::{ffi::OsStr, path::Path};
use tempfile::tempdir;

#[tokio::test]
async fn worker_health_over_real_pipes() {
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../python/mlker_worker");
    let python = std::env::var_os("MLKER_PYTHON").unwrap_or_else(|| "python3".into());
    let version = worker::ping(&python, &package).await.unwrap();
    assert!(version.starts_with("3."));
}

#[tokio::test]
async fn missing_interpreter_is_reported() {
    let result = worker::ping(OsStr::new("/nonexistent/mlker-python"), Path::new(".")).await;
    assert!(result.unwrap_err().contains("Could not start"));
}

#[test]
fn database_survives_reopen_and_protects_future_schemas() {
    let temp = tempdir().unwrap();
    let directory = temp.path().join("application");
    assert_eq!(database::initialize(&directory).unwrap(), 1);
    assert_eq!(database::initialize(&directory).unwrap(), 1);
    let connection = Connection::open(directory.join("studio.sqlite3")).unwrap();
    let version = || {
        connection
            .pragma_query_value(None, "user_version", |row| row.get::<_, i32>(0))
            .unwrap()
    };
    assert_eq!(version(), 1);
    connection.pragma_update(None, "user_version", 2).unwrap();
    assert!(database::initialize(&directory)
        .unwrap_err()
        .to_string()
        .contains("schema 2"));
    assert_eq!(version(), 2);
}

#[test]
fn database_path_failure_is_reported() {
    let temp = tempdir().unwrap();
    let file = temp.path().join("file");
    std::fs::write(&file, b"keep me").unwrap();
    assert!(database::initialize(&file).is_err());
    assert_eq!(std::fs::read(file).unwrap(), b"keep me");
}

#[tokio::test]
async fn hung_worker_is_terminated() {
    let directory = tempdir().unwrap();
    std::fs::write(
        directory.path().join("__main__.py"),
        "import time\nfrom pathlib import Path\nimport os\nPath(__file__).with_name('pid').write_text(str(os.getpid()))\ntime.sleep(60)\n",
    )
    .unwrap();
    let python = std::env::var_os("MLKER_PYTHON").unwrap_or_else(|| "python3".into());
    let error = worker::ping(&python, directory.path()).await.unwrap_err();
    assert!(error.contains("timed out"));
    let pid = std::fs::read_to_string(directory.path().join("pid")).unwrap();
    assert!(!Path::new("/proc").join(pid).exists());
}
