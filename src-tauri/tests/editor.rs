use mlker_studio::editor::EditorService;
use serde_json::json;
use std::{ffi::OsStr, path::Path};

#[tokio::test]
async fn editor_transport_recovers_after_child_failure() {
    let service = EditorService::default();
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../python/mlker_worker");
    // The second request must spawn its own process instead of reusing a dead child.
    let failure = service
        .request(OsStr::new("/bin/false"), &package, "registry", json!({}))
        .await;
    assert_eq!(failure.unwrap_err().code, "editor_transport");
    let result = service
        .request(OsStr::new("python3"), &package, "unrecognized", json!({}))
        .await;
    let code = result.unwrap_err().code;
    assert!(["unknown_method", "missing_dependency"].contains(&code.as_str()));
    let result = service
        .request(
            OsStr::new("/bin/false"),
            &package,
            "unrecognized",
            json!({}),
        )
        .await;
    assert_eq!(result.unwrap_err().code, code);
}
