//! A single health exchange per child process. Closing stdin requests a clean exit.
use serde::Deserialize;
use serde_json::json;
use std::{ffi::OsStr, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
    time::timeout,
};

const VERSION: u32 = 1;
const REQUEST_ID: &str = "health";
const MAX_FRAME: u64 = 8192;

#[derive(Deserialize)]
struct Response {
    version: u32,
    id: String,
    #[serde(flatten)]
    outcome: Outcome,
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum Outcome {
    Ok { result: PingResult },
    Error { error: WorkerError },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PingResult {
    reply: String,
    python_version: String,
}

#[derive(Deserialize)]
struct WorkerError {
    code: String,
    message: String,
}

pub async fn ping(python: &OsStr, package: &Path) -> Result<String, String> {
    let mut child = Command::new(python)
        .args(["-I", "-B"])
        .arg(package)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Could not start {}: {e}", python.to_string_lossy()))?;

    let exchange = async {
        let mut input = child.stdin.take().ok_or("Missing worker stdin")?;
        let output = child.stdout.take().ok_or("Missing worker stdout")?;
        let request = json!({"version": VERSION, "id": REQUEST_ID, "method": "ping", "params": {}});
        input
            .write_all(format!("{request}\n").as_bytes())
            .await
            .map_err(|e| e.to_string())?;
        drop(input);

        let mut frame = Vec::new();
        output
            .take(MAX_FRAME + 1)
            .read_to_end(&mut frame)
            .await
            .map_err(|e| e.to_string())?;
        if frame.len() as u64 > MAX_FRAME {
            return Err("Worker response exceeds 8192 bytes".into());
        }
        let status = child.wait().await.map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!(
                "Worker exited with {status}. Check its stderr log."
            ));
        }
        decode(&frame)
    };

    let result = timeout(Duration::from_secs(5), exchange).await;
    if !matches!(result, Ok(Ok(_))) {
        // Also reap a child that is still alive after an I/O failure or timeout.
        let _ = child.kill().await;
    }
    result.map_err(|_| "Worker timed out after 5 seconds".to_owned())?
}

fn decode(frame: &[u8]) -> Result<String, String> {
    let body = frame
        .strip_suffix(b"\n")
        .ok_or("Incomplete worker response")?;
    if body.contains(&b'\n') {
        return Err("Expected one response frame".into());
    }
    let response: Response =
        serde_json::from_slice(body).map_err(|e| format!("Invalid worker response: {e}"))?;
    if response.version != VERSION || response.id != REQUEST_ID {
        return Err("Worker response version or id mismatch".into());
    }
    match response.outcome {
        Outcome::Ok { result } if result.reply == "pong" && !result.python_version.is_empty() => {
            Ok(result.python_version)
        }
        Outcome::Ok { .. } => Err("Unexpected ping result".into()),
        Outcome::Error { error } => Err(format!("{}: {}", error.code, error.message)),
    }
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn wrong_envelope_or_unexpected_result_fails_health() {
        let reply = r#"{"version":1,"id":"health","status":"ok","result":{"reply":"pong","pythonVersion":"3.12"}}"#;
        for invalid in [
            reply.replace("health", "different"),
            reply.replace("\"version\":1", "\"version\":2"),
            reply.replace("pong", "other"),
            format!("{reply}\n{reply}"),
        ] {
            assert!(decode(format!("{invalid}\n").as_bytes()).is_err());
        }
        assert!(decode(reply.as_bytes()).is_err());
    }

    #[test]
    fn worker_error_reaches_status_detail() {
        let reply = b"{\"version\":1,\"id\":\"health\",\"status\":\"error\",\"error\":{\"code\":\"unknown_method\",\"message\":\"Missing method.\"}}\n";
        assert_eq!(
            decode(reply).unwrap_err(),
            "unknown_method: Missing method."
        );
    }
}
