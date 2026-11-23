//! One editor subprocess per desktop session, with one bounded exchange at a time.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{ffi::OsStr, io, path::Path, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

const LIMIT: usize = 4 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct EditorError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub diagnostics: Vec<Value>,
}

impl EditorError {
    pub fn transport(message: impl ToString) -> Self {
        Self {
            code: "editor_transport".into(),
            message: message.to_string(),
            diagnostics: Vec::new(),
        }
    }
}

struct Process {
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    sequence: u64,
}

impl Process {
    fn start(python: &OsStr, package: &Path) -> io::Result<Self> {
        let mut child = Command::new(python)
            .args(["-I", "-B"])
            .arg(package)
            .arg("--editor")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("Missing worker stdin"))?;
        let output = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| io::Error::other("Missing worker stdout"))?,
        );
        Ok(Self {
            _child: child,
            input,
            output,
            sequence: 0,
        })
    }

    async fn exchange(&mut self, frame: &[u8], id: &str) -> io::Result<Result<Value, EditorError>> {
        self.input.write_all(frame).await?;
        let mut line = Vec::new();
        (&mut self.output)
            .take((LIMIT + 1) as u64)
            .read_until(b'\n', &mut line)
            .await?;
        if line.len() > LIMIT || line.last() != Some(&b'\n') {
            return Err(io::Error::other(
                "Incomplete or oversized editor response. Check worker stderr.",
            ));
        }
        let response: Value = serde_json::from_slice(&line)?;
        if response["version"] != 2 || response["id"] != id {
            return Err(io::Error::other("Editor response version or ID mismatch"));
        }
        match response["status"].as_str() {
            Some("ok") if response.get("result").is_some() => Ok(Ok(response["result"].clone())),
            Some("error") => Ok(Err(serde_json::from_value(response["error"].clone())?)),
            _ => Err(io::Error::other("Invalid editor response")),
        }
    }
}

#[derive(Default)]
pub struct EditorService(Mutex<Option<Process>>);

impl EditorService {
    pub async fn request(
        &self,
        python: &OsStr,
        package: &Path,
        method: &str,
        params: Value,
    ) -> Result<Value, EditorError> {
        let mut state = self.0.lock().await;
        let mut process = match state.take() {
            Some(process) => process,
            None => Process::start(python, package).map_err(EditorError::transport)?,
        };
        process.sequence += 1;
        let id = process.sequence.to_string();
        let mut frame =
            serde_json::to_vec(&json!({"version":2, "id":id, "method":method, "params":params}))
                .map_err(EditorError::transport)?;
        frame.push(b'\n');
        if frame.len() > LIMIT {
            *state = Some(process);
            return Err(EditorError::transport("Editor request exceeds 4 MiB."));
        }
        let seconds = if method == "import_model" { 600 } else { 30 };
        let reply = timeout(Duration::from_secs(seconds), process.exchange(&frame, &id))
            .await
            .map_err(|_| {
                EditorError::transport(
                    "Editor request timed out. Check the destination before retrying a write.",
                )
            })?
            .map_err(EditorError::transport)?;
        *state = Some(process);
        reply
    }
}
