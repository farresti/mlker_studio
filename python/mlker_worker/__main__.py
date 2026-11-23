"""Serve version 1 of the MLKer JSON Lines protocol."""

import json
import logging
import platform
import sys
from pathlib import Path

VERSION = 1
MAX_FRAME = 8192


def error(request_id, code, message, version=VERSION):
    return {
        "version": version,
        "id": request_id,
        "status": "error",
        "error": {"code": code, "message": message},
    }


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant {value}")


def dispatch(frame, protocol=VERSION, handler=None):
    def reject(request_id, code, message):
        return error(request_id, code, message, protocol)

    try:
        request = json.loads(frame.decode("utf-8"), parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return reject(None, "invalid_json", "Expected a UTF-8 JSON object.")
    if not isinstance(request, dict):
        return reject(None, "invalid_request", "Expected a request object.")

    request_id = request.get("id")
    if not isinstance(request_id, str) or not (1 <= len(request_id) <= 128):
        return reject(None, "invalid_request", "Expected an id of 1 to 128 characters.")
    version = request.get("version")
    if type(version) is not int:
        return reject(request_id, "invalid_request", "Expected an integer version.")
    if version != protocol:
        return reject(request_id, "unsupported_version", f"Expected protocol version {protocol}.")
    method = request.get("method")
    if not isinstance(method, str) or not isinstance(request.get("params"), dict):
        return reject(request_id, "invalid_request", "Expected a method string and params object.")
    if handler:
        try:
            result = handler(method, request["params"])
        except Exception as exc:
            # Preserve domain diagnostics while keeping the stream usable after a bad request.
            response = reject(request_id, getattr(exc, "code", "editor_error"), str(exc))
            response["error"]["diagnostics"] = getattr(exc, "issues", [])
            return response
        return {"version": protocol, "id": request_id, "status": "ok", "result": result}
    if method != "ping":
        return reject(request_id, "unknown_method", "Unrecognized worker method.")
    if request["params"]:
        return reject(request_id, "invalid_request", "ping takes an empty params object.")
    return {
        "version": VERSION,
        "id": request_id,
        "status": "ok",
        "result": {"reply": "pong", "pythonVersion": platform.python_version()},
    }


def serve(source, sink, protocol=VERSION, limit=MAX_FRAME, handler=None):
    while frame := source.readline(limit + 1):
        if len(frame) > limit:
            while frame and not frame.endswith(b"\n"):
                frame = source.readline(limit + 1)
            response = error(None, "frame_too_large", f"Frame exceeds {limit} bytes.", protocol)
        elif not frame.endswith(b"\n"):
            response = error(None, "invalid_request", "Frame must end with a newline.", protocol)
        else:
            response = dispatch(frame, protocol, handler)
        if response["status"] == "error":
            logging.warning("Rejected worker request (%s)", response["error"]["code"])
        encoded = json.dumps(response, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        if len(encoded) > limit:
            response = error(response["id"], "frame_too_large", "Response exceeds editor frame limit.", protocol)
            encoded = json.dumps(response).encode() + b"\n"
        sink.write(encoded)
        sink.flush()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, format="mlker-worker: %(message)s")
    if sys.argv[1:] == ["--editor"]:
        # Import only from the packaged application resource directory.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from mlker_worker.editor import handle
        serve(sys.stdin.buffer, sys.stdout.buffer, 2, 4 * 1024 * 1024, handle)
    else:
        serve(sys.stdin.buffer, sys.stdout.buffer)
