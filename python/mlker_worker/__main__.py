"""Serve version 1 of the MLKer JSON Lines protocol."""

import json
import logging
import platform
import sys

VERSION = 1
MAX_FRAME = 8192


def error(request_id, code, message):
    return {
        "version": VERSION,
        "id": request_id,
        "status": "error",
        "error": {"code": code, "message": message},
    }


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant {value}")


def dispatch(frame):
    try:
        request = json.loads(frame.decode("utf-8"), parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return error(None, "invalid_json", "Expected a UTF-8 JSON object.")
    if not isinstance(request, dict):
        return error(None, "invalid_request", "Expected a request object.")

    request_id = request.get("id")
    if not isinstance(request_id, str) or not (1 <= len(request_id) <= 128):
        return error(None, "invalid_request", "Expected an id of 1 to 128 characters.")
    version = request.get("version")
    if type(version) is not int:
        return error(request_id, "invalid_request", "Expected an integer version.")
    if version != VERSION:
        return error(request_id, "unsupported_version", "Worker supports version 1.")
    method = request.get("method")
    if not isinstance(method, str) or not isinstance(request.get("params"), dict):
        return error(request_id, "invalid_request", "Expected a method string and params object.")
    if method != "ping":
        return error(request_id, "unknown_method", "Unrecognized worker method.")
    if request["params"]:
        return error(request_id, "invalid_request", "ping takes an empty params object.")
    return {
        "version": VERSION,
        "id": request_id,
        "status": "ok",
        "result": {"reply": "pong", "pythonVersion": platform.python_version()},
    }


def serve(source, sink):
    while frame := source.readline(MAX_FRAME + 1):
        if len(frame) > MAX_FRAME:
            while frame and not frame.endswith(b"\n"):
                frame = source.readline(MAX_FRAME + 1)
            response = error(None, "frame_too_large", "Frame exceeds 8192 bytes.")
        elif not frame.endswith(b"\n"):
            response = error(None, "invalid_request", "Frame must end with a newline.")
        else:
            response = dispatch(frame)
        if response["status"] == "error":
            logging.warning("Rejected worker request (%s)", response["error"]["code"])
        sink.write(json.dumps(response, separators=(",", ":")).encode("ascii") + b"\n")
        sink.flush()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, format="mlker-worker: %(message)s")
    serve(sys.stdin.buffer, sys.stdout.buffer)
