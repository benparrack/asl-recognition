"""
SignDesk: a browser dashboard for the trained letter and word models.

Run:
    python src/dashboard.py                    # opens http://127.0.0.1:8790
    python src/dashboard.py --video data/aslc_demo/pencil_2hand.mp4   # loop a file

The camera is read here, in Python, and recognition runs here
(dashboard_engine.py, which reuses demo.py's pipeline). The browser shows an
MJPEG stream of the annotated camera, gets the live state as server-sent events
and sends commands as JSON. Only the standard library is used for serving, and
the page needs no internet.

The server binds to 127.0.0.1 and refuses requests whose Host header isn't
this machine, so other websites open in the browser can't read the camera or
the text (DNS rebinding). Settings and the transcript are saved under
data/dashboard/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from dashboard_engine import Engine  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "dashboard_web"
SSE_MIN_INTERVAL = 0.05            # at most 20 state updates a second per client
MAX_BODY = 1 << 20


def make_handler(engine: Engine, port: int):
    allowed_hosts = {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):            # keep the terminal for events
            pass

        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._send(403, b"forbidden host", "text/plain")
            return False

        def _send(self, code, body: bytes, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if not self._host_ok():
                return
            path = urlparse(self.path).path
            if path == "/":
                path = "/index.html"
            if path == "/stream.mjpg":
                return self._mjpeg()
            if path == "/events":
                return self._sse()
            if path == "/api/state":
                return self._json(engine.snapshot())
            if path == "/api/vocab":
                return self._json(engine.vocabulary())
            f = (WEB_DIR / path.lstrip("/")).resolve()
            if f.parent != WEB_DIR or not f.is_file():    # no path traversal, no subdirs
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""),
                       {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"})

        def do_POST(self):
            if not self._host_ok():
                return
            if urlparse(self.path).path != "/api/command":
                return self._send(404, b"not found", "text/plain")
            # A JSON content type can't be sent cross-site without a CORS preflight.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, b"expected application/json", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                return self._send(413, b"bad body size", "text/plain")
            try:
                body = json.loads(self.rfile.read(n))
                action = body.pop("action")
                self._json(engine.command(action, **body))
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self._json({"error": str(e)}, 400)

        def _stream_headers(self, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _mjpeg(self):
            self._stream_headers("multipart/x-mixed-replace; boundary=frame")
            last = -1
            try:
                while not engine.stopped:
                    with engine.cond:
                        engine.cond.wait_for(lambda: engine.frame_id != last or engine.stopped, timeout=5)
                        if engine.frame_id == last or engine.jpeg is None:
                            continue
                        last, jpeg = engine.frame_id, engine.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _sse(self):
            self._stream_headers("text/event-stream")
            last = -1
            try:
                while not engine.stopped:
                    with engine.cond:
                        engine.cond.wait_for(lambda: engine.version != last or engine.stopped, timeout=10)
                        changed = engine.version != last
                        last = engine.version
                    if changed:
                        self.wfile.write(b"data: " + json.dumps(engine.snapshot()).encode() + b"\n\n")
                    else:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    threading.Event().wait(SSE_MIN_INTERVAL)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    ck = config.CHECKPOINT_DIR
    parser = argparse.ArgumentParser(description="SignDesk: live ASL letters and words in the browser")
    parser.add_argument("--letters-checkpoint", default=ck / "mlp_pugeault_best.pt")
    parser.add_argument("--motion-checkpoint", default=ck / "motion_v2_best.pt",
                        help="J/Z model; '' to disable")
    parser.add_argument("--words-checkpoint", default=ck / "words_transformer_full_aslc_r3_best.pt",
                        help="word model; '' to run letters only")
    parser.add_argument("--holistic-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--video", default=None, help="loop this file instead of the webcam")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()

    for name in ("motion_checkpoint", "words_checkpoint"):
        p = getattr(args, name)
        if p and not Path(p).exists():
            print(f"[dashboard] {p} not found; running without it")
            setattr(args, name, None)

    engine = Engine(args.letters_checkpoint, args.motion_checkpoint or None,
                    args.words_checkpoint or None, args.holistic_complexity, args.video)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.port))
    except OSError as e:
        sys.exit(f"[dashboard] port {args.port} is busy ({e}); is SignDesk already running? "
                 "Use --port to pick another.")
    server.daemon_threads = True
    engine.start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[dashboard] SignDesk running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopping")
    finally:
        engine.stop()
        server.server_close()


if __name__ == "__main__":
    main()
