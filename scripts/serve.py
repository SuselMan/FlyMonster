"""Web viewer for training runs: charts, milestones, replays.

Usage: python scripts/serve.py [--port 8080]   then open http://<pc>:8080
"""
import argparse
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


_explorer = None
_explorer_lock = threading.Lock()


def explorer():
    """The brain explorer loads the connectome and GPU brain on first use."""
    global _explorer
    with _explorer_lock:
        if _explorer is None:
            sys.path.insert(0, str(ROOT))
            from flysim.explorer import Explorer
            _explorer = Explorer()
        return _explorer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/brain/stimulate":
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            result = explorer().stimulate(req.get("stims", []), ms=req.get("ms", 300), stim_ms=req.get("stim_ms", 150))
            self._send(json.dumps(result).encode(), "application/json")
        except Exception as e:  # report to the page instead of dropping the connection
            self.send_error(500, str(e))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/viewer/index.html"
        if self.path == "/api/brain/meta":
            self._send(json.dumps(explorer().meta()).encode(), "application/json")
            return
        if self.path == "/api/brain/layout.bin":
            self._send(explorer().layout_bytes(), "application/octet-stream")
            return
        if self.path == "/api/runs":
            runs = []
            for d in sorted((ROOT / "results").glob("*/")):
                status = d / "status.json"
                runs.append({"name": d.name, **(json.loads(status.read_text()) if status.exists() else {})})
            body = json.dumps(runs).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if not (self.path.startswith("/viewer/") or self.path.startswith("/results/")):
            self.send_error(404)
            return
        super().do_GET()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), partial(Handler, directory=str(ROOT)))
    print(f"viewer on http://localhost:{args.port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
