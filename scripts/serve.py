"""Web viewer for training runs: charts, milestones, replays.

Usage: python scripts/serve.py [--port 8080]   then open http://<pc>:8080
"""
import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/viewer/index.html"
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
