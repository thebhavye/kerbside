"""Test the Kerbside app's HTTP API end to end, without Whisper or Ollama.

Run with:  python -m pytest tests
"""
from __future__ import annotations

import csv
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kerbside  # noqa: E402
import kerbside_app  # noqa: E402
from test_pipeline import GPX, TRANSCRIPT, fake_get, fake_post  # noqa: E402

H = {"X-Kerbside": "1"}


def fake_transcribe(audio, cache_path, size, lang, translate, fresh):
    print("  50.0%  [0:20] Footpath is broken here")  # exercises progress parsing
    return json.loads(TRANSCRIPT.read_text(encoding="utf-8"))


class Client:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def call(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw, code, ctype = r.read(), r.status, r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            raw, code, ctype = e.read(), e.code, e.headers.get("Content-Type", "")
        return code, (json.loads(raw) if "json" in ctype and raw else raw)


def test_app_flow(tmp_path: Path) -> None:
    server = kerbside_app.make_server(tmp_path / "walks", "gemma3:4b", "small", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    c = Client(server.server_address[1])
    try:
        with mock.patch.object(kerbside, "transcribe", fake_transcribe), \
                mock.patch.object(kerbside.requests, "get", fake_get), \
                mock.patch.object(kerbside.requests, "post", fake_post):
            code, page = c.call("GET", "/")
            assert code == 200 and b"Kerbside" in page and b"Find street problems" in page

            # Safety: changes need the app's header, and only localhost host names are served.
            assert c.call("POST", "/api/walks", b"{}")[0] == 403
            assert c.call("GET", "/api/walks", headers={"Host": "evil.example"})[0] == 403
            assert c.call("GET", "/api/walks/..%2F..%2Fsecret")[0] == 404

            code, walk = c.call("POST", "/api/walks", json.dumps({"title": "App Test Walk", "start": "2026-10-10 07:30:00"})
                                .encode(), {**H, "Content-Type": "application/json"})
            assert code == 201, walk
            wid = walk["id"]
            assert c.call("POST", f"/api/walks/{wid}/start", headers=H)[0] == 400, "needs both files first"
            assert c.call("PUT", f"/api/walks/{wid}/upload?kind=audio&filename=evil.exe", b"x", H)[0] == 400
            assert c.call("PUT", f"/api/walks/{wid}/upload?kind=audio&filename=walk.m4a", b"fake audio", H)[0] == 200
            assert c.call("PUT", f"/api/walks/{wid}/upload?kind=gpx&filename=t.gpx", GPX.read_bytes(), H)[0] == 200

            assert c.call("POST", f"/api/walks/{wid}/start", headers=H)[0] == 202
            deadline = time.time() + 30
            while True:
                code, st = c.call("GET", f"/api/walks/{wid}")
                if st["state"] in ("done", "error") or time.time() > deadline:
                    break
                time.sleep(0.1)
            assert st["state"] == "done", st.get("error")
            assert st["progress"] == 100 and st["findings"] == 10
            assert any("[4/5]" in line for line in st["log"])

            code, geo = c.call("GET", f"/api/walks/{wid}/findings")
            assert code == 200 and len(geo["features"]) == 10
            code, track = c.call("GET", f"/api/walks/{wid}/track")
            assert len(track["points"]) == 97

            # Review: delete the garbage finding, downgrade the crossing, move the shade tree.
            feats = [f for f in geo["features"] if f["properties"]["category"] != "garbage"]
            for f in feats:
                if f["properties"]["category"] == "unsafe_crossing":
                    f["properties"]["severity"] = 2
                if f["properties"]["category"] == "shade_tree":
                    f["geometry"]["coordinates"] = [80.2661, 13.0001]
            code, res = c.call("PUT", f"/api/walks/{wid}/findings",
                               json.dumps({"type": "FeatureCollection", "features": feats}).encode(), H)
            assert code == 200 and res["saved"] == 9

            code, report = c.call("GET", f"/api/walks/{wid}/download/report.md")
            report = report.decode("utf-8")
            assert "Garbage" not in report and "moderate (2/3)" in report and report.startswith("# App Test Walk")
            code, raw_csv = c.call("GET", f"/api/walks/{wid}/download/findings.csv")
            rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8-sig"))))
            assert len(rows) == 9 and any(r["lon"] == "80.2661" for r in rows)
            assert c.call("GET", f"/api/walks/{wid}/download/walk.json")[0] == 404, "only output files"

            code, walks = c.call("GET", "/api/walks")
            assert walks[0]["id"] == wid and walks[0]["findings"] == 9 and walks[0]["state"] == "done"
    finally:
        server.shutdown()
        server.server_close()
