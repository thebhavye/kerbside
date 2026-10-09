"""Kerbside app: a page in your browser for running walks and reviewing findings.

    python kerbside.py --app        (opens http://127.0.0.1:8765)

The server only listens on 127.0.0.1, so it is reachable from this laptop alone, and it runs the
same pipeline as the command line. Each walk gets its own folder under OUT/walks/.
"""
from __future__ import annotations

import io
import json
import re
import secrets
import threading
import traceback
import webbrowser
from contextlib import redirect_stdout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import kerbside

PAGE = Path(__file__).with_name("kerbside_app.html")
WALK_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aac", ".mp4", ".ogg", ".opus", ".flac", ".webm"}
MAX_UPLOAD = 1024 ** 3  # 1 GB
DOWNLOADS = {
    "map.html": "text/html; charset=utf-8",
    "report.md": "text/markdown; charset=utf-8",
    "findings.csv": "text/csv; charset=utf-8",
    "findings.geojson": "application/geo+json",
}
# Where each pipeline stage starts on the overall progress bar (percent).
STAGE_START = {1: 0.0, 2: 5.0, 3: 45.0, 4: 50.0, 5: 95.0}
RUN_LOCK = threading.Lock()  # one walk at a time: Whisper and Gemma already use the whole CPU


class Walk:
    """One walk: its uploaded files, settings, outputs and live progress."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.id = folder.name
        meta_path = folder / "walk.json"
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.log: list[str] = list(self.meta.get("log", []))  # kept on disk after a run ends
        self.stage = 0
        self.progress = 100.0 if self.meta.get("state") == "done" else 0.0
        self.eta = ""

    @property
    def out(self) -> Path:
        return self.folder / "out"

    def save(self) -> None:
        kerbside.write_json(self.folder / "walk.json", self.meta)

    def add_line(self, line: str) -> None:
        """Record one line of pipeline output and update progress from it."""
        self.log = (self.log + [line])[-300:]
        text = line.strip()
        if m := re.match(r"\[(\d)/5\]", text):
            self.stage = int(m.group(1))
            self.progress = STAGE_START.get(self.stage, self.progress)
            self.eta = ""
        elif self.stage == 2 and (m := re.match(r"([\d.]+)%\s+\[", text)):
            self.progress = 5.0 + 40.0 * min(100.0, float(m.group(1))) / 100
        elif m := re.match(r"Chunk (\d+)/(\d+)", text):
            self.progress = 50.0 + 45.0 * (int(m.group(1)) - 1) / int(m.group(2))
        if m := re.search(r"(about \d+ min left|less than a minute left)", text):
            self.eta = m.group(1)

    def summary(self) -> dict:
        """What the walk list shows."""
        count = None
        geo = self.out / "findings.geojson"
        if geo.exists():
            try:
                count = len(json.loads(geo.read_text(encoding="utf-8")).get("features", []))
            except (OSError, ValueError):
                pass
        return {"id": self.id, "title": self.meta.get("title", ""), "created": self.meta.get("created", ""),
                "state": self.state(), "findings": count}

    def state(self) -> str:
        state = self.meta.get("state", "new")
        if state == "running" and self.id not in LIVE:
            return "error"  # the app was closed mid-run
        return state

    def status(self) -> dict:
        error = self.meta.get("error")
        if self.state() == "error" and not error:
            error = "Processing was interrupted (the app was closed). Start it again."
        return {**self.summary(), "stage": self.stage, "progress": round(self.progress, 1), "eta": self.eta,
                "log": self.log[-200:], "error": error,
                "audio": self.meta.get("audio"), "gpx": self.meta.get("gpx")}


LIVE: dict[str, Walk] = {}  # walks currently being processed


class ProgressWriter(io.TextIOBase):
    """Stands in for stdout while the pipeline runs, feeding each line to the walk."""

    def __init__(self, walk: Walk):
        self.walk = walk
        self.buf = ""

    def write(self, s: str) -> int:
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.walk.add_line(line)
        return len(s)

    def flush(self) -> None:
        if self.buf:
            self.walk.add_line(self.buf)
            self.buf = ""


def run_walk(walk: Walk, model: str, whisper: str) -> None:
    """Run the pipeline for a walk (in a background thread). RUN_LOCK is held by the caller."""
    m = walk.meta
    argv = [str(walk.folder / m["audio"]), str(walk.folder / m["gpx"]), "--out", str(walk.out),
            "--title", m.get("title") or "Kerbside Street Audit", "--tz", m.get("tz") or "Asia/Kolkata",
            "--model", model, "--whisper", whisper]
    if m.get("start"):
        argv += ["--start", m["start"]]
    if m.get("offset"):
        argv += ["--offset", str(m["offset"])]
    writer = ProgressWriter(walk)
    try:
        with redirect_stdout(writer):
            kerbside.run(kerbside.parse_args(argv))
        m.update(state="done", error=None)
        walk.progress = 100.0
    except kerbside.KerbsideError as exc:
        m.update(state="error", error=str(exc))
    except SystemExit as exc:  # argparse rejected something
        m.update(state="error", error=f"Invalid settings ({exc}).")
    except Exception as exc:  # keep the app alive and show what happened
        walk.add_line(traceback.format_exc())
        m.update(state="error", error=f"Unexpected error: {exc}")
    finally:
        writer.flush()
        m["log"] = walk.log[-200:]
        walk.save()
        LIVE.pop(walk.id, None)
        RUN_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    """JSON API plus the app page. Configuration lives on self.server."""

    server_version = "Kerbside"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
        pass

    # -- responses
    def send(self, code: int, body: bytes, ctype: str = "application/json", headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, code: int, obj: Any) -> None:
        self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def fail(self, code: int, message: str) -> None:
        self.json(code, {"error": message})

    # -- request helpers
    def allowed(self, mutating: bool) -> bool:
        """Only this laptop's own page may use the API.

        Checking Host stops DNS-rebinding tricks; the custom header on changes forces a CORS
        preflight that we never approve, so other websites can't post to the app.
        """
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost"):
            self.fail(403, "Kerbside only accepts requests from this computer.")
            return False
        if mutating and self.headers.get("X-Kerbside") != "1":
            self.fail(403, "Missing X-Kerbside header.")
            return False
        return True

    def body(self, limit: int = 20 * 1024 * 1024) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            self.fail(413, "That file is too large.")
            return None
        return self.rfile.read(length)

    def walk(self, walk_id: str) -> Walk | None:
        folder = self.server.walks_dir / walk_id
        if not WALK_ID.match(walk_id) or not folder.is_dir():
            self.fail(404, "No such walk.")
            return None
        return LIVE.get(walk_id) or Walk(folder)

    # -- routing
    def do_GET(self) -> None:
        if not self.allowed(mutating=False):
            return
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not parts:
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif parts == ["api", "meta"]:
            self.json(200, {"categories": kerbside.CATEGORIES, "severity_labels": kerbside.SEVERITY_LABELS})
        elif parts == ["api", "walks"]:
            folders = sorted((p for p in self.server.walks_dir.iterdir() if p.is_dir() and WALK_ID.match(p.name)),
                             reverse=True)
            self.json(200, [(LIVE.get(p.name) or Walk(p)).summary() for p in folders])
        elif len(parts) >= 3 and parts[:2] == ["api", "walks"]:
            walk = self.walk(parts[2])
            if not walk:
                return
            rest = parts[3:]
            if not rest:
                self.json(200, walk.status())
            elif rest == ["findings"]:
                geo = walk.out / "findings.geojson"
                if not geo.exists():
                    return self.fail(404, "This walk has no findings yet.")
                self.send(200, geo.read_bytes(), "application/geo+json")
            elif rest == ["track"]:
                track = kerbside.load_track(walk.out / "track.json")
                self.json(200, {"points": [[round(lat, 6), round(lon, 6)] for _, lat, lon in track]})
            elif len(rest) == 2 and rest[0] == "download" and rest[1] in DOWNLOADS:
                path = walk.out / rest[1]
                if not path.exists():
                    return self.fail(404, "Not created yet.")
                inline = "inline" in parse_qs(url.query)
                name = f"kerbside-{walk.id}-{rest[1]}"
                self.send(200, path.read_bytes(), DOWNLOADS[rest[1]],
                          {"Content-Disposition": f'{"inline" if inline else "attachment"}; filename="{name}"'})
            else:
                self.fail(404, "Not found.")
        else:
            self.fail(404, "Not found.")

    def do_POST(self) -> None:
        if not self.allowed(mutating=True):
            return
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if parts == ["api", "walks"]:
            self.create_walk()
        elif len(parts) == 4 and parts[:2] == ["api", "walks"] and parts[3] == "start":
            walk = self.walk(parts[2])
            if walk:
                self.start_walk(walk)
        else:
            self.fail(404, "Not found.")

    def do_PUT(self) -> None:
        if not self.allowed(mutating=True):
            return
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if len(parts) != 4 or parts[:2] != ["api", "walks"]:
            return self.fail(404, "Not found.")
        walk = self.walk(parts[2])
        if not walk:
            return
        if parts[3] == "upload":
            self.upload(walk, parse_qs(url.query))
        elif parts[3] == "findings":
            self.save_findings(walk)
        else:
            self.fail(404, "Not found.")

    # -- actions
    def create_walk(self) -> None:
        raw = self.body()
        if raw is None:
            return
        try:
            req = json.loads(raw or b"{}")
            zone = kerbside.get_zone(str(req.get("tz") or "Asia/Kolkata"))
            start = str(req.get("start") or "").strip()
            if start:
                kerbside.parse_start(start, zone)  # friendly error now rather than after uploading
            offset = float(req.get("offset") or 0)
        except kerbside.KerbsideError as exc:
            return self.fail(400, str(exc))
        except (ValueError, TypeError):
            return self.fail(400, "Offset must be a number of seconds.")
        walk_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        folder = self.server.walks_dir / walk_id
        folder.mkdir(parents=True)
        walk = Walk(folder)
        walk.meta = {"title": str(req.get("title") or "Kerbside Street Audit")[:200], "created": datetime.now().isoformat(
            timespec="seconds"), "tz": zone.key, "start": start, "offset": offset, "state": "new"}
        walk.save()
        self.json(201, walk.status())

    def upload(self, walk: Walk, query: dict) -> None:
        kind = (query.get("kind") or [""])[0]
        suffix = Path((query.get("filename") or [""])[0]).suffix.lower()
        if kind == "audio" and suffix not in AUDIO_EXT:
            return self.fail(400, f"Audio must be one of: {', '.join(sorted(AUDIO_EXT))}.")
        if kind == "gpx" and suffix != ".gpx":
            return self.fail(400, "The GPS track must be a .gpx file.")
        if kind not in ("audio", "gpx"):
            return self.fail(400, "kind must be audio or gpx.")
        if walk.state() == "running":
            return self.fail(409, "This walk is being processed.")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self.fail(400, "The file is empty.")
        if length > MAX_UPLOAD:
            return self.fail(413, "That file is larger than 1 GB.")
        name = f"{kind}{suffix}"
        if walk.meta.get(kind) and walk.meta[kind] != name:
            (walk.folder / walk.meta[kind]).unlink(missing_ok=True)
        with (walk.folder / name).open("wb") as fh:
            left = length
            while left > 0:
                chunk = self.rfile.read(min(left, 1024 * 1024))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        walk.meta[kind] = name
        walk.meta[f"{kind}_name"] = Path((query.get("filename") or [""])[0]).name[:200]
        walk.save()
        self.json(200, walk.status())

    def start_walk(self, walk: Walk) -> None:
        if not (walk.meta.get("audio") and walk.meta.get("gpx")):
            return self.fail(400, "Add both the audio recording and the GPS track first.")
        if not RUN_LOCK.acquire(blocking=False):
            return self.fail(409, "Another walk is being processed. Wait for it to finish.")
        walk.meta.update(state="running", error=None)
        walk.save()
        LIVE[walk.id] = walk
        threading.Thread(target=run_walk, args=(walk, self.server.model, self.server.whisper), daemon=True).start()
        self.json(202, walk.status())

    def save_findings(self, walk: Walk) -> None:
        if walk.state() == "running":
            return self.fail(409, "This walk is being processed.")
        raw = self.body()
        if raw is None:
            return
        try:
            geo = json.loads(raw)
            raw_feats = geo["features"]
            assert isinstance(raw_feats, list)
        except (ValueError, KeyError, TypeError, AssertionError):
            return self.fail(400, "Expected a GeoJSON FeatureCollection.")
        features = [f for f in (kerbside.normalize_feature(r) for r in raw_feats if isinstance(r, dict)) if f]
        walk.out.mkdir(parents=True, exist_ok=True)
        kerbside.write_json(walk.out / "findings.geojson", {"type": "FeatureCollection", "features": features})
        try:
            with redirect_stdout(io.StringIO()):
                kerbside.render_outputs(walk.out, walk.meta.get("title") or "Kerbside Street Audit",
                                        kerbside.get_zone(walk.meta.get("tz") or "Asia/Kolkata"))
        except kerbside.KerbsideError as exc:
            return self.fail(500, str(exc))
        self.json(200, {"saved": len(features)})


def make_server(walks_dir: Path, model: str, whisper: str, port: int = 8765) -> ThreadingHTTPServer:
    """Create (but don't start) the app server on 127.0.0.1, trying a few ports if busy."""
    walks_dir.mkdir(parents=True, exist_ok=True)
    last: OSError | None = None
    for p in ([port] + list(range(port + 1, port + 10))) if port else [0]:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError as exc:
            last = exc
    else:
        raise kerbside.KerbsideError(f"Couldn't open a port for the app ({last}).")
    server.daemon_threads = True
    server.walks_dir, server.model, server.whisper = walks_dir, model, whisper
    return server


def serve(out: Path, model: str, whisper: str, open_browser: bool = True) -> None:
    """Run the app until Ctrl+C."""
    server = make_server(out / "walks", model, whisper)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Kerbside app running at {url}  (walks are saved in {out / 'walks'})")
    print("Only this computer can open it. Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
