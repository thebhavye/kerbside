#!/usr/bin/env python3
"""Kerbside: walk your street, talk about what's broken, get a map.

Turns a voice narration (.m4a) and a GPS track (.gpx) into a map and a report
of street problems, using only local open models: faster-whisper for speech to
text and Gemma (via Ollama) for pulling observations out of the transcript.

    python kerbside.py WALK.m4a TRACK.gpx [options]
    python kerbside.py --render-only [--out DIR]
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import html
import json
import math
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import gpxpy
    import gpxpy.gpx
    import requests
except ImportError as exc:  # pragma: no cover - depends on the user's setup
    sys.exit(f"Missing Python package ({exc.name}). Run: pip install -r requirements.txt")

OLLAMA_URL = "http://localhost:11434"
CHUNK_LINES = 12
MAX_GPS_GAP_S = 30.0
ALIGN_WARN_S = 600.0
FOOTER = ("Drafted locally from a resident's voice notes using Whisper and Gemma. "
          "Reviewed by a person before sending.")

CATEGORIES: dict[str, dict[str, Any]] = {
    "broken_footpath": {"label": "Broken footpath", "color": "#d7301f", "positive": False},
    "no_footpath": {"label": "No footpath", "color": "#7a0177", "positive": False},
    "obstruction": {"label": "Footpath obstruction", "color": "#ec7014", "positive": False},
    "blocked_drain": {"label": "Blocked or open drain", "color": "#6a51a3", "positive": False},
    "waterlogging": {"label": "Waterlogging", "color": "#2171b5", "positive": False},
    "garbage": {"label": "Garbage", "color": "#8c6d31", "positive": False},
    "streetlight_out": {"label": "Streetlight not working", "color": "#b8860b", "positive": False},
    "unsafe_crossing": {"label": "Unsafe crossing", "color": "#e7298a", "positive": False},
    "other_problem": {"label": "Other problem", "color": "#737373", "positive": False},
    "shade_tree": {"label": "Shade tree", "color": "#238b45", "positive": True},
    "good_footpath": {"label": "Good footpath", "color": "#1b9e77", "positive": True},
}
SEVERITY_LABELS = {1: "minor", 2: "moderate", 3: "severe"}

SYSTEM_PROMPT = """You help a resident audit their street for walkability.
You get numbered lines from a transcript of the resident talking while walking.
Label EVERY line, in order, with exactly one label:

- not_an_observation: chatter, filler, self-directions ("ok turning left", "let me see"),
  starting or ending the walk. Use this when the line says nothing about the street itself.
- same_as_previous: the line only adds detail or a consequence to the observation on the line
  just before it ("someone could trip on this", "it's been like this for months"). A line that
  only says how bad, risky or dangerous something is, or points back with "this", "it" or "that",
  without naming a new thing on the street, is same_as_previous.
- one of these categories, when the line is a new thing the speaker sees on the street:
  - broken_footpath: cracked, broken, missing slabs, holes in the footpath, uneven footpath
  - no_footpath: no footpath was ever built here
  - obstruction: a footpath exists but something blocks it: parked cars or bikes, shop goods,
    stalls, encroachment, construction material, animals. Use this even if people must walk on the road.
  - blocked_drain: blocked drains, open drains, missing drain or manhole covers
  - waterlogging: standing water, flooding
  - garbage: garbage, litter, dumped waste
  - streetlight_out: streetlight not working or missing
  - unsafe_crossing: dangerous or missing crossing, broken pedestrian signal, fast traffic where people cross
  - other_problem: any other problem for people walking
  - shade_tree: trees giving shade (positive)
  - good_footpath: a good, wide, clean footpath (positive)

For each line also give:
- summary: at most 15 words, plain English, written like a short complaint line. Use only
  details the speaker said; never invent places, landmarks or causes. Fix obvious transcription
  errors. Don't add phrases like "is observed". Use "" if not_an_observation.
- severity: 1 = minor annoyance (litter, smell, small crack); 2 = makes walking hard (blocked
  footpath, standing water, dark street, blocked drain); 3 = dangerous, could injure someone: a trip
  or fall hazard (missing slab, big hole, open drain or manhole), crossing fast traffic with no safe
  crossing, or the speaker says someone could get hurt. Positives and not_an_observation are always 1."""

LINE_LABELS = ["not_an_observation", "same_as_previous", *CATEGORIES]


class KerbsideError(Exception):
    """A problem we can explain to the user in plain words."""


TrackPoint = tuple[datetime, float, float]  # (UTC time, lat, lon)


# ---------------------------------------------------------------- helpers

def write_json(path: Path, data: Any) -> None:
    """Write JSON as UTF-8, keeping non-English text readable."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file with a friendly error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise KerbsideError(f"Can't find {path}.") from None
    except json.JSONDecodeError as exc:
        raise KerbsideError(f"{path} is not valid JSON (line {exc.lineno}): {exc.msg}") from None


def get_zone(name: str) -> ZoneInfo:
    """Look up a timezone by IANA name."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise KerbsideError(f"Unknown timezone '{name}'. Use a name like Asia/Kolkata. "
                            "(On Windows this needs: pip install tzdata)") from None


def to_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC and convert aware ones to UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_start(text: str, zone: ZoneInfo) -> datetime:
    """Parse --start (local time in `zone` unless it has an offset) to UTC."""
    try:
        dt = datetime.fromisoformat(text.strip())
    except ValueError:
        raise KerbsideError(f"Can't understand --start '{text}'. "
                            'Use a format like "2026-10-10 07:30:00".') from None
    return to_utc(dt.replace(tzinfo=zone) if dt.tzinfo is None else dt)


def fmt_clock(seconds: float) -> str:
    """Format seconds as m:ss."""
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"


def fmt_duration(seconds: float) -> str:
    """Format a walk duration like '8 min' or '1 h 05 min'."""
    minutes = int(round(seconds / 60))
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


def fmt_distance(metres: float) -> str:
    """Format a distance in m or km."""
    return f"{metres:.0f} m" if metres < 1000 else f"{metres / 1000:.2f} km"


def fmt_date(dt: datetime) -> str:
    """Format a date like '10 Oct 2026'."""
    return f"{dt.day} {dt:%b %Y}"


def osm_link(lat: float, lon: float) -> str:
    """OpenStreetMap link centred on a point."""
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=19/{lat:.6f}/{lon:.6f}"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- GPS track

def load_gpx(path: Path) -> list[TrackPoint]:
    """Read all timestamped points from all tracks/segments of a GPX file, sorted by time."""
    try:
        gpx = gpxpy.parse(path.read_text(encoding="utf-8-sig"))
    except (gpxpy.gpx.GPXException, UnicodeDecodeError) as exc:
        raise KerbsideError(f"Couldn't read {path.name} as a GPX file: {exc}") from None
    points = [(to_utc(p.time), p.latitude, p.longitude)
              for trk in gpx.tracks for seg in trk.segments for p in seg.points if p.time]
    if not points:
        raise KerbsideError(f"{path.name} has no GPS points with timestamps. "
                            "Was Open GPX Tracker recording during the walk?")
    points.sort(key=lambda p: p[0])
    return points


def save_track(track: list[TrackPoint], path: Path) -> None:
    """Save the track as JSON so --render-only can draw it."""
    write_json(path, {"points": [[t.isoformat(), round(lat, 7), round(lon, 7)] for t, lat, lon in track]})


def load_track(path: Path) -> list[TrackPoint]:
    """Load a track saved by save_track (empty list if missing)."""
    if not path.exists():
        return []
    return [(datetime.fromisoformat(t), lat, lon) for t, lat, lon in read_json(path).get("points", [])]


def track_stats(track: list[TrackPoint]) -> dict[str, Any]:
    """Distance (m), duration (s) and start time of a track."""
    dist = sum(haversine(a[1], a[2], b[1], b[2]) for a, b in zip(track, track[1:]))
    duration = (track[-1][0] - track[0][0]).total_seconds() if track else 0.0
    return {"distance_m": dist, "duration_s": duration, "start": track[0][0] if track else None}


def locate(track: list[TrackPoint], ts: list[float], t: float) -> tuple[float, float, float, bool]:
    """Interpolate a position at UNIX time t. Returns (lat, lon, gap to nearest GPS point in s, inside track)."""
    i = bisect.bisect_left(ts, t)
    if i < len(ts) and ts[i] == t:
        return track[i][1], track[i][2], 0.0, True
    if i == 0:
        return track[0][1], track[0][2], ts[0] - t, False
    if i == len(ts):
        return track[-1][1], track[-1][2], t - ts[-1], False
    (_, lat0, lon0), (_, lat1, lon1) = track[i - 1], track[i]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return lat0 + f * (lat1 - lat0), lon0 + f * (lon1 - lon0), min(t - ts[i - 1], ts[i] - t), True


def align_segments(segments: list[dict], track: list[TrackPoint], start: datetime,
                   offset: float = 0.0) -> list[dict]:
    """Give each transcript segment an absolute time and a position on the track."""
    ts = [p[0].timestamp() for p in track]
    base = start.timestamp() + offset
    aligned = []
    for seg in segments:
        t = base + float(seg["start"])
        lat, lon, gap, inside = locate(track, ts, t)
        aligned.append({**seg, "time": datetime.fromtimestamp(t, timezone.utc), "lat": lat, "lon": lon,
                        "gps_gap_s": round(gap, 1), "needs_review": (not inside) or gap > MAX_GPS_GAP_S})
    return aligned


# ---------------------------------------------------------------- start time

def audio_metadata(path: Path) -> tuple[datetime | None, float | None]:
    """Read (creation_time as UTC, duration in s) from the audio file with PyAV, if available."""
    try:
        import av
        with av.open(str(path)) as container:
            raw = container.metadata.get("creation_time")
            for stream in container.streams:
                raw = raw or stream.metadata.get("creation_time")
            duration = container.duration / av.time_base if container.duration else None
    except Exception:  # PyAV missing or unreadable file: fall back to other sources
        return None, None
    created = None
    if raw:
        try:
            created = to_utc(datetime.fromisoformat(raw.strip().replace("Z", "+00:00")))
        except ValueError:
            pass
    return created, duration


def resolve_start(start_arg: str | None, zone: ZoneInfo, audio_created: datetime | None,
                  track: list[TrackPoint]) -> tuple[datetime, str]:
    """Pick the recording start: --start, then audio metadata, then first GPX point."""
    if start_arg:
        return parse_start(start_arg, zone), "--start option"
    if audio_created:
        return audio_created, "audio file creation_time metadata"
    return track[0][0], "first GPX timestamp (no --start and no audio metadata)"


def time_span_warning(start: datetime, duration: float, track: list[TrackPoint]) -> str | None:
    """Warn if the audio falls more than 10 minutes outside the GPX track's time span."""
    a0, a1 = start.timestamp(), start.timestamp() + duration
    g0, g1 = track[0][0].timestamp(), track[-1][0].timestamp()
    if a0 < g0 - ALIGN_WARN_S or a1 > g1 + ALIGN_WARN_S:
        return (f"The audio ({fmt_clock(a0 - g0)} from track start, {fmt_clock(duration)} long) falls more than "
                "10 minutes outside the GPS track. Positions may be wrong: check --start, --tz or --offset.")
    return None


# ---------------------------------------------------------------- transcription

def transcribe(audio: Path, cache_path: Path, size: str, lang: str | None, translate: bool,
               fresh: bool) -> dict:
    """Transcribe with faster-whisper on CPU, reusing OUT/transcript.json when possible."""
    task = "translate" if translate else "transcribe"
    key = {"audio_file": audio.name, "whisper_model": size, "requested_language": lang, "task": task,
           "timestamps": "word"}
    if not fresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if all(cached.get(k) == v for k, v in key.items()) and isinstance(cached.get("segments"), list):
            print(f"  Using cached transcript ({len(cached['segments'])} lines). Use --fresh to redo.")
            return cached
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows/HF noise
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise KerbsideError("faster-whisper is not installed. Run: pip install -r requirements.txt") from None
    print(f"  Loading Whisper '{size}' (the first run downloads it, ~500 MB for 'small')...")
    try:
        model = WhisperModel(size, device="cpu", compute_type="int8")
        # Word timestamps: with vad_filter, a segment's own start can snap back to the end of the
        # previous speech after a long pause; the first word's start is accurate.
        seg_iter, info = model.transcribe(str(audio), language=lang, task=task, vad_filter=True,
                                          word_timestamps=True)
    except Exception as exc:
        raise KerbsideError(f"Whisper couldn't start: {exc}\n"
                            "The first run needs internet once to download the model.") from None
    print(f"  Language: {info.language} ({info.language_probability:.0%}), audio {fmt_clock(info.duration)}. "
          "Transcribing on CPU, this can take a while...")
    segments = []
    for s in seg_iter:
        text = s.text.strip()
        if not text:
            continue
        start, end = (s.words[0].start, s.words[-1].end) if s.words else (s.start, s.end)
        segments.append({"id": len(segments) + 1, "start": round(start, 2), "end": round(end, 2), "text": text})
        pct = min(100.0, end / info.duration * 100) if info.duration else 0.0
        print(f"  {pct:5.1f}%  [{fmt_clock(start)}] {text}")
    data = {**key, "language": info.language, "duration": info.duration, "segments": segments}
    write_json(cache_path, data)
    print(f"  Saved {len(segments)} lines to {cache_path}")
    return data


# ---------------------------------------------------------------- extraction (Ollama)

def check_ollama(model: str, host: str = OLLAMA_URL) -> None:
    """Make sure Ollama is running and the model is pulled."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
    except (requests.RequestException, ValueError):
        raise KerbsideError(f"Can't reach Ollama at {host}.\n"
                            "Start the Ollama app (or run: ollama serve) and try again.") from None
    names = {m.get(k, "") for m in models for k in ("name", "model")}
    wanted = {model} if ":" in model else {model, f"{model}:latest"}
    if not names & wanted:
        raise KerbsideError(f"The Ollama model '{model}' isn't downloaded yet.\nRun: ollama pull {model}")


def findings_schema() -> dict:
    """JSON schema for Ollama structured outputs: one label per transcript line."""
    return {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "label": {"type": "string", "enum": LINE_LABELS},
                "summary": {"type": "string"},  # before severity, so the model describes first
                "severity": {"type": "integer", "enum": [1, 2, 3]},
            },
            "required": ["id", "label", "summary", "severity"],
        }}},
        "required": ["lines"],
    }


def ollama_chat(model: str, system: str, user: str, schema: dict, host: str = OLLAMA_URL) -> Any:
    """One non-streaming /api/chat call; returns the parsed JSON reply."""
    payload = {"model": model, "stream": False, "format": schema, "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        r = requests.post(f"{host}/api/chat", json=payload, timeout=900)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise KerbsideError(f"Ollama request failed: {exc}") from None
    return json.loads(r.json().get("message", {}).get("content", ""))


def lines_to_findings(labels: Any, chunk: list[dict]) -> list[dict]:
    """Turn per-line labels into findings, merging same_as_previous lines into the one before.

    The model sees the chunk numbered 1..n; labels are mapped back to transcript ids by
    position. Labels with numbers outside 1..n are dropped, so a finding only covers
    consecutive lines the model actually saw.
    """
    by_pos: dict[int, dict] = {}
    for raw in labels if isinstance(labels, list) else []:
        if isinstance(raw, dict) and raw.get("id") in range(1, len(chunk) + 1) \
                and raw.get("label") in LINE_LABELS:
            by_pos.setdefault(raw["id"], raw)
    findings: list[dict] = []
    for pos, seg in enumerate(chunk, 1):
        raw = by_pos.get(pos)
        label = raw["label"] if raw else "not_an_observation"
        severity = raw.get("severity") if raw and raw.get("severity") in (1, 2, 3) else 2
        if label == "same_as_previous":
            if findings and findings[-1]["line_ids"][-1] == chunk[pos - 2]["id"]:
                last = findings[-1]
                last["line_ids"].append(seg["id"])
                if not CATEGORIES[last["category"]]["positive"]:
                    last["severity"] = max(last["severity"], severity)
        elif label in CATEGORIES:
            findings.append({"line_ids": [seg["id"]], "category": label,
                             "severity": 1 if CATEGORIES[label]["positive"] else severity,
                             "summary": " ".join(str(raw.get("summary", "")).split())})
    return findings


def extract_chunk(chunk: list[dict], model: str, host: str) -> list[dict]:
    """Ask Gemma to label each line of one chunk, then build findings from the labels."""
    user = "Transcript lines:\n" + "\n".join(f"{n}: {s['text']}" for n, s in enumerate(chunk, 1))
    data: Any = None
    for attempt in range(2):
        try:
            data = ollama_chat(model, SYSTEM_PROMPT, user, findings_schema(), host)
            break
        except ValueError:
            if attempt:
                print("    Warning: the model returned invalid JSON twice; skipping this chunk.")
                return []
    labels = data.get("lines", []) if isinstance(data, dict) else []
    stray = sum(1 for r in labels if not (isinstance(r, dict) and r.get("id") in range(1, len(chunk) + 1)))
    if stray:
        print(f"    Ignored {stray} label(s) for lines that aren't in this chunk.")
    return lines_to_findings(labels, chunk)


def transcript_hash(segments: list[dict]) -> str:
    """Fingerprint of the transcript text, to know when cached findings are stale."""
    text = "\n".join(f"{s['id']}\t{s['text']}" for s in segments)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_findings(segments: list[dict], model: str, cache_path: Path, fresh: bool,
                     host: str = OLLAMA_URL) -> list[dict]:
    """Run extraction over the transcript in chunks, caching results in OUT/extraction.json."""
    prompt = SYSTEM_PROMPT + json.dumps(findings_schema())
    key = {"model": model, "transcript_sha256": transcript_hash(segments),
           "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
    if not fresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if all(cached.get(k) == v for k, v in key.items()):
            print(f"  Using cached findings ({len(cached['findings'])}). Use --fresh to redo.")
            return cached["findings"]
    check_ollama(model, host)
    chunks = [segments[i:i + CHUNK_LINES] for i in range(0, len(segments), CHUNK_LINES)]
    findings: list[dict] = []
    seen: set[tuple] = set()
    for n, chunk in enumerate(chunks, 1):
        print(f"  Chunk {n}/{len(chunks)} (lines {chunk[0]['id']}-{chunk[-1]['id']})...", flush=True)
        for f in extract_chunk(chunk, model, host):
            sig = (tuple(f["line_ids"]), f["category"])
            if sig not in seen:
                seen.add(sig)
                findings.append(f)
        print(f"    {len(findings)} finding(s) so far")
    write_json(cache_path, {**key, "findings": findings})
    return findings


# ---------------------------------------------------------------- GeoJSON

def build_features(findings: list[dict], aligned: list[dict], zone: ZoneInfo) -> list[dict]:
    """One GeoJSON Point per finding, placed at its first transcript line."""
    by_id = {s["id"]: s for s in aligned}
    usable = sorted((f for f in findings if f["line_ids"][0] in by_id), key=lambda f: f["line_ids"][0])
    features = []
    for n, f in enumerate(usable, 1):
        first = by_id[f["line_ids"][0]]
        cat = CATEGORIES[f["category"]]
        quote = " ".join(by_id[i]["text"] for i in f["line_ids"] if i in by_id)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(first["lon"], 7), round(first["lat"], 7)]},
            "properties": {
                "id": n, "category": f["category"], "label": cat["label"], "severity": f["severity"],
                "severity_label": SEVERITY_LABELS[f["severity"]], "summary": f["summary"] or quote,
                "quote": quote, "time_local": first["time"].astimezone(zone).strftime("%Y-%m-%d %H:%M:%S"),
                "needs_review": first["needs_review"], "positive": cat["positive"], "line_ids": f["line_ids"],
            },
        })
    return features


def normalize_feature(feat: dict) -> dict | None:
    """Make a (possibly hand-edited) feature consistent: labels follow category/severity."""
    try:
        lon, lat = feat["geometry"]["coordinates"][:2]
        lat, lon = float(lat), float(lon)
    except (KeyError, TypeError, ValueError):
        return None
    p = dict(feat.get("properties") or {})
    cat = p.get("category") if p.get("category") in CATEGORIES else "other_problem"
    info = CATEGORIES[cat]
    try:
        sev = min(3, max(1, int(p.get("severity", 2))))
    except (TypeError, ValueError):
        sev = 2
    if info["positive"]:
        sev = 1
    p.update(category=cat, label=info["label"], severity=sev, severity_label=SEVERITY_LABELS[sev],
             positive=info["positive"], needs_review=bool(p.get("needs_review", False)),
             summary=str(p.get("summary") or p.get("quote") or info["label"]),
             quote=str(p.get("quote") or ""), time_local=str(p.get("time_local") or ""))
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": p}


# ---------------------------------------------------------------- rendering

MAP_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<style>
html,body{height:100%;margin:0;background:#fff;color:#1a1a1a;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
body{display:flex;flex-direction:column}
header{padding:10px 14px;border-bottom:1px solid #ddd}
h1{font-size:1.15rem;margin:0 0 4px}
.stats{font-size:.85rem;color:#444;display:flex;flex-wrap:wrap;gap:2px 14px}
#map{flex:1;min-height:320px}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);
  font-size:12px;line-height:1.6;max-height:40vh;overflow:auto}
.legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.legend i.dash{width:8px;height:8px;border:2px dashed #222}
.legend summary{cursor:pointer;font-weight:600;padding:2px 0}
.legend details:not([open]) summary{padding:0}
.search{background:#fff;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);width:min(260px,62vw)}
.search form{display:flex}
.search input{flex:1;min-width:0;border:0;padding:8px 10px;font-size:16px;border-radius:6px 0 0 6px;
  background:transparent;color:#1a1a1a}
.search button[type=submit]{border:0;border-left:1px solid #ddd;background:#f4f4f4;color:#1a1a1a;padding:0 12px;
  font-size:15px;border-radius:0 6px 6px 0;cursor:pointer}
.search .results{border-top:1px solid #ddd;max-height:40vh;overflow:auto}
.search .results button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #eee;
  background:#fff;color:#1a1a1a;padding:8px 10px;font-size:13px;line-height:1.35;cursor:pointer}
.search .msg{padding:8px 10px;font-size:13px;color:#555}
.popup h3{margin:0 0 4px;font-size:14px}
.popup .q{font-style:italic;color:#555;margin:6px 0}
.popup .warn{color:#a15c00}
@media (max-width:600px){h1{font-size:1rem}.stats{font-size:.78rem}.legend{font-size:11px}}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="stats">__STATS__</div></header>
<div id="map"></div>
<script>
const DATA = __DATA__;
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){
  return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});}
// Canvas renderer with a wide hit tolerance: small circles are easy to tap with a finger.
const map = L.map("map", {renderer: L.canvas({tolerance: 10})});
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
const layers = [];
if (DATA.track.length > 1) layers.push(L.polyline(DATA.track, {color: "#3388ff", weight: 4, opacity: 0.7}).addTo(map));
DATA.findings.slice().sort(function(a,b){return a.severity-b.severity;}).forEach(function(f){
  const c = DATA.categories[f.category] || DATA.categories.other_problem;
  const m = L.circleMarker([f.lat, f.lon], {radius: 4 + f.severity * 3, fillColor: c.color, fillOpacity: 0.8,
    color: f.needs_review ? "#222" : c.color, weight: f.needs_review ? 2.5 : 2, dashArray: f.needs_review ? "4 3" : null});
  const sev = f.positive ? "Good thing" : "Severity: " + esc(f.severity_label) + " (" + f.severity + "/3)";
  m.bindPopup('<div class="popup"><h3>' + esc(f.label) + '</h3><div><b>' + sev + '</b></div><div>' + esc(f.summary) +
    '</div>' + (f.quote ? '<div class="q">&ldquo;' + esc(f.quote) + '&rdquo;</div>' : '') + '<div>' + esc(f.time_local) +
    '</div>' + (f.needs_review ? '<div class="warn">Location approximate, please check</div>' : '') +
    '<div><a href="' + esc(f.osm) + '" target="_blank" rel="noopener">Open in OpenStreetMap</a></div></div>', {maxWidth: 260});
  layers.push(m.addTo(map));
});
if (layers.length) map.fitBounds(L.featureGroup(layers).getBounds(), {padding: [24, 24], maxZoom: 18});
else map.setView([20, 78], 4);
const legend = L.control({position: "bottomright"});
legend.onAdd = function(){
  const div = L.DomUtil.create("div", "legend");
  // Collapsed on phones so it doesn't cover the map; open on wider screens.
  const open = window.matchMedia("(min-width: 601px)").matches ? " open" : "";
  let h = "<details" + open + "><summary>Map key</summary>";
  Object.keys(DATA.categories).forEach(function(k){
    const n = DATA.findings.filter(function(f){return f.category === k;}).length;
    if (n) h += '<div><i style="background:' + DATA.categories[k].color + '"></i>' + esc(DATA.categories[k].label) + " (" + n + ")</div>";
  });
  h += '<div style="margin-top:4px;color:#555">Bigger circle = more severe</div>';
  if (DATA.findings.some(function(f){return f.needs_review;})) h += '<div><i class="dash"></i>Location needs review</div>';
  div.innerHTML = h + "</details>";
  L.DomEvent.disableScrollPropagation(div);
  L.DomEvent.disableClickPropagation(div);
  return div;
};
legend.addTo(map);
// Place search with OpenStreetMap's Nominatim. Runs only on submit, at most once per second:
// its usage policy forbids search-as-you-type on the free service.
const search = L.control({position: "topright"});
search.onAdd = function(){
  const div = L.DomUtil.create("div", "search");
  div.innerHTML = '<form role="search"><input type="search" name="q" placeholder="Search a place" ' +
    'aria-label="Search a place" enterkeyhint="search" autocomplete="off">' +
    '<button type="submit" aria-label="Search">Go</button></form><div class="results" hidden></div>';
  L.DomEvent.disableClickPropagation(div);
  L.DomEvent.disableScrollPropagation(div);
  const form = div.querySelector("form"), input = form.q, go = form.querySelector("button");
  const results = div.querySelector(".results");
  let last = 0, pin = null;
  function show(h){ results.innerHTML = h; results.hidden = !h; }
  function clearPin(){ if (pin) { pin.remove(); pin = null; } }
  form.addEventListener("submit", function(e){
    e.preventDefault();
    const q = input.value.trim();
    if (!q || go.disabled) return;
    go.disabled = true;
    show('<div class="msg">Searching…</div>');
    setTimeout(function(){
      last = Date.now();
      const b = map.getBounds();
      const box = [b.getWest(), b.getNorth(), b.getEast(), b.getSouth()].map(function(v){return v.toFixed(5);}).join(",");
      fetch("https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&viewbox=" + box +
            "&q=" + encodeURIComponent(q), {headers: {"Accept": "application/json"}})
        .then(function(r){ if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function(list){
          if (!list.length) { show('<div class="msg">No places found. Try adding the area or city.</div>'); return; }
          show(list.map(function(p, i){
            return '<button type="button" data-i="' + i + '">' + esc(p.display_name) + '</button>';
          }).join(""));
          results.querySelectorAll("button").forEach(function(el){
            el.addEventListener("click", function(ev){
              // show("") below removes this button, so Leaflet can't tell the click came from the
              // search box and would treat it as a map click that closes the new pin's popup.
              L.DomEvent.stopPropagation(ev);
              const p = list[+el.dataset.i], bb = p.boundingbox.map(Number);
              clearPin();
              pin = L.marker([+p.lat, +p.lon], {title: p.display_name}).addTo(map).bindPopup(esc(p.display_name));
              map.fitBounds([[bb[0], bb[2]], [bb[1], bb[3]]], {maxZoom: 18});
              pin.openPopup();
              show("");
              input.blur();
            });
          });
        })
        .catch(function(){ show('<div class="msg">Search failed. It needs an internet connection.</div>'); })
        .finally(function(){ go.disabled = false; });
    }, Math.max(0, last + 1000 - Date.now()));
  });
  input.addEventListener("search", function(){ if (!input.value) { show(""); clearPin(); } });  // the clear (×) button
  return div;
};
search.addTo(map);
</script>
</body>
</html>
"""


def summary_counts(features: list[dict]) -> dict[str, int]:
    """Counts of problems, severe problems, positives and items needing review."""
    props = [f["properties"] for f in features]
    return {"problems": sum(not p["positive"] for p in props),
            "severe": sum(not p["positive"] and p["severity"] == 3 for p in props),
            "positives": sum(p["positive"] for p in props),
            "review": sum(p["needs_review"] for p in props)}


def walk_date(stats: dict, features: list[dict], zone: ZoneInfo) -> str:
    """Date of the walk from the track, or from the first finding."""
    if stats["start"]:
        return fmt_date(stats["start"].astimezone(zone))
    for f in features:
        try:
            return fmt_date(datetime.fromisoformat(f["properties"]["time_local"]))
        except ValueError:
            continue
    return "unknown date"


def render_map(features: list[dict], track: list[TrackPoint], title: str, zone: ZoneInfo) -> str:
    """Self-contained Leaflet page with the track, findings, legend and header stats."""
    stats, counts = track_stats(track), summary_counts(features)
    parts = [walk_date(stats, features, zone)]
    if track:
        parts += [f"{fmt_distance(stats['distance_m'])} walked", fmt_duration(stats["duration_s"])]
    parts.append(f"{counts['problems']} problem{'s' * (counts['problems'] != 1)} ({counts['severe']} severe)")
    parts.append(f"{counts['positives']} good thing{'s' * (counts['positives'] != 1)}")
    if counts["review"]:
        parts.append(f"{counts['review']} to review")
    data = {
        "track": [[round(lat, 6), round(lon, 6)] for _, lat, lon in track],
        "categories": {k: {"label": v["label"], "color": v["color"]} for k, v in CATEGORIES.items()},
        "findings": [{**f["properties"], "lat": f["geometry"]["coordinates"][1], "lon": f["geometry"]["coordinates"][0],
                      "osm": osm_link(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])}
                     for f in features],
    }
    data_js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    stats_html = "".join(f"<span>{html.escape(p)}</span>" for p in parts)
    return (MAP_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__STATS__", stats_html)
            .replace("__DATA__", data_js))


def md_text(s: str) -> str:
    """Keep user text on one line and stop it from breaking Markdown links, tables or HTML."""
    s = " ".join(s.split()).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("|", "/").replace("[", "(").replace("]", ")")


def render_report(features: list[dict], track: list[TrackPoint], title: str, zone: ZoneInfo) -> str:
    """One-page Markdown report for a ward office or complaint portal."""
    stats, counts = track_stats(track), summary_counts(features)
    props = [(f["properties"], f["geometry"]["coordinates"]) for f in features]
    out = [f"# {md_text(title)}", "", f"**Date:** {walk_date(stats, features, zone)}  "]
    if track:
        out += [f"**Distance walked:** {fmt_distance(stats['distance_m'])}  ",
                f"**Duration:** {fmt_duration(stats['duration_s'])}  "]
    out += [f"**Problems found:** {counts['problems']} ({counts['severe']} severe) · "
            f"**Good things:** {counts['positives']}", ""]

    by_cat: dict[str, list] = {}
    for p, c in props:
        by_cat.setdefault(p["category"], []).append((p, c))
    if by_cat:
        out += ["| Category | Count |", "|---|---|"]
        out += [f"| {CATEGORIES[k]['label']} | {len(v)} |"
                for k, v in sorted(by_cat.items(), key=lambda kv: (CATEGORIES[kv[0]]["positive"], -len(kv[1])))]
        out.append("")

    problem_cats = [k for k in by_cat if not CATEGORIES[k]["positive"]]
    problem_cats.sort(key=lambda k: (-max(p["severity"] for p, _ in by_cat[k]), -len(by_cat[k]),
                                     CATEGORIES[k]["label"]))
    if not problem_cats:
        out += ["No problems were recorded on this walk.", ""]
    for k in problem_cats:
        out += [f"## {CATEGORIES[k]['label']} ({len(by_cat[k])})", ""]
        items = sorted(by_cat[k], key=lambda pc: (-pc[0]["severity"], pc[0]["time_local"]))
        for n, (p, (lon, lat)) in enumerate(items, 1):
            when = p["time_local"][11:16] or "time unknown"
            note = " *(location approximate)*" if p["needs_review"] else ""
            out.append(f"{n}. **{md_text(p['summary'])}** · {p['severity_label']} ({p['severity']}/3) · "
                       f"{when} · [map]({osm_link(lat, lon)}){note}")
        out.append("")

    positives = sorted((pc for pc in props if pc[0]["positive"]), key=lambda pc: pc[0]["time_local"])
    if positives:
        out += ["## Good things on this street", ""]
        out += [f"- {md_text(p['summary'])} ({p['label'].lower()}, [map]({osm_link(lat, lon)}))"
                for p, (lon, lat) in positives]
        out.append("")
    out += ["---", "", f"*{FOOTER}*", ""]
    return "\n".join(out)


def render_outputs(out: Path, title: str, zone: ZoneInfo) -> None:
    """(Re)build map.html and report.md from OUT/findings.geojson and OUT/track.json."""
    geo_path = out / "findings.geojson"
    if not geo_path.exists():
        raise KerbsideError(f"No {geo_path} found. Run Kerbside on a walk first (without --render-only).")
    geo = read_json(geo_path)
    raw = geo.get("features", []) if isinstance(geo, dict) else []
    features = [f for f in (normalize_feature(r) for r in raw) if f]
    if len(features) < len(raw):
        print(f"  Warning: skipped {len(raw) - len(features)} feature(s) without valid coordinates.")
    track = load_track(out / "track.json")
    if not track:
        print("  Note: no track.json found, the map will show findings only.")
    (out / "map.html").write_text(render_map(features, track, title, zone), encoding="utf-8")
    (out / "report.md").write_text(render_report(features, track, title, zone), encoding="utf-8")
    print(f"  Wrote {out / 'map.html'} and {out / 'report.md'} ({len(features)} findings)")


# ---------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    ap = argparse.ArgumentParser(prog="kerbside", description="Walk your street, talk about what's broken, get a map.")
    ap.add_argument("audio", nargs="?", help="voice narration, e.g. WALK.m4a")
    ap.add_argument("gpx", nargs="?", help="GPS track, e.g. TRACK.gpx")
    ap.add_argument("--out", default="kerbside_out", help="output folder (default: kerbside_out)")
    ap.add_argument("--model", default="gemma3:4b", help="Ollama model (default: gemma3:4b)")
    ap.add_argument("--whisper", default="small", help="faster-whisper model size (default: small)")
    ap.add_argument("--lang", default=None, help="spoken language code, e.g. en (default: auto-detect)")
    ap.add_argument("--translate", action="store_true", help="translate non-English speech to English")
    ap.add_argument("--start", help='recording start, local time, e.g. "2026-10-10 07:30:00"')
    ap.add_argument("--tz", default="Asia/Kolkata", help="timezone for --start and display (default: Asia/Kolkata)")
    ap.add_argument("--offset", type=float, default=0.0, help="shift audio timestamps by SECONDS (default: 0)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached transcript/findings")
    ap.add_argument("--render-only", action="store_true", help="rebuild map.html/report.md from findings.geojson")
    ap.add_argument("--title", default="Kerbside Street Audit", help="title for map and report")
    args = ap.parse_args(argv)
    if not args.render_only and not (args.audio and args.gpx):
        ap.error("give the audio file and the GPX track, e.g. python kerbside.py WALK.m4a TRACK.gpx")
    return args


def run(args: argparse.Namespace) -> None:
    """Run the full pipeline (or just the render step)."""
    zone = get_zone(args.tz)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        print("Re-rendering from your reviewed findings.geojson")
        render_outputs(out, args.title, zone)
        return

    audio, gpx = Path(args.audio), Path(args.gpx)
    for path, what in ((audio, "audio file"), (gpx, "GPX track")):
        if not path.is_file():
            raise KerbsideError(f"Can't find the {what}: {path}")

    print(f"[1/5] Reading GPS track {gpx.name}")
    track = load_gpx(gpx)
    save_track(track, out / "track.json")
    stats = track_stats(track)
    print(f"  {len(track)} points, {fmt_distance(stats['distance_m'])}, {fmt_duration(stats['duration_s'])}, "
          f"starting {track[0][0].astimezone(zone):%Y-%m-%d %H:%M:%S} ({args.tz})")

    extraction_cache = out / "extraction.json"
    try:
        cached_model = json.loads(extraction_cache.read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError, AttributeError):
        cached_model = None
    if args.fresh or cached_model != args.model:
        check_ollama(args.model)  # fail fast, before a long transcription

    created, audio_duration = audio_metadata(audio)
    start, source = resolve_start(args.start, zone, created, track)
    print(f"  Recording start: {start.astimezone(zone):%Y-%m-%d %H:%M:%S} ({args.tz}), from {source}")
    if args.offset:
        print(f"  Applying offset of {args.offset:+g} s")
    shifted = start + timedelta(seconds=args.offset)
    if audio_duration and (warning := time_span_warning(shifted, audio_duration, track)):
        print(f"  Warning: {warning}")

    print(f"[2/5] Transcribing {audio.name}")
    transcript = transcribe(audio, out / "transcript.json", args.whisper, args.lang, args.translate, args.fresh)
    segments = transcript["segments"]
    if not segments:
        raise KerbsideError("No speech found in the recording. Was the microphone covered?")
    if not audio_duration:
        duration = transcript.get("duration") or segments[-1]["end"]
        if warning := time_span_warning(shifted, float(duration), track):
            print(f"  Warning: {warning}")

    print("[3/5] Matching speech to GPS positions")
    aligned = align_segments(segments, track, start, args.offset)
    review = sum(s["needs_review"] for s in aligned)
    print(f"  {len(aligned)} lines placed" + (f", {review} far from any GPS point (marked for review)" if review else ""))

    print(f"[4/5] Finding street problems with {args.model}")
    findings = extract_findings(segments, args.model, extraction_cache, args.fresh)

    print("[5/5] Writing outputs")
    features = build_features(findings, aligned, zone)
    geo_path = out / "findings.geojson"
    if geo_path.exists():
        shutil.copyfile(geo_path, out / "findings.geojson.bak")
        print("  (previous findings.geojson saved as findings.geojson.bak)")
    write_json(geo_path, {"type": "FeatureCollection", "features": features})
    print(f"  Wrote {geo_path}")
    render_outputs(out, args.title, zone)
    print(f"\nDone. Open {out / 'map.html'} in a browser. To fix mistakes, edit {geo_path} "
          "and run: python kerbside.py --render-only" + (f' --out "{args.out}"' if args.out != "kerbside_out" else ""))


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        run(args)
    except KerbsideError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
