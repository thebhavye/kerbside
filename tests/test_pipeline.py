"""End-to-end test of Kerbside on the synthetic sample, without Whisper or Ollama.

Run with:  python -m pytest tests   (or)   python tests/test_pipeline.py
The plain-python form writes its outputs to kerbside_sample_out/ so you can open the map.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kerbside  # noqa: E402

SAMPLE = ROOT / "sample_data"
GPX = SAMPLE / "sample_track.gpx"
TRANSCRIPT = SAMPLE / "transcript.json"
ZONE = ZoneInfo("Asia/Kolkata")

# What the "model" says for each transcript line id: (label, severity, summary).
# Lines not listed (1, 5, 8, 14) are chatter and get "not_an_observation".
FAKE_LABELS = {
    2: ("broken_footpath", 3, "Broken footpath, large crack and missing slab"),
    3: ("same_as_previous", 3, ""),
    4: ("blocked_drain", 2, "Drain blocked with plastic"),
    6: ("shade_tree", 2, "Large rain tree giving good shade"),
    7: ("obstruction", 2, "Cars parked on footpath force walking on road"),
    9: ("streetlight_out", 2, "Streetlight on pole 42 not working"),
    10: ("garbage", 1, "Garbage dumped at corner </script><b>x</b>"),
    11: ("waterlogging", 2, "Waterlogging two days after rain"),
    12: ("good_footpath", 1, "Wide, good footpath"),
    13: ("unsafe_crossing", 3, "No zebra crossing at junction, fast traffic"),
    15: ("blocked_drain", 3, "Open drain without cover next to school gate"),
}
EXPECTED_FIRST_LINES = {2, 4, 6, 7, 9, 10, 11, 12, 13, 15}
TEXT_TO_ID = {s["text"]: s["id"] for s in json.loads(TRANSCRIPT.read_text(encoding="utf-8"))["segments"]}


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def fake_get(url: str, timeout: float = 0) -> FakeResponse:
    assert url.endswith("/api/tags")
    return FakeResponse({"models": [{"name": "gemma3:4b", "model": "gemma3:4b"}]})


UNLOADS: list[str] = []


def fake_post(url: str, json: dict, timeout: float = 0) -> FakeResponse:  # noqa: A002 - mirrors requests
    if url.endswith("/api/generate"):
        assert json["keep_alive"] == 0
        UNLOADS.append(json["model"])
        return FakeResponse({"done": True, "done_reason": "unload"})
    assert url.endswith("/api/chat")
    assert json["stream"] is False and json["options"]["temperature"] == 0
    assert "lines" in json["format"]["properties"]
    numbered = re.findall(r"^(\d+): (.*)$", json["messages"][1]["content"], re.M)
    assert 0 < len(numbered) <= kerbside.CHUNK_LINES
    assert [int(n) for n, _ in numbered] == list(range(1, len(numbered) + 1)), "lines numbered 1..n per chunk"
    labels = []
    for n, text in numbered:
        label, sev, summary = FAKE_LABELS.get(TEXT_TO_ID[text], ("not_an_observation", 1, ""))
        labels.append({"id": int(n), "label": label, "summary": summary, "severity": sev})
    labels.append({"id": 99, "label": "garbage", "summary": "Hallucinated line", "severity": 1})
    return FakeResponse({"message": {"content": __import__("json").dumps({"lines": labels})}})


def run_sample(out: Path) -> None:
    """Run the full CLI on the sample with a cached transcript and mocked Ollama."""
    out.mkdir(parents=True, exist_ok=True)
    UNLOADS.clear()
    shutil.copyfile(TRANSCRIPT, out / "transcript.json")
    audio = out / "sample_walk.m4a"
    audio.write_bytes(b"")  # placeholder: the cached transcript is used, so Whisper never runs
    with mock.patch.object(kerbside.requests, "get", fake_get), \
            mock.patch.object(kerbside.requests, "post", fake_post):
        code = kerbside.main([str(audio), str(GPX), "--out", str(out), "--start", "2026-10-10 07:30:00",
                              "--title", "Kerbside Sample Walk"])
    assert code == 0
    assert UNLOADS == ["gemma3:4b"], "Gemma should be unloaded once extraction is done"


def check_alignment() -> None:
    track = kerbside.load_gpx(GPX)
    segments = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))["segments"]
    start = kerbside.parse_start("2026-10-10 07:30:00", ZONE)
    assert start == track[0][0]
    aligned = kerbside.align_segments(segments, track, start)
    # Line 2 starts at 20 s: exactly on a GPS point, 25 m east of the start.
    l2 = aligned[1]
    assert l2["gps_gap_s"] == 0 and not l2["needs_review"]
    assert abs(kerbside.haversine(track[0][1], track[0][2], l2["lat"], l2["lon"]) - 25) < 0.5
    # Line 15 starts at 500 s, after the track ends at 480 s.
    assert aligned[-1]["needs_review"] and aligned[-1]["gps_gap_s"] == 20
    assert sum(s["needs_review"] for s in aligned) == 1
    # Interpolation halfway between two points (2.5 s gap).
    mid = kerbside.align_segments([{"id": 1, "start": 2.5, "end": 3, "text": "x"}], track, start)[0]
    assert abs(mid["lat"] - track[0][1]) < 1e-9 and track[0][2] < mid["lon"] < track[1][2]
    assert mid["gps_gap_s"] == 2.5
    # A big offset pushes everything outside the track.
    shifted = kerbside.align_segments(segments, track, start, offset=3600)
    assert all(s["needs_review"] for s in shifted)


def check_outputs(out: Path) -> None:
    for name in ("transcript.json", "track.json", "extraction.json", "findings.geojson", "map.html", "report.md"):
        assert (out / name).is_file(), f"missing {name}"

    geo = json.loads((out / "findings.geojson").read_text(encoding="utf-8"))
    feats = geo["features"]
    assert len(feats) == len(EXPECTED_FIRST_LINES), "labels for unknown lines should be dropped"
    props = {f["properties"]["line_ids"][0]: f["properties"] for f in feats}
    assert set(props) == EXPECTED_FIRST_LINES
    assert props[2]["line_ids"] == [2, 3], "same_as_previous merges into the line before"
    assert props[2]["quote"].startswith("Footpath is broken here") and "trip" in props[2]["quote"]
    assert props[2]["severity_label"] == "severe" and props[2]["label"] == "Broken footpath"
    assert props[6]["positive"] is True and props[6]["severity"] == 1, "positives are always severity 1"
    assert props[2]["time_local"] == "2026-10-10 07:30:20"
    assert props[15]["needs_review"] is True
    assert [p["needs_review"] for k, p in props.items() if k != 15] == [False] * (len(props) - 1)
    for f in feats:
        lon, lat = f["geometry"]["coordinates"]
        assert 12.99 < lat < 13.01 and 80.26 < lon < 80.27

    page = (out / "map.html").read_text(encoding="utf-8")
    assert "leaflet@1.9.4" in page and "tile.openstreetmap.org" in page and "OpenStreetMap</a> contributors" in page
    assert page.count("</script>") == 2, "embedded data must not close the script tag"
    assert "Kerbside Sample Walk" in page and "599 m walked" in page and "8 min" in page
    assert "Large rain tree giving good shade" in page
    assert "nominatim.openstreetmap.org/search" in page and 'role="search"' in page
    assert 'addEventListener("input"' not in page, "no search-as-you-type (Nominatim policy)"

    report = (out / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Kerbside Sample Walk")
    assert "**Distance walked:** 599 m" in report and "**Duration:** 8 min" in report
    assert "## Good things on this street" in report and "Large rain tree giving good shade" in report
    assert report.rstrip().endswith(f"*{kerbside.FOOTER}*")
    assert "https://www.openstreetmap.org/?mlat=13.000" in report and "#map=19/13.000" in report
    sections = re.findall(r"^## (.+)$", report, re.M)
    assert sections[0] == "Blocked or open drain (2)", "most severe category first, ties by count"
    assert sections.index("Garbage (1)") > sections.index("Waterlogging (1)")
    assert "(location approximate)" in report
    assert "</script>" not in report and "&lt;/script&gt;" in report


def check_cache_and_render_only(out: Path) -> None:
    # Second run: transcript and findings come from cache, so Ollama must not be called.
    def boom(*a, **k):
        raise AssertionError("Ollama should not be called when the cache is valid")
    with mock.patch.object(kerbside.requests, "get", boom), mock.patch.object(kerbside.requests, "post", boom):
        assert kerbside.main([str(out / "sample_walk.m4a"), str(GPX), "--out", str(out),
                              "--start", "2026-10-10 07:30:00"]) == 0
    assert (out / "findings.geojson.bak").is_file()

    # Human review: delete the garbage finding and downgrade the crossing, then re-render.
    geo_path = out / "findings.geojson"
    geo = json.loads(geo_path.read_text(encoding="utf-8"))
    geo["features"] = [f for f in geo["features"] if f["properties"]["category"] != "garbage"]
    for f in geo["features"]:
        if f["properties"]["category"] == "unsafe_crossing":
            f["properties"]["severity"] = 2
    geo_path.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf-8")
    with mock.patch.object(kerbside.requests, "get", boom), mock.patch.object(kerbside.requests, "post", boom):
        assert kerbside.main(["--render-only", "--out", str(out)]) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Garbage" not in report
    assert "No zebra crossing at junction, fast traffic** · moderate (2/3)" in report
    assert "Garbage dumped" not in (out / "map.html").read_text(encoding="utf-8")


def check_friendly_errors(out: Path) -> None:
    assert kerbside.main(["missing.m4a", str(GPX), "--out", str(out)]) == 1
    empty = out / "empty.gpx"
    empty.write_text('<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"></gpx>',
                     encoding="utf-8")
    try:
        kerbside.load_gpx(empty)
        raise AssertionError("empty GPX should fail")
    except kerbside.KerbsideError as exc:
        assert "no GPS points" in str(exc)

    def down(*a, **k):
        raise kerbside.requests.ConnectionError("refused")
    with mock.patch.object(kerbside.requests, "get", down):
        try:
            kerbside.check_ollama("gemma3:4b")
            raise AssertionError("should fail")
        except kerbside.KerbsideError as exc:
            assert "Can't reach Ollama" in str(exc)
    with mock.patch.object(kerbside.requests, "get", fake_get):
        try:
            kerbside.check_ollama("gemma3:12b")
            raise AssertionError("should fail")
        except kerbside.KerbsideError as exc:
            assert "ollama pull gemma3:12b" in str(exc)


def test_lines_to_findings() -> None:
    chunk = [{"id": i, "text": f"line {i}"} for i in range(21, 27)]
    labels = [
        {"id": 1, "label": "same_as_previous", "summary": "", "severity": 3},   # nothing before: dropped
        {"id": 2, "label": "garbage", "summary": "Trash", "severity": 1},
        {"id": 3, "label": "same_as_previous", "summary": "", "severity": 3},   # merges, raises severity
        {"id": 4, "label": "not_an_observation", "summary": "", "severity": 1},
        {"id": 5, "label": "same_as_previous", "summary": "", "severity": 2},   # after chatter: dropped
        {"id": 6, "label": "shade_tree", "summary": "Tree", "severity": 3},     # positive: forced to 1
        {"id": 6, "label": "garbage", "summary": "Duplicate", "severity": 1},   # duplicate id: ignored
        {"id": 7, "label": "garbage", "summary": "Out of chunk", "severity": 1},
        {"id": 2, "label": "made_up_label", "summary": "x", "severity": 1},
    ]
    found = kerbside.lines_to_findings(labels, chunk)
    assert found == [
        {"line_ids": [22, 23], "category": "garbage", "severity": 3, "summary": "Trash"},
        {"line_ids": [26], "category": "shade_tree", "severity": 1, "summary": "Tree"},
    ]


def test_alignment() -> None:
    check_alignment()


def test_full_pipeline(tmp_path: Path) -> None:
    run_sample(tmp_path)
    check_outputs(tmp_path)
    check_cache_and_render_only(tmp_path)


def test_friendly_errors(tmp_path: Path) -> None:
    check_friendly_errors(tmp_path)


if __name__ == "__main__":
    check_alignment()
    with tempfile.TemporaryDirectory() as tmp:
        run_sample(Path(tmp))
        check_outputs(Path(tmp))
        check_cache_and_render_only(Path(tmp))
        check_friendly_errors(Path(tmp))
    demo = ROOT / "kerbside_sample_out"
    shutil.rmtree(demo, ignore_errors=True)
    run_sample(demo)
    print(f"\nAll checks passed. Sample outputs: {demo / 'map.html'} and {demo / 'report.md'}")
