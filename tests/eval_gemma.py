"""Check extraction quality against REAL Gemma in Ollama (not run by pytest).

    python tests/eval_gemma.py [--model gemma3:4b] [--runs 2]

Runs two transcripts through the model and compares with what a person would expect:
  sample   - sample_data/transcript.json (the walk used by the tests)
  heldout  - different wording and edge cases, never used to tune the prompt's examples
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kerbside  # noqa: E402

# Expected findings: first line id -> (all line ids, accepted categories, severity)
SAMPLE_EXPECTED = {
    2: ([2, 3], {"broken_footpath"}, 3),
    4: ([4], {"blocked_drain"}, 2),
    6: ([6], {"shade_tree"}, 1),
    7: ([7], {"obstruction"}, 2),
    9: ([9], {"streetlight_out"}, 2),
    10: ([10], {"garbage"}, 1),
    11: ([11], {"waterlogging"}, 2),
    12: ([12], {"good_footpath"}, 1),
    13: ([13], {"unsafe_crossing"}, 3),
    15: ([15], {"blocked_drain"}, 3),
}

HELDOUT_LINES = [
    "Right, so I'm near the temple now.",
    "There's no footpath at all on this stretch, just mud by the road.",
    "Vegetable vendor has kept all his crates on the footpath.",
    "Let me just check the time, hmm.",
    "Manhole lid is missing, it's just a hole in the footpath.",
    "Very dangerous, a kid could fall in.",
    "Turning right onto the main road.",
    "Traffic signal for pedestrians is not working, everyone is just running across.",
    "Lots of plastic covers and food waste thrown by the wall.",
    "A cow is sitting in the middle of the footpath.",
    "Nice neem trees here, the whole stretch is shaded.",
    "Okay, that's it, going back home.",
]
HELDOUT_EXPECTED = {
    2: ([2], {"no_footpath"}, 2),
    3: ([3], {"obstruction"}, 2),
    5: ([5, 6], {"broken_footpath", "blocked_drain"}, 3),
    8: ([8], {"unsafe_crossing"}, 3),
    9: ([9], {"garbage"}, 1),
    10: ([10], {"obstruction", "other_problem"}, 1),
    11: ([11], {"shade_tree"}, 1),
}

# Observations split across lines, as Whisper often does. The detail must reach the summary.
DETAIL_LINES = [
    "Okay, walking towards the market.",
    "Streetlight is off here.",
    "Pole number seventeen.",
    "Big pothole in the footpath.",
    "About two feet wide.",
    "Drain overflowing near the tea shop.",
    "It's been like this since last month.",
    "Now I'm near the bus stop.",
    "Nice wide footpath.",
]
DETAIL_EXPECTED = {
    2: ([2, 3], {"streetlight_out"}, 2),
    4: ([4, 5], {"broken_footpath"}, 3),
    6: ([6, 7], {"blocked_drain", "waterlogging"}, 2),
    9: ([9], {"good_footpath"}, 1),
}
DETAIL_IN_SUMMARY = {2: ("17", "seventeen"), 4: ("two", "2"), 6: ("month",)}

# Place words that would only be right if the speaker said them.
PLACE_WORDS = ("shop", "bakery", "corner", "school", "gate", "temple", "wall", "junction", "pole", "bus stop")


def evaluate(name: str, segments: list[dict], expected: dict, model: str,
             details: dict | None = None) -> dict:
    """Run extraction once and print a line-by-line comparison."""
    text = {s["id"]: s["text"] for s in segments}
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        found = kerbside.extract_findings(segments, model, Path(tmp) / "extraction.json", fresh=True)
        elapsed = time.perf_counter() - t0
    got = {f["line_ids"][0]: f for f in found}
    print(f"\n== {name}: {elapsed:.1f} s for {len(segments)} lines ==")
    print(f"{'line':>4}  {'expected':<26} {'gemma':<18} {'sev':<5} ok  summary")
    score = {"cat": 0, "sev": 0, "ids": 0, "missed": 0, "extra": 0, "invented": 0, "seconds": elapsed}
    for lid in sorted(set(expected) | set(got)):
        e, g = expected.get(lid), got.get(lid)
        cat_ok = bool(e and g and g["category"] in e[1])
        score["cat"] += cat_ok
        score["sev"] += bool(cat_ok and g["severity"] == e[2])
        score["ids"] += bool(cat_ok and g["line_ids"] == e[0])
        score["missed"] += bool(e and not g)
        score["extra"] += bool(g and not e)
        quote = " ".join(text[i] for i in (g or {}).get("line_ids", [lid])).lower()
        invented = [w for w in PLACE_WORDS if g and w in g["summary"].lower() and w not in quote]
        score["invented"] += bool(invented)
        lost = bool(details and lid in details and g and
                    not any(w in g["summary"].lower() for w in details[lid]))
        score["lost_detail"] = score.get("lost_detail", 0) + lost
        flags = (f" ids={g['line_ids']}" if g and e and g["line_ids"] != e[0] else "") + \
                (f"  INVENTED: {', '.join(invented)}" if invented else "") + \
                ("  LOST DETAIL" if lost else "")
        sev = f"{e[2] if e else '-'}/{g['severity'] if g else '-'}"
        print(f"{lid:>4}  {'/'.join(sorted(e[1])) if e else '(not an observation)':<26} "
              f"{g['category'] if g else 'MISSED':<18} {sev:<5} "
              f"{'Y' if cat_ok else 'N'}   {g['summary'] if g else text[lid]}{flags}")
    n = len(expected)
    print(f"categories {score['cat']}/{n} · severities {score['sev']}/{n} · line ids {score['ids']}/{n} · "
          f"missed {score['missed']} · extra {score['extra']} · invented places {score['invented']}" +
          (f" · lost details {score.get('lost_detail', 0)}/{len(details)}" if details else ""))
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--runs", type=int, default=1, help="repeat to check run-to-run stability")
    args = ap.parse_args()
    try:
        kerbside.check_ollama(args.model)
    except kerbside.KerbsideError as exc:
        sys.exit(str(exc))
    sample = json.loads((ROOT / "sample_data" / "transcript.json").read_text(encoding="utf-8"))["segments"]
    heldout = [{"id": i, "start": i * 10.0, "end": i * 10.0 + 4, "text": t} for i, t in enumerate(HELDOUT_LINES, 1)]
    detail = [{"id": i, "start": i * 10.0, "end": i * 10.0 + 4, "text": t} for i, t in enumerate(DETAIL_LINES, 1)]
    for run in range(1, args.runs + 1):
        print(f"\n######## run {run}/{args.runs}")
        evaluate("sample", sample, SAMPLE_EXPECTED, args.model)
        evaluate("heldout", heldout, HELDOUT_EXPECTED, args.model)
        evaluate("details", detail, DETAIL_EXPECTED, args.model, DETAIL_IN_SUMMARY)


if __name__ == "__main__":
    main()
