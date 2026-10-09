# Kerbside

**Walk your street, talk about what's broken, get a map. Everything runs on your own laptop.**

**[Live demo map](https://thebhavye.github.io/kerbside/demo/map.html)** · [demo report](demo/report.md)
(a synthetic 600 m walk, narrated by a text-to-speech voice and processed by real Whisper and Gemma)

You walk your street with your iPhone in your pocket, recording a voice memo and a GPS track. Kerbside
transcribes what you said with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), uses
[Gemma](https://ai.google.dev/gemma) running in [Ollama](https://ollama.com) to pick out the street
problems you mentioned ("footpath broken here", "drain blocked", "streetlight not working") and the good
things ("huge rain tree, nice shade"), places each one on your walked route, and gives you:

- `map.html`: a phone-friendly map of everything you found, with a place search box
- `report.md`: a one-page report ready for your ward office or municipal complaint portal
- `findings.geojson`: the data, which you can review and edit

Built for the Hacktoberfest "Touch Grass" challenge. Open models only, CPU only, no cloud.

## Setup on Windows (PowerShell)

Tested on a Windows 11 laptop with an Intel i5, 16 GB RAM and no GPU.

1. Install **Python 3.11 or newer** from [python.org](https://www.python.org/downloads/windows/)
   (tick "Add python.exe to PATH" in the installer).
2. Install **Ollama for Windows** from [ollama.com/download](https://ollama.com/download) and start it.
3. Download the Gemma model (about 3.3 GB):

   ```powershell
   ollama pull gemma3:4b
   ```

4. Set up Kerbside in its folder:

   ```powershell
   cd kerbside
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   If PowerShell refuses to run `Activate.ps1`, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

The first run downloads the Whisper `small` model (~500 MB), so it needs internet once. After that
everything works offline. (The map's background tiles come from OpenStreetMap, so viewing the map
still needs internet.)

## Recording a walk on your iPhone

1. Open **Open GPX Tracker** (free, open source, on the App Store) and start tracking.
2. Open **Voice Memos** and start recording.
3. Lock the phone and put it in your pocket. The microphone picks you up fine.
4. Walk and talk. Pause briefly between observations and name the problem clearly:
   "footpath broken here, big crack", "drain blocked", "streetlight not working",
   "cars parked on the footpath", "huge rain tree, nice shade". Say it *where* you see it.
5. At the end, stop both recordings.
6. Get the files to your laptop: in Open GPX Tracker, share the `.gpx` file; in Voice Memos, share the
   memo (`.m4a`). Email them to yourself, upload to Google Drive, or use iCloud for Windows.

## Running it

```powershell
python kerbside.py "New Recording.m4a" "walk.gpx"
```

Then open `kerbside_out\map.html`.

**How long it takes** (measured on an Intel i5 laptop, CPU only): transcribing an 8.5-minute walk took
about 30 seconds, and Gemma needed 1 to 2 minutes per 12 sentences. A 15-minute walk where you talk a
lot (about 100 sentences) takes roughly 15 to 20 minutes, mostly Gemma. The very first run also spends
about 5 minutes downloading Whisper.

**Memory:** Gemma needs about 4 GB of free RAM and Whisper about 1 GB. On a 16 GB laptop, close
browsers, Discord, Slack and similar apps before a run, or Windows will slow to a crawl swapping to disk.

Options:

| Option | What it does |
|---|---|
| `--out DIR` | output folder (default `kerbside_out`) |
| `--model NAME` | Ollama model (default `gemma3:4b`) |
| `--whisper SIZE` | faster-whisper model: `tiny`, `base`, `small` (default), `medium`... |
| `--lang CODE` | spoken language, e.g. `en` (default: auto-detect; set `en` if Whisper guesses wrong) |
| `--translate` | turn non-English speech into English text (untested beyond English) |
| `--start DATETIME` | when the voice recording started, e.g. `"2026-10-10 07:30:00"` (local time) |
| `--tz ZONE` | timezone for `--start` and for display (default `Asia/Kolkata`) |
| `--offset SECONDS` | shift the audio timeline to fix alignment |
| `--fresh` | redo transcription and extraction instead of using the cache |
| `--render-only` | rebuild `map.html` and `report.md` from your edited `findings.geojson` |
| `--title TEXT` | title for the map and report |

The transcript (`transcript.json`) and Gemma's findings (`extraction.json`) are cached in the output
folder, so re-running with a different `--offset` or `--title` is quick.

## Reviewing and fixing findings

AI makes mistakes, so review before you send anything. In testing, Gemma picked the right category very reliably but sometimes rated severity one level off (for example an open drain as moderate instead of severe), so check severities in particular. Open `kerbside_out\findings.geojson` in a text
editor (or drag it into [geojson.io](https://geojson.io), fix things on the map, and save it back):

- delete a feature that is wrong,
- change `summary`, `category` or `severity` (1 minor, 2 moderate, 3 severe),
- drag a point to where the problem really is.

Then rebuild the map and report without any AI calls:

```powershell
python kerbside.py --render-only
```

Labels and colours follow the `category` and `severity` you set. Points with a dashed outline (and
"location approximate" in the report) were spoken when the GPS had no fix nearby; check those first.

Note: a full re-run (without `--render-only`) regenerates `findings.geojson`. The previous file is kept
as `findings.geojson.bak`.

## Fixing alignment

Kerbside matches each sentence to the GPS track by time. It takes the recording's start time from, in
order: `--start`, the time stored inside the `.m4a`, or the first GPS point. It prints which one it used
and warns if the audio and the track don't overlap.

- If the points are clearly off, give the start time you see in Voice Memos:
  `--start "2026-10-10 07:30:00"` (and `--tz` if you're not in India).
- If points are consistently a bit early or late along the route, nudge them with `--offset`:
  `--offset 20` moves everything 20 seconds later along the walk, `--offset -20` earlier.

## Sharing

- **Public map:** import `findings.geojson` into [uMap](https://umap.openstreetmap.fr) (open-source map
  tool built on OpenStreetMap) to publish a shareable map for your neighbourhood.
- **Complaints:** paste or attach `report.md` to your ward office email or municipal complaint portal.
  Each item has an OpenStreetMap link to its exact spot.
- **OpenStreetMap:** please do **not** bulk-upload findings to OpenStreetMap. OSM forbids unreviewed
  automated edits. If you've confirmed something that belongs on the map (a missing footpath, a
  crossing), add it by hand in the OSM editor.

## Privacy

Your audio and GPS track never leave your laptop. Whisper and Gemma run locally; the only network
traffic is the one-time model downloads, the map tiles when you view the map, and any place names you
type into the map's search box (sent to OpenStreetMap's Nominatim search service).

## Try it without recording anything

`sample_data/` has a synthetic ~600 m walk (`sample_track.gpx`, a point every 5 seconds) and a matching
`transcript.json`. The test runs the whole pipeline on it with Gemma mocked out, so you don't need
Whisper or Ollama:

```powershell
python tests\test_pipeline.py      # writes a demo map to kerbside_sample_out\map.html
python -m pytest tests             # same checks, if you have pytest
```

## How it works

1. **Start time:** `--start`, else the `.m4a`'s `creation_time` metadata (read with PyAV), else the
   first GPX timestamp.
2. **Speech to text:** faster-whisper on CPU (`int8`), with voice-activity detection to skip silence.
3. **Positioning:** each sentence's time is interpolated along the GPX track. Sentences more than 30 s
   from a GPS point, or outside the track, are flagged for review.
4. **Extraction:** the transcript goes to Gemma 12 lines at a time. Gemma labels *every* line as a
   problem category, a good thing, "same as previous line" (extra detail about the last observation)
   or "not an observation" (chatter, directions). Ollama structured outputs (a JSON schema) mean it can
   only answer with valid labels and severities, and Kerbside joins continuation lines into one finding.
   Labelling each line, instead of asking for a free-form list of findings, stops a small model from
   skipping lines or lumping unrelated lines together.
5. **Outputs:** GeoJSON, a self-contained Leaflet map and a Markdown report.

## Checking extraction quality

`tests/eval_gemma.py` runs the sample transcript and a held-out set of differently worded lines through
your real Ollama model and prints each finding next to what a person would expect, plus timing:

```powershell
python tests\eval_gemma.py --runs 2
```

Use it after changing the prompt or trying another model (`--model gemma3:12b`). Results can vary a
little between runs even at temperature 0, so look at more than one run.
