# Kuku Promo Finder

Reads the screenplays of a microdrama series and returns a ranked list of promo clips: 20 to 60 seconds long, with a hook in the first 3 seconds, ending on a cliffhanger, and understandable by someone who has never seen the show. Each promo comes with exact start and end timestamps for every segment, the hook, the cliffhanger and one line on why it works.

> The screenplays are Kuku's confidential content, so they are **not** in this repo. The results of running on all 50 episodes are in the raw outputs folder linked in the solution doc.

---

## Quick start

If you only want to see it run, this takes about 2 minutes and needs no API key (step 5 uses a built-in fake model):

```bash
git clone https://github.com/yamiamit/KukuFM_Task.git && cd KukuFM_Task
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir input                                   # copy the episode .txt files into input/
python parser.py input                        # check: every episode listed, no WARNING lines
LLM_PROVIDER=mock python promo_finder.py --out outputs_mock
```

For a real run, follow the steps below.

---

## Step by step

### 1. Requirements

- Python 3.10 or newer (`python3 --version`)
- One dependency, `requests`, installed in step 2
- For real results, an API key for a language model (step 6). A free Google Gemini key is enough.

### 2. Set up

```bash
git clone https://github.com/yamiamit/KukuFM_Task.git
cd KukuFM_Task
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the screenplays

Create an `input/` folder and copy the episode files into it:

```bash
mkdir input
cp /path/to/episodes/*.txt input/
```

Either one file per episode or one big file holding all episodes works. Each episode should look roughly like this:

```text
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: One-line summary of the episode.

**CAMERA:**
Start: (00:00.00) | End: (00:04.80)
Wide shot as the church doors open.

**ANNA (Shocked):**
Start: (00:11.30) | End: (00:13.10)
"Where is everyone?"
```

Small differences are fine: labels without `**`, times written as `00:01.00 - 00:04.80` or `hh:mm:ss`, times on the label's own line, no episode header (the number is then taken from the file name), Windows line endings, `.md` files.

### 4. Check that the screenplays parse

```bash
python parser.py input
```

You should see one line per episode and `0 warnings`:

```text
Episode 1: 95 events (37 dialogue), 02:18.00 long  [episode-001.txt]
...
50 episodes parsed, 0 warnings
```

Any `WARNING` line names the file and the problem (for example, a file in a format it could not read). Fix those before running. Optional: `python check_formats.py` rewrites episode 1 in 9 other screenplay formats and checks the parser reads them all the same.

### 5. Test the pipeline offline (no API key)

```bash
LLM_PROVIDER=mock python promo_finder.py --judge --out outputs_mock
```

This uses a fake model, so the promos are meaningless, but it proves parsing, validation, ranking and file output all work. It finishes in about a second and writes to `outputs_mock/`.

### 6. Choose a model

The model is set only through environment variables; no code changes. Pick one:

| Provider | Commands | Get a key |
|---|---|---|
| Google Gemini (free tier, default) | `export LLM_PROVIDER=gemini GEMINI_API_KEY=your_key` | [aistudio.google.com](https://aistudio.google.com) → Get API key |
| Groq (free tier) | `export LLM_PROVIDER=groq GROQ_API_KEY=your_key` | [console.groq.com](https://console.groq.com) |
| OpenAI | `export LLM_PROVIDER=openai OPENAI_API_KEY=your_key` | platform.openai.com |
| OpenRouter | `export LLM_PROVIDER=openrouter OPENROUTER_API_KEY=your_key` | openrouter.ai |
| Local Ollama (no key) | `export LLM_PROVIDER=ollama` | [ollama.com](https://ollama.com) |
| Any other OpenAI-compatible API | `export LLM_PROVIDER=proxy LLM_BASE_URL=https://host/v1 LLM_API_KEY=your_key LLM_MODEL=model-name` | from that service |

To use a different model from the same provider, also set `LLM_MODEL=<model name>`. On Windows PowerShell, use `$env:LLM_PROVIDER="gemini"` and `$env:GEMINI_API_KEY="your_key"` instead of `export`.

The submitted results were produced with `LLM_PROVIDER=proxy` through a local OpenAI-compatible proxy.

### 7. Run

First try 3 episodes (about 2 minutes) and read the results:

```bash
python promo_finder.py --episodes 1-3 --judge
```

Then the whole show (about 20 to 30 minutes for 50 episodes on a free tier):

```bash
python promo_finder.py --judge
```

It prints progress per episode and finishes with the number of valid and rejected promos. Open `outputs/promos.csv`: rows are ranked, and ranks 1 to 15 are the ones to post first. If a run stops halfway, just run the same command again: finished model calls are cached in `.llm_cache/`, so nothing is paid for twice.

### 8. Check the quality

```bash
python evaluate.py --top 8
```

This runs two checks and prints a table for each:

- **Judge sanity test:** for the top 8 promos, the same judge also scores a version that starts 10 s earlier (late hook) and a random clip of the same length. If those score as well as the real picks, the scores cannot be trusted.
- **Agreement with a human editor:** compares the tool's promos with the hand-picked ones in `my_picks.csv` (episodes 1 to 3). To use your own picks, edit that file: one row per segment, columns `pick,episode,start,end`; rows sharing a `pick` id form one multi-segment promo.

---

## Options

`promo_finder.py`

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `input` | Folder with the screenplay files |
| `--out` | `outputs` | Folder for the results |
| `--episodes` | all | A range such as `1-5` |
| `--per-episode` | `3` | Most promos to ask for per episode |
| `--cross` | `5` | Cross-episode promos to ask for (`0` turns them off) |
| `--judge` | off | Run the cold-viewer judge on every promo (recommended) |

`evaluate.py`

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `outputs` | Folder holding the `promos.json` to check |
| `--top` | `8` | How many top promos to use in the judge sanity test (`0` skips it) |
| `--picks` | `my_picks.csv` | Human picks to compare against |

Environment variables: `LLM_PROVIDER`, `LLM_MODEL`, the provider's key (table above), `LLM_BASE_URL` (proxy only), `LLM_MIN_GAP_SEC` (pause between calls, default 4 s to stay under free-tier limits; lower it on a paid key), `LLM_CACHE` (cache folder, default `.llm_cache`).

---

## Outputs

Everything is written to `outputs/` (or the `--out` folder):

| File | What it holds |
|---|---|
| `promos.csv` | The ranked promos: rank, score, type (`single` or `cross`), episodes, segment timestamps, duration, hook, cliffhanger, why, notes |
| `promos.json` | The same plus the model's own scores, the judge's scores and comments, the clip transcript, and the rejected candidates with the reason |
| `raw_llm_outputs.jsonl` | Every model response, unedited |
| `run_stats.json` | Calls, tokens (as reported by the API, and estimated) and run time, for working out cost |
| `judge_check.csv` | Written by `evaluate.py`: judge scores for the real picks against the worse versions |

The `notes` column records anything the code changed or noticed, for example `FIXED: hook was at +4.3s, trimmed start` or `hook_id missing: assumed the first line is the hook`.

---

## Troubleshooting

| Message | Fix |
|---|---|
| `needs an API key: set GEMINI_API_KEY` | Run the `export` line for your provider (step 6) in the same terminal |
| `LLM_PROVIDER='…' is not known` | Check the spelling; the message lists the valid names |
| `No episodes could be read` or a `WARNING` from the parser | Run `python parser.py input`; each warning names the file and what format it expected |
| `404` or "model not found" | The default model name has changed: set `LLM_MODEL` to a current model from your provider |
| `429` or rate-limit errors | Free-tier limit: wait a minute and rerun (cached calls are skipped), or raise `LLM_MIN_GAP_SEC` |
| `No outputs/promos.json yet` (from `evaluate.py`) | Run `promo_finder.py` first, or pass `--out` with the folder it wrote |

---

## How it works

| Step | Done by | Why |
|---|---|---|
| Parse each screenplay into timestamped lines with IDs | Code | Timestamps must be exact; models make them up |
| Build a "story so far" from the episode loglines | Code | Free context: who is who and what matters in the series |
| Propose up to 3 promos per episode, by line ID | Model | Judging drama needs language understanding |
| Join the best moments of different episodes into cross-episode promos | Model (1 call) | The brief allows multi-episode promos; one call over the top 12 moments keeps it cheap |
| Map IDs to timestamps; enforce 20–60 s, hook within 3 s, no mid-line cut | Code | Hard rules shouldn't depend on the model |
| A "cold viewer" judge scores each clip, seeing only the clip | Model (2nd call) | Tests whether a stranger would follow it, without the model grading its own work |
| Rank by weighted score, at most 2 per episode in the top 15 | Code | Transparent and easy to tune |

## Files

| File | Purpose |
|---|---|
| `promo_finder.py` | The main pipeline (steps 5 and 7) |
| `parser.py` | Reads screenplays into timestamped lines; `python parser.py <folder>` checks a folder |
| `llm.py` | One adapter for every model provider, with caching and usage counts |
| `evaluate.py` | Quality checks (step 8) |
| `check_formats.py` | Parser robustness test on 9 other screenplay formats |
| `mock_llm.py` | Fake model for offline testing |
| `my_picks.csv` | Hand-picked promos for episodes 1 to 3 |
