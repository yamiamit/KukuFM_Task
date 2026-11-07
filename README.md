# Kuku Promo Finder

Reads microdrama screenplays and finds and ranks 20–60 s promo clips. Each promo has a hook in the first 3 s, ends on a cliffhanger, and makes sense on its own.

The screenplays are confidential, so they are not in this repo. The full results from running on all 50 episodes are in the raw outputs folder linked in the solution doc.

## Run

```bash
python3 -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt                      # only dependency: requests
mkdir -p input    # put the episode .txt files here (one file per episode, or one big file: both work)

# 0) new show? check it parses first: every episode listed, no WARNING lines
python parser.py input
python check_formats.py          # optional: parser survives 9 other screenplay formats

# 1) offline smoke test, no API key needed
LLM_PROVIDER=mock python promo_finder.py

# 2) real run on a few episodes first
export LLM_PROVIDER=gemini GEMINI_API_KEY=your_key      # free key: aistudio.google.com
python promo_finder.py --episodes 1-3 --judge

# 3) full pack (per-episode + cross-episode promos)
python promo_finder.py --judge

# 4) quality check: is the judge trustworthy? does the tool agree with a human editor?
python evaluate.py --top 8              # compares against my_picks.csv (my own picks for episodes 1-3)
```

The submitted results were produced with `LLM_PROVIDER=proxy` through a local OpenAI-compatible proxy.

To swap models, change env vars only: `LLM_PROVIDER=groq GROQ_API_KEY=...` or `LLM_MODEL=<name>`.
Any OpenAI-compatible endpoint works with `LLM_PROVIDER=proxy LLM_BASE_URL=<url>/v1 LLM_API_KEY=... LLM_MODEL=<name>`.
LLM responses are cached in `.llm_cache/`, so reruns cost nothing.

## Outputs (`outputs/`)
- `promos.csv`: ranked promos (episode, segment timestamps, hook, cliffhanger, why)
- `promos.json`: the same, plus scores, clip transcripts, and rejected candidates with the reason
- `raw_llm_outputs.jsonl`: unedited model responses (the brief requires raw outputs)
- `run_stats.json`: calls, tokens (reported by the API and estimated) and run time, for cost at scale
- `judge_check.csv`: from `evaluate.py`, judge scores for the real picks vs deliberately worse clips

## How it works

| Step | Done by | Why |
|---|---|---|
| Parse screenplay into timestamped events with IDs | Code | Timestamps must be exact; LLMs make them up |
| Build a "story so far" from the screenplay loglines | Code | Free context: who is who, what matters in the series |
| Propose promos per episode (with the story so far) | LLM | Judging drama needs language understanding |
| Join the best moments of different episodes into cross-episode promos | LLM (1 call) | The brief allows multi-episode promos; one call over the top 12 moments keeps it cheap |
| Map IDs to timestamps; enforce 20–60 s, hook ≤ 3 s, no mid-line cut | Code | Hard rules shouldn't depend on the model |
| "Cold viewer" judge sees only the clip, never the story so far | LLM (2nd call) | Tests whether the clip stands alone without the self-grading bias |
| Judge sanity test: score late-hook and random clips too | Code + LLM | If bad clips score as high as picks, the ranking can't be trusted |
| Rank: weighted score + max 2 per episode in the top 15 | Code | Transparent, easy to tune |

The input screenplays are confidential: `input/` is git-ignored. Don't commit them.
