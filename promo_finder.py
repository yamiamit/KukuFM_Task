"""Find and rank promo clips across a microdrama's episodes.

Pipeline (what's AI vs code):
  1. parse      (code)  screenplay -> timestamped events with IDs
  2. propose    (LLM)   per episode, suggest promos by referring to line IDs only.
                        The prompt includes a free "story so far" built from the
                        loglines, so the model knows who is who and what matters.
  3. cross      (LLM)   one extra call joins the strongest moments from DIFFERENT
                        episodes into cross-episode promos (the brief allows this)
  4. validate   (code)  map IDs -> exact timestamps, enforce 20-60s, hook in first 3s,
                        no mid-line cuts; reject or auto-fix and record why
  5. judge      (LLM)   a fresh "cold viewer" call that sees ONLY the clip text, so
                        standalone-ness is tested honestly (optional: --judge).
                        The judge never gets the story so far.
  6. rank       (code)  weighted score + diversity across episodes

Usage:
  LLM_PROVIDER=gemini GEMINI_API_KEY=... python promo_finder.py --input input --judge
"""
import argparse, csv, json, re, time
from pathlib import Path
from parser import load_episodes, timeline_for_prompt, fmt, print_warnings
from llm import call_llm, model_name, STATS

MIN_LEN, MAX_LEN, HOOK_WINDOW = 20.0, 60.0, 3.0

PROPOSE_SYSTEM = """You are a senior promo editor for vertical microdrama apps like ReelShort and DramaBox.
You pick 20-60 second clips from a show that are posted as ads on TikTok/Instagram/YouTube to make
strangers stop scrolling and install the app. You answer with valid JSON only."""

PROPOSE_USER = """SHOW: episode {num} of a vertical microdrama.

STORY SO FAR (earlier episodes, for YOUR understanding only; the person seeing the promo has NOT
seen them, so the clip itself must make clear who is wronged and what is at stake):
{story}

THIS EPISODE: {logline}
CONTINUITY: {continuity}

TIMELINE (each line: ID [start-end mm:ss] TYPE or SPEAKER (emotion): text)
CAM = camera, ACT = action, SFX = sound.
{timeline}

TASK: propose up to {k} promo clips from THIS episode, best first.

What makes a good promo:
- HOOK in the first 3 seconds: a slap, a reveal, a shocking or humiliating line, a threat. The clip must
  START at (or within 1-2 s before) the hook. No slow build-up, no establishing shots first.
- CLIFFHANGER at the end: cut on an unresolved moment (a question unanswered, a threat, a reveal just
  before the reaction, a character's resolve right before they act). Never end on a resolution.
- STANDALONE: someone who has never seen the show must grasp who is wronged and what is at stake.
- Length 20-60 seconds total. Prefer one continuous segment. You may join up to 3 segments from this
  episode, in time order, if skipping a slow middle makes it stronger.
- Different promos should use different moments, not overlapping copies of the same one.

STRICT RULES:
- Refer to moments ONLY by the IDs above. Never invent IDs or timestamps.
- end_id must be a line where the moment is complete (don't cut a sentence).
- EVERY promo must include hook_id: the ID of the line/action that is the hook.

Return JSON:
{{"promos": [{{"segments": [{{"start_id": "L010", "end_id": "L045"}}],
  "hook_id": "L010", "hook": "what happens in the first 3 seconds",
  "cliffhanger": "the unresolved moment it ends on", "why": "one line on why it works",
  "scores": {{"hook": 1-10, "cliffhanger": 1-10, "standalone": 1-10}}}}]}}
If nothing in this episode works as a promo, return {{"promos": []}}."""

CROSS_USER = """SHOW: a vertical microdrama. One line per episode:
{story}

STRONGEST MOMENTS already found inside single episodes. IDs are written E<episode>:<line>,
timestamps are positions inside that episode.
{moments}

TASK: build up to {k} CROSS-EPISODE promos. Each one joins 2-3 segments from DIFFERENT episodes above
into a single 20-60 second clip (total of all segments).
A strong shape: open on a moment of shock or injustice, then jump to a later reversal or reveal, and
cut on the unresolved instant right before the payoff.

RULES:
- Segments play in the order you list them. The FIRST segment must open on the hook (first 3 seconds).
- Trim each segment to its strongest 8-25 seconds so the total stays within 20-60 seconds.
- Use ONLY the IDs listed above. Never invent IDs or timestamps.
- A stranger who has never seen the show must still follow it.

Return JSON:
{{"promos": [{{"segments": [{{"start_id": "E01:L010", "end_id": "E01:L020"}},
                            {{"start_id": "E07:L030", "end_id": "E07:L041"}}],
  "hook_id": "E01:L010", "hook": "...", "cliffhanger": "...", "why": "one line on why it works",
  "scores": {{"hook": 1-10, "cliffhanger": 1-10, "standalone": 1-10}}}}]}}"""

JUDGE_SYSTEM = """You are a 25-year-old in the US scrolling TikTok. You have NEVER seen this show.
Be a harsh, honest critic. Answer with valid JSON only."""

JUDGE_USER = """Below is the TRANSCRIPT OF THE CLIP you just saw ({dur:.0f} seconds), exactly as it plays,
including what the camera shows. You know nothing else about the show.
Speaker names and character descriptions are screenplay labels: on screen you would only see faces and
hear voices, never the labels. Judge the story the scene tells, not whether the labels are consistent.

{clip}

Score 1-10 (10 = best):
- hook: would the first 3 seconds stop you scrolling?
- cliffhanger: does the ending make you NEED to know what happens next?
- standalone: did you understand who is who and what is at stake, with no prior context?
Return JSON: {{"hook": n, "cliffhanger": n, "standalone": n, "would_watch_next": true/false,
"confusing": "anything you didn't understand, or empty", "summary": "what you think the story is, 1 line"}}"""


# ---------------------------------------------------------------- context (code, free)
def story_so_far(eps, before=None):
    """One line per episode from the screenplay loglines. No LLM call needed."""
    lines = [f"Ep {e.number}: {e.logline}" for e in eps if before is None or e.number < before]
    return "\n".join(lines) or "(this is the first episode)"


# ---------------------------------------------------------------- validation (code)
def snap_end(ep, t):
    """Extend the cut so no dialogue line is chopped mid-sentence."""
    for e in ep.events:
        if e.kind == "DLG" and e.start < t < e.end:
            t = e.end
    return t


def parse_id(raw, default_ep):
    """'L010' -> (default_ep, 'L010'); 'E07:L010' -> (7, 'L010')."""
    m = re.match(r"\s*E?0*(\d+)\s*:\s*(L\d+)", str(raw))
    return (int(m.group(1)), m.group(2)) if m else (default_ep, str(raw).strip())


def build_promo(eps_by_num, cand, default_ep=None):
    """Turn an LLM proposal (IDs) into exact timestamps; return (promo, problems).
    Segments may come from one episode or, for cross-episode promos, from several."""
    problems, segs = [], []
    for s in cand.get("segments", [])[:3]:
        na, ia = parse_id(s.get("start_id", ""), s.get("episode", default_ep))
        nb, ib = parse_id(s.get("end_id", ""), na)
        ep = eps_by_num.get(na)
        a, b = (ep.by_id(ia), ep.by_id(ib)) if ep and na == nb else (None, None)
        if not a or not b:
            problems.append(f"unknown id in {s}")
            continue
        start, end = a.start, snap_end(ep, max(b.end, a.end))
        if end <= start:
            problems.append(f"segment ends before it starts: {s}")
            continue
        segs.append([na, start, end])
    if not segs:
        return None, problems or ["no valid segments"]

    if len({s[0] for s in segs}) == 1:
        segs.sort()                            # one episode: keep time order
    merged = [segs[0]]
    for s in segs[1:]:                         # merge overlapping segments of the same episode
        m = merged[-1]
        if s[0] == m[0] and m[1] <= s[1] <= m[2]:
            m[2] = max(m[2], s[2])
        else:
            merged.append(s)

    status = "ok"
    first_ep = eps_by_num[merged[0][0]]
    if not cand.get("hook_id"):               # model left it out: assume the clip opens on its hook
        problems.append("hook_id missing: assumed the first line is the hook")
        cand = dict(cand, hook_id=next((e.id for e in first_ep.events if e.start >= merged[0][1] - 0.05), ""))
    hn, hid = parse_id(cand.get("hook_id", ""), merged[0][0])
    hook = first_ep.by_id(hid) if hn == merged[0][0] else None
    if hook is None:
        problems.append("hook_id not found in the first segment's episode")
    elif hook.start >= merged[0][2] or hook.end <= merged[0][1]:
        status, _ = "rejected", problems.append("hook is not inside the first segment")
    elif hook.start - merged[0][1] > HOOK_WINDOW:   # auto-fix: trim the slow opening
        delay = hook.start - merged[0][1]
        new_start = max(merged[0][1], hook.start - 1.0)
        problems.append(f"FIXED: hook was at +{delay:.1f}s, trimmed start to {fmt(new_start)}")
        merged[0][1] = new_start

    dur = sum(b - a for _, a, b in merged)
    if status == "ok" and dur < MIN_LEN:
        status, _ = "rejected", problems.append(f"too short: {dur:.1f}s")
    elif status == "ok" and dur > MAX_LEN:
        status, _ = "rejected", problems.append(f"too long: {dur:.1f}s")

    episodes = sorted({n for n, _, _ in merged})
    promo = {
        "type": "cross" if len(episodes) > 1 else "single",
        "episode": merged[0][0], "episodes": episodes,
        "segments": [{"episode": n, "start": fmt(a), "end": fmt(b),
                      "start_sec": a, "end_sec": b} for n, a, b in merged],
        "duration_sec": round(dur, 1),
        "hook": cand.get("hook", ""), "cliffhanger": cand.get("cliffhanger", ""),
        "why": cand.get("why", ""), "self_scores": cand.get("scores", {}),
        "status": status, "notes": problems,
    }
    return promo, problems


def clip_transcript(eps_by_num, promo, id_prefix=False):
    """Exactly what a viewer would see/hear in the clip, in order, re-timed from 0.
    With id_prefix=True, lines keep their E<ep>:<ID> and original times (for the cross prompt)."""
    out, offset = [], 0.0
    for i, s in enumerate(promo["segments"]):
        ep, a, b = eps_by_num[s["episode"]], s["start_sec"], s["end_sec"]
        if i:
            out.append("--- CUT ---")
        for e in ep.events:
            if e.start >= a - 0.05 and e.start < b:
                if id_prefix:
                    tag = f"E{ep.number:02d}:{e.id} [{fmt(e.start)}-{fmt(e.end)}]"
                else:
                    tag = f"[{fmt(offset + e.start - a)}]"
                out.append(f'{tag} {e.speaker}: "{e.text}"' if e.kind == "DLG" else f"{tag} ({e.kind}) {e.text}")
        offset += b - a
    return "\n".join(out)


# ---------------------------------------------------------------- ranking (code)
def score(p):
    s = p.get("judge_scores") or p["self_scores"]       # prefer the unbiased cold-viewer score
    try:
        base = 0.35 * float(s.get("hook", 0)) + 0.40 * float(s.get("cliffhanger", 0)) \
             + 0.25 * float(s.get("standalone", 0))
    except (TypeError, ValueError):
        base = 0.0
    if p.get("judge_scores", {}).get("would_watch_next") is False:
        base -= 1.0
    if p["type"] == "single" and p["episode"] <= 5:      # early eps: no spoilers, easy to start watching
        base += 0.3
    return round(base, 2)


def rank(promos, per_episode_cap=2, cross_cap=3, top_n=15):
    """Greedy: highest score first, but at most `cap` per episode (or cross-episode) in the top list."""
    pool = sorted(promos, key=lambda p: -p["score"])
    top, rest, used = [], [], {}
    for p in pool:
        key = "cross" if p["type"] == "cross" else p["episode"]
        cap = cross_cap if key == "cross" else per_episode_cap
        if len(top) < top_n and used.get(key, 0) < cap:
            top.append(p)
            used[key] = used.get(key, 0) + 1
        else:
            rest.append(p)
    ranked = top + rest
    for i, p in enumerate(ranked, 1):
        p["rank"] = i
    return ranked


def finish(eps_by_num, promo, args, raw_log, stage):
    """Attach transcript, run the cold-viewer judge, score."""
    promo["clip_transcript"] = clip_transcript(eps_by_num, promo)
    if args.judge and promo["status"] == "ok":
        try:
            j = call_llm(JUDGE_SYSTEM, JUDGE_USER.format(dur=promo["duration_sec"],
                                                         clip=promo["clip_transcript"]), temperature=0)
            promo["judge_scores"] = j
            raw_log.write(json.dumps({"episode": promo["episodes"], "stage": f"judge_{stage}",
                                      "response": j}) + "\n")
        except Exception as e:
            promo["notes"].append(f"judge failed: {e}")
    promo["score"] = score(promo)
    return promo


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--cross", type=int, default=5, help="cross-episode promos to request (0 = off)")
    ap.add_argument("--judge", action="store_true", help="run the cold-viewer judge pass")
    ap.add_argument("--episodes", default="", help="e.g. 1-5 to run a subset")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    all_eps = load_episodes(args.input)
    print_warnings()
    if not all_eps:
        raise SystemExit(f"No episodes could be read from '{args.input}'. Check the warnings above, "
                         f"or run: python parser.py {args.input}")
    eps = all_eps
    if args.episodes:
        lo, hi = map(int, args.episodes.split("-"))
        eps = [e for e in all_eps if lo <= e.number <= hi]
    eps_by_num = {e.number: e for e in all_eps}
    print(f"Model: {model_name()} | episodes: {len(eps)}")

    raw_log = open(out / "raw_llm_outputs.jsonl", "w")
    all_promos, t0 = [], time.time()
    for ep in eps:
        prompt = PROPOSE_USER.format(num=ep.number, story=story_so_far(all_eps, before=ep.number),
                                     logline=ep.logline, continuity=ep.continuity,
                                     timeline=timeline_for_prompt(ep), k=args.per_episode)
        try:
            resp = call_llm(PROPOSE_SYSTEM, prompt)
        except Exception as e:
            print(f"  ep {ep.number}: LLM error {e}"); continue
        raw_log.write(json.dumps({"episode": ep.number, "stage": "propose", "response": resp}) + "\n")

        for cand in resp.get("promos", []):
            promo, _ = build_promo(eps_by_num, cand, default_ep=ep.number)
            if promo:
                all_promos.append(finish(eps_by_num, promo, args, raw_log, "single"))
        ok = sum(p["status"] == "ok" and p["episode"] == ep.number for p in all_promos)
        print(f"  ep {ep.number}: {len(resp.get('promos', []))} proposed, {ok} valid")

    # ---- cross-episode: join the strongest single-episode moments (best one per episode, top 12)
    if args.cross:
        best, seen = [], set()
        for p in sorted((p for p in all_promos if p["status"] == "ok"), key=lambda p: -p["score"]):
            if p["episode"] not in seen:
                best.append(p); seen.add(p["episode"])
        best = sorted(best[:12], key=lambda p: p["episode"])
        if len(best) >= 2:
            moments = "\n\n".join(
                f"# Episode {p['episode']} | hook: {p['hook']} | cliffhanger: {p['cliffhanger']}\n"
                + clip_transcript(eps_by_num, p, id_prefix=True) for p in best)
            try:
                resp = call_llm(PROPOSE_SYSTEM, CROSS_USER.format(
                    story=story_so_far(eps), moments=moments, k=args.cross))
                raw_log.write(json.dumps({"episode": "cross", "stage": "propose_cross", "response": resp}) + "\n")
                n_ok = 0
                for cand in resp.get("promos", []):
                    promo, _ = build_promo(eps_by_num, cand)
                    if promo:
                        all_promos.append(finish(eps_by_num, promo, args, raw_log, "cross"))
                        n_ok += promo["status"] == "ok"
                print(f"  cross-episode: {len(resp.get('promos', []))} proposed, {n_ok} valid")
            except Exception as e:
                print(f"  cross-episode: LLM error {e}")
    raw_log.close()

    valid = rank([p for p in all_promos if p["status"] == "ok"])
    rejected = [p for p in all_promos if p["status"] != "ok"]
    secs = time.time() - t0

    (out / "promos.json").write_text(json.dumps({"model": model_name(), "ranked": valid,
                                                 "rejected": rejected}, indent=2))
    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "type", "episodes", "segments", "duration_sec",
                    "hook", "cliffhanger", "why", "notes"])
        for p in valid:
            segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
            w.writerow([p["rank"], p["score"], p["type"], "+".join(map(str, p["episodes"])), segs,
                        p["duration_sec"], p["hook"], p["cliffhanger"], p["why"], "; ".join(p["notes"])])

    stats = dict(STATS, model=model_name(), episodes=len(eps), seconds=round(secs, 1),
                 valid_promos=len(valid), rejected_promos=len(rejected))
    if eps:
        stats["est_tokens_per_episode"] = round((STATS["est_prompt_tokens"] + STATS["est_completion_tokens"]) / len(eps))
    (out / "run_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n{len(valid)} valid promos, {len(rejected)} rejected, {secs:.0f}s. "
          f"See {out}/promos.csv, promos.json and run_stats.json")


if __name__ == "__main__":
    main()
