"""Quality check: can we trust the scores, and does the tool beat chance?

Run AFTER promo_finder.py. Two checks:

1. Judge sanity test (needs an LLM). For the top single-segment promos, build two deliberately
   worse versions and ask the same cold-viewer judge to score them:
     - slow_open: same clip but starting ~10 s earlier, so the hook arrives late
     - random:    a random window of the same length from the same episode
   If the judge scores these about as high as the real picks, its scores (and our ranking)
   can't be trusted. Results: <out>/judge_check.csv

2. Agreement with a human editor (optional, no LLM). Write your own picks for a few episodes in
   my_picks.csv (columns: pick,episode,start,end  e.g. ep3_slap,3,00:41.00,01:15.50), then rerun.
   Rows with the same pick id are segments of one multi-segment promo (pick may be left empty).
   Reports how many of your picks the tool also found (recall), and how many of the tool's
   picks in those episodes you agree with (precision).

Usage:
  python evaluate.py --top 8                 # judge sanity test on the top 8 promos
  python evaluate.py --picks my_picks.csv    # also compare against your own picks
"""
import argparse, csv, json, random
from pathlib import Path
from statistics import mean
from parser import load_episodes, fmt, to_sec
from llm import call_llm, config_error
from promo_finder import JUDGE_SYSTEM, JUDGE_USER, MAX_LEN, clip_transcript, snap_end

METRICS = ("hook", "cliffhanger", "standalone")


def clip(ep_num, a, b):
    return {"segments": [{"episode": ep_num, "start_sec": a, "end_sec": b}], "duration_sec": round(b - a, 1)}


def judge(eps, c):
    return call_llm(JUDGE_SYSTEM, JUDGE_USER.format(dur=c["duration_sec"], clip=clip_transcript(eps, c)),
                    temperature=0)


def controls(ep, a, b, rng):
    """Worse versions of the clip [a, b]."""
    out, dur = {}, b - a
    earlier = [e.start for e in ep.events if e.start <= a - 8]
    if earlier and b - max(earlier) <= MAX_LEN:
        out["slow_open"] = (max(earlier), b)
    starts = [e.start for e in ep.events if e.start + dur <= ep.duration and abs(e.start - a) > dur / 2]
    if starts:
        r = rng.choice(starts)
        out["random"] = (r, snap_end(ep, r + dur))
    return out


def judge_check(eps, ranked, top, out):
    rng = random.Random(42)
    picks = [p for p in ranked if p["type"] == "single" and len(p["segments"]) == 1][:top]
    rows = []
    for p in picks:
        ep, s = eps[p["episode"]], p["segments"][0]
        variants = {"picked": (s["start_sec"], s["end_sec"])}
        variants.update(controls(ep, s["start_sec"], s["end_sec"], rng))
        for name, (a, b) in variants.items():
            j = p["judge_scores"] if name == "picked" and p.get("judge_scores") else judge(eps, clip(ep.number, a, b))
            rows.append({"rank": p["rank"], "episode": ep.number, "variant": name, "start": fmt(a), "end": fmt(b),
                         **{m: j.get(m) for m in METRICS}, "would_watch_next": j.get("would_watch_next"),
                         "confusing": j.get("confusing", "")})
    if not rows:
        print("No single-segment promos to test. Run promo_finder.py first.")
        return
    with open(out / "judge_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\nJudge sanity test on the top {len(picks)} promos (mean score, 1-10):")
    print(f"  {'variant':<10} " + " ".join(f"{m:>12}" for m in METRICS) + "   n")
    for v in ("picked", "slow_open", "random"):
        vr = [r for r in rows if r["variant"] == v]
        if vr:
            vals = [mean(float(r[m] or 0) for r in vr) for m in METRICS]
            print(f"  {v:<10} " + " ".join(f"{x:>12.1f}" for x in vals) + f"   {len(vr)}")
    for v, m in (("slow_open", "hook"), ("random", "cliffhanger")):
        pairs = [(r, c) for r in rows if r["variant"] == "picked"
                 for c in rows if c["variant"] == v and c["rank"] == r["rank"]]
        if pairs:
            wins = sum(float(r[m] or 0) > float(c[m] or 0) for r, c in pairs)
            print(f"  picked beat {v} on {m}: {wins}/{len(pairs)}")
    print(f"  details: {out / 'judge_check.csv'}")


def overlap(a1, b1, a2, b2):
    """Overlap as a share of the shorter clip."""
    inter = max(0.0, min(b1, b2) - max(a1, a2))
    return inter / max(0.01, min(b1 - a1, b2 - a2))


def compare_picks(ranked, picks_file, top_n=15):
    groups = {}                                # rows sharing a "pick" id are segments of one promo
    with open(picks_file) as f:
        for i, r in enumerate(csv.DictReader(f)):
            st, en = r["start"].strip(), r["end"].strip()
            seg = (int(r["episode"]), to_sec(st) if ":" in st else float(st), to_sec(en) if ":" in en else float(en))
            groups.setdefault((r.get("pick") or "").strip() or f"row{i}", []).append(seg)
    human = [seg for segs in groups.values() for seg in segs]
    labelled = {h[0] for h in human}
    tool = [(s["episode"], s["start_sec"], s["end_sec"], p["rank"]) for p in ranked for s in p["segments"]
            if s["episode"] in labelled]

    def match(e, a, b, pool):
        return any(e == pe and overlap(a, b, pa, pb) >= 0.5 for pe, pa, pb, *_ in pool)

    found = sum(any(match(*h, tool) for h in segs) for segs in groups.values())
    agreed = [t for t in tool if match(t[0], t[1], t[2], human)]
    top = [t for t in tool if t[3] <= top_n]
    print(f"\nAgreement with your picks ({len(groups)} picks, episodes {sorted(labelled)}):")
    print(f"  recall:    tool found {found}/{len(groups)} of your picks")
    print(f"  precision: you picked {len(agreed)}/{len(tool)} of the tool's clips in those episodes")
    if top:
        print(f"  top {top_n}:    {sum(match(t[0], t[1], t[2], human) for t in top)}/{len(top)} of its top clips match yours")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--picks", default="my_picks.csv")
    args = ap.parse_args()

    out = Path(args.out)
    if not (out / "promos.json").exists():
        raise SystemExit(f"No {out / 'promos.json'} yet: run promo_finder.py first (or pass --out <its folder>).")
    if args.top and config_error():
        raise SystemExit(config_error())
    ranked = json.loads((out / "promos.json").read_text())["ranked"]
    eps = {e.number: e for e in load_episodes(args.input)}
    if args.top:
        judge_check(eps, ranked, args.top, out)
    if Path(args.picks).exists():
        compare_picks(ranked, args.picks)
    else:
        print(f"\n(no {args.picks}: add your own picks to measure agreement with a human editor)")


if __name__ == "__main__":
    main()
