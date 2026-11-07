"""Offline stand-in for the LLM so the pipeline can be tested without an API key.
It is deliberately dumb (picks the first line with '?' or '!' as hook), so real runs
must use a real model. Lets you check parsing, validation and ranking end to end."""
import re

TS = r"\[(\d+):([\d.]+)-(\d+):([\d.]+)\]"


def _rows(pattern, user):
    return [(i, int(a) * 60 + float(b), int(c) * 60 + float(d), t)
            for i, a, b, c, d, t in re.findall(pattern, user, re.M)]


def mock_response(user: str) -> dict:
    if "TRANSCRIPT OF THE CLIP" in user:          # judge prompt
        return {"hook": 6, "cliffhanger": 6, "standalone": 6, "would_watch_next": True,
                "confusing": "", "summary": "mock judge"}

    if "CROSS-EPISODE" in user:                   # cross prompt: first ~12 s of the first two moments
        rows = _rows(rf"^(E\d+:L\d+) {TS} (.+)$", user)
        by_ep = {}
        for r in rows:
            by_ep.setdefault(r[0].split(":")[0], []).append(r)
        segs = []
        for ep_rows in list(by_ep.values())[:2]:
            first = ep_rows[0]
            last = [r for r in ep_rows if r[2] <= first[1] + 12][-1] if ep_rows else first
            segs.append({"start_id": first[0], "end_id": last[0]})
        if len(segs) < 2:
            return {"promos": []}
        return {"promos": [{"segments": segs, "hook_id": segs[0]["start_id"], "hook": "mock cross hook",
                            "cliffhanger": "mock cross cliffhanger", "why": "mock cross pick",
                            "scores": {"hook": 5, "cliffhanger": 5, "standalone": 5}}]}

    rows = _rows(rf"^(L\d{{3}}) {TS} (.+)$", user)
    if not rows:
        return {"promos": []}
    hook = next((r for r in rows if '"' in r[3] and ("!" in r[3] or "?" in r[3])), rows[0])
    end = [r for r in rows if r[2] <= hook[1] + 40][-1]
    return {"promos": [{
        "segments": [{"start_id": hook[0], "end_id": end[0]}], "hook_id": hook[0],
        "hook": hook[3][:80], "cliffhanger": end[3][:80], "why": "mock pick",
        "scores": {"hook": 5, "cliffhanger": 5, "standalone": 5}}]}
