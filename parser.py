"""Parse Kuku screenplay text files into structured, timestamped events.

Pure code, no AI: timestamps must be exact, so we never let the LLM produce them.
Works whether the pack is one file per episode or one big file with many episodes.

Tolerant of small format differences a new show may have:
  - labels with or without **bold** / trailing colon:  **ACTION:**  ACTION:  ANNA (Firm):
  - time lines:  Start: (00:01.00) | End: (00:04.80)   Start 00:01.00 End 00:04.80
                 00:01.00 - 00:04.80   [00:01.00 --> 00:04.80]   hh:mm:ss   00:04,80
  - times on the label's own line:  **CAMERA:** (00:00.00 - 00:04.80)
  - episode headers:  EPISODE NUMBER: Episode 3   Episode 3   EPISODE 3:   or none (number from file name)
  - Windows line endings, a byte-order mark, .txt or .md files
Anything it cannot read is reported in WARNINGS instead of being silently skipped.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

WARNINGS = []

T = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
TIME_RE = re.compile(rf"Start\s*:?\s*\(?\s*({T})\s*\)?\s*[|,;/-]?\s*End\s*:?\s*\(?\s*({T})\s*\)?", re.I)
RANGE_RE = re.compile(rf"[(\[]?\s*({T})\s*(?:-->|->|–|—|-|to)\s*({T})\s*[)\]]?", re.I)
EP_PATTERN = r"^[ \t#*]*EPISODE(?:[ \t]+NUMBER)?[ \t]*[:#-]?[ \t]*(?:Episode[ \t]*)?(\d+)"
EP_RE = re.compile(EP_PATTERN, re.I | re.M)
LOGLINE_RE = re.compile(r"^[ \t*]*(?:EPISODE[ \t]+)?(?:LOGLINE|SYNOPSIS|SUMMARY)[ \t*]*:[ \t*]*(.+)", re.I | re.M)
CONTINUITY_RE = re.compile(r"^[ \t*]*CONTINUITY(?:[ \t]+NOTES)?[ \t*]*:[ \t*]*(.+)", re.I | re.M)

# Label (upper-cased) -> event kind. Anything not listed is a speaker, i.e. dialogue.
NON_DIALOGUE = {"CAMERA": "CAM", "CAM": "CAM", "SHOT": "CAM", "TRANSITION": "CAM",
                "ACTION": "ACT", "ACT": "ACT", "DESCRIPTION": "ACT", "SCENE DESCRIPTION": "ACT",
                "STAGE DIRECTION": "ACT", "DIRECTION": "ACT", "VISUAL": "ACT",
                "SOUND FX": "SFX", "SFX": "SFX", "SOUND": "SFX", "SOUND EFFECT": "SFX", "SOUND EFFECTS": "SFX",
                "MUSIC": "MUS", "TEXT": "TXT", "TEXT ON SCREEN": "TXT", "ON-SCREEN TEXT": "TXT",
                "ON SCREEN TEXT": "TXT", "CAPTION": "TXT", "SUPER": "TXT"}


def to_sec(ts: str) -> float:
    """'01:05.60' -> 65.6 ; also handles 'hh:mm:ss.xx' and a comma decimal."""
    parts = [float(p) for p in ts.strip().replace(",", ".").split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return round(sec, 2)


def fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    id: str            # e.g. L023 (unique within episode)
    scene: int
    kind: str          # DLG / CAM / ACT / SFX / ...
    speaker: str       # only for DLG
    emotion: str       # parenthetical, only for DLG
    start: float
    end: float
    text: str


@dataclass
class Episode:
    number: int
    logline: str = ""
    continuity: str = ""
    events: list = field(default_factory=list)
    source: str = ""

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)

    def by_id(self, eid: str):
        return next((e for e in self.events if e.id == eid), None)


def _parse_label(label: str):
    """'MATRIARCH (O.S.) (Firm)' -> ('DLG', 'MATRIARCH', 'Firm')."""
    up = label.strip().upper()
    if up in NON_DIALOGUE:
        return NON_DIALOGUE[up], "", ""
    first = up.split("(")[0].strip()
    for key, kind in NON_DIALOGUE.items():       # e.g. "CAMERA DIRECTION", "ACTION (cont'd)"
        if first == key or first.startswith(key + " "):
            return kind, "", ""
    name = label.split("(")[0].strip()
    parens = re.findall(r"\(([^)]*)\)", label)
    emotion = parens[-1] if parens else ""
    return "DLG", name, emotion


def _find_times(line: str):
    """Return (start, end, match) if the line carries a time range, else None."""
    m = TIME_RE.search(line)
    if not m:
        m = RANGE_RE.search(line)
        # a bare range only counts on a line that is mostly the range, or right after a label
        # ('**ANNA (Firm):** (00:15.50 - 00:17.50)'), not on a time mentioned inside dialogue
        if not m:
            return None
        short = len(line.strip()) - len(m.group(0)) <= 24
        if not short and not _clean_label(line[:m.start()], need_colon=True):
            return None
    return to_sec(m.group(1)), to_sec(m.group(2)), m


def _clean_label(text: str, need_colon: bool):
    """'**LADY ANNA (Firm):**' -> 'LADY ANNA (Firm)'; None if it doesn't look like a label."""
    s = text.strip().strip("*_#> \t").strip()
    had_colon = s.endswith(":")
    s = s.rstrip(":").strip().strip("*_ ").strip()
    if s.upper() in NON_DIALOGUE:                  # 'camera:', 'Action' are labels whatever their case
        return s
    if not s or len(s) > 80 or not s[0].isalpha() or not s[0].isupper():
        return None
    if need_colon and not had_colon and s.split("(")[0].strip().upper() != s.split("(")[0].strip():
        return None                                # a plain sentence, not a label
    return s


def _event_start(lines, i):
    """If an event starts at line i, return (label, start, end, body_start_index, inline_body)."""
    line = lines[i]
    if not line.strip():
        return None
    same = _find_times(line)
    if same:                                       # label and times on one line
        a, b, m = same
        label = _clean_label(line[:m.start()], need_colon=False)
        rest = line[m.end():].strip().lstrip(":-–— ").strip()
        if label:
            return label, a, b, i + 1, rest
        return None
    label = _clean_label(line, need_colon=True)
    if label and i + 1 < len(lines):
        nxt = _find_times(lines[i + 1])
        if nxt and not _clean_label(lines[i + 1][:nxt[2].start()], need_colon=False):
            return label, nxt[0], nxt[1], i + 2, lines[i + 1][nxt[2].end():].strip()
    return None


def parse_episode_text(text: str, number: int = 0) -> Episode:
    num = EP_RE.search(text)
    ep = Episode(number=int(num.group(1)) if num else number)
    m = LOGLINE_RE.search(text)
    ep.logline = m.group(1).strip() if m else ""
    m = CONTINUITY_RE.search(text)
    ep.continuity = m.group(1).strip() if m else ""

    lines = text.splitlines()
    scene, i, n, flipped = 0, 0, 0, 0
    while i < len(lines):
        sm = re.match(r"\s*SCENE\s+(\d+)", lines[i], re.I)
        if sm:
            scene = int(sm.group(1))
        ev = _event_start(lines, i)
        if ev:
            label, start, end, j, inline = ev
            kind, speaker, emotion = _parse_label(label)
            body = [inline] if inline else []
            while j < len(lines) and lines[j].strip() and not _event_start(lines, j):
                body.append(lines[j].strip())
                j += 1
            if end < start:
                start, end, flipped = end, start, flipped + 1
            n += 1
            ep.events.append(Event(
                id=f"L{n:03d}", scene=scene, kind=kind, speaker=speaker, emotion=emotion,
                start=start, end=end, text=" ".join(body).strip().strip('"')))
            i = j
            continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.id))
    stamps = sum(1 for line in lines if TIME_RE.search(line) or RANGE_RE.search(line))
    if ep.events and len(ep.events) < 0.9 * stamps:
        WARNINGS.append(f"episode {ep.number}: {stamps} timestamps in the text but only {len(ep.events)} lines "
                        f"parsed; some lines may be in a format the parser does not know")
    if flipped:
        WARNINGS.append(f"episode {ep.number}: {flipped} lines had end before start; swapped them")
    return ep


def load_episodes(folder: str) -> list:
    """Reads every .txt/.md in a folder; splits files that contain several episodes."""
    WARNINGS.clear()
    episodes = []
    paths = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".txt", ".md"))
    if not paths:
        WARNINGS.append(f"no .txt or .md files found in '{folder}'")
    for path in paths:
        raw = path.read_text(encoding="utf-8-sig", errors="ignore")
        chunks = [c for c in re.split(f"(?={EP_PATTERN})", raw, flags=re.I | re.M) if EP_RE.search(c)]
        if not chunks:                             # no episode header: one episode, number from file name
            digits = re.search(r"\d+", path.stem)
            chunks = [raw]
            fallback = int(digits.group()) if digits else len(episodes) + 1
            WARNINGS.append(f"{path.name}: no 'EPISODE <n>' header, using episode {fallback} from the file name")
        else:
            fallback = 0
        found = 0
        for chunk in chunks:
            ep = parse_episode_text(chunk, number=fallback)
            ep.source = path.name
            if ep.events:
                episodes.append(ep)
                found += 1
                if not any(e.kind == "DLG" for e in ep.events):
                    WARNINGS.append(f"{path.name}: episode {ep.number} has no dialogue lines; check speaker labels")
                if not ep.logline:
                    WARNINGS.append(f"{path.name}: episode {ep.number} has no logline, so the 'story so far' is thinner")
        if not found:
            WARNINGS.append(f"{path.name}: found 0 timed lines, so this file was SKIPPED. Expected a label line "
                            f"(e.g. '**ACTION:**' or 'ANNA:') followed by a time line "
                            f"(e.g. 'Start: (00:01.00) | End: (00:04.80)' or '00:01.00 - 00:04.80')")

    nums = [e.number for e in episodes]
    if len(set(nums)) != len(nums):
        WARNINGS.append("duplicate episode numbers found; renumbered 1..N in file order")
        for k, e in enumerate(episodes, 1):
            e.number = k
    episodes.sort(key=lambda e: e.number)
    return episodes


def print_warnings():
    for w in WARNINGS:
        print(f"  WARNING: {w}")


def timeline_for_prompt(ep: Episode) -> str:
    """Compact, ID-tagged timeline the LLM reads. The LLM answers with IDs only."""
    out = []
    for e in ep.events:
        t = f"[{fmt(e.start)}-{fmt(e.end)}]"
        if e.kind == "DLG":
            emo = f" ({e.emotion})" if e.emotion else ""
            out.append(f'{e.id} {t} {e.speaker}{emo}: "{e.text}"')
        else:
            out.append(f"{e.id} {t} {e.kind}: {e.text}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    eps = load_episodes(sys.argv[1] if len(sys.argv) > 1 else "input")
    for ep in eps:
        dlg = sum(e.kind == "DLG" for e in ep.events)
        print(f"Episode {ep.number}: {len(ep.events)} events ({dlg} dialogue), {fmt(ep.duration)} long  [{ep.source}]")
    print_warnings()
    print(f"\n{len(eps)} episodes parsed, {len(WARNINGS)} warnings")
    if eps:
        print("\n--- first 12 timeline lines of first episode ---")
        print("\n".join(timeline_for_prompt(eps[0]).splitlines()[:12]))
