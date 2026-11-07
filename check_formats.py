"""Robustness check: rewrite episode 1 of the input in 9 other plausible screenplay formats
(Windows line endings, no bold, bare time ranges, hh:mm:ss, times on the label line, no header, .md, ...)
plus edge cases (one big file, an unreadable file, duplicate numbers) and check the parser reads the same events.

Usage: python check_formats.py [input_folder]     (needs episodes 1-3 in the folder)
"""
from pathlib import Path
import re, sys, shutil, tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parser as P

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "input")
OUT = Path(tempfile.mkdtemp(prefix="fmt_check_"))
ep1 = sorted(SRC.glob("episode-001*"))[0].read_text()
ep2 = sorted(SRC.glob("episode-002*"))[0].read_text()
ep3 = sorted(SRC.glob("episode-003*"))[0].read_text()
TIME = re.compile(r"Start: \((\S+)\) \| End: \((\S+)\)")
LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$", re.M)

def hms(t):  # 01:05.60 -> 00:01:05.60
    return "00:" + t

variants = {
    "windows_bom":        "﻿" + ep1.replace("\n", "\r\n"),
    "no_bold":            LABEL.sub(r"\1:", ep1),
    "bare_range":         TIME.sub(r"\1 - \2", ep1),
    "arrow_brackets":     TIME.sub(r"[\1 --> \2]", ep1),
    "hh_mm_ss_comma":     TIME.sub(lambda m: f"Start: ({hms(m[1]).replace('.', ',')}) | End: ({hms(m[2]).replace('.', ',')})", ep1),
    "times_on_label_line": re.sub(r"^\*\*(.+?):\*\*\s*\n" + TIME.pattern, r"**\1:** (\2 - \3)", ep1, flags=re.M),
    "short_header":       ep1.replace("EPISODE NUMBER: Episode 1", "Episode 1").replace("EPISODE LOGLINE:", "Logline:"),
    "no_header":          re.sub(r"^EPISODE NUMBER:.*\n", "", ep1, flags=re.M),
    "markdown_file":      ep1.replace("EPISODE NUMBER: Episode 1", "# EPISODE 1"),
}
names = {"no_header": "ep1.txt", "markdown_file": "episode1.md"}

ref = P.load_episodes(str(SRC))[0]
want = [(e.kind, e.speaker, e.start, e.end, e.text) for e in ref.events]
ok_all = True
print(f"Reference: episode 1 from {SRC} ({len(want)} events)\n")
for name, text in variants.items():
    d = OUT / name; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    (d / names.get(name, "episode.txt")).write_text(text, newline="")
    eps = P.load_episodes(str(d))
    got = [(e.kind, e.speaker, e.start, e.end, e.text) for e in eps[0].events] if eps else []
    ok = got == want and eps[0].number == 1
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:20} events {len(got):3}/{len(want)}  ep#{eps[0].number if eps else '-'}  warnings: {P.WARNINGS}")

# one big file holding three episodes
d = OUT / "one_big_file"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
(d / "whole_show.txt").write_text(ep1 + "\n\n" + ep2 + "\n\n" + ep3)
eps = P.load_episodes(str(d))
print(f"{'PASS' if [e.number for e in eps] == [1, 2, 3] else 'FAIL'}  one_big_file         episodes {[e.number for e in eps]}  events {[len(e.events) for e in eps]}")

# a file the parser cannot read must be reported, not silently skipped
d = OUT / "unreadable"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
(d / "notes.txt").write_text("Anna walks in. Mark leaves.\nThe end.")
eps = P.load_episodes(str(d))
print(f"{'PASS' if not eps and any('SKIPPED' in w for w in P.WARNINGS) else 'FAIL'}  unreadable_file      warnings: {P.WARNINGS[-1][:90]}...")

# duplicate episode numbers get renumbered with a warning
d = OUT / "duplicates"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
(d / "a.txt").write_text(ep1); (d / "b.txt").write_text(ep2.replace("Episode 2", "Episode 1"))
eps = P.load_episodes(str(d))
print(f"{'PASS' if [e.number for e in eps] == [1, 2] and P.WARNINGS else 'FAIL'}  duplicate_numbers    episodes {[e.number for e in eps]}  warnings: {P.WARNINGS}")

shutil.rmtree(OUT, ignore_errors=True)
