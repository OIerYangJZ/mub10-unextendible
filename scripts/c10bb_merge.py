#!/usr/bin/env python
"""Merge-verify split runs of scripts/c10bb6.cpp (completeness of the biunimodular vectors of length 10).
Each run log = the program's stdout and stderr (plus the job script's provenance lines) concatenated.
Requirements for a log to count:
  * 'VERSION c10bb6', the catalogue line 'CATALOGUE: 3040 of 3040 roots exactly unimodular ; enclosures pairwise
    disjoint: yes ; OK', 'modulation closure: OK', 'dilation closure: OK';
  * provenance lines 'SRC_SHA256 <hex>' and 'ROOTS_SHA256 <hex>' written by the job script; all logs must agree on both.
A subproblem s in 0..255 is CERTIFIED if some valid log has a line 'SUBDONE s ...: nodes N fails 0'.  Any SUBDONE
line with fails > 0 is reported and does not count.  Duplicated subproblems across logs are allowed.
Usage: python c10bb_merge.py LOG1 LOG2 ...    (each LOG may be 'a.out+a.err' to concatenate files)
"""
import re, sys

def read(spec):
    return "\n".join(open(f, errors="replace").read() for f in spec.split("+"))

def main():
    logs = sys.argv[1:]
    cert = {}; problems = []; src = set(); roots = set(); valid = 0
    for spec in logs:
        txt = read(spec)
        need = ["VERSION c10bb6", "CATALOGUE: 3040 of 3040 roots exactly unimodular ; enclosures pairwise disjoint: yes ; OK",
                "modulation closure: OK", "dilation closure: OK"]
        miss = [n for n in need if n not in txt]
        m1 = re.search(r"SRC_SHA256 ([0-9a-f]{64})", txt); m2 = re.search(r"ROOTS_SHA256 ([0-9a-f]{64})", txt)
        if miss or not m1 or not m2:
            problems.append("%s: missing %s%s" % (spec, miss, "" if (m1 and m2) else " provenance")); continue
        src.add(m1.group(1)); roots.add(m2.group(1)); valid += 1
        for m in re.finditer(r"^SUBDONE\s+(\d+) eps=\S+ ch=\S+: nodes (\d+) fails (\d+)", txt, re.M):
            sb, nodes, fails = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not 0 <= sb <= 255: problems.append("%s: bad id %d" % (spec, sb)); continue
            if fails == 0: cert[sb] = cert.get(sb, 0) + 1
            else: problems.append("%s: subproblem %d has %d failures" % (spec, sb, fails))
    if len(src) > 1 or len(roots) > 1:
        problems.append("provenance mismatch: %d source hashes, %d roots hashes" % (len(src), len(roots)))
    missing = [s for s in range(256) if s not in cert]
    print("valid logs %d / %d ; source %s ; roots %s" % (valid, len(logs), ",".join(x[:16] for x in src), ",".join(x[:16] for x in roots)))
    print("subproblems certified: %d / 256" % (256 - len(missing)))
    if missing: print("missing:", missing)
    for p in problems: print("problem:", p)
    ok = not missing and len(src) == 1 and len(roots) == 1 and not any("provenance" in p for p in problems)
    print("C10-BB-MERGE: %s" % ("COMPLETE -- every biunimodular vector of length 10 is one of the 3040 certified rays" if ok else "INCOMPLETE"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
