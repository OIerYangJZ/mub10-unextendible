#!/usr/bin/env python
"""Manifest verification for the product-triple theorem (Astra-B round-6 audit, BLOCKING items 1-2).

For every orbit representative in scripts/orbit_reps.txt:
  * regenerate the expected chart inputs from the EXACT exponent tables (product_triple_charts.py --rep, which uses
    exact_tables.py) into a fresh temporary directory;
  * require EXACTLY the 91 charts {affine} U {b_i_j : 0 <= i != j <= 9} (no missing / unexpected chart files);
  * require byte equality (sha256) of each expected input with the input used on the server (runs/charts_all/oN/*.ms);
  * verify each certificate (oN/<chart>_cof.txt) against the EXPECTED input with verify_cert.py (independent integer
    expansion of 1 = sum h_v g_v mod p).
For orbits with a rank certificate (runs/rank_witness/*.json): recompute the sha256 of the square witness and re-rank it
with rankmodp.  Writes results/product_triples_manifest.json and .md; exit status 0 iff every orbit is covered by at
least one fully verified method and nothing is missing, mismatched or failing.
The certificate checks (one verify_cert.py subprocess per chart) run in parallel, largest certificates first
(MANIFEST_JOBS workers, default 12); the checks themselves and the report are unchanged.
Usage: python verify_manifest.py   (run from ~/mub10)
"""
import os, sys, json, hashlib, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
JOBS = int(os.environ.get("MANIFEST_JOBS", "12"))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
P, R = 1073741741, 95182621

def sha(fn):
    return hashlib.sha256(open(fn, "rb").read()).hexdigest()

# exact integer description of the symmetry action on label assignments (positions Z0,Z1,X0,X1,Y0,Y1; labels
# 'inf',0..4), as in the proof document: H2, S2 permute positions; F5, D, M2 act on labels; conjugation does both.
INF = "inf"
def _inv5(t): return {1: 1, 2: 3, 3: 2, 4: 4}[t]
GENS = [((2, 3, 0, 1, 5, 4), lambda t: t),                                         # H2
        ((0, 1, 4, 5, 3, 2), lambda t: t),                                         # S2
        ((0, 1, 2, 3, 4, 5), lambda t: 0 if t == INF else (INF if t == 0 else _inv5(t))),   # F5: t -> 1/t
        ((0, 1, 2, 3, 4, 5), lambda t: t if t == INF else (t + 1) % 5),            # D: t -> t+1
        ((0, 1, 2, 3, 4, 5), lambda t: t if t == INF else (-t) % 5),               # M2: t -> -t
        ((0, 1, 2, 3, 5, 4), lambda t: t if t == INF else (-t) % 5)]               # global conjugation

def roster_check(reps):
    """independent exact enumeration: the 4080 admissible assignments, their orbits, and a one-to-one roster"""
    import itertools
    labs = [INF, 0, 1, 2, 3, 4]
    adm = [a for a in itertools.product(labs, repeat=6)
           if not ({a[0], a[1]} & {a[2], a[3]} or {a[0], a[1]} & {a[4], a[5]} or {a[2], a[3]} & {a[4], a[5]})]
    idx = {a: i for i, a in enumerate(adm)}; seen = {}; orbits = []
    for a in adm:
        if a in seen: continue
        k = len(orbits); orb = [a]; seen[a] = k; q = [a]
        while q:
            b = q.pop()
            for sig, pi in GENS:
                c = [None] * 6
                for s in range(6): c[sig[s]] = pi(b[s])
                c = tuple(c)
                if c not in seen: seen[c] = k; orb.append(c); q.append(c)
        orbits.append(orb)
    parse = lambda r: tuple(x if x == INF else int(x) for x in r.split(","))
    hit = [seen.get(parse(rep)) for _, rep in reps]
    ok = (len(adm) == 4080 and len(orbits) == 10 and len(reps) == 10 and None not in hit and sorted(hit) == list(range(10)))
    return ok, len(adm), sorted(len(o) for o in orbits), hit

def main():
    t0 = time.time()
    reps = [l.split() for l in open(os.path.join(HERE, "orbit_reps.txt")) if l.strip()]
    r_ok, n_adm, sizes, hit = roster_check(reps)
    print("roster: %d admissible assignments, %d orbits %s, representatives hit orbits %s -> %s" % (
        n_adm, len(sizes), sizes, hit, "COMPLETE" if r_ok else "INCOMPLETE"), flush=True)
    expected_names = ["affine"] + ["b_%d_%d" % (i, j) for i in range(10) for j in range(10) if i != j]
    assert len(expected_names) == 91
    tmp = tempfile.mkdtemp(prefix="manifest_")
    report = {"p": P, "zeta20": R, "orbits": [], "generated": time.ctime()}
    all_ok = True
    # phase 1: regenerate inputs, compare inputs, collect the certificate checks
    pending = []; per_orbit = []
    for oid, rep in reps:
        od = os.path.join(ROOT, "runs", "charts_all", "o%s" % oid)
        ed = os.path.join(tmp, "o%s" % oid)
        subprocess.run([PY, "-I", os.path.join(HERE, "product_triple_charts.py"), ed, "--rep", rep, str(P), str(R)],
                       check=True, capture_output=True)
        used = sorted(f[:-3] for f in os.listdir(od) if f.endswith(".ms") and (f.startswith("b_") or f == "affine.ms"))
        missing = sorted(set(expected_names) - set(used)); unexpected = sorted(set(used) - set(expected_names))
        charts = []; n_ok = 0
        for name in expected_names:
            e_in = os.path.join(ed, name + ".ms"); u_in = os.path.join(od, name + ".ms"); cof = os.path.join(od, name + "_cof.txt")
            rec = {"chart": name, "input_sha256": sha(e_in)}
            rec["input_matches_used"] = os.path.exists(u_in) and sha(u_in) == rec["input_sha256"]
            if os.path.exists(cof):
                rec["certificate_sha256"] = sha(cof)
                pending.append((os.path.getsize(cof), e_in, cof, rec))
            else:
                rec["verified"] = False; rec["verifier"] = "no certificate file"
            charts.append(rec)
        per_orbit.append((oid, rep, missing, unexpected, charts))
    # phase 2: all certificate checks in parallel, largest first
    def check(job):
        _, e_in, cof, rec = job
        v = subprocess.run([PY, "-I", os.path.join(HERE, "verify_cert.py"), e_in, cof], capture_output=True, text=True)
        rec["verified"] = v.returncode == 0 and v.stdout.startswith("CERTIFICATE OK")
        rec["verifier"] = v.stdout.strip()[:160]
    pending.sort(key=lambda j: -j[0])
    print("checking %d certificates with %d workers" % (len(pending), JOBS), flush=True)
    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(check, pending))
    # phase 3: per-orbit verdicts (unchanged logic)
    for oid, rep, missing, unexpected, charts in per_orbit:
        n_ok = sum(1 for rec in charts if rec["verified"] and rec["input_matches_used"])
        chart_ok = (n_ok == 91 and not missing and not unexpected)
        # rank witness
        wj = os.path.join(ROOT, "runs", "rank_witness", "%s_n3.json" % rep.replace(",", "-"))
        rank = None
        if os.path.exists(wj):
            meta = json.load(open(wj)); wb = wj[:-5] + ".bin"
            h = sha(wb) if os.path.exists(wb) else None
            out = subprocess.run([os.path.join(HERE, "rankmodp"), wb], capture_output=True, text=True) if h else None
            rk = int(out.stdout.split()[1]) if out and out.returncode == 0 else -1
            rank = {"witness_sha256_recorded": meta["witness_sha256"], "witness_sha256_now": h,
                    "invariant_dim": meta["invariant_dim"], "rerank": rk,
                    "ok": h == meta["witness_sha256"] and rk == meta["invariant_dim"] and meta["p"] == P and meta["zeta20"] == R}
        orbit_ok = chart_ok or (rank is not None and rank["ok"])
        all_ok &= orbit_ok
        report["orbits"].append({"orbit": int(oid), "rep": rep, "charts_verified": n_ok, "missing": missing,
                                 "unexpected": unexpected, "chart_method_ok": chart_ok, "rank_method": rank,
                                 "orbit_ok": orbit_ok, "charts": charts})
        print("o%s %-22s charts %3d/91 verified%s%s ; rank %s ; ORBIT %s" % (
            oid, rep, n_ok, " missing=%d" % len(missing) if missing else "", " unexpected=%d" % len(unexpected) if unexpected else "",
            ("OK (%d)" % rank["rerank"]) if rank and rank["ok"] else ("n/a" if rank is None else "FAILED"),
            "OK" if orbit_ok else "NOT OK"), flush=True)
    all_ok = all_ok and r_ok
    report["roster"] = {"ok": r_ok, "assignments": n_adm, "orbit_sizes": sizes, "rep_orbits": hit}
    report["all_ok"] = all_ok
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "results", "product_triples_manifest.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "results", "product_triples_manifest.md"), "w") as fh:
        fh.write("# Product-triple proof manifest (%s)\n\np = %d, zeta20 -> %d. Inputs regenerated from exact formulas; "
                 "certificates verified against the regenerated inputs.\n\n| orbit | rep | charts verified | chart method | rank method | orbit |\n|---|---|---|---|---|---|\n" % (report["generated"], P, R))
        for o in report["orbits"]:
            rk = o["rank_method"]
            fh.write("| %d | %s | %d/91 | %s | %s | %s |\n" % (o["orbit"], o["rep"], o["charts_verified"], "OK" if o["chart_method_ok"] else "no",
                     ("OK (rank %d)" % rk["rerank"]) if rk and rk["ok"] else ("n/a" if rk is None else "FAILED"), "OK" if o["orbit_ok"] else "NOT OK"))
        fh.write("\nRoster: %d admissible assignments in %d orbits %s; one representative per orbit: %s.\n" % (
            n_adm, len(sizes), sizes, "yes" if r_ok else "NO"))
        fh.write("\n**ALL ORBITS COVERED: %s** (%.0f s)\n" % ("YES" if all_ok else "NO", time.time() - t0))
    print("ALL ORBITS COVERED: %s" % ("YES" if all_ok else "NO"))
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
