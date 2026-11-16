#!/usr/bin/env python
"""Independent verifier for a finite-field Nullstellensatz certificate 1 = sum_v h_v g_v (mod p).
Reads the generators g_v from an msolve input file (vars / p / polys) and the cofactors h_v (one polynomial per line,
Singular output, same order) and expands the sum with plain Python integer arithmetic.  No Groebner software.
Exit code 0 and 'CERTIFICATE OK' iff the expansion equals the constant 1 modulo p.
Usage: python verify_cert.py <chart.ms> <cofactors.txt>
"""
import sys, re

def parse_poly(s, idx, p):
    """sum of terms c*x1^e*x2*..., c an integer (possibly absent), returns {exponent tuple: coef mod p}"""
    s = s.replace(" ", "").replace("\n", "")
    if s in ("", "0"):
        return {}
    terms = re.findall(r"[+-]?[^+-]+", s)
    n = len(idx)
    out = {}
    for t in terms:
        sign = -1 if t[0] == "-" else 1
        t = t.lstrip("+-")
        coef = 1
        e = [0] * n
        for f in t.split("*"):
            if not f:
                continue
            if f[0].isdigit():
                if "/" in f:
                    a, b = f.split("/"); coef = coef * int(a) * pow(int(b), -1, p) % p
                else:
                    coef = coef * int(f) % p
            else:
                if "^" in f:
                    v, k = f.split("^"); e[idx[v]] += int(k)
                else:
                    e[idx[f]] += 1
        key = tuple(e)
        out[key] = (out.get(key, 0) + sign * coef) % p
    return {k: v for k, v in out.items() if v}

def mul_acc(acc, a, b, p):
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(x + y for x, y in zip(ka, kb))
            acc[k] = (acc.get(k, 0) + va * vb) % p

def main():
    ms, cof = sys.argv[1], sys.argv[2]
    lines = open(ms).read().split("\n")
    V = [v.strip() for v in lines[0].split(",")]
    p = int(lines[1])
    idx = {v: i for i, v in enumerate(V)}
    gens = [parse_poly(q, idx, p) for q in "\n".join(lines[2:]).strip().split(",\n") if q.strip()]
    hs = [parse_poly(l, idx, p) for l in open(cof).read().strip().split("\n")]
    if len(hs) != len(gens):
        print("MISMATCH: %d generators, %d cofactors" % (len(gens), len(hs))); sys.exit(2)
    acc = {}
    for g, h in zip(gens, hs):
        mul_acc(acc, g, h, p)
    acc = {k: v for k, v in acc.items() if v}
    one = tuple([0] * len(V))
    ok = acc == {one: 1}
    print("%s: p=%d, %d generators, %d cofactor terms, result %s" % (
        "CERTIFICATE OK" if ok else "CERTIFICATE FAILED", p, len(gens), sum(len(h) for h in hs),
        "1" if ok else "%d nonzero terms" % len(acc)))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
