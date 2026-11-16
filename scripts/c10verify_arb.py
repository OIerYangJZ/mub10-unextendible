#!/usr/bin/env sage-python
"""Independent second implementation of the cyclic-10 certification (cross-check of scripts/c10krawczyk.cpp).
Different encoding and arithmetic: the system and its Jacobian are built symbolically in Sage over QQ
(PolynomialRing + derivative, compiled with fast_callable), evaluation in Arb ball arithmetic (ComplexBallField, 160
bits).  Krawczyk boxes have dyadic radii and centres snapped to 2^-100, so their endpoints are exact and the inclusion tests on
Arb's outward-rounded bounds are rigorous.
Checks:
  1. every root: K(X) inside int(X) (unique root in X); the root lies in K(X);
  2. unimodularity: 1/conj(K) inside X  => fixed by the involution => unimodular; |K_j|^2 excluding 1 => not;
  3. the K enclosures are pairwise disjoint;
  4. rays z (z_0 = 1, z_{j+1} = x_j z_j), Gram matrix by Arb matrix product, O+ / U+ edges (value 0 / 10 possible),
     degrees, and equality of the edge sets with the C++ graph file.
Usage: sage -python c10verify_arb.py roots.bin graph_plus.bin [nproc]
"""
import sys, time, struct, math
from array import array
from multiprocessing import Pool
from sage.all import (PolynomialRing, QQ, ComplexBallField, RealBallField, CDF, fast_callable, prod, matrix, vector,
                      identity_matrix)

T0 = time.time()
def log(*a): print("[%6.1fs]" % (time.time() - T0), *a, flush=True)

n = 10
R = PolynomialRing(QQ, ["x%d" % i for i in range(n)]); X = R.gens()
POLYS = [sum(prod(X[(j + r) % n] for r in range(k)) for j in range(n)) for k in range(1, n)] + [prod(X) - 1]
JAC = [[p.derivative(X[m]) for m in range(n)] for p in POLYS]
CB = ComplexBallField(160); RB = RealBallField(160)
fF = [fast_callable(p, vars=X, domain=CB) for p in POLYS]
fJ = [[fast_callable(q, vars=X, domain=CB) for q in row] for row in JAC]
dF = [fast_callable(p, vars=X, domain=CDF) for p in POLYS]
dJ = [[fast_callable(q, vars=X, domain=CDF) for q in row] for row in JAC]
EXPS = [-40, -37, -33, -30, -27, -23, -20, -17]

def box(c, e):
    r = RB(2.0 ** e)
    return CB(RB(c.real()).add_error(r), RB(c.imag()).add_error(r))

def cert_one(x):
    xc = vector(CDF, x)
    for _ in range(3):
        Jm = matrix(CDF, [[f(*xc) for f in row] for row in dJ]); Fv = vector(CDF, [f(*xc) for f in dF])
        xc = xc - Jm.solve_right(Fv)
    Y = matrix(CDF, [[f(*xc) for f in row] for row in dJ]).inverse()
    snap = lambda v: float(round(float(v) * 2.0 ** 100)) / 2.0 ** 100 if abs(float(v)) < 2.0 ** 50 else None
    xc = vector(CDF, [CDF(snap(c.real()), snap(c.imag())) for c in xc])   # centre +- 2^e exact in 160 bits
    Yb = matrix(CB, Y); xb = vector(CB, [CB(c) for c in xc])
    Fx = vector(CB, [f(*xb) for f in fF])
    for e0 in EXPS:
        es = [e0 + max(0, math.ceil(math.log2(max(1.0, abs(c))))) for c in xc]
        Xb = [box(xc[i], es[i]) for i in range(n)]
        JX = matrix(CB, [[f(*Xb) for f in row] for row in fJ])
        D = vector(CB, [Xb[i] - xb[i] for i in range(n)])
        K = xb - Yb * Fx + (identity_matrix(CB, n) - Yb * JX) * D
        ok = all(K[i].real().lower() > Xb[i].real().lower() and K[i].real().upper() < Xb[i].real().upper() and
                 K[i].imag().lower() > Xb[i].imag().lower() and K[i].imag().upper() < Xb[i].imag().upper() for i in range(n))
        if not ok:
            continue
        nonuni = any((K[i].real() ** 2 + K[i].imag() ** 2 < 1) or (K[i].real() ** 2 + K[i].imag() ** 2 > 1) for i in range(n))
        s = [1 / K[i].conjugate() for i in range(n)]
        sin = all(s[i].real().lower() >= Xb[i].real().lower() and s[i].real().upper() <= Xb[i].real().upper() and
                  s[i].imag().lower() >= Xb[i].imag().lower() and s[i].imag().upper() <= Xb[i].imag().upper() for i in range(n))
        uni = 1 if sin and not nonuni else (0 if nonuni and not sin else (-2 if sin and nonuni else -1))
        if uni == -1 and e0 != EXPS[-1]:
            continue
        return (1, uni, e0, [(float(K[i].real().lower()), float(K[i].real().upper()), float(K[i].imag().lower()),
                              float(K[i].imag().upper())) for i in range(n)],
                [(K[i].real().mid(), K[i].real().rad(), K[i].imag().mid(), K[i].imag().rad()) for i in range(n)] if uni == 1 else None)
    return (0, -1, None, None, None)

def chunk(args):
    return [cert_one(x) for x in args]

def main():
    fn, gfile = sys.argv[1], sys.argv[2]; nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    raw = open(fn, "rb").read(); N = struct.unpack("<i", raw[:4])[0]
    a = array("d"); a.frombytes(raw[4:4 + 16 * n * N])
    roots = [[complex(a[2 * (t * n + i)], a[2 * (t * n + i) + 1]) for i in range(n)] for t in range(N)]
    import hashlib, sage.version
    log("roots %d ; sha256(roots.bin) %s ; Sage %s ; system built symbolically (%d polynomials, Jacobian by derivative)" % (
        N, hashlib.sha256(raw).hexdigest(), sage.version.version, len(POLYS)))
    parts = [roots[i::nproc * 8] for i in range(nproc * 8)]
    with Pool(nproc) as P:
        res_parts = P.map(chunk, parts)
    res = [None] * N
    for i, part in enumerate(res_parts):
        for j, r in enumerate(part):
            res[i + j * nproc * 8] = r
    cert = sum(r[0] for r in res); uni = [r[1] for r in res]
    log("certified %d / %d ; unimodular %d ; non-unimodular %d ; undecided %d ; contradictions %d ; largest radius 2^%d" % (
        cert, N, uni.count(1), uni.count(0), uni.count(-1), uni.count(-2), max(r[2] for r in res if r[0])))
    # pairwise disjointness of the K enclosures (float bounds widened by a relative 4 ulp)
    def wid(lo, hi):
        return lo - 4e-16 * (1 + abs(lo)), hi + 4e-16 * (1 + abs(hi))
    boxes = []
    for t, r in enumerate(res):
        if r[0]:
            boxes.append((t, [wid(b[0], b[1]) + wid(b[2], b[3]) for b in r[3]]))
    boxes.sort(key=lambda tb: tb[1][0][0])
    over = 0; tests = 0
    for i in range(len(boxes)):
        bi = boxes[i][1]
        for j in range(i + 1, len(boxes)):
            bj = boxes[j][1]
            if bj[0][0] > bi[0][1]:
                break
            tests += 1
            if all(not (bj[k][1] < bi[k][0] or bi[k][1] < bj[k][0] or bj[k][3] < bi[k][2] or bi[k][3] < bj[k][2]) for k in range(n)):
                over += 1
    log("enclosures pairwise disjoint: %s (box tests %d, overlapping %d)" % ("YES" if over == 0 else "NO", tests, over))
    # rays and Gram matrix
    U = [t for t in range(N) if res[t][0] and res[t][1] == 1]
    Z = []
    for t in U:
        K = [CB(RB(a).add_error(b), RB(c).add_error(d)) for (a, b, c, d) in res[t][4]]   # contains the worker's ball
        z = [CB(1)]
        for j in range(n - 1):
            z.append(K[j] * z[-1])
        Z.append(z)
    M = matrix(CB, Z)
    G = M.conjugate() * M.transpose()
    log("Gram matrix of %d rays computed in Arb" % len(U))
    m = len(U); W = (m + 63) // 64
    Ob = [[0] * W for _ in range(m)]; Ub = [[0] * W for _ in range(m)]
    nO = nU = both = 0
    for i in range(m):
        for j in range(i + 1, m):
            e = G[i, j]
            o = e.contains_zero()
            a2 = e.real() ** 2 + e.imag() ** 2
            u = a2.contains_exact(10)
            if o:
                Ob[i][j // 64] |= 1 << (j % 64); Ob[j][i // 64] |= 1 << (i % 64); nO += 2
            if u:
                Ub[i][j // 64] |= 1 << (j % 64); Ub[j][i // 64] |= 1 << (i % 64); nU += 2
            if o and u:
                both += 2
    degU = [sum(bin(w).count("1") for w in row) for row in Ub]
    log("O+ ordered pairs %d ; U+ ordered pairs %d ; both %d ; max U+ degree %d" % (nO, nU, both, max(degU)))
    raw = open(gfile, "rb").read(); mm = struct.unpack("<i", raw[:4])[0]
    q = array("Q"); q.frombytes(raw[4:4 + 16 * mm * ((mm + 63) // 64)])
    same = (mm == m) and all(q[i * W + k] == Ob[i][k] for i in range(m) for k in range(W)) and \
        all(q[m * W + i * W + k] == Ub[i][k] for i in range(m) for k in range(W))
    log("edge sets identical to the C++ graph file: %s" % ("YES" if same else "NO"))
    ok = (N == 34940 and cert == N and uni.count(-1) == 0 and uni.count(-2) == 0 and over == 0 and len(U) == 3040
          and max(degU) <= 8 and both == 0 and same)
    print("CONTRACT: N=34940 %s, all certified %s, unimodularity decided %s, disjoint %s, 3040 unimodular %s, max U+ degree <= 8 %s, O+/U+ disjoint %s, graphs equal C++ %s" % (
        N == 34940, cert == N, uni.count(-1) == 0 and uni.count(-2) == 0, over == 0, len(U) == 3040, max(degU) <= 8, both == 0, same))
    print("ARB-VERIFY: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
