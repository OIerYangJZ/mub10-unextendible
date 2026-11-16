#!/usr/bin/env python
"""Projective (P^9 x P^9) chart decomposition for strong unextendibility of the indirect 2x5 product triple
(Astra-A round 5).  Bihomogeneous bidegree-(1,1) system over K = Q(zeta20):
    d_k     = X_k Y_k - X_0 Y_0                                 (k = 1..9)
    f_{t,j} = sum_{k != l} H_{tjk} H_{tjl}^{-1} X_k Y_l          (all 20 rows (t,j) of U1^*U0, U2^*U0, times sqrt10)
A common unbiased vector gives a point with X_0 Y_0 != 0.  If the system has NO point in P^9 x P^9 over the algebraic
closure of F_p (for one prime above p, zeta20 -> r), it has none in characteristic 0 (reduce an integral, primitive
representative).  Charts covering P^9 x P^9 on {d = 0}:
    affine:      X_0 = Y_0 = 1                         (18 variables)
    boundary(i,j), i != j:  X_i = Y_j = 1, Y_i = X_j = 0, X_k Y_k = 0 for all k   (16 variables)   -- 90 charts
Each chart is written as an msolve file over F_p.  Controls: affine U0+U1 with the |+i>(x)G3(:,0) phases (solvable);
boundary charts with only the U0 equations (X = e_i, Y = e_j is a solution).
Usage: python product_triple_charts.py <outdir> [p r]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import product_triple as PT


U = [PT.U0, PT.U1, PT.U2]          # default: the fully indirect triple (orbit 8); --rep inf,0,1,2,3,4 overrides
GAUSS_CHECK = True


REP = "inf,0,1,2,3,4"


def exponents():
    """exact table from integer formulas (exact_tables.py); the numerical matrices serve only as a cross-check"""
    import exact_tables as ET
    E = np.array(ET.exponent_table(REP), dtype=int)
    M = np.vstack([U[1].conj().T @ U[0], U[2].conj().T @ U[0]]) * np.sqrt(10)
    assert np.abs(M - np.exp(2j * np.pi * E / 20)).max() < 1e-12, "exact table disagrees with the numerical triple"
    # independent cross-check against the Gauss-sum formula (Astra-A round 5, eq. (6)):
    #   H_{t,j,x} = w^{-t x^2 - j x},  H_{t,j,5+k} = rho_t w^{c_t (k-j)^2},  w = z^4, i = z^5
    c = {1: 4, 2: 2, 3: 3, 4: 1}; rho = {1: 0, 2: 0, 3: 5, 4: 5}
    E2 = np.zeros((20, 10), dtype=int)
    for t in range(1, 5):
        for j in range(5):
            row = (t - 1) * 5 + j
            for x in range(5):
                E2[row, x] = (4 * (-t * x * x - j * x)) % 20
                E2[row, 5 + x] = (rho[t] + 4 * c[t] * (x - j) ** 2) % 20
    if GAUSS_CHECK:
        assert (E == E2).all(), "exponent table disagrees with the Gauss-sum formula"
    return E


def polys(E, p, r, chart, rows=range(20)):
    rp = [pow(r, e, p) for e in range(20)]
    X = ["X%d" % k for k in range(10)]; Y = ["Y%d" % k for k in range(10)]
    fixed = {}
    if chart == "affine":
        fixed = {"X0": 1, "Y0": 1}
    else:
        i, j = chart
        fixed = {X[i]: 1, Y[j]: 1, Y[i]: 0, X[j]: 0}
    var = [v for v in X + Y if v not in fixed]

    def mon(a, b):                       # product of two variables with fixed values substituted -> (coef, monomial)
        c = 1; m = []
        for v in (a, b):
            if v in fixed:
                c *= fixed[v]
            else:
                m.append(v)
        return c, "*".join(m)

    out = []
    # d equations
    if chart == "affine":
        for k in range(1, 10):
            out.append("%s*%s-1" % (X[k], Y[k]))
    else:
        for k in range(10):
            c, m = mon(X[k], Y[k])
            if c and m:
                out.append(m)
            elif c and not m:
                raise SystemExit("inconsistent chart")
    # f equations
    for row in rows:
        terms = {}
        for k in range(10):
            for l in range(10):
                if k == l:
                    continue
                coef = rp[(E[row, k] - E[row, l]) % 20]
                c, m = mon(X[k], Y[l])
                if c == 0:
                    continue
                terms[m] = (terms.get(m, 0) + c * coef) % p
        const = terms.pop("", 0)
        ts = ["%d*%s" % (v, m) for m, v in sorted(terms.items()) if v]
        if not ts and const:
            raise SystemExit("chart has a nonzero constant equation")
        if not ts:
            continue
        s = "+".join(ts) + ("+%d" % const if const else "")       # constant LAST (msolve parser, N7)
        out.append(s)
    return var, out


def write(path, var, eqs, p):
    with open(path, "w") as fh:
        fh.write(",".join(var) + "\n%d\n" % p + ",\n".join(eqs) + "\n")


def main():
    global U, GAUSS_CHECK, REP
    args = [a for a in sys.argv[1:]]
    if "--rep" in args:                      # labels for (Z0,Z1,X0,X1,Y0,Y1): inf or 0..4 (G_inf = I, G_t)
        k = args.index("--rep"); REP = args[k + 1]; rep = REP.split(","); del args[k:k + 2]
        import product_triples_all as PA
        ell = [0 if x == "inf" else int(x) + 1 for x in rep]
        U = PA.triple(ell)
        GAUSS_CHECK = rep == ["inf", "0", "1", "2", "3", "4"]
        for a in range(3):
            assert np.abs(U[a].conj().T @ U[a] - np.eye(10)).max() < 1e-12
            for b in range(a + 1, 3):
                assert np.abs(np.abs(U[a].conj().T @ U[b]) ** 2 - 0.1).max() < 1e-12
    od = args[0]
    p, r = (int(args[1]), int(args[2])) if len(args) > 2 else (1073741741, 95182621)
    assert pow(r, 20, p) == 1 and (pow(r, 8, p) - pow(r, 6, p) + pow(r, 4, p) - pow(r, 2, p) + 1) % p == 0
    os.makedirs(od, exist_ok=True)
    E = exponents()
    n = 0
    var, eqs = polys(E, p, r, "affine"); write(os.path.join(od, "affine.ms"), var, eqs, p); n += 1
    for i in range(10):
        for j in range(10):
            if i != j:
                var, eqs = polys(E, p, r, (i, j)); write(os.path.join(od, "b_%d_%d.ms" % (i, j)), var, eqs, p); n += 1
                if (i, j) in ((0, 1), (0, 5), (5, 0), (5, 6)):      # controls: U0 equations only (X = e_i, Y = e_j solves)
                    var, eqs = polys(E, p, r, (i, j), rows=[]); write(os.path.join(od, "ctrl_b_%d_%d.ms" % (i, j)), var, eqs, p)
    # affine control: U0 + U1 rows only, pinned to the |+i>(x)G3(:,0) phase vector
    var, eqs = polys(E, p, r, "affine", rows=range(10))
    psi = U[2][:, 0]; x = np.sqrt(10) * U[0].conj().T @ psi; x = x / x[0]
    f = np.rint(np.angle(x) / (2 * np.pi / 20)).astype(int) % 20
    rp = [pow(r, e, p) for e in range(20)]
    eqs += ["X%d+%d" % (k, (-rp[f[k]]) % p) for k in range(1, 5)]
    write(os.path.join(od, "ctrl_affine.ms"), var, eqs, p)
    print("p %d r %d: wrote %d charts + controls to %s ; exponent table from exact formulas (exact_tables.py)%s" % (
        p, r, n, od, "; also matches Astra-A's Gauss-sum table" if GAUSS_CHECK else ""))


if __name__ == "__main__":
    main()
