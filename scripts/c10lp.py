#!/usr/bin/env python
"""Residual 2-design LP for the Fourier pair (I, F_d) (Astra-B round 8 request 2 = Astra-A round 8 request 2, primal side).
A complete set containing (I, F_d) is a 2-design, so its other d(d-1) rays r (all unbiased to I and F_d, hence in the
catalogue) satisfy
        sum_r w_r A_r = R := I + S - sum_{v in I u F_d} A_v ,   A_r = (|r><r|)^{(x)2},   w_r in {0,1}.
Coordinates: A_r restricted to Sym^2 in the monomial basis, M_r[(ij),(kl)] = r_i r_j conj(r_k r_l) (i<=j, k<=l); the
real coordinates are Re M[a,b] (a<=b) and Im M[a,b] (a<b).  Equality on these entries is equality of the operators.
Runs (HiGHS through scipy.optimize.linprog):
  mix   : w >= 0 only (ray-mixture, Astra-A)          -- phase 1 (min L1 slack), then min product weight
  box   : 0 <= w <= 1 (Astra-B LP)                     -- same
  cuts  : box + per-ray cuts  sum_{s ~O r} w_s >= (d-1) w_r ,  sum_{s ~U r} w_s >= d(d-2) w_r  (valid for 0/1 points)
  cap   : the product cap sum_{product} w <= 5 is checked against the minimum product weight of each run.
Self-test: d=5 with the 20 chirp rays (w = 1) must satisfy the equation exactly.
Usage: python c10lp.py <rays.npz> [graph_tol]
"""
import sys, time
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

T0 = time.time()
def log(*a): print("[%6.1fs]" % (time.time() - T0), *a, flush=True)

def coords(V, d):
    """V: (n, d) rays -> (n, d(d+1)/2 squared) real coordinates of (|v><v|)^{(x)2} on Sym^2"""
    iu, ju = np.triu_indices(d)
    P2 = V[:, iu] * V[:, ju]                                    # (n, m) with m = d(d+1)/2
    M = P2[:, :, None] * P2.conj()[:, None, :]                  # (n, m, m)
    m = P2.shape[1]; a, b = np.triu_indices(m); a1, b1 = np.triu_indices(m, 1)
    return np.concatenate([M[:, a, b].real, M[:, a1, b1].imag], axis=1)

def target(d):
    iu, ju = np.triu_indices(d); m = len(iu)
    T = np.zeros((m, m))
    for x in range(m):
        for y in range(m):
            i, j, k, l = iu[x], ju[x], iu[y], ju[y]
            T[x, y] = int(i == k and j == l) + int(i == l and j == k)   # int(): numpy bool + bool is OR
    a, b = np.triu_indices(m); a1, b1 = np.triu_indices(m, 1)
    IS = np.concatenate([T[a, b], np.zeros(len(a1))])
    n = np.arange(d); F = np.exp(2j * np.pi * np.outer(n, n) / d) / np.sqrt(d)
    pair = np.concatenate([np.eye(d), F.T])                     # rows: the 2d rays of I and F_d
    return IS - coords(pair, d).sum(0)

def selftest():
    d = 5; n = np.arange(d); w5 = np.exp(2j * np.pi / 5)
    rays = np.array([w5 ** ((t * x * x + j * x) % 5) / np.sqrt(5) for t in range(1, 5) for j in range(5) for x in [n]]).reshape(20, 5)
    err = np.abs(coords(rays, d).sum(0) - target(d)).max()
    log("self-test d=5 (complete set I, F5, G1..G4): max |sum A_r - R| = %.1e" % err)
    assert err < 1e-12

def solve(A, b, prod, N, ub, extra=None, label=""):
    """phase 1: min sum(s+ + s-) s.t. A w + s+ - s- = b ; then min product weight s.t. A w = b"""
    k = A.shape[0]
    c = np.concatenate([np.zeros(N), np.ones(2 * k)])
    Aeq = sp.hstack([sp.csr_matrix(A), sp.identity(k), -sp.identity(k)]).tocsr()
    bounds = [(0, ub)] * N + [(0, None)] * (2 * k)
    Aub = bub = None
    if extra is not None:
        Aub = sp.hstack([extra[0], sp.csr_matrix((extra[0].shape[0], 2 * k))]).tocsr(); bub = extra[1]
    r1 = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=b, bounds=bounds, method="highs")
    log("%-5s phase 1: status %d, min L1 residual %.3e (scale |b|_1 = %.1f)" % (label, r1.status, r1.fun, np.abs(b).sum()))
    out = {"phase1": r1.fun}
    if r1.status != 0 or r1.fun > 1e-7:
        y = r1.eqlin.marginals if r1.status == 0 else None
        out["dual"] = y
        return out
    cp = np.zeros(N); cp[prod] = 1
    for sense in (1, -1):
        r2 = linprog(sense * cp, A_ub=extra[0] if extra is not None else None, b_ub=extra[1] if extra is not None else None,
                     A_eq=A, b_eq=b, bounds=[(0, ub)] * N, method="highs")
        if r2.status != 0:
            log("%-5s %s product weight: status %d (%s)" % (label, "min" if sense == 1 else "max", r2.status, r2.message)); continue
        w = r2.x; res = np.abs(A @ w - b).max()
        log("%-5s %s product weight %.6f ; support %d ; max w %.4f ; sum w %.4f ; residual %.1e" % (
            label, "min" if sense == 1 else "max", w[prod].sum(), int((w > 1e-9).sum()), w.max(), w.sum(), res))
        out["min" if sense == 1 else "max"] = w
    return out

def main():
    selftest()
    src = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
    d = 10; D = np.load(src); psi = D["psi"]; prod = D["product"]; N = len(psi)
    A = coords(psi, d).T; b = target(d)                         # (k, N), (k,)
    log("rays %d ; coordinates %d ; R trace check: sum w = %.4f required" % (N, len(b), d * (d - 1)))
    # reduce to the row space of A; the part of b outside it must vanish for any real solution
    U_, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int((s > s[0] * 1e-10).sum())
    Q = U_[:, :r]; out_of_span = np.linalg.norm(b - Q @ (Q.T @ b))
    log("rank of {A_r} = %d (largest sv %.3e, sv[r-1] %.3e, sv[r] %.3e) ; |b - proj b| = %.3e" % (
        r, s[0], s[r - 1], s[r] if r < len(s) else 0, out_of_span))
    Ar = Q.T @ A; br = Q.T @ b
    # local cuts from the overlap graphs
    G = np.abs(psi.conj() @ psi.T) ** 2; np.fill_diagonal(G, -1)
    isO = np.abs(G) < tol; isU = np.abs(G - 0.1) < tol
    cutA = sp.vstack([sp.csr_matrix(-(isO.astype(float))) + (d - 1) * sp.identity(N),
                      sp.csr_matrix(-(isU.astype(float))) + d * (d - 2) * sp.identity(N)]).tocsr()
    cutb = np.zeros(2 * N)
    res = {}
    res["mix"] = solve(Ar, br, prod, N, None, label="mix")
    res["box"] = solve(Ar, br, prod, N, 1.0, label="box")
    res["cuts"] = solve(Ar, br, prod, N, 1.0, extra=(cutA, cutb), label="cuts")
    for k, v in res.items():
        if "min" in v:
            pw = v["min"][prod].sum()
            log("%-5s product cap 5: %s (min product weight %.4f)" % (k, "satisfiable" if pw <= 5 + 1e-7 else "VIOLATED", pw))
        else:
            log("%-5s INFEASIBLE (phase-1 residual %.3e) -> candidate separator" % (k, v["phase1"]))
    np.savez("runs/c10lp_result.npz", **{"%s_%s" % (k, kk): vv for k, v in res.items() for kk, vv in v.items() if isinstance(vv, np.ndarray)})
    log("saved runs/c10lp_result.npz")

if __name__ == "__main__":
    main()
