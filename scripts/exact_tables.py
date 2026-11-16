#!/usr/bin/env python
"""Exact (integer-formula) exponent tables and Weyl actions for MU product triples in C^2 (x) C^5.
No floating point is used to produce the outputs; floats appear only in the optional cross-check against the
numerical matrices.  All phases are exponents of zeta20 = e^{2 pi i/20} (omega = zeta20^4, i = zeta20^5).

Labels: 'inf' -> G_inf = I_5;  t in F5 -> G_t(x, j) = omega^{t x^2 + j x} / sqrt5.
Triple: basis b (Z, X, Y) = [ |q_{b,0}> (x) G_{l(b,0)} , |q_{b,1}> (x) G_{l(b,1)} ],  qubit states
  Z: |0>, |1>;  X: (|0> +- |1>)/sqrt2;  Y: (|0> +- i|1>)/sqrt2.
Facts used (classical):
  * sqrt2 <q'|q> for q in Z and q' in X or Y, and for X vs Y, is in {1, -1, i, -i}  (direct evaluation);
  * sqrt5 (G_inf^* G_t)_{jk} = omega^{t j^2 + k j};   sqrt5 (G_t^* G_inf)_{jk} = omega^{-(t k^2 + j k)};
  * for s != t: sqrt5 (G_s^* G_t)_{jk} = (1/sqrt5) sum_x omega^{(t-s) x^2 + (k-j) x}
        = chi(t-s) omega^{-(k-j)^2 (4(t-s))^{-1}}     (quadratic Gauss sum over F5, sum_x omega^{a x^2} = chi(a) sqrt5);
  * Weyl W = X^a Z^b (X|x> = |x+1>, Z|x> = omega^x |x>):  W G_t e_k = omega^{t a^2 - k a - a b} G_t e_{k + b - 2 t a},
    W G_inf e_k = omega^{b k} G_inf e_{k + a}.
"""
CHI5 = {1: 1, 4: 1, 2: -1, 3: -1}          # quadratic character of F5
INV5 = {1: 1, 2: 3, 3: 2, 4: 4}

# qubit states as (component0, component1) in exponents of i (None = 0), times 1/sqrt2 except Z states
QUBIT = {("Z", 0): ("e", 0), ("Z", 1): ("e", 1),
         ("X", 0): ("s", (0, 0)), ("X", 1): ("s", (0, 2)),     # (|0> + i^0 |1>)/sqrt2, (|0> + i^2 |1>)/sqrt2
         ("Y", 0): ("s", (0, 1)), ("Y", 1): ("s", (0, 3))}     # (|0> + i |1>)/sqrt2,  (|0> + i^3 |1>)/sqrt2

def qubit_overlap_exp(bq, q):
    """exponent e (mod 4) with sqrt2 <bq | q> = i^e  (bq, q in different Pauli bases)."""
    kb, vb = QUBIT[bq]; kq, vq = QUBIT[q]
    if kb == "e" and kq == "s":                 # <z_k | (|0> + i^m |1>)/sqrt2> * sqrt2 = i^{m if k=1 else 0}
        return vq[1] if vb == 1 else 0
    if kb == "s" and kq == "e":                 # conj
        return (-vb[1]) % 4 if vq == 1 else 0
    if kb == "s" and kq == "s":                 # sqrt2 <a|b> = (1 + i^{-mb} i^{mq}) / sqrt2 ... must be a unit
        m = (vq[1] - vb[1]) % 4                 # (1 + i^m)/sqrt2 * ... ; for X vs Y: m odd -> (1 + i^m)/sqrt2 = zeta8^{m}
        raise ValueError("X/Y overlap is a zeta8 power, handled by caller via exp20")
    raise ValueError("same basis")

def qubit_overlap_exp20(bq, q):
    """exponent e (mod 20)?  sqrt2 <bq|q> as a power of zeta_40 is needed for X vs Y; we only use U0 = Z basis here."""
    return 5 * qubit_overlap_exp(bq, q)

def qudit_overlap_exp(u, v, j, k):
    """exponent e (mod 20) with sqrt5 (G_u^* G_v)_{jk} = zeta20^e  (u != v labels; 'inf' or 0..4)."""
    if u == "inf" and v == "inf":
        raise ValueError("same basis")
    if u == "inf":
        t = v; return (4 * (t * j * j + k * j)) % 20
    if v == "inf":
        t = u; return (-4 * (t * k * k + j * k)) % 20
    a = (v - u) % 5
    if a == 0:
        raise ValueError("same basis")
    e = (-4 * (((k - j) ** 2) * INV5[(4 * a) % 5])) % 20
    if CHI5[a] == -1:
        e = (e + 10) % 20
    return e

def parse_rep(rep):
    lab = [x if x == "inf" else int(x) for x in rep.split(",")]
    return {("Z", 0): lab[0], ("Z", 1): lab[1], ("X", 0): lab[2], ("X", 1): lab[3], ("Y", 0): lab[4], ("Y", 1): lab[5]}

def exponent_table(rep):
    """20 x 10 table E with sqrt10 [U1^* U0 ; U2^* U0]_{row, col} = zeta20^E.  U0 = Z basis, rows: X then Y blocks,
    each block = (half 0: 5 rows, half 1: 5 rows); columns: Z half 0 (5), Z half 1 (5)."""
    L = parse_rep(rep)
    E = [[0] * 10 for _ in range(20)]
    row = 0
    for B in ("X", "Y"):
        for h in (0, 1):
            for j in range(5):
                for zh in (0, 1):
                    for k in range(5):
                        e = (qubit_overlap_exp20((B, h), ("Z", zh)) + qudit_overlap_exp(L[(B, h)], L[("Z", zh)], j, k)) % 20
                        E[row][5 * zh + k] = e
                row += 1
    return E

def weyl_substitution(rep, a, b):
    """Exact action of I2 (x) X^a Z^b in U0 (Z-basis) coordinates as a substitution z_i -> zeta20^{ph_i} z_{sig_i}:
    with z = sqrt10 U0^* psi and psi -> W psi, (M z)_i = M[i, sig_i] z_{sig_i}, M = U0^* (I (x) W) U0 (monomial).
    Column action: W G e_k = zeta20^{e_k} G e_{pi(k)}  =>  M[pi(k), k] = zeta20^{e_k}; rows: sig(pi(k)) = k."""
    L = parse_rep(rep)
    sig = [0] * 10; ph = [0] * 10
    for zh in (0, 1):
        t = L[("Z", zh)]
        for k in range(5):
            if t == "inf":
                pk, e = (k + a) % 5, (4 * (b * k)) % 20
            else:
                pk, e = (k + b - 2 * t * a) % 5, (4 * (t * a * a - k * a - a * b)) % 20
            sig[5 * zh + pk] = 5 * zh + k; ph[5 * zh + pk] = e
    return sig, ph

if __name__ == "__main__":
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import product_triples_all as PA
    reps = [l.split()[1] for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbit_reps.txt")) if l.strip()]
    X5 = np.roll(np.eye(5), 1, axis=0); Z5 = np.diag(np.exp(2j * np.pi * np.arange(5) / 5))
    bad = 0
    for rep in reps:
        U = PA.triple([0 if x == "inf" else int(x) + 1 for x in rep.split(",")])
        C = np.vstack([U[1].conj().T @ U[0], U[2].conj().T @ U[0]]) * np.sqrt(10)
        E = np.array(exponent_table(rep))
        err = np.abs(C - np.exp(2j * np.pi * E / 20)).max()
        werr = 0.0
        for a in range(5):
            for b in range(5):
                Wm = np.linalg.matrix_power(X5, a) @ np.linalg.matrix_power(Z5, b)
                M = U[0].conj().T @ np.kron(np.eye(2), Wm) @ U[0]
                sig, ph = weyl_substitution(rep, a, b)
                Mx = np.zeros((10, 10), complex)
                for i in range(10):
                    Mx[i, sig[i]] = np.exp(2j * np.pi * ph[i] / 20)
                werr = max(werr, np.abs(M - Mx).max())
        ok = err < 1e-12 and werr < 1e-12
        bad += not ok
        print("%-22s exact table vs numeric: %.1e ; exact Weyl actions vs numeric (25 elements): %.1e  %s" % (rep, err, werr, "OK" if ok else "MISMATCH"))
    print("ALL OK" if bad == 0 else "%d MISMATCHES" % bad)
