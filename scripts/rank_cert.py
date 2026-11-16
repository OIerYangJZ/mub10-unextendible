#!/usr/bin/env python
"""Astra-B round-5 rank certificate for strong unextendibility of an MU product triple in C^2 (x) C^5.

R = F_p[c, z_0..z_9, w_0..w_9] / (z_k w_k - c), graded by bidegree (n, n).  Generators (bidegree (1,1)):
    f_g = sum_{k != l} C_{g,k} C_{g,l}^{-1} z_k w_l,   C = sqrt10 [U1^* U0 ; U2^* U0] = zeta20^E  (g = 0..19).
A physical common unbiased vector gives z = sqrt10 U0^* psi, w = conj z, c = 1 with all f_g = 0, so an identity
c^n = sum_g q_g f_g in R (over K = Q(zeta20)) is a contradiction.  The Weyl group W = <A = I(x)X5, B = I(x)Z5> (order 25)
acts on R by monomial substitutions and preserves span{f_g}; c^n is W-invariant, so it suffices that the W-invariant
part of the image, spanned by F_psi * R_{n-1,n-1}^{psi^-1} (F_psi = psi-isotypic generator combinations), is the whole
invariant subspace R_{n,n}^W.  We test this by a rank computation over F_p (zeta20 -> r).  rank = dim R_{n,n}^W mod one
prime implies a nonzero minor over K, hence surjectivity over K, hence the identity exists in characteristic 0.
Usage: python rank_cert.py <rep inf,0,1,2,3,4> <n> [p r]
"""
import sys, os, itertools, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import product_triples_all as PA

T0 = time.time()

def log(*a):
    print("[%6.1fs]" % (time.time() - T0), *a, flush=True)

def zeta_exp(M, tol=1e-9):
    """matrix of 20th roots of unity (times a positive scale) -> exponents mod 20"""
    E = np.rint(np.angle(M) / (2 * np.pi / 20)).astype(int) % 20
    return E

def monomial_action(U0, W):
    """psi -> (I2 (x) W) psi gives z -> M z with M = U0^* (I2 (x) W) U0 monomial.  As a substitution in polynomials,
    z_i -> (M z)_i = M[i, sig(i)] z_sig(i) and w_i -> conj(M[i, sig(i)]) w_sig(i).  Returns (sig, phase exponents)."""
    M = U0.conj().T @ np.kron(np.eye(2), W) @ U0
    sig = np.argmax(np.abs(M), axis=1)           # row i -> its unique nonzero column sig[i]
    for i in range(10):
        assert abs(abs(M[i, sig[i]]) - 1) < 1e-9, "not monomial"
    assert sorted(sig) == list(range(10))
    vals = np.array([M[i, sig[i]] for i in range(10)])
    ph = zeta_exp(vals)
    assert np.abs(vals - np.exp(2j * np.pi * ph / 20)).max() < 1e-9
    return sig, ph

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rep = args[0].split(","); n = int(args[1])
    p, r = (int(args[2]), int(args[3])) if len(args) > 3 else (1073741741, 95182621)
    assert pow(r, 20, p) == 1 and (pow(r, 8, p) - pow(r, 6, p) + pow(r, 4, p) - pow(r, 2, p) + 1) % p == 0
    zp = [pow(r, e, p) for e in range(20)]
    ell = [0 if x == "inf" else int(x) + 1 for x in rep]
    U = PA.triple(ell)
    import exact_tables as ET
    C = np.vstack([U[1].conj().T @ U[0], U[2].conj().T @ U[0]]) * np.sqrt(10)
    E = np.array(ET.exponent_table(args[0]), dtype=int)          # exact integer formulas
    assert np.abs(C - np.exp(2j * np.pi * E / 20)).max() < 1e-12, "exact table disagrees with the numerical triple"
    # Weyl generators on C^5
    X5 = np.roll(np.eye(5), 1, axis=0); Z5 = np.diag(np.exp(2j * np.pi * np.arange(5) / 5))
    acts = []
    for a in range(5):
        for b in range(5):
            Wm = np.linalg.matrix_power(X5, a) @ np.linalg.matrix_power(Z5, b)
            sig_n, ph_n = monomial_action(U[0], Wm)                  # numerical cross-check only
            sig, ph = ET.weyl_substitution(args[0], a, b)            # exact integer formulas
            assert list(sig) == list(sig_n) and list(np.array(ph) % 20) == list(np.array(ph_n) % 20), "Weyl action mismatch"
            acts.append(((a, b), (np.array(sig), np.array(ph))))
    log("orbit %s: exact exponent table and exact Weyl actions (25 elements), both cross-checked numerically" % args[0])

    # ---- normal monomials of R_{m,m}: (h, alpha, beta) with alpha_k beta_k = 0
    def basis(m):
        out = []
        for h in range(m, -1, -1):
            d = m - h
            for A in itertools.combinations_with_replacement(range(10), d):
                al = [0] * 10
                for k in A: al[k] += 1
                free = [k for k in range(10) if al[k] == 0]
                for B in itertools.combinations_with_replacement(free, d):
                    be = [0] * 10
                    for k in B: be[k] += 1
                    out.append((h, tuple(al), tuple(be)))
        return out
    def nf(h, al, be):
        mn = [min(x, y) for x, y in zip(al, be)]
        return (h + sum(mn), tuple(x - m for x, m in zip(al, mn)), tuple(y - m for y, m in zip(be, mn)))

    # action of a Weyl element on a monomial: z_k -> mu_k z_{sig k}, w_k -> mu_k^-1 w_{sig k}  (substitution z -> M z)
    def act_mono(act, mono):
        perm, ph = act
        h, al, be = mono
        e = 0; al2 = [0] * 10; be2 = [0] * 10
        for k in range(10):
            if al[k]: al2[perm[k]] += al[k]; e += ph[k] * al[k]
            if be[k]: be2[perm[k]] += be[k]; e -= ph[k] * be[k]
        return (h, tuple(al2), tuple(be2)), e % 20

    def char_subspace(monos, chi):
        """basis (list of {mono: coef}) of the chi-isotypic part, chi = (s, t): A^a B^b acts by zeta5^{s a + t b}"""
        idx = {m: i for i, m in enumerate(monos)}
        seen = set(); vecs = []
        for m in monos:
            if m in seen: continue
            v = {}
            orbit = set()
            for (a, b), act in acts:
                m2, e = act_mono(act, m)
                orbit.add(m2)
                # projector: (1/25) sum_g chi(g)^-1 g.m ; chi(g) = zeta20^{4(s a + t b)}
                ee = (e - 4 * (chi[0] * a + chi[1] * b)) % 20
                v[m2] = (v.get(m2, 0) + zp[ee]) % p
            seen |= orbit
            v = {k: x for k, x in v.items() if x}
            if v: vecs.append(v)
        return vecs

    chars = [(s, t) for s in range(5) for t in range(5)]
    Bn = basis(n); Bn1 = basis(n - 1); B11 = basis(1)
    log("dim R_{%d,%d} = %d, dim R_{%d,%d} = %d, dim R_{1,1} = %d" % (n, n, len(Bn), n - 1, n - 1, len(Bn1), len(B11)))
    # generators as vectors in R_{1,1}
    gens = []
    for g in range(20):
        v = {}
        for k in range(10):
            for l in range(10):
                if k != l:
                    al = [0] * 10; be = [0] * 10; al[k] = 1; be[l] = 1
                    v[(0, tuple(al), tuple(be))] = zp[(E[g, k] - E[g, l]) % 20]
        gens.append(v)
    # isotypic components of span{f_g}: project every generator, keep a linearly independent set per character
    def proj(vec, chi):
        out = {}
        for (a, b), act in acts:
            for m, x in vec.items():
                m2, e = act_mono(act, m)
                ee = (e - 4 * (chi[0] * a + chi[1] * b)) % 20
                out[m2] = (out.get(m2, 0) + x * zp[ee]) % p
        return {k: x for k, x in out.items() if x}
    def indep(vecs):
        keys = sorted({k for v in vecs for k in v})
        if not keys: return []
        kid = {k: i for i, k in enumerate(keys)}
        Mx = np.zeros((len(vecs), len(keys)), dtype=np.int64)
        for i, v in enumerate(vecs):
            for k, x in v.items(): Mx[i, kid[k]] = x
        piv_rows = rank_mod(Mx.copy(), p, return_rows=True)
        return [vecs[i] for i in piv_rows]
    Fchar = {}
    for chi in chars:
        Fchar[chi] = indep([proj(gv, chi) for gv in gens])
    # soundness: every projected generator combination must lie in span{f_g} (W-invariance of the generator span)
    def rank_of(vecs):
        keys = sorted({k for v in vecs for k in v}); kid = {k: i for i, k in enumerate(keys)}
        Mx = np.zeros((len(vecs), len(keys)), dtype=np.int64)
        for i, v in enumerate(vecs):
            for k, x in v.items(): Mx[i, kid[k]] = x
        return rank_mod(Mx, p)
    rg = rank_of(gens); allF = [v for vs in Fchar.values() for v in vs]
    assert rank_of(gens + allF) == rg and len(allF) == rg, "generator span not W-invariant / isotypic split wrong"
    log("generator span: rank %d, W-invariant, isotypic decomposition complete" % rg)
    log("generator isotypic dims:", {chi: len(v) for chi, v in Fchar.items() if v}, "total", sum(len(v) for v in Fchar.values()))
    # invariant target subspace and columns
    inv = char_subspace(Bn, (0, 0))
    log("dim R_{%d,%d}^W = %d" % (n, n, len(inv)))
    sub1 = {chi: char_subspace(Bn1, chi) for chi in chars}
    cols = []
    for psi, Fs in Fchar.items():
        if not Fs: continue
        need = ((-psi[0]) % 5, (-psi[1]) % 5)
        for F in Fs:
            for q in sub1[need]:
                v = {}
                for m1, x1 in F.items():
                    for m2, x2 in q.items():
                        m = nf(m1[0] + m2[0], tuple(a + b for a, b in zip(m1[1], m2[1])), tuple(a + b for a, b in zip(m1[2], m2[2])))
                        v[m] = (v.get(m, 0) + x1 * x2) % p
                cols.append({k: x for k, x in v.items() if x})
    log("columns: %d" % len(cols))
    # express everything in coordinates of the invariant subspace via one representative monomial per invariant vector
    rep_of = []
    for v in inv:
        mrep = min(v); rep_of.append((mrep, v[mrep]))
    rows = len(inv)
    Mx = np.zeros((len(cols), rows), dtype=np.int64)
    for j, v in enumerate(cols):
        for i, (mrep, c0) in enumerate(rep_of):
            x = v.get(mrep, 0)
            if x: Mx[j, i] = x * pow(c0, -1, p) % p
    # consistency: each column must lie in the invariant span (spot-check 20 columns exactly)
    for j in range(len(cols)):                     # every column (audit hardening; was a 20-column spot check)
        recon = {}
        for i, (mrep, c0) in enumerate(rep_of):
            if Mx[j, i]:
                for k, x in inv[i].items():
                    recon[k] = (recon.get(k, 0) + int(Mx[j, i]) * x) % p
        recon = {k: x for k, x in recon.items() if x}
        assert recon == cols[j], "column %d not in the invariant span (construction error)" % j
    # the target c^n must be invariant and in coordinates
    target = (n, tuple([0] * 10), tuple([0] * 10))
    assert any(target in v for v in inv)
    if "--cpp" in sys.argv:                        # dense C++ rank (rankmodp) for the large n=4 systems
        import subprocess, tempfile
        fn = tempfile.mktemp(suffix=".bin", dir=os.environ.get("TMPDIR", "/tmp"))
        A = np.ascontiguousarray(Mx.T % p).astype(np.uint32)      # rows = invariant coordinates
        with open(fn, "wb") as fh:
            fh.write(np.array([A.shape[0], A.shape[1], p], dtype=np.int64).tobytes()); fh.write(A.tobytes())
        log("matrix %d x %d written, running rankmodp" % A.shape)
        out = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "rankmodp"), fn], capture_output=True, text=True).stdout
        os.remove(fn)
        rk = int(out.split()[1]); log(out.strip())
    else:
        rk = rank_mod(Mx, p)
    if rk == rows:                                 # export a square witness: the pivot columns, re-ranked independently
        import hashlib, json, subprocess, tempfile
        piv = rank_mod(Mx.copy(), p, return_rows=True)
        S = np.ascontiguousarray(Mx[piv, :] % p).astype(np.uint32)   # rows x rows, rows = selected columns
        wd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "rank_witness"); os.makedirs(wd, exist_ok=True)
        tag = "%s_n%d" % (args[0].replace(",", "-"), n)
        fn = os.path.join(wd, tag + ".bin")
        with open(fn, "wb") as fh:
            fh.write(np.array([S.shape[0], S.shape[1], p], dtype=np.int64).tobytes()); fh.write(S.tobytes())
        out = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "rankmodp"), fn], capture_output=True, text=True)
        assert out.returncode == 0, "rankmodp failed"
        rk2 = int(out.stdout.split()[1])
        h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
        json.dump({"rep": args[0], "n": n, "p": p, "zeta20": r, "invariant_dim": rows, "columns": len(cols),
                   "selected_columns": [int(x) for x in piv], "square_rank_cpp": rk2, "witness_sha256": h},
                  open(os.path.join(wd, tag + ".json"), "w"), indent=1)
        log("witness: %d x %d square submatrix, independent C++ rank %d, sha256 %s" % (S.shape[0], S.shape[1], rk2, h[:16]))
        assert rk2 == rows, "square witness is singular"
    log("RESULT orbit %s n=%d: rank %d of %d (invariant dim) ; columns %d ; %s" % (
        sys.argv[1], n, rk, rows, len(cols), "FULL RANK -> certificate exists over K" if rk == rows else "not full rank"))

def rank_mod(M, p, return_rows=False):
    """rank of an integer matrix mod p by row reduction (rows are vectors); optionally the pivot row indices"""
    M = M % p
    nr, nc = M.shape
    rk = 0; piv_rows = []
    order = list(range(nr))
    for c in range(nc):
        if rk == nr: break
        nz = np.nonzero(M[rk:, c])[0]
        if len(nz) == 0: continue
        i = rk + nz[0]
        if i != rk:
            M[[rk, i]] = M[[i, rk]]; order[rk], order[i] = order[i], order[rk]
        inv = pow(int(M[rk, c]), -1, p)
        M[rk] = (M[rk] * inv) % p
        below = np.nonzero(M[rk + 1:, c])[0] + rk + 1
        if len(below):
            f = M[below, c].copy()
            # (f * M[rk]) mod p without overflow: split M[rk] into 15-bit halves
            lo = M[rk] & 0x7FFF; hi = M[rk] >> 15
            M[below] = (M[below] - ((f[:, None] * hi[None, :]) % p * 32768 + f[:, None] * lo[None, :]) % p) % p
        piv_rows.append(order[rk]); rk += 1
    return piv_rows if return_rows else rk

if __name__ == "__main__":
    main()
