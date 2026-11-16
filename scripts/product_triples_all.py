#!/usr/bin/env python
"""All mutually unbiased product triples in C^2 (x) C^5, up to symmetry, and evidence on strong unextendibility.

Classification (McNulty-Pammer-Weigert, JMP 57, 032202 (2016), Cor. 3): up to local equivalence every MU product
triple is {B_Z, B_X, B_Y} with B_b = [ |0_b> (x) G_l(b,0) , |1_b> (x) G_l(b,1) ], G_inf = I_5,
G_t(x,j) = w^{t x^2 + j x}/sqrt5 (t in F5), and the label sets {l(b,0), l(b,1)} pairwise disjoint (4080 assignments).

An assignment is a 6-tuple of label indices at positions (Z0, Z1, X0, X1, Y0, Y1); label index 0 = inf, t+1 = t.

Subcommands (run on the server; output dir default runs/ptall):
  enum    [outdir]            enumerate the 4080 assignments, build and verify the symmetry generators, orbits -> orbits.json
  numeric [outdir] [nproc]    per orbit: MU checks, zeta20 encoding check, F_min for every reference basis (rep + 2 members)
  gb      [outdir]            msolve mod-p inputs (3 prime/embedding pairs per rep) + pinned solvable controls -> ms/, jobs.txt
  collect [outdir] [results_prefix]   merge everything -> results_prefix.md / .json
"""
import os, sys, json, itertools, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

W5 = np.exp(2j * np.pi / 5)
Z20 = np.exp(2j * np.pi / 20)
LAB = ["inf", "0", "1", "2", "3", "4"]
POS = ["Z0", "Z1", "X0", "X1", "Y0", "Y1"]
BAS = "ZXY"
TYPES = ["ddd", "ddi", "dii", "iii"]
X5 = np.arange(5)
I5 = np.eye(5, dtype=complex)
I2 = np.eye(2, dtype=complex)
S2 = np.sqrt(2.0)


def G(lab):
    if lab == 0:
        return I5.copy()
    t = lab - 1
    return W5 ** ((t * X5[:, None] ** 2 + X5[:, None] * X5[None, :]) % 5) / np.sqrt(5)


Q = [np.array([1, 0], complex), np.array([0, 1], complex),
     np.array([1, 1], complex) / S2, np.array([1, -1], complex) / S2,
     np.array([1, 1j], complex) / S2, np.array([1, -1j], complex) / S2]


def basis(ell, b):
    return np.hstack([np.kron(Q[2 * b][:, None], G(ell[2 * b])), np.kron(Q[2 * b + 1][:, None], G(ell[2 * b + 1]))])


def triple(ell):
    return [basis(ell, b) for b in range(3)]


def lname(ell):
    return " ".join("%s:%s,%s" % (BAS[b], LAB[ell[2 * b]], LAB[ell[2 * b + 1]]) for b in range(3))


def ttype(ell):
    return TYPES[sum(ell[2 * b] != ell[2 * b + 1] for b in range(3))]


def all_assignments():
    out = []
    for ell in itertools.product(range(6), repeat=6):
        s = [set(ell[0:2]), set(ell[2:4]), set(ell[4:6])]
        if s[0] & s[1] or s[0] & s[2] or s[1] & s[2]:
            continue
        out.append(ell)
    return out


def abs_is_perm(A, tol=1e-10):
    a = np.abs(A)
    if not np.all((a < tol) | (np.abs(a - 1) < tol)):
        return False
    r = np.rint(a)
    return bool(np.all(r.sum(0) == 1) and np.all(r.sum(1) == 1))


def same_basis(A, B):
    return abs_is_perm(B.conj().T @ A)


# ---------------------------------------------------------------- symmetry generators
def qubit_perm(f):
    """f maps a qubit vector to a qubit vector (unitary or antiunitary); returns sigma with f(Q[p]) ~ Q[sigma[p]]."""
    sig = []
    for p in range(6):
        v = f(Q[p])
        hits = [q for q in range(6) if abs(abs(np.vdot(Q[q], v)) - 1) < 1e-12]
        assert len(hits) == 1, (p, hits)
        sig.append(hits[0])
    assert sorted(sig) == list(range(6))
    return sig


def c5_perm(f):
    """f maps a 5x5 basis matrix to a 5x5 matrix; returns pi with f(G(a)) = G(pi[a]) up to column perm + phases."""
    pi = []
    for a in range(6):
        A = f(G(a))
        hits = [b for b in range(6) if same_basis(A, G(b))]
        assert len(hits) == 1, (a, hits)
        pi.append(hits[0])
    assert sorted(pi) == list(range(6))
    return pi


def generators():
    H = np.array([[1, 1], [1, -1]], complex) / S2
    S = np.diag([1, 1j])
    F5 = W5 ** np.outer(X5, X5) / np.sqrt(5)
    P5 = np.diag(W5 ** (X5 ** 2 % 5))
    M2 = np.zeros((5, 5), complex)
    for x in range(5):
        M2[(2 * x) % 5, x] = 1
    ident6 = list(range(6))
    gens = []
    for name, V in (("qubit H", H), ("qubit S", S)):
        assert np.abs(V.conj().T @ V - I2).max() < 1e-14
        gens.append(dict(name=name, sigma=qubit_perm(lambda v, V=V: V @ v), pi=ident6,
                         op=(lambda Bs, V=V: [np.kron(V, I5) @ B for B in Bs])))
    for name, Wm in (("C5 F (DFT)", F5), ("C5 P=diag(w^{x^2})", P5), ("C5 M2: |x>->|2x>", M2)):
        assert np.abs(Wm.conj().T @ Wm - I5).max() < 1e-14
        gens.append(dict(name=name, sigma=ident6, pi=c5_perm(lambda A, Wm=Wm: Wm @ A),
                         op=(lambda Bs, Wm=Wm: [np.kron(I2, Wm) @ B for B in Bs])))
    gens.append(dict(name="complex conjugation K", sigma=qubit_perm(lambda v: v.conj()), pi=c5_perm(lambda A: A.conj()),
                     op=(lambda Bs: [B.conj() for B in Bs])))
    return gens


def act(g, ell):
    new = [None] * 6
    for p in range(6):
        new[g["sigma"][p]] = g["pi"][ell[p]]
    return tuple(new)


def verify_gen(g, ell):
    """transformed triple must equal the triple of the relabelled assignment, basis-by-basis bijectively."""
    A = g["op"](triple(ell))
    T = triple(act(g, ell))
    used = []
    for Ab in A:
        hits = [c for c in range(3) if same_basis(Ab, T[c])]
        if len(hits) != 1:
            return False
        used.append(hits[0])
    return sorted(used) == [0, 1, 2]


def perm_group_order(perms):
    seen = {tuple(range(len(perms[0])))}
    frontier = list(seen)
    while frontier:
        nf = []
        for h in frontier:
            for g in perms:
                c = tuple(g[h[i]] for i in range(len(h)))
                if c not in seen:
                    seen.add(c)
                    nf.append(c)
        frontier = nf
    return len(seen)


def cmd_enum(outdir):
    os.makedirs(outdir, exist_ok=True)
    A = all_assignments()
    print("assignments:", len(A))
    cnt = {}
    for ell in A:
        cnt[ttype(ell)] = cnt.get(ttype(ell), 0) + 1
    print("by type:", cnt)
    gens = generators()
    idx = {ell: i for i, ell in enumerate(A)}
    gen_info = []
    t0 = time.time()
    for g in gens:
        bad = 0
        for ell in A:
            if act(g, ell) not in idx or not verify_gen(g, ell):
                bad += 1
        print("generator %-22s sigma(pos)=%s pi(lab)=%s : verified on %d assignments, failures %d" % (
            g["name"], [POS[s] for s in g["sigma"]], [LAB[s] for s in g["pi"]], len(A), bad))
        assert bad == 0
        gen_info.append(dict(name=g["name"], sigma=[POS[s] for s in g["sigma"]], pi=[LAB[s] for s in g["pi"]],
                             verified_on=len(A), failures=bad))
    print("verification time %.1fs" % (time.time() - t0))
    qorder = perm_group_order([g["sigma"] for g in gens[:2]])
    corder = perm_group_order([g["pi"] for g in gens[2:5]])
    print("qubit Clifford group on the 6 stabilizer states: order %d; C5 Clifford group on the 6 labels: order %d" % (qorder, corder))
    # union-find
    par = list(range(len(A)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i, ell in enumerate(A):
        for g in gens:
            a, b = find(i), find(idx[act(g, ell)])
            if a != b:
                par[a] = b
    orb = {}
    for i, ell in enumerate(A):
        orb.setdefault(find(i), []).append(ell)
    orbits = sorted((sorted(m) for m in orb.values()), key=lambda m: (TYPES.index(ttype(m[0])), m[0]))
    out = []
    for k, m in enumerate(orbits):
        types = {ttype(e) for e in m}
        assert len(types) == 1
        out.append(dict(id=k, rep=list(m[0]), rep_name=lname(m[0]), type=ttype(m[0]), size=len(m),
                        members=[list(e) for e in m]))
        print("orbit %2d  %-28s %s size %d" % (k, lname(m[0]), ttype(m[0]), len(m)))
    assert sum(o["size"] for o in out) == len(A)
    json.dump(dict(n_assignments=len(A), by_type=cnt, generators=gen_info, qubit_group_order=qorder,
                   c5_group_order=corder, orbits=out), open(os.path.join(outdir, "orbits.json"), "w"))
    print("orbits: %d -> %s" % (len(out), os.path.join(outdir, "orbits.json")))


# ---------------------------------------------------------------- numerics
def checks(ell):
    Bs = triple(ell)
    uerr = max(np.abs(B.conj().T @ B - np.eye(10)).max() for B in Bs)
    mu = max(np.abs(np.abs(Bs[a].conj().T @ Bs[b]) ** 2 - 0.1).max() for a, b in ((0, 1), (0, 2), (1, 2)))
    M = np.vstack([Bs[1].conj().T @ Bs[0], Bs[2].conj().T @ Bs[0]]) * np.sqrt(10)
    E = np.rint(np.angle(M) / (2 * np.pi / 20)).astype(int) % 20
    zerr = np.abs(M - Z20 ** E).max()
    return dict(unitary_err=float(uerr), mu_err=float(mu), zeta20_err=float(zerr)), E


def fmin_task(args):
    from scipy.optimize import minimize
    ell, ref, nstarts, seed = args
    Bs = triple(tuple(ell))
    U0 = Bs[ref]
    M = np.vstack([Bs[i].conj().T @ U0 for i in range(3) if i != ref])
    s10 = np.sqrt(10)

    def fg(th):
        x = np.concatenate([[1.0 + 0j], np.exp(1j * th)])
        amp = M @ x / s10
        r = 10 * np.abs(amp) ** 2 - 1
        D = (M[:, 1:] * (1j * np.exp(1j * th))[None, :]) / s10
        dp = 2 * np.real(np.conj(amp)[:, None] * D)
        return (r ** 2).sum(), (20 * r) @ dp

    rng = np.random.default_rng(seed)
    vals, best = [], (np.inf, None)
    for _ in range(nstarts):
        res = minimize(fg, rng.uniform(0, 2 * np.pi, 9), jac=True, method="BFGS", options={"gtol": 1e-12, "maxiter": 3000})
        vals.append(float(res.fun))
        if res.fun < best[0]:
            best = (float(res.fun), res.x)
    vals = np.sort(vals)
    x = np.concatenate([[1.0 + 0j], np.exp(1j * best[1])])
    dev = float(np.abs(10 * np.abs(M @ x / s10) ** 2 - 1).max())
    u, c = np.unique(np.round(vals, 6), return_counts=True)
    n = len(vals)
    return dict(ell=list(ell), ref=BAS[ref], nstarts=nstarts, seed=seed, Fmin=float(vals[0]),
                q01=float(vals[int(0.01 * n)]), q10=float(vals[int(0.1 * n)]), q50=float(vals[n // 2]),
                hits_min=int(np.sum(vals < vals[0] + 1e-6)), maxdev_at_best=dev,
                distinct_minima=[[float(a), int(b)] for a, b in zip(u[:8], c[:8])],
                best_theta=[float(t) for t in best[1]])


def cmd_numeric(outdir, nproc):
    from concurrent.futures import ProcessPoolExecutor
    O = json.load(open(os.path.join(outdir, "orbits.json")))
    rng = np.random.default_rng(20261115)
    tasks, meta = [], []
    chk = {}
    for o in O["orbits"]:
        rep = tuple(o["rep"])
        c, _ = checks(rep)
        chk[o["id"]] = c
        others = [tuple(m) for m in o["members"] if tuple(m) != rep]
        pick = [others[i] for i in rng.choice(len(others), size=min(2, len(others)), replace=False)] if others else []
        for role, ell, ns in [("rep", rep, 2000)] + [("member", e, 500) for e in pick]:
            for ref in range(3):
                tasks.append((list(ell), ref, ns, int(rng.integers(1 << 30))))
                meta.append((o["id"], role))
    print("orbits %d, F_min tasks %d, workers %d" % (len(O["orbits"]), len(tasks), nproc), flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(nproc) as ex:
        for k, r in enumerate(ex.map(fmin_task, tasks)):
            r["orbit"], r["role"] = meta[k]
            res.append(r)
            if (k + 1) % 20 == 0:
                print("  %d/%d done, %.0fs" % (k + 1, len(tasks), time.time() - t0), flush=True)
    json.dump(dict(checks=chk, fmin=res), open(os.path.join(outdir, "numeric.json"), "w"))
    print("numeric done in %.0fs" % (time.time() - t0))


# ---------------------------------------------------------------- msolve inputs (mod p)
def is_prime(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_1mod20(below, count):
    n = below - 1
    n -= (n - 1) % 20
    out = []
    while len(out) < count:
        if is_prime(n):
            out.append(n)
        n -= 20
    return out


def prim20(p):
    for a in range(2, 10000):
        r = pow(a, (p - 1) // 20, p)
        if pow(r, 10, p) != 1 and pow(r, 4, p) != 1:
            return r
    raise ValueError(p)


def embeddings():
    ps = primes_1mod20(2 ** 30, 4)
    ks = [1, 3, 7, 9]
    return [dict(tag="r%d" % i, p=p, r0=prim20(p), k=k, r=pow(prim20(p), k, p)) for i, (p, k) in enumerate(zip(ps, ks))]


ZS = ["z%d" % k for k in range(1, 10)]
WS = ["w%d" % k for k in range(1, 10)]


def coeff_table(E, j, pw):
    """list of (k, l, coeff) for the equation of row j (constant (0,0) term excluded)."""
    return [(k, l, pw[(E[j, k] - E[j, l]) % 20]) for k in range(10) for l in range(10) if (k, l) != (0, 0)]


def ms_text(E, p, r, rows, pins=()):
    pw = [pow(r, e, p) for e in range(20)]
    polys = ["%s*%s-1" % (ZS[k], WS[k]) for k in range(9)]
    for j in rows:
        terms = []
        for k, l, c in coeff_table(E, j, pw):
            mon = "*".join(v for v in ((ZS[k - 1] if k else ""), (WS[l - 1] if l else "")) if v)
            terms.append(mon if c == 1 else "%d*%s" % (c, mon))
        polys.append("+".join(terms) + "-9")          # constant LAST (msolve 0.7.3 misparses a leading constant, N7)
    for k, f in pins:
        polys.append("%s-%d" % (ZS[k - 1], pw[f % 20]))
    for s in polys:
        assert not s[0].isdigit() or "*" in s.split("+")[0], s
    return ",".join(ZS + WS) + "\n%d\n" % p + ",\n".join(polys) + "\n"


def modp_residuals(E, p, r, f):
    """evaluate the 20 equations mod p at z_k = r^{f_k}, w_k = r^{-f_k} (f_0 = 0)."""
    pw = [pow(r, e, p) for e in range(20)]
    zv = [1] + [pw[f[k] % 20] for k in range(1, 10)]
    wv = [1] + [pw[(-f[k]) % 20] for k in range(1, 10)]
    out = []
    for j in range(20):
        s = 1 - 10                                   # (0,0) term is 1, constant -10
        for k, l, c in coeff_table(E, j, pw):
            s += c * zv[k] * wv[l]
        out.append(s % p)
    return out


def cmd_gb(outdir):
    O = json.load(open(os.path.join(outdir, "orbits.json")))
    msdir = os.path.join(outdir, "ms")
    os.makedirs(msdir, exist_ok=True)
    EMB = embeddings()
    print("embeddings:", EMB)
    jobs, info = [], {}
    for o in O["orbits"]:
        rep = tuple(o["rep"])
        c, E = checks(rep)
        assert c["zeta20_err"] < 1e-12 and c["mu_err"] < 1e-12, c
        Bs = triple(rep)
        # pinned control: column col of U2 (= B_Y) is unbiased to U0, U1; x = sqrt10 U0^* v normalised so x_0 = 1
        col = 0
        x = np.sqrt(10) * Bs[0].conj().T @ Bs[2][:, col]
        x = x / x[0]
        f = (np.rint(np.angle(x) / (2 * np.pi / 20)).astype(int) % 20).tolist()
        assert np.abs(x - Z20 ** np.array(f)).max() < 1e-12
        assert f == [(E[10 + col, 0] - E[10 + col, k]) % 20 for k in range(10)]
        # complex check of the exact system at x: U1 rows equal 10, U2 row col equals 100
        val = np.array([sum(Z20 ** ((E[j, k] - E[j, l]) % 20) * x[k] * np.conj(x[l]) for k in range(10) for l in range(10))
                        for j in range(20)])
        assert np.abs(val[:10] - 10).max() < 1e-9 and abs(val[10 + col] - 100) < 1e-9
        info[o["id"]] = dict(pins=[[k, f[k]] for k in range(1, 5)], control_column=col)
        for e in EMB:
            resid = modp_residuals(E, e["p"], e["r"], f)
            assert all(v == 0 for v in resid[:10]), resid     # U0+U1 equations vanish mod p at the U2-column point
            assert resid[10 + col] == 90 % e["p"]
            name = "o%02d_%s" % (o["id"], e["tag"])
            fn = os.path.join(msdir, name + ".ms")
            open(fn, "w").write(ms_text(E, e["p"], e["r"], range(20)))
            jobs.append((name, fn))
            cname = "o%02d_ctrl_%s" % (o["id"], e["tag"])
            fn = os.path.join(msdir, cname + ".ms")
            open(fn, "w").write(ms_text(E, e["p"], e["r"], range(10), pins=[(k, f[k]) for k in range(1, 5)]))
            jobs.append((cname, fn))
    # order: all triple runs for r0, r1, r2 first, then controls r0, then the rest
    def prio(j):
        n = j[0]
        tag = int(n[-1])
        ctrl = "_ctrl_" in n
        return (0 if not ctrl and tag <= 2 else 1 if ctrl and tag == 0 else 2 if ctrl and tag <= 2 else 3, n)
    jobs.sort(key=prio)
    with open(os.path.join(outdir, "jobs.txt"), "w") as fh:
        for name, fn in jobs:
            fh.write("%s %s %s %s\n" % (name, fn, fn[:-3] + ".out", fn[:-3] + ".log"))
    json.dump(dict(embeddings=EMB, controls=info), open(os.path.join(outdir, "gb_info.json"), "w"))
    print("wrote %d msolve inputs; job list %s" % (len(jobs), os.path.join(outdir, "jobs.txt")))


# ---------------------------------------------------------------- collect
def parse_ms(out_fn):
    if not os.path.exists(out_fn):
        return dict(result="MISSING")
    s = open(out_fn).read().strip()
    if not s:
        return dict(result="EMPTY")
    if s.startswith("[-1]"):
        return dict(result="[-1]")
    if s.startswith("[1,"):
        return dict(result="POSDIM", raw=s[:80])
    if s.startswith("[0,"):
        toks = s.replace("[", " ").replace("]", " ").replace(",", " ").split()
        try:
            if toks[1] == "1":                      # [0, [1, [] ]] : over Q, no real solutions (should not occur mod p)
                return dict(result="SOL", raw=s[:80])
            return dict(result="SOL", deg=int(toks[3]))
        except Exception:
            return dict(result="SOL", raw=s[:80])
    return dict(result="OTHER", raw=s[:80])


def fmt(x):
    return "%.4g" % x


def galois_map(O):
    """Galois automorphism zeta20 -> zeta20^17 (omega -> omega^2, i fixed) sends G_t to G_2t (up to column perm and a sign)
    and fixes the qubit states, so it maps the triple of l to the triple of 2l (label-wise).  Check on the exponent level that the
    complexified system of the image is exactly the conjugate system: every row of 17*E(rep) equals a row of E(image) + const."""
    where = {tuple(m): o["id"] for o in O["orbits"] for m in o["members"]}
    mul2 = [0] + [1 + (2 * t) % 5 for t in range(5)]
    out = []
    for o in O["orbits"]:
        rep = tuple(o["rep"])
        img = tuple(mul2[a] for a in rep)
        _, E = checks(rep)
        _, E2 = checks(img)
        # sigma(G_t) = -G_2t with columns j -> 2j (sigma(sqrt5) = -sqrt5); sigma(I) = I; qubit states fixed (sqrt2 fixed).
        # Hence U0 column (h, j) -> (h, 2j) with sign -1 (exponent +10) if label l(Z,h) is finite.
        tau = [h * 5 + ((2 * j) % 5 if rep[h] else j) for h in range(2) for j in range(5)]
        bsh = np.array([10 if rep[h] else 0 for h in range(2) for j in range(5)])
        G17 = (17 * E + bsh[None, :]) % 20
        ok = all(any(len(set(((E2[jj, tau] - G17[j]) % 20).tolist())) == 1 for jj in range(20)) for j in range(20))
        out.append(dict(id=o["id"], image=lname(img), image_orbit=where[img], exponent_rows_match=bool(ok)))
    return out


def cmd_collect(outdir, prefix):
    O = json.load(open(os.path.join(outdir, "orbits.json")))
    N = json.load(open(os.path.join(outdir, "numeric.json"))) if os.path.exists(os.path.join(outdir, "numeric.json")) else None
    GI = json.load(open(os.path.join(outdir, "gb_info.json")))
    status = {}
    sf = os.path.join(outdir, "jobs.txt.status.jsonl")
    if os.path.exists(sf):
        for line in open(sf):
            d = json.loads(line)
            status[d["name"]] = d
    EMB = GI["embeddings"]
    rows = []
    for o in O["orbits"]:
        i = o["id"]
        row = dict(id=i, rep=[LAB[a] for a in o["rep"]], rep_name=o["rep_name"], type=o["type"], size=o["size"])
        if N:
            row["checks"] = N["checks"][str(i)]
            fr = [r for r in N["fmin"] if r["orbit"] == i]
            rep = [r for r in fr if r["role"] == "rep"]
            row["fmin_rep"] = {r["ref"]: {k: r[k] for k in ("Fmin", "q10", "q50", "hits_min", "nstarts", "maxdev_at_best",
                                                             "distinct_minima")} for r in rep}
            sig_rep = sorted(r["Fmin"] for r in rep)
            val = []
            for ell in sorted({tuple(r["ell"]) for r in fr if r["role"] == "member"}):
                mm = sorted(r["Fmin"] for r in fr if r["role"] == "member" and tuple(r["ell"]) == ell)
                val.append(dict(member=lname(ell), sorted_Fmin=mm,
                                agree=bool(np.allclose(mm, sig_rep, atol=1e-5, rtol=0))))
            row["fmin_orbit_validation"] = dict(rep_sorted_Fmin=sig_rep, members=val)
        ms = {}
        for e in EMB:
            for kind, name in (("triple", "o%02d_%s" % (i, e["tag"])), ("control", "o%02d_ctrl_%s" % (i, e["tag"]))):
                st = status.get(name)
                d = parse_ms(os.path.join(outdir, "ms", name + ".out")) if st else dict(result="NOT RUN")
                if st:
                    d.update(rc=st["rc"], wall=round(st["wall"], 1), maxrss_mb=st.get("maxrss_mb"))
                    if st["rc"] == 124:
                        d["result"] = "TIMEOUT"
                    elif st["rc"] != 0 and d["result"] != "[-1]":
                        d["result"] = "ERROR(rc=%d)" % st["rc"] if d["result"] in ("EMPTY", "MISSING") else d["result"]
                    elif st["rc"] != 0:
                        d["result"] = "ERROR(rc=%d)" % st["rc"]       # never trust [-1] from a failed process
                d.update(p=e["p"], r=e["r"], k=e["k"])
                ms.setdefault(kind, {})[e["tag"]] = d
        row["msolve"] = ms
        row["control_pins"] = GI["controls"][str(i)]
        rows.append(row)
    summ = dict(n_orbits=len(rows),
                n_triple_minus1_all_run=sum(1 for r in rows if r["msolve"]["triple"] and all(
                    v["result"] == "[-1]" for v in r["msolve"]["triple"].values() if v["result"] != "NOT RUN") and any(
                    v["result"] != "NOT RUN" for v in r["msolve"]["triple"].values())))
    ctrl_all = [v["result"] for r in rows for v in r["msolve"]["control"].values() if v["result"] != "NOT RUN"]
    trip_all = [v["result"] for r in rows for v in r["msolve"]["triple"].values() if v["result"] != "NOT RUN"]
    summ.update(n_triple_runs=len(trip_all), n_triple_runs_minus1=trip_all.count("[-1]"),
                n_control_runs=len(ctrl_all), n_control_runs_SOL=ctrl_all.count("SOL"),
                n_control_runs_minus1=ctrl_all.count("[-1]"))
    if N:
        summ["min_Fmin_over_orbits_and_refs"] = min(r["fmin_rep"][b]["Fmin"] for r in rows for b in "ZXY")
    GAL = galois_map(O)
    out = dict(summary=summ, n_assignments=O["n_assignments"], by_type=O["by_type"], generators=O["generators"],
               qubit_group_order=O["qubit_group_order"], c5_group_order=O["c5_group_order"], embeddings=EMB,
               galois_zeta20_to_zeta20_17=GAL, orbits=rows)
    json.dump(out, open(prefix + ".json", "w"), indent=1)
    # markdown
    L = []
    L.append("# Mutually unbiased product triples in C^2 (x) C^5: orbit scan\n")
    L.append("Generated by `scripts/product_triples_all.py collect` from `%s` on %s.\n" % (outdir, time.strftime("%Y-%m-%d %H:%M")))
    L.append("Assignments: %d (by type %s). Orbits: %d.\n" % (O["n_assignments"], O["by_type"], len(rows)))
    L.append("Labels: inf = I_5, t = G_t (t in F5). Basis b in {Z,X,Y}: `b:l(b,0),l(b,1)` means [|0_b>(x)G_l(b,0), |1_b>(x)G_l(b,1)].\n")
    L.append("\n## Summary\n")
    L.append("- Orbits: %d. msolve triple runs: %d, of which `[-1]`: %d. Orbits with `[-1]` for every tested prime/embedding: %d/%d." % (
        len(rows), summ["n_triple_runs"], summ["n_triple_runs_minus1"], summ["n_triple_minus1_all_run"], len(rows)))
    L.append("- Pinned solvable controls: %d runs, %d returned a solution parametrization, %d returned `[-1]` (must be 0)." % (
        summ["n_control_runs"], summ["n_control_runs_SOL"], summ["n_control_runs_minus1"]))
    if N:
        L.append("- Smallest numerical F_min over all orbits and reference bases: %.6g (F = 0 iff a common unbiased vector exists)." %
                 summ["min_Fmin_over_orbits_and_refs"])
    L.append("- Status: evidence only. A modular `[-1]` is not a proof over Q(zeta20); F_min > 0 from random starts is not a proof either.\n")
    L.append("msolve embeddings (t -> r = r0^k mod p): " + "; ".join("%s: p=%d, r0=%d, k=%d, r=%d" % (e["tag"], e["p"], e["r0"], e["k"], e["r"]) for e in EMB) + "\n")
    L.append("\n## Orbit table\n")
    L.append("F_min(ref) = min over vectors unbiased to basis `ref` of sum over the other 20 vectors of (10p-1)^2 (representative: 2000 random BFGS starts per ref; orbit-check members: 500 per ref). "
             "msolve column: result per embedding (`[-1]` = no solution over the algebraic closure of F_p). Control: pinned solvable system (U0+U1 equations + z1..z4 pinned to a column of U2); must NOT be `[-1]`.\n")
    hdr = "| # | Z: l0,l1 | X: l0,l1 | Y: l0,l1 | type | size | F_min ref Z | F_min ref X | F_min ref Y | orbit F check | " + \
          " | ".join("msolve %s" % e["tag"] for e in EMB) + " | controls |"
    L.append(hdr)
    L.append("|" + "---|" * (hdr.count("|") - 1))
    for r in rows:
        lab = r["rep"]
        fz = fx = fy = chk = "-"
        if "fmin_rep" in r:
            fz, fx, fy = (fmt(r["fmin_rep"][b]["Fmin"]) for b in "ZXY")
            vs = r["fmin_orbit_validation"]["members"]
            chk = "%d/%d agree" % (sum(v["agree"] for v in vs), len(vs)) if vs else "n/a (size 1)"
        mcells = []
        for e in EMB:
            d = r["msolve"]["triple"][e["tag"]]
            mcells.append(d["result"] + ("" if d["result"] == "NOT RUN" else " (%ss)" % d.get("wall")) +
                          (" deg %d" % d["deg"] if "deg" in d else ""))
        cc = []
        for e in EMB:
            d = r["msolve"]["control"][e["tag"]]
            if d["result"] != "NOT RUN":
                cc.append("%s:%s%s" % (e["tag"], d["result"], " deg %d" % d["deg"] if "deg" in d else ""))
        L.append("| %d | %s,%s | %s,%s | %s,%s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
            r["id"], lab[0], lab[1], lab[2], lab[3], lab[4], lab[5], r["type"], r["size"], fz, fx, fy, chk,
            " | ".join(mcells), "; ".join(cc) if cc else "-"))
    if N:
        L.append("\n## Numerical details (representatives, ref Z)\n")
        L.append("| # | checks (unitary / MU / zeta20 err) | F_min | hits within 1e-6 of min | q10 | q50 | max abs(10p-1) at best | first distinct local minima (value x count) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            c = r["checks"]
            z = r["fmin_rep"]["Z"]
            L.append("| %d | %.1e / %.1e / %.1e | %.6g | %d/%d | %.4g | %.4g | %.3g | %s |" % (
                r["id"], c["unitary_err"], c["mu_err"], c["zeta20_err"], z["Fmin"], z["hits_min"], z["nstarts"], z["q10"],
                z["q50"], z["maxdev_at_best"], ", ".join("%.6g x%d" % (a, b) for a, b in z["distinct_minima"][:5])))
    L.append("\n## Symmetries used and their validation\n")
    for g in O["generators"]:
        L.append("- **%s**: positions (Z0,Z1,X0,X1,Y0,Y1) -> %s; labels (inf,0,1,2,3,4) -> %s; verified on all %d assignments "
                 "(transformed triple equals the relabelled triple up to column permutations and phases, bijectively on the 3 bases), failures %d." % (
                     g["name"], ",".join(g["sigma"]), ",".join(g["pi"]), g["verified_on"], g["failures"]))
    L.append("- Group orders: qubit Clifford (H,S) acts on the 6 stabilizer states as a group of order %d (octahedral rotations; "
             "it induces every permutation of the three bases Z,X,Y, so permuting the bases of the triple is included); "
             "C5 Clifford (F, P, M2) acts on the 6 labels as a group of order %d. Complex conjugation K (antiunitary) is added; "
             "it maps common unbiased vectors to common unbiased vectors." % (O["qubit_group_order"], O["c5_group_order"]))
    if N:
        nag = sum(v["agree"] for r in rows for v in r["fmin_orbit_validation"]["members"])
        ntot = sum(len(r["fmin_orbit_validation"]["members"]) for r in rows)
        L.append("- F_min orbit check: for every orbit, up to 2 random other members; the sorted triple (F_min ref Z, ref X, ref Y) "
                 "of each member must equal that of the representative within 1e-5 (a symmetry may move the reference basis). "
                 "Agreement: %d/%d." % (nag, ntot))
    L.append("\n## Galois conjugation (remark on independence of the msolve evidence)\n")
    L.append("The automorphism zeta20 -> zeta20^17 of Q(zeta20) (omega -> omega^2, i fixed) maps G_t to G_2t up to column permutation and sign and "
             "fixes the six qubit states, so it maps the triple of l to the triple of 2l. On the exponent level (checked below) the complexified "
             "system of the image is exactly the Galois conjugate of the system of the representative. Emptiness over the algebraic closure is "
             "Galois invariant (the check allows the predicted permutation/sign of the U0 columns and a constant per row), so Galois-conjugate orbits must get the same exact verdict, and running orbit A with embedding r^17 is the same "
             "computation as running its image with r. The real question (w = conj z, F_min) is NOT Galois invariant: the F_min values of "
             "conjugate orbits differ.\n")
    L.append("| orbit | image under t -> 2t | image orbit | exponent rows match |")
    L.append("|---|---|---|---|")
    for g in GAL:
        L.append("| %d | %s | %d | %s |" % (g["id"], g["image"], g["image_orbit"], g["exponent_rows_match"]))
    L.append("\n## Reproduce (on school-server, in ~/mub10)\n")
    L.append("```")
    L.append("PY=~/miniforge3/envs/mub10/bin/python")
    L.append("$PY -I scripts/product_triples_all.py enum %s        # assignments, symmetry verification, orbits" % outdir)
    L.append("$PY -I scripts/product_triples_all.py numeric %s 8   # checks + F_min (about 75 s with 8 workers)" % outdir)
    L.append("$PY -I scripts/product_triples_all.py gb %s          # msolve mod-p inputs + pinned controls, %s/jobs.txt" % (outdir, outdir))
    L.append("$PY -I scripts/product_triples_runms.py %s/jobs.txt 4 16   # msolve -t 2, ulimit -v 12000000, timeout 1200, <=4 parallel (about 3 min)" % outdir)
    L.append("$PY -I scripts/product_triples_all.py collect %s %s" % (outdir, prefix))
    L.append("```")
    L.append("Inputs/outputs: `%s/ms/oNN_rK.{ms,out,log}` (triple), `%s/ms/oNN_ctrl_rK.{ms,out,log}` (control), run status `%s/jobs.txt.status.jsonl`." % (outdir, outdir, outdir))
    L.append("Cross-check: `%s/ms/o08_r0.ms` and `o08_ctrl_r0.ms` (orbit 8 = the fully indirect triple of E46/E48) are byte-identical to the earlier "
             "`runs/gb/triple2_modp.ms` and `runs/gb/ctrl3_modp.ms`." % outdir)
    open(prefix + ".md", "w").write("\n".join(L) + "\n")
    print("wrote %s.md / .json" % prefix)


if __name__ == "__main__":
    cmd = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "runs/ptall"
    if cmd == "enum":
        cmd_enum(outdir)
    elif cmd == "numeric":
        cmd_numeric(outdir, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    elif cmd == "gb":
        cmd_gb(outdir)
    elif cmd == "collect":
        cmd_collect(outdir, sys.argv[3] if len(sys.argv) > 3 else "results/product_triples_scan")
    else:
        raise SystemExit(__doc__)
