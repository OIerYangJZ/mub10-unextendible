#!/usr/bin/env python
"""Exact (integer) verification of the inputs of the MU-product-triple proof in C^2 (x) C^5.

All checks marked EXACT use only Python integers: elements of Z[zeta_N] (N = 20 or 40) are integer coefficient
vectors modulo the cyclotomic polynomial Phi_N, which is computed (not typed in) and asserted to be
Phi_20 = x^8 - x^6 + x^4 - x^2 + 1 and Phi_40 = x^16 - x^12 + x^8 - x^4 + 1.  Standard complex embedding
zeta_N -> exp(2 pi i / N); then zeta_40^2 = zeta_20, zeta_40^8 = w = zeta_5, zeta_40^10 = i, and
    sqrt2 = zeta_8 + zeta_8^-1 = 2 cos(pi/4) > 0,     sqrt5 = 1 + 2 (zeta_5 + zeta_5^4) = 1 + 4 cos(2 pi/5) > 0,
so these exact elements are the positive square roots used by numpy (np.sqrt) in the proof scripts.
Complex conjugation is the ring automorphism zeta -> zeta^-1.

Scaled exact basis matrices (no denominators):  N_b = sqrt10 * U_b,  columns  Q2[p] (x) G5[l](:, j)  with
    Q2 = sqrt2 * (|0>, |1>, |+>, |->, |+i>, |-i>) = (sqrt2,0), (0,sqrt2), (1,1), (1,-1), (1,i), (1,-i)
    G5[inf] = sqrt5 * I_5,   G5[t](x, j) = w^(t x^2 + j x)   (= sqrt5 * G_t of product_triples_all.py).

Part A (10 orbit representatives, scripts/orbit_reps.txt)
  A1 EXACT  N_b^* N_b = 10 I (b = 0,1,2) and every entry s of N_a^* N_b (a < b) has s * conj(s) = 10   (unitary, MU)
  A2 EXACT  full-matrix identity in Z[zeta_40]:  N_b^* N_0 = sqrt10 * zeta_20^E   (b = 1, 2; 200 entries), i.e.
            sqrt10 * [U1^* U0 ; U2^* U0] = zeta_20^E, with E the floating-point table used by the proof scripts;
            E_exact (the unique e in 0..19 for each entry) is also derived independently and compared.
  A3 EXACT  factorised identity in Z[zeta_20]: (sqrt10 * entry) * zeta_20^-E = 1 with
            sqrt10 * entry = (sqrt2 <q'|q>) * (sqrt5 <g'|g>), the qubit factor a unit in {+-1, +-i} and the C^5
            factor w^(...) or the exact Gauss sum S(a,b) = sum_y w^(a y^2 + b y) divided by sqrt5 (S * sqrt5 / 5,
            divisibility by 5 asserted).
  A4        the float tables come from the proof code itself: product_triples_all.checks, product_triple_charts.exponents,
            rank_cert.zeta_exp (numpy; this is the only floating-point part, used only to obtain the table under test).
  A5 EXACT  every chart file runs/charts_all/oNN/{affine,b_i_j}.ms (the generators the Singular certificates were
            verified against) equals the chart system rebuilt here from E_exact with an independent encoder
            (zeta_20 -> r mod p, r read from oNN.gen.txt and checked to be a root of Phi_20 mod p).
Part B (symmetry generators of the orbit reduction; generators() of product_triples_all.py)
  B1 EXACT  generator matrices: H = (sqrt2/2)[[1,1],[1,-1]], S = diag(1,i), F = (sqrt5/5) w^(xy), P = diag(w^(x^2)),
            M2: |x> -> |2x>, and K (complex conjugation); num^* num = den^2 I.
  B2 EXACT  induced permutations sigma (6 qubit states) and pi (6 labels) derived exactly; compared with the numeric
            ones of product_triples_all.generators().
  B3 EXACT  for every generator g and every basis B = [Q2[2b] (x) G5[l0], Q2[2b+1] (x) G5[l1]] (3 x 36 = 108 bases):
            A = g(N_B) satisfies  T^* A = 10 den M  with M monomial, |M_ij| = 1 (M M^* = I exactly), and
            10 A = T (T^* A) entrywise, where T = N of the relabelled basis (positions via sigma, labels via pi);
            T^* T = 10 I is checked for all 108 bases.  Hence g(U_B) = U_T * (column permutation x column phases).
  B4        combinatorial: all 4080 assignments, sigma maps position pairs to position pairs bijectively, every
            relabelled assignment is again one of the 4080, orbits recomputed from the exact permutations and compared
            with scripts/orbit_reps.txt and runs/ptall/orbits.json; group orders 24 and 60.
Bonus EXACT: all 4080 assignments are MU triples (factorised: |<q'|q>|^2 = 1/2 across qubit bases, |<g'|g>|^2 = 1/5 for
  distinct labels).

Usage (server, ~/mub10):  python -I scripts/exact_checks.py [--md results/product_triples_exact_checks.md]
"""
import os, sys, json, time, re, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


# =============================================================== exact cyclotomic arithmetic (Python integers only)
def poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def poly_divexact(a, b):
    """a / b for integer polynomials (low -> high), b monic; asserts zero remainder"""
    a = list(a)
    db = len(b) - 1
    assert b[-1] == 1
    q = [0] * (len(a) - db)
    for k in range(len(a) - 1, db - 1, -1):
        c = a[k]
        if c:
            q[k - db] = c
            for m in range(db + 1):
                a[k - db + m] -= c * b[m]
    assert not any(a), "inexact division"
    return q


def cyclotomic(N):
    f = [-1] + [0] * (N - 1) + [1]          # x^N - 1
    for d in range(1, N):
        if N % d == 0:
            f = poly_divexact(f, cyclotomic(d))
    return f


class Cyc:
    """Z[zeta_N] = Z[x]/(Phi_N); elements are tuples of phi(N) Python ints (coefficients of 1, x, ..., x^(phi-1))."""

    def __init__(self, N):
        self.N = N
        self.phi = cyclotomic(N)
        d = self.d = len(self.phi) - 1
        red, cur = [], [1] + [0] * (d - 1)
        for k in range(N):
            red.append(tuple(cur))
            top = cur[d - 1]
            nxt = [0] + cur[:d - 1]
            if top:
                for m in range(d):
                    nxt[m] -= top * self.phi[m]
            cur = nxt
        assert tuple(cur) == red[0], "x^N != 1 mod Phi_N"
        self.red = red
        self.rs = [[(m, c) for m, c in enumerate(r) if c] for r in red]
        self.zero = tuple([0] * d)
        self.one = red[0]

    def z(self, k):                       # zeta_N^k
        return self.red[k % self.N]

    def from_acc(self, acc):              # acc[k] = coefficient of zeta^k, k mod N
        out = [0] * self.d
        for k, c in enumerate(acc):
            if c:
                for m, r in self.rs[k]:
                    out[m] += c * r
        return tuple(out)

    def add(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def sub(self, a, b):
        return tuple(x - y for x, y in zip(a, b))

    def neg(self, a):
        return tuple(-x for x in a)

    def smul(self, c, a):
        return tuple(c * x for x in a)

    def mul(self, a, b):
        N = self.N
        acc = [0] * N
        bn = [(j, y) for j, y in enumerate(b) if y]
        if not bn:
            return self.zero
        for i, x in enumerate(a):
            if x:
                for j, y in bn:
                    acc[(i + j) % N] += x * y
        return self.from_acc(acc)

    def conj(self, a):
        N = self.N
        acc = [0] * N
        for i, x in enumerate(a):
            if x:
                acc[(-i) % N] += x
        return self.from_acc(acc)

    def norm2(self, a):                   # a * conj(a)
        return self.mul(a, self.conj(a))

    def const(self, c):
        return self.smul(c, self.one)

    def divexact_int(self, a, n):
        assert all(x % n == 0 for x in a), "not divisible by %d" % n
        return tuple(x // n for x in a)

    def unit_exp(self, a, scale):
        """k with a == scale * zeta^k (scale a Python int), else None"""
        for k in range(self.N):
            if self.smul(scale, self.z(k)) == a:
                return k
        return None

    def to_complex(self, a):              # only for the float cross-check (server)
        import cmath
        return sum(c * cmath.exp(2j * cmath.pi * m / self.N) for m, c in enumerate(a))


# --------------------------------------------------------------- exact matrices
def mat_mul(R, A, B):
    n, m, q = len(A), len(B), len(B[0])
    Bt = [[B[k][j] for k in range(m)] for j in range(q)]
    nzA = [[k for k in range(m) if any(A[i][k])] for i in range(n)]
    out = []
    for i in range(n):
        row = []
        for j in range(q):
            s = R.zero
            for k in nzA[i]:
                b = Bt[j][k]
                if any(b):
                    s = R.add(s, R.mul(A[i][k], b))
            row.append(s)
        out.append(row)
    return out


def adj(R, A):
    return [[R.conj(A[i][j]) for i in range(len(A))] for j in range(len(A[0]))]


def is_scalar_identity(R, A, c):
    cc = R.const(c)
    return all(A[i][j] == (cc if i == j else R.zero) for i in range(len(A)) for j in range(len(A[0])))


# =============================================================== exact model of the triples
R40 = Cyc(40)
R20 = Cyc(20)
assert R20.phi == [1, 0, -1, 0, 1, 0, -1, 0, 1], R20.phi                         # x^8 - x^6 + x^4 - x^2 + 1
assert R40.phi == [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 1], R40.phi  # x^16 - x^12 + x^8 - x^4 + 1

LAB = ["inf", "0", "1", "2", "3", "4"]
POS = ["Z0", "Z1", "X0", "X1", "Y0", "Y1"]


class Model:
    """exact objects over R = Z[zeta_40]"""

    def __init__(self, R):
        self.R = R
        z = R.z
        self.i = z(10)
        self.w = lambda k: z(8 * (k % 5))                       # w^k, w = zeta_5 = zeta_40^8
        self.z20 = lambda k: z(2 * (k % 20))                    # zeta_20^k
        self.sqrt2 = R.add(z(5), z(35))                         # zeta_8 + zeta_8^-1
        self.sqrt5 = R.add(R.one, R.smul(2, R.add(z(8), z(32))))  # 1 + 2(zeta_5 + zeta_5^4)
        assert R.mul(self.sqrt2, self.sqrt2) == R.const(2)
        assert R.mul(self.sqrt5, self.sqrt5) == R.const(5)
        self.sqrt10 = R.mul(self.sqrt2, self.sqrt5)
        assert R.mul(self.sqrt10, self.sqrt10) == R.const(10)
        for s in (self.sqrt2, self.sqrt5, self.sqrt10):
            assert R.conj(s) == s
        o, Z, i = R.one, R.zero, self.i
        self.Q2 = [[self.sqrt2, Z], [Z, self.sqrt2], [o, o], [o, R.neg(o)], [o, i], [o, R.neg(i)]]
        self._G5 = {}

    def G5(self, lab):
        if lab not in self._G5:
            R = self.R
            if lab == 0:
                M = [[self.sqrt5 if x == j else R.zero for j in range(5)] for x in range(5)]
            else:
                t = lab - 1
                M = [[self.w(t * x * x + j * x) for j in range(5)] for x in range(5)]
            self._G5[lab] = M
        return self._G5[lab]

    def basis(self, p0, l0, p1, l1):
        """N = sqrt10 * [Q[p0] (x) G[l0], Q[p1] (x) G[l1]]  (10 x 10, row a*5+x, column h*5+j)"""
        R = self.R
        N = [[R.zero] * 10 for _ in range(10)]
        for h, (p, l) in enumerate(((p0, l0), (p1, l1))):
            q, G = self.Q2[p], self.G5(l)
            for a in range(2):
                if not any(q[a]):
                    continue
                for x in range(5):
                    for j in range(5):
                        N[a * 5 + x][h * 5 + j] = R.mul(q[a], G[x][j])
        return N

    def triple(self, ell):
        return [self.basis(2 * b, ell[2 * b], 2 * b + 1, ell[2 * b + 1]) for b in range(3)]


M40 = Model(R40)


def read_reps():
    reps = []
    for line in open(os.path.join(HERE, "orbit_reps.txt")):
        if line.strip():
            k, r = line.split()
            reps.append((int(k), r, tuple(0 if x == "inf" else int(x) + 1 for x in r.split(","))))
    return reps


# =============================================================== Part A
def md5_of(name):
    import hashlib
    fn = os.path.join(HERE, name)
    return hashlib.md5(open(fn, "rb").read()).hexdigest() if os.path.exists(fn) else None


CODE_VERSIONS = {}
NOTES = []


def float_tables(ell, rep_str):
    """the exponent tables as the proof code paths compute them: numpy rounding of angles (product_triples_all.checks,
    rank_cert.zeta_exp; the chart files were generated with the same rounding), the integer-formula table of
    exact_tables.py (if present) and product_triple_charts.exponents() of the current version (U and REP set)."""
    sys.path.insert(0, HERE)
    import numpy as np
    import product_triples_all as PA
    for nm in ("product_triples_all.py", "product_triple_charts.py", "rank_cert.py", "exact_tables.py"):
        CODE_VERSIONS.setdefault(nm, md5_of(nm))
    out = {}
    U = PA.triple(ell)
    out["product_triples_all.checks"] = PA.checks(ell)[1]
    C = np.vstack([U[1].conj().T @ U[0], U[2].conj().T @ U[0]]) * np.sqrt(10)
    try:
        import rank_cert as RC
        out["rank_cert.zeta_exp"] = RC.zeta_exp(C)
    except Exception as e:                                   # pragma: no cover
        out["rank_cert.zeta_exp"] = "ERROR %r" % e
    if os.path.exists(os.path.join(HERE, "exact_tables.py")):
        try:
            import exact_tables as ET
            out["exact_tables.exponent_table"] = ET.exponent_table(rep_str)
        except Exception as e:                               # pragma: no cover
            out["exact_tables.exponent_table"] = "ERROR %r" % e
    try:
        import product_triple_charts as PTC
        PTC.U = U
        if hasattr(PTC, "REP"):
            PTC.REP = rep_str
        PTC.GAUSS_CHECK = rep_str == "inf,0,1,2,3,4"
        out["product_triple_charts.exponents"] = PTC.exponents()
    except Exception as e:                                   # pragma: no cover
        out["product_triple_charts.exponents"] = "ERROR %r" % e
    res = {}
    for k, v in out.items():
        res[k] = v if isinstance(v, str) else [[int(x) % 20 for x in row] for row in np.asarray(v)]
    # float cross-check that the exact model is the same triple as the script's (standard embedding)
    dev = 0.0
    Nx = M40.triple(ell)
    for b in range(3):
        for r in range(10):
            for c in range(10):
                dev = max(dev, abs(R40.to_complex(Nx[b][r][c]) - np.sqrt(10) * U[b][r, c]))
    res["_float_dev_exact_model_vs_PA_triple"] = dev
    return res


def part_A6(ell, rep_str):
    """EXACT Weyl actions on U0 coordinates: N0^* (I2 (x) X^a Z^b) N0 = 10 * monomial; compared with
    exact_tables.weyl_substitution (used by rank_cert.py) and with rank_cert.monomial_action (numeric)."""
    R, Mo = R40, M40
    N0 = Mo.triple(ell)[0]
    try:
        sys.path.insert(0, HERE)
        import exact_tables as ET
    except Exception:
        ET = None
    try:
        import numpy as np
        import rank_cert as RC
        import product_triples_all as PA
        U0 = PA.triple(ell)[0]
    except Exception:
        RC = None
    n_ok = n_et = n_rc = 0
    for a in range(5):
        for b in range(5):
            W = [[Mo.w(b * y) if x == (y + a) % 5 else R.zero for y in range(5)] for x in range(5)]
            IW = [[W[r % 5][c % 5] if r // 5 == c // 5 else R.zero for c in range(10)] for r in range(10)]
            C = mat_mul(R, adj(R, N0), mat_mul(R, IW, N0))
            sig, ph, ok = [], [], True
            for i in range(10):
                nz = [j for j in range(10) if any(C[i][j])]
                if len(nz) != 1:
                    ok = False; break
                e = R.unit_exp(C[i][nz[0]], 10)
                if e is None or e % 2:
                    ok = False; break
                sig.append(nz[0]); ph.append((e // 2) % 20)
            ok = ok and sorted(sig) == list(range(10))
            n_ok += ok
            if ok and ET is not None:
                s2, p2 = ET.weyl_substitution(rep_str, a, b)
                n_et += list(s2) == sig and [x % 20 for x in p2] == ph
            if ok and RC is not None:
                X5 = np.roll(np.eye(5), 1, axis=0)                     # as in rank_cert.main
                Z5 = np.diag(np.exp(2j * np.pi * np.arange(5) / 5))
                Wm = np.linalg.matrix_power(X5, a) @ np.linalg.matrix_power(Z5, b)
                s3, p3 = RC.monomial_action(U0, Wm)
                n_rc += list(int(x) for x in s3) == sig and [int(x) % 20 for x in p3] == ph
    return dict(exact_monomial=n_ok, matches_exact_tables=n_et if ET else None,
                matches_rank_cert_numeric=n_rc if RC else None, total=25)


def part_A1_A2(ell):
    R, Mo = R40, M40
    Ns = Mo.triple(ell)
    unit_ok = [is_scalar_identity(R, mat_mul(R, adj(R, N), N), 10) for N in Ns]
    mu = {}
    ten = R.const(10)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        P = mat_mul(R, adj(R, Ns[a]), Ns[b])
        mu["%d%d" % (a, b)] = sum(R.norm2(P[i][j]) == ten for i in range(10) for j in range(10))
    # C = sqrt10 [U1^*U0; U2^*U0] = [N1^*N0; N2^*N0] / sqrt10
    rows = []
    for b in (1, 2):
        rows += mat_mul(R, adj(R, Ns[b]), Ns[0])
    E_exact = []
    for g in range(20):
        er = []
        for k in range(10):
            hits = [e for e in range(20) if rows[g][k] == R.mul(Mo.sqrt10, Mo.z20(e))]
            assert len(hits) <= 1
            er.append(hits[0] if hits else None)
        E_exact.append(er)
    return dict(unit_ok=unit_ok, mu=mu, rows=rows, E_exact=E_exact)


def check_A2(rows, E):
    R, Mo = R40, M40
    return sum(rows[g][k] == R.mul(Mo.sqrt10, Mo.z20(E[g][k])) for g in range(20) for k in range(10))


def part_A3(ell, E):
    """factorised check in Z[zeta_20]: (sqrt2<q'|q>) (sqrt5<g'|g>) zeta20^-E == 1"""
    R = R20
    w = lambda k: R.z(4 * (k % 5))
    i = R.z(5)
    sqrt5 = R.add(R.one, R.smul(2, R.add(R.z(4), R.z(16))))
    assert R.mul(sqrt5, sqrt5) == R.const(5)
    o = R.one
    V = {2: (o, o), 3: (o, R.neg(o)), 4: (o, i), 5: (o, R.neg(i))}     # sqrt2 * |+>, |->, |+i>, |-i>
    npass, cases = 0, {}
    for g in range(20):
        b, c = 1 + g // 10, g % 10
        h, j = c // 5, c % 5
        pp, s = 2 * b + h, ell[2 * b + h]          # row: column j of the half h of basis b
        for k in range(10):
            h0, x = k // 5, k % 5
            p, t = h0, ell[h0]                     # column: column x of the half h0 of U0 (qubit |0> or |1>)
            qf = R.conj(V[pp][p])                  # sqrt2 <q_pp | e_p>
            assert R.norm2(qf) == R.one and qf in (o, R.neg(o), i, R.neg(i))
            if s == 0 and t == 0:
                raise AssertionError("label inf in two bases (not an MU triple)")
            elif s == 0:                           # <e_j | G_t(:,x)> sqrt5 = w^(t' j^2 + x j)
                tt = t - 1
                cf = w(tt * j * j + x * j); case = "inf-t"
            elif t == 0:                           # <G_s(:,j) | e_x> sqrt5 = w^-(s' x^2 + j x)
                ss = s - 1
                cf = w(-(ss * x * x + j * x)); case = "s-inf"
            else:
                ss, tt = s - 1, t - 1
                assert ss != tt
                S = R.zero                         # Gauss sum sum_y w^((t-s) y^2 + (x-j) y)
                for y in range(5):
                    S = R.add(S, w((tt - ss) * y * y + (x - j) * y))
                cf = R.divexact_int(R.mul(S, sqrt5), 5)            # S / sqrt5
                case = "gauss"
            val = R.mul(R.mul(qf, cf), R.z(-E[g][k]))
            ok = val == R.one
            npass += ok
            cases[case] = cases.get(case, 0) + 1
    return npass, cases


# --------------------------------------------------------------- A5: chart files
def roots_phi20(p):
    out = []
    for a in range(2, 200):
        r0 = pow(a, (p - 1) // 20, p)
        if pow(r0, 10, p) != 1 and pow(r0, 4, p) != 1:
            break
    else:
        raise RuntimeError("no primitive 20th root found")
    for k in (1, 3, 7, 9, 11, 13, 17, 19):
        out.append(pow(r0, k, p))
    return sorted(out)


def parse_ms(path):
    txt = open(path).read()
    lines = txt.split("\n")
    V = [v.strip() for v in lines[0].split(",")]
    p = int(lines[1])
    polys = []
    for q in "\n".join(lines[2:]).strip().split(",\n"):
        q = q.replace(" ", "").replace("\n", "")
        if not q:
            continue
        d = {}
        for t in re.findall(r"[+-]?[^+-]+", q):
            sg = -1 if t[0] == "-" else 1
            t = t.lstrip("+-")
            coef, mono = 1, []
            for f in t.split("*"):
                if f[0].isdigit():
                    coef *= int(f)
                elif "^" in f:
                    v, e = f.split("^"); mono += [v] * int(e)
                else:
                    mono.append(f)
            key = tuple(sorted(mono))
            d[key] = (d.get(key, 0) + sg * coef) % p
        polys.append({k: v for k, v in d.items() if v})
    return V, p, polys


def chart_system(E, p, r, chart):
    """independent encoder: d_k = X_k Y_k - X_0 Y_0, f_g = sum_{k != l} zeta20^(E[g][k]-E[g][l]) X_k Y_l, on a chart"""
    X = ["X%d" % k for k in range(10)]
    Y = ["Y%d" % k for k in range(10)]
    if chart == "affine":
        fixed = {"X0": 1, "Y0": 1}
    else:
        i, j = chart
        fixed = {X[i]: 1, Y[j]: 1, Y[i]: 0, X[j]: 0}
    var = [v for v in X + Y if v not in fixed]
    rp = [pow(r, e, p) for e in range(20)]

    def prod(vs, coef):
        c, m = coef, []
        for v in vs:
            if v in fixed:
                c = c * fixed[v]
            else:
                m.append(v)
        return c % p, tuple(sorted(m))

    polys = []
    if chart == "affine":                       # X_k Y_k - 1, k = 1..9
        for k in range(1, 10):
            polys.append({(X[k], Y[k]): 1, (): p - 1})
    else:                                       # boundary: X_0 Y_0 = 0, hence X_k Y_k = 0 for all k
        for k in range(10):
            c, m = prod([X[k], Y[k]], 1)
            if c:
                assert m, "chart inconsistent"
                polys.append({m: c})
    for g in range(20):
        d = {}
        for k in range(10):
            for l in range(10):
                if k != l:
                    c, m = prod([X[k], Y[l]], rp[(E[g][k] - E[g][l]) % 20])
                    if c:
                        d[m] = (d.get(m, 0) + c) % p
        d = {k: v for k, v in d.items() if v}
        if d:
            assert any(k for k in d), "nonzero constant equation"
            polys.append(d)
    return var, polys


def canon(poly):
    return tuple(sorted(poly.items()))


def part_A5(oid, E, d=None):
    d = d or os.path.join(ROOT, "runs", "charts_all", "o%d" % oid)
    if not os.path.isdir(d):
        return dict(dir=d, present=False)
    p, r = 1073741741, 95182621                  # defaults of product_triple_charts.py
    gen = d.rstrip("/") + ".gen.txt"
    if os.path.exists(gen):
        m = re.search(r"p (\d+) r (\d+)", open(gen).read())
        if m:
            p, r = int(m.group(1)), int(m.group(2))
    R = R20
    r_ok = pow(r, 20, p) == 1 and sum(c * pow(r, m, p) for m, c in enumerate(R.phi)) % p == 0
    charts = ["affine"] + [(i, j) for i in range(10) for j in range(10) if i != j]
    n_files = n_same_order = n_same_set = n_vars = 0
    bad = []
    for ch in charts:
        fn = os.path.join(d, ("affine" if ch == "affine" else "b_%d_%d" % ch) + ".ms")
        if not os.path.exists(fn):
            bad.append((str(ch), "missing"))
            continue
        n_files += 1
        V, pf, polys = parse_ms(fn)
        var, exp = chart_system(E, p, r, ch)
        vok = V == var and pf == p
        n_vars += vok
        so = [canon(q) for q in polys] == [canon(q) for q in exp]
        ss = sorted(canon(q) for q in polys) == sorted(canon(q) for q in exp)
        n_same_order += so
        n_same_set += ss
        if not (vok and so and ss):
            # diagnose: which root of Phi_20 would match?
            alt = [rr for rr in roots_phi20(p)
                   if sorted(canon(q) for q in chart_system(E, p, rr, ch)[1]) == sorted(canon(q) for q in polys)]
            bad.append((str(ch), "vars %s order %s set %s; matching roots %s" % (vok, so, ss, alt)))
    return dict(dir=os.path.relpath(d, ROOT), present=True, p=p, r=r, r_root_of_phi20=r_ok, charts=len(charts),
                files=n_files, vars_p_ok=n_vars, same_order=n_same_order, same_multiset=n_same_set, bad=bad)


# =============================================================== Part B
def generators_exact():
    R, Mo = R40, M40
    o, Z = R.one, R.zero
    s2 = Mo.sqrt2
    gens = []
    gens.append(dict(name="qubit H", kind="qubit", den=2, M=[[s2, s2], [s2, R.neg(s2)]]))
    gens.append(dict(name="qubit S", kind="qubit", den=1, M=[[o, Z], [Z, Mo.i]]))
    gens.append(dict(name="C5 F (DFT)", kind="c5", den=5,
                     M=[[R.mul(Mo.sqrt5, Mo.w(x * y)) for y in range(5)] for x in range(5)]))
    gens.append(dict(name="C5 P=diag(w^{x^2})", kind="c5", den=1,
                     M=[[Mo.w(x * x) if x == y else Z for y in range(5)] for x in range(5)]))
    gens.append(dict(name="C5 M2: |x>->|2x>", kind="c5", den=1,
                     M=[[o if x == (2 * y) % 5 else Z for y in range(5)] for x in range(5)]))
    gens.append(dict(name="complex conjugation K", kind="conj", den=1, M=None))
    for g in gens:
        if g["M"] is not None:
            g["unitary_exact"] = is_scalar_identity(R, mat_mul(R, adj(R, g["M"]), g["M"]), g["den"] ** 2)
        else:
            g["unitary_exact"] = True               # antiunitary; conj is a ring automorphism
    return gens


def apply_gen(g, A, qubit_dim):
    """apply the numerator of g to a matrix A (rows indexed a*5+x if qubit_dim == 10, else a 2- or 5-row block)"""
    R = R40
    if g["kind"] == "conj":
        return [[R.conj(x) for x in row] for row in A]
    if qubit_dim == "q2":
        return mat_mul(R, g["M"], A) if g["kind"] == "qubit" else None
    if qubit_dim == "c5":
        return mat_mul(R, g["M"], A) if g["kind"] == "c5" else None
    # full 10 x n: (V (x) I5) or (I2 (x) W)
    n = len(A[0])
    out = [[R.zero] * n for _ in range(10)]
    for a in range(2):
        for x in range(5):
            for c in range(n):
                s = R.zero
                if g["kind"] == "qubit":
                    for a2 in range(2):
                        m = g["M"][a][a2]
                        if any(m) and any(A[a2 * 5 + x][c]):
                            s = R.add(s, R.mul(m, A[a2 * 5 + x][c]))
                else:
                    for x2 in range(5):
                        m = g["M"][x][x2]
                        if any(m) and any(A[a * 5 + x2][c]):
                            s = R.add(s, R.mul(m, A[a * 5 + x2][c]))
                out[a * 5 + x][c] = s
    return out


def monomial_check(C, scale):
    """C must have exactly one nonzero per row and column, each with c conj(c) == scale^2; returns (ok, perm, phase exps)"""
    R = R40
    n = len(C)
    perm, ph = [None] * n, []
    s2 = R.const(scale * scale)
    for j in range(n):
        nz = [i for i in range(n) if any(C[i][j])]
        if len(nz) != 1 or R.norm2(C[nz[0]][j]) != s2:
            return False, None, None
        perm[j] = nz[0]
        ph.append(R.unit_exp(C[nz[0]][j], scale))
    if sorted(perm) != list(range(n)):
        return False, None, None
    return True, perm, ph


def derive_sigma_pi(g):
    """exact induced permutations on the 6 qubit states and the 6 labels"""
    R, Mo = R40, M40
    den = g["den"]
    sigma = list(range(6))
    if g["kind"] in ("qubit", "conj"):
        sigma = []
        for p in range(6):
            q = [[Mo.Q2[p][0]], [Mo.Q2[p][1]]]
            a = apply_gen(g, q, "q2")
            hits = []
            for p2 in range(6):
                t = Mo.Q2[p2]
                s = R.add(R.mul(R.conj(t[0]), a[0][0]), R.mul(R.conj(t[1]), a[1][0]))
                if (all(R.smul(2, a[k][0]) == R.mul(s, t[k]) for k in range(2))
                        and R.norm2(s) == R.const(4 * den * den)):
                    hits.append(p2)
            assert len(hits) == 1, (g["name"], p, hits)
            sigma.append(hits[0])
    pi = list(range(6))
    if g["kind"] in ("c5", "conj"):
        pi = []
        for l in range(6):
            A = apply_gen(g, Mo.G5(l), "c5")
            hits = [l2 for l2 in range(6) if monomial_check(mat_mul(R, adj(R, Mo.G5(l2)), A), 5 * den)[0]]
            assert len(hits) == 1, (g["name"], l, hits)
            pi.append(hits[0])
    assert sorted(sigma) == list(range(6)) and sorted(pi) == list(range(6))
    return sigma, pi


def all_assignments():
    out = []
    for ell in itertools.product(range(6), repeat=6):
        s = [set(ell[0:2]), set(ell[2:4]), set(ell[4:6])]
        if s[0] & s[1] or s[0] & s[2] or s[1] & s[2]:
            continue
        out.append(ell)
    return out


def act(sigma, pi, ell):
    new = [None] * 6
    for p in range(6):
        new[sigma[p]] = pi[ell[p]]
    return tuple(new)


def group_order(perms):
    seen = {tuple(range(len(perms[0])))}
    fr = list(seen)
    while fr:
        nf = []
        for h in fr:
            for g in perms:
                c = tuple(g[h[i]] for i in range(len(h)))
                if c not in seen:
                    seen.add(c); nf.append(c)
        fr = nf
    return len(seen)


def part_B(gens):
    R, Mo = R40, M40
    info = []
    # T^*T = 10 I for all 108 bases
    bases = {}
    for b in range(3):
        for l0 in range(6):
            for l1 in range(6):
                bases[(b, l0, l1)] = Mo.basis(2 * b, l0, 2 * b + 1, l1)
    unit108 = sum(is_scalar_identity(R, mat_mul(R, adj(R, N), N), 10) for N in bases.values())
    log("B: T^*T = 10 I exactly for %d/108 bases" % unit108)
    A_all = all_assignments()
    Aset = set(A_all)
    basis_ok = {}
    for gi, g in enumerate(gens):
        sigma, pi = g["sigma"], g["pi"]
        den = g["den"]
        # pair structure of sigma
        cmap = {}
        pairs_ok = True
        for b in range(3):
            s0, s1 = sigma[2 * b], sigma[2 * b + 1]
            if s0 // 2 != s1 // 2:
                pairs_ok = False
            cmap[b] = s0 // 2
        pairs_ok = pairs_ok and sorted(cmap.values()) == [0, 1, 2]
        nb_ok, phases, t0 = 0, set(), time.time()
        for (b, l0, l1), N in bases.items():
            if not pairs_ok:
                basis_ok[(gi, b, l0, l1)] = False
                continue
            c = cmap[b]
            lab = {sigma[2 * b]: pi[l0], sigma[2 * b + 1]: pi[l1]}
            T = bases[(c, lab[2 * c], lab[2 * c + 1])]
            A = apply_gen(g, N, "full")
            C = mat_mul(R, adj(R, T), A)
            ok, perm, ph = monomial_check(C, 10 * den)
            if ok:
                # 10 A == T C entrywise (C monomial: column j of T C is T[:, perm[j]] * C[perm[j]][j])
                for j in range(10):
                    cj = C[perm[j]][j]
                    for r_ in range(10):
                        if R.smul(10, A[r_][j]) != R.mul(T[r_][perm[j]], cj):
                            ok = False
                phases.update(ph)
            basis_ok[(gi, b, l0, l1)] = ok
            nb_ok += ok
        # assignment-level coverage
        n_ass = 0
        for ell in A_all:
            new = act(sigma, pi, ell)
            if new in Aset and pairs_ok and all(basis_ok[(gi, b, ell[2 * b], ell[2 * b + 1])] for b in range(3)):
                n_ass += 1
        info.append(dict(name=g["name"], den=den, unitary_exact=g["unitary_exact"], sigma=[POS[s] for s in sigma],
                         pi=[LAB[x] for x in pi], pairs_ok=pairs_ok, bases_ok=nb_ok, bases=len(bases),
                         assignments_ok=n_ass, assignments=len(A_all),
                         phases=sorted(phases, key=lambda x: (x is None, x)),
                         time=round(time.time() - t0, 1)))
        log("B: %-24s bases %d/108 assignments %d/%d phases zeta40^%s (%.1fs)" % (
            g["name"], nb_ok, n_ass, len(A_all), sorted(x for x in phases if x is not None), time.time() - t0))
    return info, unit108, A_all


def orbits_from(gens, A_all):
    idx = {e: i for i, e in enumerate(A_all)}
    par = list(range(len(A_all)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i, ell in enumerate(A_all):
        for g in gens:
            a, b = find(i), find(idx[act(g["sigma"], g["pi"], ell)])
            if a != b:
                par[a] = b
    orb = {}
    for i, ell in enumerate(A_all):
        orb.setdefault(find(i), []).append(ell)
    return [sorted(m) for m in orb.values()]


def mu_all_assignments():
    """exact factorised MU of all 4080 assignments"""
    R, Mo = R40, M40
    ok_q = all(R.norm2(R.add(R.mul(R.conj(Mo.Q2[p2][0]), Mo.Q2[p][0]), R.mul(R.conj(Mo.Q2[p2][1]), Mo.Q2[p][1])))
               == R.const(2) for p in range(6) for p2 in range(6) if p // 2 != p2 // 2)
    ok_g = True
    for s in range(6):
        for t in range(6):
            if s != t:
                P = mat_mul(R, adj(R, Mo.G5(s)), Mo.G5(t))
                ok_g &= all(R.norm2(P[i][j]) == R.const(5) for i in range(5) for j in range(5))
    return ok_q, ok_g


# =============================================================== Part C: Heisenberg-Weyl triples of Z_10
def part_C(orbs, reps):
    """EXACT: with the CRT permutation |n> -> |n mod 2> (x) |n mod 5>, the eigenbases of Z_10, X_10 and X_10 Z_10^k
    (k = 1, 3, 7, 9) are product bases of the family; spectra are nondegenerate, so the eigenbases are unique up to
    phases and order.  Returns the assignment and orbit of each triple {Z, X, X Z^k}."""
    R, Mo = R40, M40
    crt = [((n % 2) * 5 + (n % 5)) for n in range(10)]           # product-coordinate row of standard vector |n>
    inv = [None] * 10
    for n, r in enumerate(crt):
        inv[r] = n

    def op(k, xpow):                                              # X^xpow Z^k on C^10, |n> -> zeta10^(k n) |n + xpow>
        M = [[R.zero] * 10 for _ in range(10)]
        for n in range(10):
            M[(n + xpow) % 10][n] = R.z(4 * k * n)
        return M

    def pull(v):                                                  # product coordinates -> standard coordinates
        return [v[crt[n]] for n in range(10)]

    def eig_info(A, N):
        """all 10 columns of N (product coords) are eigenvectors of A; returns (ok, distinct eigenvalues)"""
        lams = []
        for c in range(10):
            v = pull([N[r][c] for r in range(10)])
            Av = [R.zero] * 10
            for i in range(10):
                for j in range(10):
                    if any(A[i][j]) and any(v[j]):
                        Av[i] = R.add(Av[i], R.mul(A[i][j], v[j]))
            i0 = [i for i in range(10) if any(v[i])][0]
            if not all(R.mul(Av[r], v[i0]) == R.mul(v[r], Av[i0]) for r in range(10)):
                return False, None
            # eigenvalue: lambda * (v_i0 conj v_i0) = Av_i0 conj v_i0 ; keep the pair to compare exactly
            lams.append((R.mul(Av[i0], R.conj(v[i0])), R.norm2(v[i0])))
        distinct = all(R.mul(a1, n2) != R.mul(a2, n1) for x, (a1, n1) in enumerate(lams) for (a2, n2) in lams[x + 1:])
        return True, distinct

    out = {}
    zb = Mo.basis(0, 0, 1, 0)                                     # Z: inf, inf
    xb = Mo.basis(2, 1, 3, 1)                                     # X: 0, 0
    out["Z_10 eigenbasis = Z:inf,inf"] = eig_info(op(1, 0), zb)
    out["X_10 eigenbasis = X:0,0"] = eig_info(op(0, 1), xb)
    trip = {}
    for k in (1, 3, 7, 9):
        A = op(k, 1)
        hits = []
        for t in range(5):
            ok, dist = eig_info(A, Mo.basis(4, t + 1, 5, t + 1))  # Y: t, t
            if ok:
                hits.append((t, dist))
        ell = (0, 0, 1, 1, hits[0][0] + 1, hits[0][0] + 1) if len(hits) == 1 else None
        orb = [o for o, (oid, rep, e) in enumerate(reps) if ell is not None and ell in next(m for m in orbs if e in m)]
        trip["X Z^%d" % k] = dict(hits=hits, assignment="Z:inf,inf X:0,0 Y:%d,%d" % (hits[0][0], hits[0][0]) if ell else None,
                                  orbit=orb[0] if orb else None)
    out["triples"] = trip
    return out


# =============================================================== negative controls
def negative_controls(reps, resA, gens):
    """each corrupted input must be rejected; returns {name: detected (bool)}"""
    out = {}
    oid, rep, ell = reps[8]
    E = [row[:] for row in resA[8]["E"]]
    Ebad = [row[:] for row in E]
    Ebad[3][7] = (Ebad[3][7] + 1) % 20
    a = part_A1_A2(ell)
    out["A2 rejects E with one entry +1"] = check_A2(a["rows"], Ebad) == 199
    out["A3 rejects E with one entry +1"] = part_A3(ell, Ebad)[0] == 199
    Ebad2 = [row[:] for row in E]
    Ebad2[12][0] = (Ebad2[12][0] + 10) % 20                       # sign flip of one entry
    out["A2 rejects a sign flip (+10)"] = check_A2(a["rows"], Ebad2) == 199
    c5 = part_A5(oid, Ebad)
    out["A5 rejects chart files vs corrupted E"] = c5.get("present", False) and c5["same_multiset"] < 91
    # wrong relabelling: generator P with the identity label permutation must fail on bases with finite labels
    R, Mo = R40, M40
    g = [h for h in gens if h["name"].startswith("C5 P")][0]
    N = Mo.basis(2, 1, 3, 2)                                      # X basis, labels 0,1
    A = apply_gen(g, N, "full")
    out["B rejects P with identity relabelling"] = not monomial_check(mat_mul(R, adj(R, N), A), 10 * g["den"])[0]
    T = Mo.basis(2, g["pi"][1], 3, g["pi"][2])
    out["B accepts P with its true relabelling (sanity)"] = monomial_check(mat_mul(R, adj(R, T), A), 10 * g["den"])[0]
    # wrong qubit target: H maps Z basis to X basis; comparing with the Y basis must fail
    h = gens[0]
    N = Mo.basis(0, 0, 1, 0)
    A = apply_gen(h, N, "full")
    out["B rejects H(Z) compared with Y"] = not monomial_check(mat_mul(R, adj(R, Mo.basis(4, 0, 5, 0)), A), 10 * h["den"])[0]
    # non-MU control: two bases sharing label 0 must fail the exact MU test
    P = mat_mul(R, adj(R, Mo.basis(0, 1, 1, 1)), Mo.basis(2, 1, 3, 2))
    out["A1 rejects a non-MU pair (shared label)"] = not all(R.norm2(P[i][j]) == R.const(10) for i in range(10) for j in range(10))
    return out


# =============================================================== main
def main():
    md_path = None
    if "--md" in sys.argv:
        md_path = sys.argv[sys.argv.index("--md") + 1]
    NOTES.extend(sys.argv[k + 1] for k in range(len(sys.argv) - 1) if sys.argv[k] == "--note")
    reps = read_reps()
    log("Phi_20, Phi_40 computed and asserted; sqrt2^2 = 2, sqrt5^2 = 5, sqrt10^2 = 10 exactly")
    resA = []
    for oid, rep, ell in reps:
        t0 = time.time()
        a = part_A1_A2(ell)
        ft = float_tables(ell, rep)
        Ef = ft["product_triples_all.checks"]
        same = {k: (v == a["E_exact"]) if not isinstance(v, str) else v for k, v in ft.items()
                if not k.startswith("_")}
        a2 = check_A2(a["rows"], Ef)
        a3, cases = part_A3(ell, Ef)
        a5 = part_A5(oid, a["E_exact"])
        a6 = part_A6(ell, rep)
        extra = None
        if oid == 8:
            extra = part_A5(8, a["E_exact"], os.path.join(ROOT, "runs", "charts_o8"))
        resA.append(dict(orbit=oid, rep=rep, unit_ok=a["unit_ok"], mu=a["mu"], A2_pass=a2, A3_pass=a3, A3_cases=cases,
                         E_exact_complete=all(e is not None for row in a["E_exact"] for e in row),
                         E_float_equals_exact=same, float_dev=ft["_float_dev_exact_model_vs_PA_triple"],
                         charts=a5, charts_o8=extra, weyl=a6, E=a["E_exact"]))
        log("A: orbit %d %-22s unitary %s MU %s A2 %d/200 A3 %d/200 E_float==E_exact %s charts %s/%s/%s (%.1fs)" % (
            oid, rep, a["unit_ok"], a["mu"], a2, a3, same, a5.get("same_order"), a5.get("same_multiset"),
            a5.get("charts"), time.time() - t0))
        log("   Weyl actions (A6): %s" % a6)
        if extra:
            log("   runs/charts_o8: %s" % {k: v for k, v in extra.items() if k != "bad"}, extra.get("bad"))
    # ---- Part B
    gens = generators_exact()
    for g in gens:
        g["sigma"], g["pi"] = derive_sigma_pi(g)
    sys.path.insert(0, HERE)
    import product_triples_all as PA
    pg = PA.generators()
    for g, h in zip(gens, pg):
        assert g["name"] == h["name"]
        g["matches_numeric"] = (list(g["sigma"]) == list(h["sigma"]) and list(g["pi"]) == list(h["pi"]))
    log("B: exact sigma/pi equal numeric generators(): %s" % [g["matches_numeric"] for g in gens])
    infoB, unit108, A_all = part_B(gens)
    for g, ib in zip(gens, infoB):
        ib["matches_numeric"] = g["matches_numeric"]
    assert A_all == PA.all_assignments()
    qord = group_order([g["sigma"] for g in gens[:2]])
    cord = group_order([g["pi"] for g in gens[2:5]])
    orbs = orbits_from(gens, A_all)
    O = json.load(open(os.path.join(ROOT, "runs", "ptall", "orbits.json")))
    json_orbits = {o["id"]: set(tuple(m) for m in o["members"]) for o in O["orbits"]}
    rep_orbit = []
    for oid, rep, ell in reps:
        mine = [set(m) for m in orbs if ell in m]
        rep_orbit.append(dict(orbit=oid, rep=rep, size=len(mine[0]), rep_is_min=min(mine[0]) == ell,
                              equals_orbits_json=mine[0] == json_orbits[oid]))
    distinct = len({min(next(m for m in orbs if ell in m)) for _, _, ell in reps}) == len(reps)
    mu_q, mu_g = mu_all_assignments()
    log("B: orbits %d (sizes %s), reps in distinct orbits %s; group orders %d, %d; MU-all factor checks %s %s" % (
        len(orbs), sorted(len(m) for m in orbs), distinct, qord, cord, mu_q, mu_g))
    hw = part_C(orbs, reps)
    log("C: Heisenberg-Weyl:", hw)
    controls = negative_controls(reps, resA, gens)
    log("controls (each must detect the corruption):", controls)
    summary = dict(resA=resA, generators=infoB, controls=controls, code_versions=CODE_VERSIONS, hw=hw, unit108=unit108, n_assignments=len(A_all), n_orbits=len(orbs),
                   orbit_sizes=sorted(len(m) for m in orbs), reps_distinct=distinct, rep_orbit=rep_orbit,
                   qubit_group_order=qord, c5_group_order=cord, mu_all_qubit=mu_q, mu_all_c5=mu_g,
                   runtime_s=round(time.time() - T0, 1))
    # overall verdict
    allA = all(r["A2_pass"] == 200 and r["A3_pass"] == 200 and all(r["unit_ok"]) and all(v == 100 for v in r["mu"].values())
               and all(v is True for v in r["E_float_equals_exact"].values()) and r["E_exact_complete"]
               and r["charts"].get("same_order") == 91 and r["charts"].get("vars_p_ok") == 91 and r["charts"].get("r_root_of_phi20")
               and r["weyl"]["exact_monomial"] == 25 and r["weyl"]["matches_exact_tables"] in (25, None)
               and r["weyl"]["matches_rank_cert_numeric"] in (25, None)
               for r in resA)
    allB = all(i["bases_ok"] == 108 and i["assignments_ok"] == 4080 and i["unitary_exact"] and i["matches_numeric"]
               and i["pairs_ok"] for i in infoB) and unit108 == 108 and len(orbs) == 10 and distinct and \
        all(r["rep_is_min"] and r["equals_orbits_json"] for r in rep_orbit) and mu_q and mu_g and qord == 24 and cord == 60
    allC = all(controls.values())
    allHW = hw["Z_10 eigenbasis = Z:inf,inf"] == (True, True) and hw["X_10 eigenbasis = X:0,0"] == (True, True) and \
        all(len(v["hits"]) == 1 and v["hits"][0][1] and v["orbit"] in (0, 1) for v in hw["triples"].values())
    summary["verdict"] = dict(partA=allA, partB=allB, controls=allC, partC=allHW)
    js = os.path.join(ROOT, "results", "product_triples_exact_checks.json")
    json.dump(summary, open(js, "w"), indent=1, default=str)
    log("wrote", js)
    if md_path:
        write_md(summary, md_path)
        log("wrote", md_path)
    log("OVERALL Part A %s, Part B %s, Part C %s, controls %s" % ("PASS" if allA else "FAIL", "PASS" if allB else "FAIL",
                                                                 "PASS" if allHW else "FAIL", "OK" if allC else "FAILED"))
    if not (allA and allB and allC and allHW):    # audit (AUD1 item 4): a reported FAIL must give a nonzero exit status
        sys.exit(1)


def write_md(S, path):
    L = []
    A = L.append
    A("# Exact checks for the MU product triples in C^2 (x) C^5")
    A("")
    A("Generated by `scripts/exact_checks.py` on the school server (single process, pure Python integers for every "
      "check marked exact; numpy only to reproduce the floating-point exponent tables of the proof scripts). "
      "Runtime %.0f s. Machine-readable output: `results/product_triples_exact_checks.json` (includes the 10 exponent tables)."
      % S["runtime_s"])
    A("")
    v = S["verdict"]
    A("**Overall: Part A (exponent tables) %s; Part B (symmetry generators) %s; Part C (Heisenberg-Weyl identification) %s; negative controls %s.**" % (
        "PASS" if v["partA"] else "FAIL", "PASS" if v["partB"] else "FAIL", "PASS" if v["partC"] else "FAIL",
        "all OK" if v["controls"] else "FAILED"))
    A("")
    A("## Arithmetic")
    A("")
    A("- Elements of Z[zeta_N] (N = 20, 40) are integer coefficient vectors modulo Phi_N; Phi_N is computed by exact "
      "polynomial division of x^N - 1 and asserted to be x^8 - x^6 + x^4 - x^2 + 1 (N = 20) and "
      "x^16 - x^12 + x^8 - x^4 + 1 (N = 40). Equality is coefficientwise equality of reduced vectors.")
    A("- sqrt2 = zeta_8 + zeta_8^-1, sqrt5 = 1 + 2(zeta_5 + zeta_5^4), sqrt10 = sqrt2 sqrt5 (squares 2, 5, 10 checked "
      "exactly). Under the standard embedding zeta_N -> exp(2 pi i/N) these are 2cos(pi/4) > 0 and 1 + 4cos(2pi/5) > 0, i.e. "
      "the positive roots used by `np.sqrt`. Complex conjugation = the automorphism zeta -> zeta^-1.")
    A("- Scaled exact bases N_b = sqrt10 U_b with columns sqrt2 q (x) sqrt5 G_l(:, j): qubit vectors (sqrt2,0), (0,sqrt2), "
      "(1,1), (1,-1), (1,i), (1,-i); sqrt5 G_inf = sqrt5 I, sqrt5 G_t(x,j) = w^(t x^2 + j x), w = zeta_5. "
      "Float cross-check (server, standard embedding): max |N_b - sqrt10 * PA.triple(ell)[b]| over the 10 representatives = %.1e."
      % max(r["float_dev"] for r in S["resA"]))
    A("")
    A("## Part A: exponent tables of the 10 orbit representatives")
    A("")
    A("Columns: (A1) N_b^* N_b = 10 I for b = 0,1,2, and s conj(s) = 10 for all 300 entries s of N_a^* N_b, a < b "
      "(pairwise MU, exact); (A2) full-matrix identity N_b^* N_0 = sqrt10 zeta_20^E in Z[zeta_40], i.e. "
      "sqrt10 [U1^*U0; U2^*U0] = zeta_20^E entrywise, with E the floating-point table of the proof scripts; "
      "(A3) factorised identity (sqrt2<q'|q>)(sqrt5<g'|g>) zeta_20^-E = 1 in Z[zeta_20] with exact Gauss sums; "
      "(A4) E_exact (unique exponent per entry, derived from the exact products) equals the float table produced by "
      "each of `product_triples_all.checks`, `product_triple_charts.exponents` and `rank_cert.zeta_exp`; "
      "(A5) chart files `runs/charts_all/oNN/{affine,b_i_j}.ms` (the generator lists against which the Singular "
      "certificates were verified) equal, polynomial by polynomial and in the same order, the chart systems rebuilt from "
      "E_exact by an independent encoder (zeta_20 -> r mod p; r checked to be a root of Phi_20 mod p).")
    A("")
    A("| # | rep (Z0,Z1,X0,X1,Y0,Y1) | A1 unitary | A1 MU entries | A2 pass | A3 pass (cases) | A4 code tables = E_exact | A5 chart files (vars+p / same list / same multiset) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in S["resA"]:
        c = r["charts"]
        cs = "%d/%d / %d/%d / %d/%d" % (c["vars_p_ok"], c["charts"], c["same_order"], c["charts"], c["same_multiset"],
                                        c["charts"]) if c.get("present") else "dir missing"
        if c.get("present") and not c["r_root_of_phi20"]:
            cs += " (r NOT a root of Phi_20!)"
        A("| %d | %s | %d/3 | %d/300 | %d/200 | %d/200 (%s) | %s | %s |" % (
            r["orbit"], r["rep"], sum(r["unit_ok"]), sum(r["mu"].values()), r["A2_pass"], r["A3_pass"],
            ", ".join("%s %d" % kv for kv in sorted(r["A3_cases"].items())),
            "%d/%d" % (sum(v is True for v in r["E_float_equals_exact"].values()), len(r["E_float_equals_exact"])), cs))
    tot2 = sum(r["A2_pass"] for r in S["resA"])
    tot3 = sum(r["A3_pass"] for r in S["resA"])
    A("")
    A("Totals: A2 %d/2000 entries, A3 %d/2000 entries; A4 %d/%d table comparisons; A5 %d/910 chart files identical." % (
        tot2, tot3, sum(sum(v is True for v in r["E_float_equals_exact"].values()) for r in S["resA"]),
        sum(len(r["E_float_equals_exact"]) for r in S["resA"]),
        sum(r["charts"].get("same_order", 0) for r in S["resA"])))
    A("")
    A("A4 code paths: %s. Script versions (md5) at run time: %s. The chart files in A5 were generated earlier by the "
      "float-rounding version of `product_triple_charts.py`; A5 compares the files themselves, so it does not depend on "
      "which version is current." % (", ".join("`%s`" % k for k in S["resA"][0]["E_float_equals_exact"]),
                                     ", ".join("`%s` %s" % (k, v) for k, v in S["code_versions"].items())))
    for r in S["resA"]:
        bad = {k: v for k, v in r["E_float_equals_exact"].items() if v is not True}
        if bad:
            A("Orbit %d A4 non-matches: %s" % (r["orbit"], bad))
    A("")
    A("(A6) Weyl actions used by the rank certificate (E57): for each representative and each of the 25 elements "
      "W = X^a Z^b of C^5, N_0^* (I_2 (x) W) N_0 = 10 M exactly with M monomial and entries in zeta_20^Z; compared with "
      "`exact_tables.weyl_substitution` (integer formulas, used by the current `rank_cert.py`) and with the numeric "
      "`rank_cert.monomial_action`.")
    A("")
    A("| # | exact monomial | = exact_tables.weyl_substitution | = rank_cert.monomial_action |")
    A("|---|---|---|---|")
    for r in S["resA"]:
        w_ = r["weyl"]
        A("| %d | %d/25 | %s | %s |" % (r["orbit"], w_["exact_monomial"],
                                       "n/a" if w_["matches_exact_tables"] is None else "%d/25" % w_["matches_exact_tables"],
                                       "n/a" if w_["matches_rank_cert_numeric"] is None else "%d/25" % w_["matches_rank_cert_numeric"]))
    A("")
    for r in S["resA"]:
        if r.get("charts_o8"):
            c = r["charts_o8"]
            if c.get("present"):
                A("Also `%s` (the first orbit-8 chart run, E51/E56): vars+p %d/%d, same list %d/%d, same multiset %d/%d (p = %d, r = %d)." % (
                    c["dir"], c["vars_p_ok"], c["charts"], c["same_order"], c["charts"], c["same_multiset"], c["charts"], c["p"], c["r"]))
        if r["charts"].get("bad"):
            A("Orbit %d chart mismatches: %s" % (r["orbit"], r["charts"]["bad"][:10]))
    A("")
    A("## Part B: symmetry generators of the orbit reduction")
    A("")
    A("Exact generator matrices (numerator / den): H = sqrt2[[1,1],[1,-1]] / 2, S = diag(1, i), F = sqrt5 w^(xy) / 5, "
      "P = diag(w^(x^2)), M2: |x> -> |2x>, K = complex conjugation; num^* num = den^2 I checked exactly. "
      "sigma (positions) and pi (labels) are derived exactly (V q_p proportional to q_sigma(p) with a unimodular factor; "
      "W G_l = G_pi(l) x monomial) and compared with the numeric `generators()` of `product_triples_all.py`. "
      "Basis check, for every generator and every one of the 108 bases [q_2b (x) G_l0, q_2b+1 (x) G_l1]: with A = num(g) N_B "
      "and T = N of the relabelled basis, T^* A = 10 den M with M monomial and every nonzero entry m of M satisfying "
      "m conj(m) = 1 exactly, and 10 A = T (T^* A) entrywise (so g(U_B) = U_T x permutation x phases). "
      "T^* T = 10 I holds exactly for %d/108 bases. Assignment check: for all %d assignments, the relabelled assignment is "
      "again an assignment, sigma maps the three position pairs bijectively onto position pairs, and all three bases pass the basis check."
      % (S["unit108"], S["n_assignments"]))
    A("")
    A("| generator | exactly unitary | sigma (Z0..Y1 ->) | pi (inf,0..4 ->) | = numeric | bases pass | assignments pass | column phases seen (zeta_40^k) |")
    A("|---|---|---|---|---|---|---|---|")
    for g in S["generators"]:
        A("| %s | %s | %s | %s | %s | %d/%d | %d/%d | %s |" % (
            g["name"].replace("|", "\\|"), g["unitary_exact"], ",".join(g["sigma"]), ",".join(g["pi"]), g["matches_numeric"],
            g["bases_ok"], g["bases"], g["assignments_ok"], g["assignments"],
            " ".join(str(x) for x in g["phases"])))
    A("")
    A("Orbits recomputed from the exact permutations: %d orbits, sizes %s; group orders: qubit part %d, C^5 part %d. "
      "All 4080 assignments are MU triples exactly (factorised: |<q'|q>|^2 = 1/2 across qubit bases: %s; |<g'|g>|^2 = 1/5 for "
      "distinct labels: %s)." % (S["n_orbits"], S["orbit_sizes"], S["qubit_group_order"], S["c5_group_order"],
                                 S["mu_all_qubit"], S["mu_all_c5"]))
    A("")
    A("| orbit | rep | size | rep = smallest member | orbit = members in runs/ptall/orbits.json |")
    A("|---|---|---|---|---|")
    for r in S["rep_orbit"]:
        A("| %d | %s | %d | %s | %s |" % (r["orbit"], r["rep"], r["size"], r["rep_is_min"], r["equals_orbits_json"]))
    A("")
    A("Representatives lie in distinct orbits: %s." % S["reps_distinct"])
    A("")
    A("## Scope")
    A("")
    A("- Proved exactly here: for the 10 representatives, sqrt10 [U1^*U0; U2^*U0] = zeta_20^E entrywise for the tables E "
      "used by the proof scripts (float rounding is therefore harmless); the triples are exactly unitary and pairwise MU; "
      "the chart inputs `runs/charts_all/oNN/*.ms` (and `runs/charts_o8/*.ms`) are exactly the reductions of the "
      "chart systems built from these tables under zeta_20 -> r (r a root of Phi_20 mod p); every symmetry generator maps "
      "every basis of every one of the 4080 assignments to the relabelled basis up to column permutation and column "
      "phases, and the exact permutations reproduce the 10 orbits and representatives.")
    A("- Not re-checked here: the certificate expansions themselves (`scripts/verify_cert.py`, status in "
      "`runs/charts_all/certs_status.txt`); the chart-cover / reduction-mod-p argument; the rank-certificate linear algebra "
      "(`scripts/rank_cert.py` computes its table with `rank_cert.zeta_exp`, which is covered by A4); the msolve evidence "
      "files in `runs/ptall/ms/` (evidence only, not part of the proof); the scope of MPW Cor. 3 (that the 4080 "
      "assignments exhaust all MU product triples up to local equivalence).")
    A("- The exact checks use the standard embedding zeta_N -> exp(2 pi i/N); the identification of the exact objects with "
      "the numpy objects of the scripts is by construction (same formulas, sqrt2 and sqrt5 positive under this embedding), "
      "cross-checked numerically to 5e-16.")
    A("")
    A("## Part C: Heisenberg-Weyl triples of Z_10 (exact)")
    A("")
    A("With the CRT permutation |n> -> |n mod 2> (x) |n mod 5> of the computational basis of C^10 (a global unitary), "
      "X_10 = X_2 (x) X_5 and Z_10 = Z_2 (x) Z_5^3. Checked exactly: every column of the product basis is an eigenvector of "
      "the operator (cross-multiplication test), and the 10 eigenvalues are pairwise distinct (so the eigenbasis is unique "
      "up to phases and order).")
    A("")
    hw = S["hw"]
    A("- Z_10: eigenbasis = Z:inf,inf: %s (distinct eigenvalues: %s)" % tuple(hw["Z_10 eigenbasis = Z:inf,inf"]))
    A("- X_10: eigenbasis = X:0,0: %s (distinct eigenvalues: %s)" % tuple(hw["X_10 eigenbasis = X:0,0"]))
    for k_, v_ in hw["triples"].items():
        A("- %s: eigenbasis = Y:t,t for t in %s (distinct eigenvalues: %s); triple {Z, X, %s} = %s, in orbit %s" % (
            k_, [h[0] for h in v_["hits"]], [h[1] for h in v_["hits"]], k_, v_["assignment"], v_["orbit"]))
    A("")
    if NOTES:
        A("Notes:")
        A("")
        for n_ in NOTES:
            A("- " + n_)
        A("")
    A("## Negative controls")
    A("")
    A("Each corrupted input must be rejected by the same code paths:")
    A("")
    for k, v in S["controls"].items():
        A("- %s: %s" % (k, "OK" if v else "FAILED"))
    A("")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
