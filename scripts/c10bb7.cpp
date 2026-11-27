// c10bb6.cpp -- rigorous completeness of the biunimodular vectors of length 10 (second version: interval AD,
// mean-value exclusion, 4D Krawczyk on the reduced system, degenerate directions by an interval linear solve,
// modulation symmetry).  See c10bb.cpp for the reduction:
//   z_n = u_{n mod 5} (n even), v_{n mod 5} (n odd);  U_k = sum_m u_m w^{-3km}, V_k likewise (w = e^{2 pi i/5});
//   z biunimodular  <=>  |u_m| = |v_m| = 1,  Re(U_k conj V_k) = 0,  |U_k|^2 + |V_k|^2 = 10  (k = 0..4).
// Normalisation u_0 = z_0 = 1; u_m (m = 1..4) on two overlapping rational charts  u = +-((1 - s^2) + 2 i s)/(1 + s^2), s in [-17/16, 17/16].
// Symmetry (modulation z_n -> e^{2 pi i a n/10} z_n, which fixes z_0 and maps (U_k, V_k) -> (U_{k-a5}, (-1)^{a2} V_{k-a5})):
//   every biunimodular z is a modulation of one with |U_0|^2 = max_k |U_k|^2 and eps_0 = +1 (U_0 != 0 since
//   sum_k |U_k|^2 = 25).  The certified set is checked to be closed under the 10 modulations.
// Per box S (s-box, sign pattern eps, chart signs):
//   nondegenerate k:  V_k = eps_k i U_k sqrt(10/|U_k|^2 - 1);
//   degenerate k (direction of U_k undetermined, or |V_k| small): V_k = X + iY from two of the linear equations
//      E_m:  Re(a_m V_k) = (1 - |c_m|^2 - rho^2/25) 5/2,  c_m = (1/5) sum_{j != k} V_j w^{3jm},  a_m = conj(c_m) w^{3km},
//      rho^2 = 10 - |U_k|^2, plus the equation |V_k|^2 = rho^2  (all signs of V_k covered);
//   residuals F_m = |v_m|^2 - 1.  EXCLUDE if a residual (natural or mean-value enclosure) misses 0, or |U_k|^2 > 10,
//   or |U_0|^2 < |U_k|^2 certainly.  KRAWCZYK on 4 equations (nondegenerate: F_0..F_4 minus one; degenerate:
//   |V_k|^2 - rho^2 and F_m for the 3 unused m): if K(S) in int S, S holds one zero s*; then either the leftover
//   equation excludes it, or the enclosure of x_n = z_{n+1}/z_n at K lies in a certified cyclic-10 uniqueness box (COVERED).
// Build: g++ -O2 -std=c++17 -fopenmp -ffp-contract=off -frounding-math c10bb6.cpp -o c10bb6
// Usage: c10bb2 roots.bin [maxsubs]     env: ANGMAX (0.5), VSMALL (0.05), KTRY (2^-4), NODECAP
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <string>
#include <sstream>
#include <omp.h>
using namespace std;
typedef complex<double> cd;

// ---------------- intervals ----------------
struct I { double lo, hi; };
static inline double dn(double x) { return nextafter(x, -INFINITY); }
static inline double up(double x) { return nextafter(x, INFINITY); }
static inline I ipt(double x) { return {x, x}; }
static atomic<long long> nOverflow(0);   // c10bb7 diagnostic: finite inputs whose enclosure overflowed
static const I UNIV = {-INFINITY, INFINITY};
static inline bool fin(I a) { return std::isfinite(a.lo) && std::isfinite(a.hi); }
// c10bb7 rule: a nonfinite input endpoint gives the universal interval; a finite computation that overflows is
// counted and its (valid, possibly infinite) enclosure kept.  Hence 0*inf, inf-inf, inf/inf never occur: no NaN.
static inline I chk(I r) { if (!fin(r)) nOverflow++; return r; }
static inline I iadd(I a, I b) { if (!fin(a) || !fin(b)) { return UNIV; } return chk({dn(a.lo + b.lo), up(a.hi + b.hi)}); }
static inline I isub(I a, I b) { if (!fin(a) || !fin(b)) { return UNIV; } return chk({dn(a.lo - b.hi), up(a.hi - b.lo)}); }
static inline I ineg(I a) { return {-a.hi, -a.lo}; }
static inline I imul(I a, I b) {
    if (!fin(a) || !fin(b)) { return UNIV; }
    double p1 = a.lo * b.lo, p2 = a.lo * b.hi, p3 = a.hi * b.lo, p4 = a.hi * b.hi;
    return chk({dn(min(min(p1, p2), min(p3, p4))), up(max(max(p1, p2), max(p3, p4)))});
}
static inline I isq(I a) {
    if (!fin(a)) { return {0.0, INFINITY}; }
    double l = fabs(a.lo), h = fabs(a.hi);
    if (a.lo <= 0 && a.hi >= 0) return chk({0.0, up(max(l, h) * max(l, h))});
    double m = min(l, h), M = max(l, h); return chk({dn(m * m), up(M * M)});
}
static inline bool zin(I a) { return !(a.lo > 0) && !(a.hi < 0); }      // NaN-safe: "zero not excluded"
static inline I irecip(I b) { if (!fin(b)) { return UNIV; } return chk({dn(1.0 / b.hi), up(1.0 / b.lo)}); }   // 0 not in b
static inline I idiv(I a, I b) { return imul(a, irecip(b)); }               // 0 not in b
static inline I isqrt(I a) { if (!fin(a)) { return {0.0, INFINITY}; } double lo = max(0.0, a.lo); return {lo > 0 ? dn(sqrt(lo)) : 0.0, up(sqrt(max(0.0, a.hi)))}; }
static inline I meet(I a, I b) { return {max(a.lo, b.lo), min(a.hi, b.hi)}; }
static inline I hull(I a, I b) { return {min(a.lo, b.lo), max(a.hi, b.hi)}; }
static inline bool inside(I a, I b) { return a.lo >= b.lo && a.hi <= b.hi; }
static inline double wid(I a) { return a.hi - a.lo; }
static inline double mid(I a) { return 0.5 * (a.lo + a.hi); }

// ---------------- AD intervals (value + gradient in s_0..s_3) ----------------
const int NV = 4;
struct D { I v; I g[NV]; };
static inline D dc(I v) { D r; r.v = v; for (int i = 0; i < NV; i++) r.g[i] = ipt(0.0); return r; }
static inline D dvar(I v, int j) { D r = dc(v); r.g[j] = ipt(1.0); return r; }
static inline D dadd(const D& a, const D& b) { D r; r.v = iadd(a.v, b.v); for (int i = 0; i < NV; i++) r.g[i] = iadd(a.g[i], b.g[i]); return r; }
static inline D dsub(const D& a, const D& b) { D r; r.v = isub(a.v, b.v); for (int i = 0; i < NV; i++) r.g[i] = isub(a.g[i], b.g[i]); return r; }
static inline D dneg(const D& a) { D r; r.v = ineg(a.v); for (int i = 0; i < NV; i++) r.g[i] = ineg(a.g[i]); return r; }
static inline D dmul(const D& a, const D& b) { D r; r.v = imul(a.v, b.v); for (int i = 0; i < NV; i++) r.g[i] = iadd(imul(a.v, b.g[i]), imul(b.v, a.g[i])); return r; }
static inline D dsq(const D& a) { D r; r.v = isq(a.v); I t = imul(ipt(2.0), a.v); for (int i = 0; i < NV; i++) r.g[i] = imul(t, a.g[i]); return r; }
static inline D dscal(const D& a, I c) { D r; r.v = imul(a.v, c); for (int i = 0; i < NV; i++) r.g[i] = imul(a.g[i], c); return r; }
static inline bool ddiv(const D& a, const D& b, D& r) {      // a/b, 0 not in b
    if (zin(b.v)) return false;
    I q = idiv(a.v, b.v); r.v = q;
    for (int i = 0; i < NV; i++) r.g[i] = idiv(isub(a.g[i], imul(q, b.g[i])), b.v);
    return true;
}
static inline bool dsqrt(const D& a, D& r) {                  // needs a > 0 on the box
    if (a.v.lo <= 0) return false;
    I s = isqrt(a.v); r.v = s; I h = irecip(imul(ipt(2.0), s));
    for (int i = 0; i < NV; i++) r.g[i] = imul(a.g[i], h);
    return true;
}
struct CD { D re, im; };
static inline CD cdc(I re, I im) { return {dc(re), dc(im)}; }
static inline CD cadd(const CD& a, const CD& b) { return {dadd(a.re, b.re), dadd(a.im, b.im)}; }
static inline CD csub(const CD& a, const CD& b) { return {dsub(a.re, b.re), dsub(a.im, b.im)}; }
static inline CD cmul(const CD& a, const CD& b) { return {dsub(dmul(a.re, b.re), dmul(a.im, b.im)), dadd(dmul(a.re, b.im), dmul(a.im, b.re))}; }
static inline CD cmulc(const CD& a, I cre, I cim) { return {dsub(dscal(a.re, cre), dscal(a.im, cim)), dadd(dscal(a.re, cim), dscal(a.im, cre))}; }
static inline CD cscal(const CD& a, const D& s) { return {dmul(a.re, s), dmul(a.im, s)}; }
static inline CD ctimesi(const CD& a) { return {dneg(a.im), a.re}; }
static inline D cabs2(const CD& a) { return dadd(dsq(a.re), dsq(a.im)); }
static inline CD cconj(const CD& a) { return {a.re, dneg(a.im)}; }

// ---------------- cyclic-10 Krawczyk (uniqueness boxes of the certified roots) ----------------
const int n10 = 10;
struct C { I re, im; };
static inline C pc(cd z) { return {ipt(z.real()), ipt(z.imag())}; }
static inline C padd(C a, C b) { return {iadd(a.re, b.re), iadd(a.im, b.im)}; }
static inline C psub(C a, C b) { return {isub(a.re, b.re), isub(a.im, b.im)}; }
static inline C pmul(C a, C b) { return {isub(imul(a.re, b.re), imul(a.im, b.im)), iadd(imul(a.re, b.im), imul(a.im, b.re))}; }
template <class T, class Mul, class Add, class Sub>
static void FJ(const T* x, T* F, T (*J)[n10], T one, T zero, Mul mul, Add add, Sub sub, bool wantJ) {
    for (int k = 1; k <= n10; k++) {
        T acc = zero; int nw = (k < n10) ? n10 : 1;
        T rowJ[n10]; for (int m = 0; m < n10; m++) rowJ[m] = zero;
        for (int j = 0; j < nw; j++) {
            int w[n10]; for (int r = 0; r < k; r++) w[r] = (j + r) % n10;
            T pre[n10 + 1], suf[n10 + 1]; pre[0] = one; for (int r = 0; r < k; r++) pre[r + 1] = mul(pre[r], x[w[r]]);
            acc = add(acc, pre[k]);
            if (wantJ) { suf[k] = one; for (int r = k - 1; r >= 0; r--) suf[r] = mul(suf[r + 1], x[w[r]]);
                for (int r = 0; r < k; r++) rowJ[w[r]] = add(rowJ[w[r]], mul(pre[r], suf[r + 1])); }
        }
        F[k - 1] = (k < n10) ? acc : sub(acc, one);
        if (wantJ) for (int m = 0; m < n10; m++) J[k - 1][m] = rowJ[m];
    }
}
static void FJd(const cd* x, cd* F, cd (*J)[n10], bool wantJ) {
    FJ<cd>(x, F, J, cd(1), cd(0), [](cd a, cd b) { return a * b; }, [](cd a, cd b) { return a + b; }, [](cd a, cd b) { return a - b; }, wantJ);
}
static void FJi(const C* x, C* F, C (*J)[n10], bool wantJ) { FJ<C>(x, F, J, pc(1.0), pc(0.0), pmul, padd, psub, wantJ); }
template <int N> static bool cinverse(cd (*A)[N], cd (*Y)[N]) {
    cd M[N][2 * N];
    for (int i = 0; i < N; i++) for (int j = 0; j < 2 * N; j++) M[i][j] = j < N ? A[i][j] : cd(i == j - N ? 1 : 0);
    for (int c = 0; c < N; c++) {
        int p = c; for (int i = c + 1; i < N; i++) if (abs(M[i][c]) > abs(M[p][c])) p = i;
        if (abs(M[p][c]) == 0) return false;
        for (int j = 0; j < 2 * N; j++) swap(M[c][j], M[p][j]);
        cd f = 1.0 / M[c][c]; for (int j = 0; j < 2 * N; j++) M[c][j] *= f;
        for (int i = 0; i < N; i++) if (i != c && M[i][c] != 0.0) { cd g = M[i][c]; for (int j = 0; j < 2 * N; j++) M[i][j] -= g * M[c][j]; }
    }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) Y[i][j] = M[i][N + j];
    return true;
}
static void krawczyk10(const cd* xt, cd (*Y)[n10], const C* X, C* K) {
    C xp[n10], Fx[n10], JX[n10][n10], dummy[n10][n10];
    for (int i = 0; i < n10; i++) xp[i] = pc(xt[i]);
    FJi(xp, Fx, dummy, false); FJi(X, dummy[0], JX, true);
    C Dl[n10]; for (int j = 0; j < n10; j++) Dl[j] = psub(X[j], xp[j]);
    for (int i = 0; i < n10; i++) {
        C yf = pc(0.0); for (int k = 0; k < n10; k++) yf = padd(yf, pmul(pc(Y[i][k]), Fx[k]));
        C s = psub(xp[i], yf);
        for (int j = 0; j < n10; j++) { C m = pc(i == j ? 1.0 : 0.0);
            for (int k = 0; k < n10; k++) m = psub(m, pmul(pc(Y[i][k]), JX[k][j])); s = padd(s, pmul(m, Dl[j])); }
        K[i] = s;
    }
}
struct RootBox { cd ctr[n10]; C X[n10]; C K[n10]; double rad; };
static vector<RootBox> RB;
static vector<pair<double,int>> RBsorted;
static double RBmaxrad = 0;
static int find_box(const C* x) {           // a certified box containing the enclosure x, or -1
    double mx = mid(x[0].re), xw = 0;
    for (int n = 0; n < n10; n++) xw = max(xw, max(wid(x[n].re), wid(x[n].im)));
    auto it = lower_bound(RBsorted.begin(), RBsorted.end(), make_pair(mx - 2 * RBmaxrad - xw, -1));
    for (; it != RBsorted.end() && it->first <= mx + 2 * RBmaxrad + xw; ++it) {
        const RootBox& r = RB[it->second]; bool ok = true;
        for (int n = 0; n < n10 && ok; n++) ok = inside(x[n].re, r.X[n].re) && inside(x[n].im, r.X[n].im);
        if (ok) return it->second;
    }
    return -1;
}

// ---------------- reduced system ----------------
static I W5re[5], W5im[5];                   // w^j
static double ANGMAX = 0.5, VSMALL = 0.05, KTRY = 1.0 / 16, ADW = 1.0 / 8;
static bool NOSYM = false, DBGP = false;
static long long NODECAP = 0;
static atomic<long long> nNodes{0}, nExcl{0}, nCov{0}, nFail{0}, nKraw{0}, nDegen{0};
static vector<atomic<int>> hit;
static atomic<long long> HPX[64], HAX[64], HK[64], HC[64], HS[64], HBIG[64];   // prescreen-excl, AD-excl, Kraw tried, covered, split, split above ADW

struct Box { I s[NV]; };
struct Eval {  // residuals: all vanish at a genuine solution                                // result of one evaluation
    bool ok;                                 // false: AD failed (singular) -> only natural tests were possible
    bool excluded;
    int nres; D res[8];                      // residual list (all must vanish at a genuine solution)
    int kraw[4]; int nk; int extra;          // indices of 4 Krawczyk equations and the leftover one (-1 none)
    int degen, m1, m2;                       // BRANCH DESCRIPTOR: degenerate index (-1 none) and linear-solve pair
    CD z[n10];
};

static inline CD chartD(int sgn, const D& s) {
    D s2 = dsq(s); D den = dadd(dc(ipt(1.0)), s2);                 // den >= 1
    D re, im; ddiv(dsub(dc(ipt(1.0)), s2), den, re); ddiv(dscal(s, ipt(2.0)), den, im);
    if (sgn < 0) { re = dneg(re); im = dneg(im); }
    re.v = meet(re.v, I{-1.0, 1.0}); im.v = meet(im.v, I{-1.0, 1.0});
    return {re, im};
}

// evaluate on box (point box for centre evaluation); returns residual enclosures (value + gradient)
// force != nullptr: evaluate EXACTLY the residual functions of force's branch (same degenerate index and pair),
// so that values, derivatives and Krawczyk data always refer to the same functions (audit AUD3, blocking item 1).
static void evaluate(const Box& B, const int* eps, const int* ch, bool point, Eval& E, bool sym = true, const Eval* force = nullptr) {
    E.ok = true; E.excluded = false; E.nres = 0; E.extra = -1; E.nk = 0; E.degen = -1; E.m1 = -1; E.m2 = -1;
    CD u[5]; u[0] = cdc(ipt(1.0), ipt(0.0));
    for (int m = 1; m < 5; m++) {
        D s = point ? dc(B.s[m - 1]) : dvar(B.s[m - 1], m - 1);
        if (point) s.g[m - 1] = ipt(1.0);
        u[m] = chartD(ch[m - 1], s);
    }
    CD U[5]; D A[5];
    for (int k = 0; k < 5; k++) {
        CD acc = cdc(ipt(0.0), ipt(0.0));
        for (int m = 0; m < 5; m++) { int j = ((-3 * k * m) % 5 + 5) % 5; acc = cadd(acc, cmulc(u[m], W5re[j], W5im[j])); }
        U[k] = acc; A[k] = cabs2(acc);
        if (A[k].v.lo > 10.0) { E.excluded = true; return; }
    }
    if (!NOSYM && sym) { for (int k = 1; k < 5; k++) if (A[0].v.hi < A[k].v.lo) { E.excluded = true; return; }   // |U_0|^2 maximal (symmetry)
    for (int k = 2; k < 5; k++) if (A[1].v.hi < A[k].v.lo) { E.excluded = true; return; } }  // |U_1|^2 maximal among 1..4 (dilations)
    // classify k.  The linear solve goes to an index whose DIRECTION is undetermined (U_k may vanish / poorly
    // resolved); only if there is none, to the index with the smallest |V_k| (|U_k|^2 near 10).  Other problematic
    // indices use value-only enclosures (tight when |V_k| is small).
    int degen = -1; double worst = -1, vbest = INFINITY; int vcand = -1;
    for (int k = 0; k < 5; k++) {
        double wU = max(wid(U[k].re.v), wid(U[k].im.v));
        double angunc = (A[k].v.lo > 0) ? wU / sqrt(A[k].v.lo) : INFINITY;
        bool dirbad = (A[k].v.lo <= 0) || angunc > ANGMAX;
        if (dirbad && angunc > worst) { worst = angunc; degen = k; }
        double gap = 10.0 - A[k].v.hi;
        if (gap < VSMALL && gap < vbest) { vbest = gap; vcand = k; }
    }
    if (degen < 0) degen = vcand;
    if (force) degen = force->degen;
    E.degen = degen;
    CD V[5];
    for (int k = 0; k < 5; k++) {
        if (k == degen) continue;
        bool valid = A[k].v.lo > 0 && A[k].v.hi < 10.0;
        if (valid) {
            D tenoverA; ddiv(dc(ipt(10.0)), A[k], tenoverA);
            D t = dsub(tenoverA, dc(ipt(1.0))), g;
            if (dsqrt(t, g)) {
                CD vk = cscal(ctimesi(U[k]), g);
                if (eps[k] < 0) vk = {dneg(vk.re), dneg(vk.im)};
                V[k] = vk; continue;
            }
        }
        // value-only fallback (no usable derivative): V_k = eps i dir * sqrt(10 - A), dir clipped to the unit box
        E.ok = false;
        I rho = isqrt(isub(ipt(10.0), meet(A[k].v, I{0.0, 10.0})));
        I dre = {-1.0, 1.0}, dim = {-1.0, 1.0};
        if (A[k].v.lo > 0) { I inv = {dn(1.0 / up(sqrt(A[k].v.hi))), up(1.0 / dn(sqrt(A[k].v.lo)))};
            dre = meet(imul(U[k].re.v, inv), I{-1.0, 1.0}); dim = meet(imul(U[k].im.v, inv), I{-1.0, 1.0}); }
        I vr = imul(ineg(dim), rho), vi = imul(dre, rho);
        if (eps[k] < 0) { vr = ineg(vr); vi = ineg(vi); }
        { double wU = max(wid(U[k].re.v), wid(U[k].im.v));
          if (!(A[k].v.lo > 0) || wU / sqrt(A[k].v.lo) > 2.0) { vr = I{-rho.hi, rho.hi}; vi = I{-rho.hi, rho.hi}; } }   // both signs
        V[k] = {dc(vr), dc(vi)};
        for (int i = 0; i < NV; i++) { V[k].re.g[i] = I{-INFINITY, INFINITY}; V[k].im.g[i] = I{-INFINITY, INFINITY}; }
    }
    CD v[5];
    if (degen < 0) {
        for (int m = 0; m < 5; m++) {
            CD acc = cdc(ipt(0.0), ipt(0.0));
            for (int k = 0; k < 5; k++) { int j = (3 * k * m) % 5; acc = cadd(acc, cmulc(V[k], W5re[j], W5im[j])); }
            v[m] = {dscal(acc.re, idiv(ipt(1.0), ipt(5.0))), dscal(acc.im, idiv(ipt(1.0), ipt(5.0)))};
            E.res[E.nres++] = dsub(cabs2(v[m]), dc(ipt(1.0)));
        }
        // Krawczyk set: drop the residual with the widest gradient (heuristic); leftover is checked separately
        E.nk = 4; int drop = 4; E.extra = drop;
        for (int i = 0, c = 0; i < 5; i++) if (i != drop) E.kraw[c++] = i;
    } else {
        int k = degen; nDegen++;
        I fifth = idiv(ipt(1.0), ipt(5.0));
        CD c[5], a[5]; D b[5];
        D rho2 = dsub(dc(ipt(10.0)), A[k]);
        for (int m = 0; m < 5; m++) {
            CD acc = cdc(ipt(0.0), ipt(0.0));
            for (int j = 0; j < 5; j++) if (j != k) { int e = (3 * j * m) % 5; acc = cadd(acc, cmulc(V[j], W5re[e], W5im[e])); }
            c[m] = {dscal(acc.re, fifth), dscal(acc.im, fifth)};
            int e = (3 * k * m) % 5;
            a[m] = cmulc(cconj(c[m]), W5re[e], W5im[e]);
            D rhs = dsub(dsub(dc(ipt(1.0)), cabs2(c[m])), dscal(rho2, idiv(ipt(1.0), ipt(25.0))));
            b[m] = dscal(rhs, idiv(ipt(5.0), ipt(2.0)));
        }
        // equations: Re(a_m) X - Im(a_m) Y = b_m ; choose the pair with the best determinant
        int m1 = -1, m2 = -1; double best = 0;
        if (force) {
            m1 = force->m1; m2 = force->m2;
            I det = isub(imul(ineg(a[m1].re.v), a[m2].im.v), imul(ineg(a[m1].im.v), a[m2].re.v));
            if (zin(det)) { E.ok = false; return; }
        } else
        for (int i = 0; i < 5; i++) for (int j = i + 1; j < 5; j++) {
            I det = isub(imul(ineg(a[i].re.v), a[j].im.v), imul(ineg(a[i].im.v), a[j].re.v));   // -Re a_i Im a_j + Im a_i Re a_j
            double q = zin(det) ? 0 : min(fabs(det.lo), fabs(det.hi));
            if (q > best) { best = q; m1 = i; m2 = j; }
        }
        if (m1 < 0) { E.ok = false; return; }   // (no residuals: the box is split)
        E.m1 = m1; E.m2 = m2;
        D det = dsub(dmul(dneg(a[m1].re), a[m2].im), dmul(dneg(a[m1].im), a[m2].re));
        D Xn = dsub(dmul(b[m1], dneg(a[m2].im)), dmul(dneg(a[m1].im), b[m2]));
        D Yn = dsub(dmul(a[m1].re, b[m2]), dmul(a[m2].re, b[m1]));
        D X, Y; if (!ddiv(Xn, det, X) || !ddiv(Yn, det, Y)) { E.ok = false; return; }
        V[k] = {X, Y};
        for (int m = 0; m < 5; m++) {
            int e = (3 * k * m) % 5;
            CD add = cmulc(V[k], W5re[e], W5im[e]); add = {dscal(add.re, fifth), dscal(add.im, fifth)};
            v[m] = cadd(c[m], add);
        }
        // residuals: |V_k|^2 - rho^2 and F_m for the three unused m (F_m1, F_m2 follow from these)
        E.res[E.nres++] = dsub(cabs2(V[k]), rho2);
        for (int m = 0; m < 5; m++) if (m != m1 && m != m2) E.res[E.nres++] = dsub(cabs2(v[m]), dc(ipt(1.0)));
        // also check F_m1, F_m2 naturally (consistency)
        E.res[E.nres++] = dsub(cabs2(v[m1]), dc(ipt(1.0)));
        E.res[E.nres++] = dsub(cabs2(v[m2]), dc(ipt(1.0)));
        // the orthogonality Re(conj(U_k) V_k) = 0 is a genuine equation of the system and must be imposed
        E.res[E.nres++] = dadd(dmul(U[k].re, V[k].re), dmul(U[k].im, V[k].im));
        E.nk = 4; for (int i = 0; i < 4; i++) E.kraw[i] = i; E.extra = 0;     // extra >= 0: choose the best 4
    }
    for (int n = 0; n < n10; n++) E.z[n] = (n % 2 == 0) ? u[n % 5] : v[n % 5];
}

// value-only evaluation (cheap pre-screen); 0 = excluded, 1 = not excluded
static inline C cmulcC(C a, I cre, I cim) { return {isub(imul(a.re, cre), imul(a.im, cim)), iadd(imul(a.re, cim), imul(a.im, cre))}; }
static int prescreen(const Box& B, const int* eps, const int* ch) {
    C u[5]; u[0] = {ipt(1.0), ipt(0.0)};
    for (int m = 1; m < 5; m++) {
        I s = B.s[m - 1]; I s2 = isq(s); I den = iadd(ipt(1.0), s2);
        I re = meet(idiv(isub(ipt(1.0), s2), den), I{-1.0, 1.0}), im = meet(idiv(imul(ipt(2.0), s), den), I{-1.0, 1.0});
        if (ch[m - 1] < 0) { re = ineg(re); im = ineg(im); }
        u[m] = {re, im};
    }
    C U[5]; I A[5];
    for (int k = 0; k < 5; k++) {
        C acc = {ipt(0.0), ipt(0.0)};
        for (int m = 0; m < 5; m++) { int j = ((-3 * k * m) % 5 + 5) % 5; acc = padd(acc, cmulcC(u[m], W5re[j], W5im[j])); }
        U[k] = acc; A[k] = iadd(isq(acc.re), isq(acc.im));
        if (A[k].lo > 10.0) return 0;
    }
    if (!NOSYM) { for (int k = 1; k < 5; k++) if (A[0].hi < A[k].lo) return 0;          // |U_0|^2 maximal
    for (int k = 2; k < 5; k++) if (A[1].hi < A[k].lo) return 0; }       // |U_1|^2 maximal among 1..4 (dilations)
    C V[5];
    for (int k = 0; k < 5; k++) {
        I Ak = meet(A[k], I{0.0, 10.0});
        I rho = isqrt(isub(ipt(10.0), Ak));
        I dre = {-1.0, 1.0}, dim = {-1.0, 1.0};
        if (Ak.lo > 0) { I inv = {dn(1.0 / up(sqrt(Ak.hi))), up(1.0 / dn(sqrt(Ak.lo)))};
            dre = meet(imul(U[k].re, inv), I{-1.0, 1.0}); dim = meet(imul(U[k].im, inv), I{-1.0, 1.0}); }
        I vr = imul(ineg(dim), rho), vi = imul(dre, rho);
        double wU = max(wid(U[k].re), wid(U[k].im));
        bool dirok = Ak.lo > 0 && wU / sqrt(Ak.lo) <= 2.0;
        if (!dirok) { vr = I{-rho.hi, rho.hi}; vi = I{-rho.hi, rho.hi}; }          // direction free (covers both signs)
        else if (eps[k] < 0) { vr = ineg(vr); vi = ineg(vi); }
        V[k] = {vr, vi};
    }
    I fifth = idiv(ipt(1.0), ipt(5.0));
    for (int m = 0; m < 5; m++) {
        C acc = {ipt(0.0), ipt(0.0)};
        for (int k = 0; k < 5; k++) { int j = (3 * k * m) % 5; acc = padd(acc, cmulcC(V[k], W5re[j], W5im[j])); }
        I a2 = imul(iadd(isq(acc.re), isq(acc.im)), imul(fifth, fifth));
        if (a2.lo > 1.0 || a2.hi < 1.0) return 0;
    }
    return 1;
}

static bool x_of_z(const CD* z, C* x) {     // x_n = z_{n+1}/z_n = conj(z_n) z_{n+1} / |z_n|^2
    for (int n = 0; n < n10; n++) {
        I a2 = iadd(isq(z[n].re.v), isq(z[n].im.v)); if (a2.lo <= 0) return false;
        const CD& p = z[n]; const CD& q = z[(n + 1) % n10];
        I re = iadd(imul(p.re.v, q.re.v), imul(p.im.v, q.im.v)), im = isub(imul(p.re.v, q.im.v), imul(p.im.v, q.re.v));
        x[n] = {idiv(re, a2), idiv(im, a2)};
    }
    return true;
}

static int krawczyk_cover(const Box& B0, const Box& B, Eval& E, Eval& Ec, Box& Cb, const int* eps, const int* ch, int& rootout);
// 0 excluded, 1 covered, 2 split
static int process(const Box& B, const int* eps, const int* ch, int& rootout) {
    rootout = -1;
    double w0 = 0; for (int i = 0; i < NV; i++) w0 = max(w0, wid(B.s[i]));
    int lv = min(63, max(0, (int)floor(-log2(w0))));
    if (!prescreen(B, eps, ch)) { HPX[lv]++; return 0; }
    static atomic<int> deepdbg{0};
    bool deep = (lv >= 36) && getenv("DEEPDBG") && deepdbg.fetch_add(1) < 3;
    if (deep) { DBGP = true; fprintf(stderr, "DEEP box lv %d s=(%.15g %.15g %.15g %.15g) eps=%d%d%d%d%d ch=%d%d%d%d\n", lv, mid(B.s[0]), mid(B.s[1]), mid(B.s[2]), mid(B.s[3]), eps[0], eps[1], eps[2], eps[3], eps[4], ch[0], ch[1], ch[2], ch[3]); }
    struct Off { ~Off() { } } off_;
    if (w0 > ADW) { HBIG[lv]++; return 2; }
    Eval E; evaluate(B, eps, ch, false, E);
    if (E.excluded) { HAX[lv]++; return 0; }
    for (int r = 0; r < E.nres; r++) if (!zin(E.res[r].v)) { HAX[lv]++; return 0; }
    if (!E.ok) { HS[lv]++; return 2; }
    Box Cb; for (int i = 0; i < NV; i++) Cb.s[i] = ipt(mid(B.s[i]));
    Eval Ec; evaluate(Cb, eps, ch, true, Ec, false, &E);   // same branch as E; no symmetry test at the centre (a point)
    if (Ec.excluded || !Ec.ok || Ec.nres != E.nres || Ec.degen != E.degen || Ec.m1 != E.m1 || Ec.m2 != E.m2) return 2;
    // mean-value enclosures
    for (int r = 0; r < E.nres; r++) {
        I mv = Ec.res[r].v;
        for (int i = 0; i < NV; i++) mv = iadd(mv, imul(E.res[r].g[i], isub(B.s[i], Cb.s[i])));
        if (!zin(meet(mv, E.res[r].v)) || mv.lo > E.res[r].v.hi || mv.hi < E.res[r].v.lo) { HAX[lv]++; return 0; }
    }
    double w = 0; for (int i = 0; i < NV; i++) w = max(w, wid(B.s[i]));
    if (DBGP) fprintf(stderr, "   [dbg] passed MV, w=%.2e ok=%d nres=%d extra=%d\n", w, E.ok, E.nres, E.extra);
    if (w > KTRY) { HS[lv]++; return 2; }
    HK[lv]++;
    // Krawczyk on an INFLATED box B' (10% each side): a unique zero of the 4 equations in B' contains every genuine
    // solution in B, so solutions lying on faces of the bisection grid (e.g. u_m = +-1, s = 0) are interior to B'.
    nKraw++;
    {
        Box Bi; for (int i = 0; i < NV; i++) { double h = 0.1 * wid(B.s[i]); Bi.s[i] = {dn(B.s[i].lo - h), up(B.s[i].hi + h)}; }
        Eval Ei; evaluate(Bi, eps, ch, false, Ei, false, &E);   // inflated box, same branch; no symmetry test
        if (Ei.excluded || !Ei.ok || Ei.nres != E.nres || Ei.degen != E.degen || Ei.m1 != E.m1 || Ei.m2 != E.m2) return 2;
        E = Ei;                                // gradients over B' (used below with the same centre)
        // the box used by the Krawczyk test is B'
        return krawczyk_cover(B, Bi, E, Ec, Cb, eps, ch, rootout);
    }
}
static int krawczyk_cover(const Box& B0, const Box& B, Eval& E, Eval& Ec, Box& Cb, const int* eps, const int* ch, int& rootout) {
    double w00 = 0; for (int i = 0; i < NV; i++) w00 = max(w00, wid(B0.s[i]));
    int lv = min(63, max(0, (int)floor(-log2(w00))));
    if (E.extra >= 0) {             // choose the 4 residuals with the best-conditioned point Jacobian
        double bestdet = -1; int best[4] = {0, 1, 2, 3}; int nr = E.nres;
        for (int a = 0; a < nr; a++) for (int b = a + 1; b < nr; b++) for (int c = b + 1; c < nr; c++) for (int d = c + 1; d < nr; d++) {
            int idx[4] = {a, b, c, d}; double T[4][4];
            for (int r2 = 0; r2 < 4; r2++) { double nrm = 0; for (int i = 0; i < 4; i++) { T[r2][i] = mid(Ec.res[idx[r2]].g[i]); nrm += T[r2][i] * T[r2][i]; }
                nrm = sqrt(nrm); if (nrm > 0) for (int i = 0; i < 4; i++) T[r2][i] /= nrm; }        // row-normalised
            double det = 1;
            for (int cc = 0; cc < 4; cc++) { int pp = cc; for (int a2 = cc + 1; a2 < 4; a2++) if (fabs(T[a2][cc]) > fabs(T[pp][cc])) pp = a2;
                if (T[pp][cc] == 0) { det = 0; break; } if (pp != cc) for (int j = 0; j < 4; j++) swap(T[cc][j], T[pp][j]);
                det *= T[cc][cc]; for (int a2 = cc + 1; a2 < 4; a2++) { double g = T[a2][cc] / T[cc][cc]; for (int j = cc; j < 4; j++) T[a2][j] -= g * T[cc][j]; } }
            if (fabs(det) > bestdet) { bestdet = fabs(det); for (int i = 0; i < 4; i++) best[i] = idx[i]; }
        }
        for (int i = 0; i < 4; i++) E.kraw[i] = best[i];
    }
    double Jm[4][4]; double Gc[4];
    for (int a = 0; a < 4; a++) { const D& R = E.res[E.kraw[a]]; Gc[a] = mid(Ec.res[E.kraw[a]].v);
        for (int i = 0; i < 4; i++) Jm[a][i] = mid(Ec.res[E.kraw[a]].g[i]); }
    // invert the point Jacobian (real 4x4) with partial pivoting
    double M[4][8]; for (int a = 0; a < 4; a++) for (int j = 0; j < 8; j++) M[a][j] = j < 4 ? Jm[a][j] : (a == j - 4 ? 1.0 : 0.0);
    for (int c = 0; c < 4; c++) { int p = c; for (int a = c + 1; a < 4; a++) if (fabs(M[a][c]) > fabs(M[p][c])) p = a;
        if (M[p][c] == 0) return 2; for (int j = 0; j < 8; j++) swap(M[c][j], M[p][j]);
        double f = 1.0 / M[c][c]; for (int j = 0; j < 8; j++) M[c][j] *= f;
        for (int a = 0; a < 4; a++) if (a != c) { double g = M[a][c]; for (int j = 0; j < 8; j++) M[a][j] -= g * M[c][j]; } }
    double Y[4][4]; for (int a = 0; a < 4; a++) for (int j = 0; j < 4; j++) Y[a][j] = M[a][4 + j];
    Box K;
    for (int i = 0; i < 4; i++) {
        I yg = ipt(0.0);
        for (int a = 0; a < 4; a++) yg = iadd(yg, imul(ipt(Y[i][a]), Ec.res[E.kraw[a]].v));
        I s = isub(Cb.s[i], yg);
        for (int j = 0; j < 4; j++) {
            I m = ipt(i == j ? 1.0 : 0.0);
            for (int a = 0; a < 4; a++) m = isub(m, imul(ipt(Y[i][a]), E.res[E.kraw[a]].g[j]));
            s = iadd(s, imul(m, isub(B.s[j], Cb.s[j])));
        }
        K.s[i] = s;
        if (!(s.lo > B.s[i].lo && s.hi < B.s[i].hi)) { if (DBGP) fprintf(stderr, "   [dbg] Krawczyk not inside: i=%d K=[%.6g,%.6g] B=[%.6g,%.6g]\n", i, s.lo, s.hi, B.s[i].lo, B.s[i].hi); return 2; }
    }
    // unique zero s* of the 4 equations in B, s* in K.  Tighten by iterating on K (s* in K, K in B).
    for (int it = 0; it < 3; it++) {
        Eval Ek; evaluate(K, eps, ch, false, Ek, false, &E);
        Box Kc; for (int i = 0; i < NV; i++) Kc.s[i] = ipt(mid(K.s[i]));
        Eval Ekc; evaluate(Kc, eps, ch, true, Ekc, false, &E);
        if (!Ek.ok || !Ekc.ok || Ek.excluded || Ekc.excluded || Ek.nres != E.nres || Ekc.nres != E.nres) break;   // AUD3 item 2
        Box K2; bool good = true;
        for (int i = 0; i < 4 && good; i++) {
            I yg = ipt(0.0);
            for (int a = 0; a < 4; a++) yg = iadd(yg, imul(ipt(Y[i][a]), Ekc.res[E.kraw[a]].v));
            I s = isub(Kc.s[i], yg);
            for (int j = 0; j < 4; j++) { I m = ipt(i == j ? 1.0 : 0.0);
                for (int a = 0; a < 4; a++) m = isub(m, imul(ipt(Y[i][a]), Ek.res[E.kraw[a]].g[j]));
                s = iadd(s, imul(m, isub(K.s[j], Kc.s[j]))); }
            K2.s[i] = meet(s, K.s[i]); if (K2.s[i].lo > K2.s[i].hi) good = false;
        }
        if (!good) return 2;          // cannot happen in exact arithmetic (the zero lies in K); be conservative
        K = K2;
    }
    Eval Ek; evaluate(K, eps, ch, false, Ek, true, &E);
    if (Ek.excluded) return 0;
    for (int r = 0; r < Ek.nres; r++) if (!zin(Ek.res[r].v)) return 0;     // leftover equation excludes s*
    if (!Ek.ok) return 2;
    C x[n10]; if (!x_of_z(Ek.z, x)) return 2;
    int r = find_box(x);
    if (DBGP) { double xw = 0; for (int n = 0; n < n10; n++) xw = max(xw, max(wid(x[n].re), wid(x[n].im)));
        double kw = 0; for (int i = 0; i < NV; i++) kw = max(kw, wid(K.s[i]));
        fprintf(stderr, "   [dbg] Krawczyk OK, K width %.2e, x width %.2e, box found %d, x0=(%.12f,%.12f)\n", kw, xw, r, mid(x[0].re), mid(x[0].im)); }
    if (r < 0) return 2;
    rootout = r; HC[lv]++; return 1;
}

struct TaskStat { long long nodes = 0, fails = 0, covered = 0, excluded = 0; bool capped = false; };
static TaskStat solve_box(const int* eps, const int* ch, const Box& root) {
    TaskStat T;
    vector<Box> st; st.push_back(root);
    while (!st.empty()) {
        Box B = st.back(); st.pop_back(); T.nodes++;
        if (NODECAP && T.nodes > NODECAP) { T.capped = true; T.fails++; nFail++; break; }
        int rt; int res = process(B, eps, ch, rt);
        if (res == 0) { nExcl++; T.excluded++; continue; }
        if (res == 1) { nCov++; T.covered++; hit[rt].store(1); continue; }
        int wv = 0; double w = 0; for (int i = 0; i < NV; i++) if (wid(B.s[i]) > w) { w = wid(B.s[i]); wv = i; }
        if (w < 1e-12) { T.fails++; nFail++; continue; }
        Box a = B, b = B; double md = mid(B.s[wv]); a.s[wv].hi = md; b.s[wv].lo = md;
        st.push_back(a); st.push_back(b);
    }
    nNodes += T.nodes;
    return T;
}
static void solve_sub(const int* eps, const int* ch) {
    Box root; for (int i = 0; i < NV; i++) root.s[i] = I{-1.0625, 1.0625};
    solve_box(eps, ch, root);
}

int main(int argc, char** argv) {
    auto T0 = chrono::steady_clock::now();
    auto el = [&]() { return chrono::duration<double>(chrono::steady_clock::now() - T0).count(); };
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: c10bb2 roots.bin [maxsubs]\n"); return 2; }
    if (getenv("ANGMAX")) ANGMAX = atof(getenv("ANGMAX"));
    if (getenv("VSMALL")) VSMALL = atof(getenv("VSMALL"));
    if (getenv("KTRY")) KTRY = atof(getenv("KTRY"));
    if (getenv("NODECAP")) NODECAP = atoll(getenv("NODECAP"));
    if (getenv("ADW")) ADW = atof(getenv("ADW"));
    {   volatile double tiny = 2.2250738585072014e-308, half = tiny / 2; volatile double a = 1.0, b = 3.0; volatile double q = a / b;
        bool ok = (half > 0) && (half * 2 == tiny) && (q == 0.333333333333333314829616256247390992939472198486328125);
        printf("ENV: binary64 round-to-nearest gradual-underflow %d ; compiler %s\n", (int)ok, __VERSION__); if (!ok) return 4; }
    printf("VERSION c10bb7 (c10bb6 + nonfinite-safe interval primitives, AUD38) built %s %s\n", __DATE__, __TIME__);
    I s5 = isqrt(ipt(5.0)), q4 = ipt(0.25);
    I c1 = imul(isub(s5, ipt(1.0)), q4), sn1 = imul(isqrt(iadd(ipt(10.0), imul(ipt(2.0), s5))), q4);
    I c2 = imul(ineg(iadd(s5, ipt(1.0))), q4), sn2 = imul(isqrt(isub(ipt(10.0), imul(ipt(2.0), s5))), q4);
    W5re[0] = ipt(1.0); W5im[0] = ipt(0.0); W5re[1] = c1; W5im[1] = sn1; W5re[2] = c2; W5im[2] = sn2;
    W5re[3] = c2; W5im[3] = ineg(sn2); W5re[4] = c1; W5im[4] = ineg(sn1);
    // certified cyclic-10 uniqueness boxes of the (numerically) unimodular roots, largest radius that passes
    FILE* f = fopen(argv[1], "rb"); int32_t N; if (fread(&N, 4, 1, f) != 1) return 3;
    vector<cd> R((size_t)N * n10); if (fread(R.data(), 16, R.size(), f) != R.size()) return 3; fclose(f);
    vector<int> uni;
    for (int t = 0; t < N; t++) { bool u = true; for (int i = 0; i < n10; i++) if (fabs(abs(R[(size_t)t * n10 + i]) - 1) > 1e-6) u = false; if (u) uni.push_back(t); }
    RB.resize(uni.size());
    const double rads[] = {3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12};
    int nok = 0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:nok)
    for (int a = 0; a < (int)uni.size(); a++) {
        cd x[n10]; for (int i = 0; i < n10; i++) x[i] = R[(size_t)uni[a] * n10 + i];
        cd F[n10], J[n10][n10], Y[n10][n10];
        for (int it = 0; it < 4; it++) { FJd(x, F, J, true); if (!cinverse<n10>(J, Y)) break;
            for (int i = 0; i < n10; i++) { cd dx = 0; for (int k = 0; k < n10; k++) dx += Y[i][k] * F[k]; x[i] -= dx; } }
        FJd(x, F, J, true); cinverse<n10>(J, Y);
        RB[a].rad = 0;
        for (double rho : rads) {
            C X[n10], K[n10];
            for (int i = 0; i < n10; i++) X[i] = {{dn(x[i].real() - rho), up(x[i].real() + rho)}, {dn(x[i].imag() - rho), up(x[i].imag() + rho)}};
            krawczyk10(x, Y, X, K);
            bool ok = true;
            for (int i = 0; i < n10 && ok; i++) ok = K[i].re.lo > X[i].re.lo && K[i].re.hi < X[i].re.hi && K[i].im.lo > X[i].im.lo && K[i].im.hi < X[i].im.hi;
            if (ok) {
                C Z[n10]; for (int i = 0; i < n10; i++) Z[i] = {meet(K[i].re, X[i].re), meet(K[i].im, X[i].im)};
                for (int it = 0; it < 4; it++) {          // the root lies in Z; K(Z) with centre mid(Z) in Z contains it
                    cd zc[n10]; for (int i = 0; i < n10; i++) zc[i] = cd(mid(Z[i].re), mid(Z[i].im));
                    C K2[n10]; krawczyk10(zc, Y, Z, K2);
                    for (int i = 0; i < n10; i++) Z[i] = {meet(K2[i].re, Z[i].re), meet(K2[i].im, Z[i].im)};
                }
                for (int i = 0; i < n10; i++) { RB[a].X[i] = X[i]; RB[a].K[i] = Z[i]; RB[a].ctr[i] = x[i]; } RB[a].rad = rho; break; }
        }
        if (RB[a].rad > 0) nok++;
    }
    double minr = 1; for (auto& r : RB) { RBmaxrad = max(RBmaxrad, r.rad); if (r.rad > 0) minr = min(minr, r.rad); }
    for (int a = 0; a < (int)RB.size(); a++) if (RB[a].rad > 0) RBsorted.push_back({RB[a].ctr[0].real(), a});
    sort(RBsorted.begin(), RBsorted.end());
    printf("roots %d ; unimodular (numerically) %zu ; uniqueness boxes certified %d, radius [%.0e, %.0e] ; %.1fs\n", N, uni.size(), nok, minr, RBmaxrad, el());
    // closure of the certified set under the 10 modulations x -> e^{2 pi i a/10} x (rigorous enclosures)
    I cpi5 = imul(iadd(s5, ipt(1.0)), q4), spi5 = imul(isqrt(isub(ipt(10.0), imul(ipt(2.0), s5))), q4);   // cos, sin (pi/5)
    C w10[10]; w10[0] = pc(1.0); C w1 = {cpi5, spi5}; for (int a = 1; a < 10; a++) w10[a] = pmul(w10[a - 1], w1);
    int closure_bad = 0;
    for (int a = 0; a < (int)RB.size(); a++) for (int b = 1; b < 10; b++) {
        // the root of box a lies in K = K(X) n X; its modulation is a cyclic-10 root (C_k(w x) = w^k C_k(x), prod = 1)
        C xs[n10]; for (int n = 0; n < n10; n++) xs[n] = pmul(RB[a].K[n], w10[b]);
        if (find_box(xs) < 0) closure_bad++;
    }
    // catalogue self-certification (AUD3): exact unimodularity of each root via the involution x -> 1/conj(x)
    // (sigma(K) inside the uniqueness box X  =>  sigma(root) = root), and pairwise disjointness of the enclosures K
    int uni_exact = 0;
    for (int a = 0; a < (int)RB.size(); a++) {
        bool ok = RB[a].rad > 0;
        for (int n = 0; n < n10 && ok; n++) {
            I a2 = iadd(isq(RB[a].K[n].re), isq(RB[a].K[n].im)); if (a2.lo <= 0) { ok = false; break; }
            I inv = irecip(a2); C sg = {imul(RB[a].K[n].re, inv), imul(RB[a].K[n].im, inv)};
            ok = inside(sg.re, RB[a].X[n].re) && inside(sg.im, RB[a].X[n].im);
        }
        uni_exact += ok;
    }
    vector<pair<double,int>> sw; for (int a = 0; a < (int)RB.size(); a++) sw.push_back({RB[a].K[0].re.lo, a});
    sort(sw.begin(), sw.end()); long long overlaps = 0;
    for (size_t i = 0; i < sw.size(); i++) for (size_t j = i + 1; j < sw.size() && sw[j].first <= RB[sw[i].second].K[0].re.hi; j++) {
        const RootBox &P = RB[sw[i].second], &Qb = RB[sw[j].second]; bool dis = false;
        for (int n = 0; n < n10 && !dis; n++) dis = P.K[n].re.hi < Qb.K[n].re.lo || Qb.K[n].re.hi < P.K[n].re.lo || P.K[n].im.hi < Qb.K[n].im.lo || Qb.K[n].im.hi < P.K[n].im.lo;
        if (!dis) overlaps++;
    }
    bool catalogue_ok = (uni_exact == 3040) && (overlaps == 0) && (nok == 3040) && (uni.size() == 3040);
    printf("CATALOGUE: %d of %zu roots exactly unimodular ; enclosures pairwise disjoint: %s ; %s\n", uni_exact, RB.size(), overlaps == 0 ? "yes" : "NO",
           catalogue_ok ? "OK" : "FAILED");
    printf("modulation closure: %s (images not found: %d)\n", closure_bad == 0 ? "OK" : "FAILED", closure_bad);
    int dil_bad = 0;
    for (int a = 0; a < (int)RB.size(); a++) {
        // z_n -> z_{3n} maps roots to roots (a dilation of Z_10 fixing 0); x'_n = z_{3n+3}/z_{3n} = x_{3n} x_{3n+1} x_{3n+2}
        C xs[n10];
        for (int n = 0; n < n10; n++) { int b = (3 * n) % 10; xs[n] = pmul(pmul(RB[a].K[b], RB[a].K[(b + 1) % 10]), RB[a].K[(b + 2) % 10]); }
        if (find_box(xs) < 0) dil_bad++;
    }
    printf("dilation closure: %s (images not found: %d)\n", dil_bad == 0 ? "OK" : "FAILED", dil_bad);
    closure_bad += dil_bad;
    hit = vector<atomic<int>>(RB.size()); for (auto& h : hit) h.store(0);
    if (getenv("TESTROOT")) {
        NOSYM = !getenv("SYMON"); DBGP = true;
        int a = atoi(getenv("TESTROOT"));
        cd x[n10]; for (int i = 0; i < n10; i++) x[i] = RB[a].ctr[i];
        cd z[n10]; z[0] = 1; for (int n = 0; n + 1 < n10; n++) z[n + 1] = x[n] * z[n];
        cd u[5], v[5]; for (int n = 0; n < n10; n++) (n % 2 == 0 ? u[n % 5] : v[n % 5]) = z[n];
        cd w5 = polar(1.0, 2 * M_PI / 5); cd U[5], V[5];
        for (int k = 0; k < 5; k++) { U[k] = 0; V[k] = 0; for (int m = 0; m < 5; m++) { U[k] += u[m] * pow(w5, -3 * k * m); V[k] += v[m] * pow(w5, -3 * k * m); } }
        int eps[5], ch[4]; double sv[4];
        for (int k = 0; k < 5; k++) eps[k] = (real(V[k] / (cd(0, 1) * U[k])) >= 0) ? 1 : -1;
        for (int m = 1; m < 5; m++) { cd q = u[m]; ch[m - 1] = q.real() >= 0 ? 1 : -1; if (ch[m - 1] < 0) q = -q; sv[m - 1] = q.imag() / (1 + q.real()); }
        printf("TEST root %d: |U|^2 = %.4f %.4f %.4f %.4f %.4f ; eps %d%d%d%d%d ; ch %d%d%d%d ; s = %.12f %.12f %.12f %.12f\n", a,
               norm(U[0]), norm(U[1]), norm(U[2]), norm(U[3]), norm(U[4]), eps[0], eps[1], eps[2], eps[3], eps[4], ch[0], ch[1], ch[2], ch[3], sv[0], sv[1], sv[2], sv[3]);
        for (double h : {3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6}) {
            Box B; for (int i = 0; i < NV; i++) B.s[i] = {sv[i] - h, sv[i] + h};
            int rt; int res = process(B, eps, ch, rt);
            printf("  half-width %.0e : result %d (0 excl, 1 covered, 2 split) root %d (expected %d)\n", h, res, rt, a);
            Box Bf = B; int fi = getenv("FACEI") ? atoi(getenv("FACEI")) : 2; Bf.s[fi] = {sv[fi], sv[fi] + 2 * h};
            res = process(Bf, eps, ch, rt);
            printf("  FACE box (coord %d lower face at the root) half-width %.0e : result %d root %d\n", fi, h, res, rt);
        }
        return 0;
    }
    vector<pair<int,int>> subs; for (int e = 0; e < 16; e++) for (int c = 0; c < 16; c++) subs.push_back({e << 1, c});   // eps_0 = +1
    if (getenv("SUBLIST")) {   // work-stealing mode over a list of whole subproblems (e.g. "40-139" or "0,1,5-9")
        vector<int> L; { string sl = getenv("SUBLIST"); stringstream ss(sl); string tok;
            while (getline(ss, tok, ',')) { size_t d = tok.find('-'); if (d == string::npos) L.push_back(stoi(tok)); else for (int q = stoi(tok.substr(0, d)); q <= stoi(tok.substr(d + 1)); q++) L.push_back(q); } }
        { vector<int> chk = L; sort(chk.begin(), chk.end());
          bool bad = chk.empty() || chk.front() < 0 || chk.back() > 255 || adjacent_find(chk.begin(), chk.end()) != chk.end();
          if (bad) { printf("SUBLIST invalid (empty, out of range or duplicate ids)\n"); return 5; } }
        struct Item { int sb; Box B; };
        deque<Item> Q; mutex mtx; condition_variable cv; int waiting = 0; bool finished = false; atomic<int> waitingA{0};
        static atomic<long long> outstanding[256], subfail[256], subnodes[256];
        for (int q = 0; q < 256; q++) { outstanding[q] = 0; subfail[q] = 0; subnodes[q] = 0; }
        for (int sb : L) { Item it; it.sb = sb; for (int i = 0; i < NV; i++) it.B.s[i] = I{-1.0625, 1.0625}; Q.push_back(it); outstanding[sb] = 1; }
        printf("WORK-STEALING MODE: %zu subproblems: %s ; threads %d\n", L.size(), getenv("SUBLIST"), omp_get_max_threads());
        atomic<int> subsdone{0}; long long totfail = 0;
        auto t0 = chrono::steady_clock::now();
        #pragma omp parallel
        {
            int NT = omp_get_num_threads();
            vector<Item> loc;
            for (;;) {
                if (loc.empty()) {
                    unique_lock<mutex> lk(mtx);
                    waiting++; waitingA++;
                    if (Q.empty() && waiting == NT) { finished = true; cv.notify_all(); }
                    cv.wait(lk, [&] { return finished || !Q.empty(); });
                    if (Q.empty()) break;                       // finished
                    waiting--; waitingA--;
                    loc.push_back(Q.front()); Q.pop_front();
                }
                Item it = loc.back(); loc.pop_back();
                int sb = it.sb;
                int eps[5], ch[4];
                for (int k = 0; k < 5; k++) eps[k] = (subs[sb].first >> k & 1) ? -1 : 1;
                for (int k = 0; k < 4; k++) ch[k] = (subs[sb].second >> k & 1) ? -1 : 1;
                subnodes[sb]++; nNodes++;
                int rt; int res = process(it.B, eps, ch, rt);
                long long delta = -1;
                if (res == 0) nExcl++;
                else if (res == 1) { nCov++; hit[rt].store(1); }
                else {
                    int wv = 0; double w = 0; for (int i = 0; i < NV; i++) if (wid(it.B.s[i]) > w) { w = wid(it.B.s[i]); wv = i; }
                    if (w < 1e-12) { subfail[sb]++; nFail++; }
                    else { Item a = it, b = it; double md = mid(it.B.s[wv]); a.B.s[wv].hi = md; b.B.s[wv].lo = md; loc.push_back(a); loc.push_back(b); delta = 1; }
                }
                long long left = (outstanding[sb] += delta);
                if (left == 0) {
                    int nd = ++subsdone;
                    fprintf(stderr, "SUBDONE %3d eps=%d%d%d%d%d ch=%d%d%d%d: nodes %lld fails %lld ; %d/%zu at %.0fs\n", sb, eps[0], eps[1], eps[2], eps[3], eps[4],
                            ch[0], ch[1], ch[2], ch[3], subnodes[sb].load(), subfail[sb].load(), nd, L.size(), chrono::duration<double>(chrono::steady_clock::now() - t0).count());
                }
                if (loc.size() >= 4 && waitingA.load() > 0) {          // share the oldest (largest) half
                    lock_guard<mutex> lk(mtx);
                    size_t h = loc.size() / 2;
                    for (size_t q = 0; q < h; q++) Q.push_back(loc[q]);
                    loc.erase(loc.begin(), loc.begin() + h);
                    cv.notify_all();
                }
            }
        }
        for (int sb : L) totfail += subfail[sb].load();
        printf("WORK-STEALING DONE: subproblems %d/%zu ; nodes %lld ; fails %lld ; %.1fs\n", subsdone.load(), L.size(), nNodes.load(), totfail, el());
        bool pass = (subsdone.load() == (int)L.size()) && totfail == 0 && catalogue_ok && closure_bad == 0;
        printf(pass ? "C10-BB7-SUBLIST: PASS\n" : "C10-BB7-SUBLIST: FAIL\n");
        printf("C10-BB7 NONFINITE: overflowed enclosures %lld (0 => identical arithmetic to c10bb6 on these nodes)\n", nOverflow.load());
        return pass ? 0 : 1;
    }
    if (getenv("TASK_LO") && getenv("TASK_HI")) {   // task mode: task t = piece (t % 16) of subproblem (t / 16)
        int lo = atoi(getenv("TASK_LO")), hi = atoi(getenv("TASK_HI"));
        printf("TASK MODE: tasks [%d, %d) of 4096 (subproblems %d..%d) ; NODECAP %lld per task\n", lo, hi, lo / 16, (hi - 1) / 16, NODECAP);
        long long tf = 0, tc = 0; int ndone = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:tf, tc, ndone)
        for (int q = 0; q < hi - lo; q++) {
            int t = hi - 1 - q;                                    // highest indices first
            int sb = t / 16, pc = t % 16;
            int eps[5], ch[4];
            for (int k = 0; k < 5; k++) eps[k] = (subs[sb].first >> k & 1) ? -1 : 1;
            for (int k = 0; k < 4; k++) ch[k] = (subs[sb].second >> k & 1) ? -1 : 1;
            Box root; for (int i = 0; i < NV; i++) root.s[i] = ((pc >> i) & 1) ? I{0.0, 1.0625} : I{-1.0625, 0.0};
            auto t1 = chrono::steady_clock::now();
            TaskStat T = solve_box(eps, ch, root);
            fprintf(stderr, "TASK %4d sub %3d piece %2d eps=%d%d%d%d%d ch=%d%d%d%d: nodes %lld excluded %lld covered %lld fails %lld capped %d ; %.1fs\n",
                    t, sb, pc, eps[0], eps[1], eps[2], eps[3], eps[4], ch[0], ch[1], ch[2], ch[3], T.nodes, T.excluded, T.covered, T.fails, (int)T.capped,
                    chrono::duration<double>(chrono::steady_clock::now() - t1).count());
            tf += T.fails; tc += T.capped; ndone++;
        }
        printf("TASKS [%d, %d): done %d ; fails %lld ; capped %lld ; %.1fs\n", lo, hi, ndone, tf, tc, el());
        printf((tf == 0 && tc == 0 && nok == (int)uni.size() && uni.size() == 3040 && closure_bad == 0) ? "C10-BB7-TASKS: PASS\n" : "C10-BB7-TASKS: FAIL\n");
        printf("C10-BB7 NONFINITE: overflowed enclosures %lld\n", nOverflow.load());
        return (tf == 0 && tc == 0 && catalogue_ok && closure_bad == 0) ? 0 : 1;
    }
    if (getenv("ONLYSUB")) { int o = atoi(getenv("ONLYSUB")); subs = {subs[o]}; printf("ONLY subproblem %d\n", o); }
    int maxsubs = (argc > 2) ? atoi(argv[2]) : (int)subs.size();
    if (maxsubs < (int)subs.size()) { subs.resize(maxsubs); printf("TIMING RUN: %d of 256 subproblems\n", maxsubs); }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < (int)subs.size(); i++) {
        int eps[5], ch[4];
        for (int k = 0; k < 5; k++) eps[k] = (subs[i].first >> k & 1) ? -1 : 1;
        for (int k = 0; k < 4; k++) ch[k] = (subs[i].second >> k & 1) ? -1 : 1;
        long long before = nNodes.load(); auto t1 = chrono::steady_clock::now();
        solve_sub(eps, ch);
        fprintf(stderr, "sub %3d eps=%d%d%d%d%d ch=%d%d%d%d: nodes %lld, %.1fs\n", i, eps[0], eps[1], eps[2], eps[3], eps[4], ch[0], ch[1], ch[2], ch[3],
                nNodes.load() - before, chrono::duration<double>(chrono::steady_clock::now() - t1).count());
    }
    printf("lvl: prescreen-excl  AD-excl  split(>ADW)  split(AD)  kraw-tried  covered\n");
    for (int l = 0; l < 64; l++) if (HPX[l] + HAX[l] + HBIG[l] + HS[l] + HK[l]) printf("%3d: %12lld %10lld %12lld %10lld %10lld %8lld\n", l, HPX[l].load(), HAX[l].load(), HBIG[l].load(), HS[l].load(), HK[l].load(), HC[l].load());
    int nh = 0; for (auto& h : hit) nh += h.load();
    printf("B&B: subproblems %zu ; nodes %lld ; excluded %lld ; covered %lld ; Krawczyk tries %lld ; degenerate evals %lld ; FAILED %lld ; certified rays reached %d ; %.1fs\n",
           subs.size(), nNodes.load(), nExcl.load(), nCov.load(), nKraw.load(), nDegen.load(), nFail.load(), nh, el());
    bool pass = nFail.load() == 0 && catalogue_ok && closure_bad == 0 && subs.size() == 256 && !getenv("ONLYSUB");
    printf(pass ? "C10-BB7: COMPLETE -- every biunimodular vector of length 10 is one of the 3040 certified rays\n" : "C10-BB7: INCOMPLETE\n");
        printf("C10-BB7 NONFINITE: overflowed enclosures %lld (0 => identical arithmetic to c10bb6 on these nodes)\n", nOverflow.load());
    return pass ? 0 : 1;
}
