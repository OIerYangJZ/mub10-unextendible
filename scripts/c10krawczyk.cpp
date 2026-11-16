// c10krawczyk.cpp -- rigorous enclosures for the cyclic-10 roots and the conservative overlap graphs (E63/E64 upgrade).
// System (complex, n = 10):  C_k(x) = sum_j prod_{r<k} x_{j+r} = 0 (k = 1..9),  x_0 ... x_9 - 1 = 0.
// For every approximate root:  Newton polish (double), then the Krawczyk test on the box X = xt + [-rho, rho](1 + i):
//     K(X) = xt - Y F(xt) + (I - Y J(X)) (X - xt),   Y ~ J(xt)^{-1};   K(X) in int X  =>  X holds exactly one root
// (real Krawczyk on R^20; complex rectangular interval arithmetic encloses the real 2x2 Jacobian blocks).  The root then
// lies in Xs = K(X) n X (iterated once more for tightness).
// Unimodularity (exact):  s(x) = 1/conj(x) maps roots to roots.  If s(Xs) is inside X, then s(x*) = x* by uniqueness,
// so |x*_j| = 1 for all j.  If some |Xs_j|^2 excludes 1, x* is not unimodular.  Anything else is reported UNDECIDED.
// Distinctness: the enclosures Xs are checked pairwise disjoint (sort on a random projection + sweep).
// Rays: z_0 = 1, z_{j+1} = x_j z_j (interval), |<z,z'>|^2 enclosed; with psi = z/sqrt10 the MU value 1/10 is 10 here.
//   O+ edge unless the enclosure excludes 0; U+ edge unless it excludes 10.  These contain every genuine edge.
// Interval arithmetic: round-to-nearest followed by one-ulp outward widening (nextafter) after every operation;
// compile WITHOUT fast-math and with -ffp-contract=off.
// Input: roots.bin = int32 n, then n x 10 complex128 (re, im).  Output: summary on stdout, graph_plus.bin (c10clique
// format), rays_plus.txt (root index of each certified unimodular root, and its centre).
// Build: g++ -O2 -std=c++17 -fopenmp -ffp-contract=off -frounding-math c10krawczyk.cpp -o c10krawczyk
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
using namespace std;
typedef complex<double> cd;
const int n = 10;

struct I { double lo, hi; };
static inline double dn(double x) { return nextafter(x, -INFINITY); }
static inline double up(double x) { return nextafter(x, INFINITY); }
static inline I ipt(double x) { return {x, x}; }
static inline I iadd(I a, I b) { return {dn(a.lo + b.lo), up(a.hi + b.hi)}; }
static inline I isub(I a, I b) { return {dn(a.lo - b.hi), up(a.hi - b.lo)}; }
static inline I imul(I a, I b) {
    double p1 = a.lo * b.lo, p2 = a.lo * b.hi, p3 = a.hi * b.lo, p4 = a.hi * b.hi;
    return {dn(min(min(p1, p2), min(p3, p4))), up(max(max(p1, p2), max(p3, p4)))};
}
static inline I isq(I a) {                                   // a^2 (>= 0)
    double l = fabs(a.lo), h = fabs(a.hi);
    if (a.lo <= 0 && a.hi >= 0) return {0.0, up(max(l, h) * max(l, h))};
    double m = min(l, h), M = max(l, h); return {dn(m * m), up(M * M)};
}
static inline I irecip_pos(I b) { return {dn(1.0 / b.hi), up(1.0 / b.lo)}; }   // b.lo > 0
static inline bool inside_strict(I a, I b) { return a.lo > b.lo && a.hi < b.hi; }
static inline bool inside(I a, I b) { return a.lo >= b.lo && a.hi <= b.hi; }
static inline bool disjoint(I a, I b) { return a.hi < b.lo || b.hi < a.lo; }
static inline I meet(I a, I b) { return {max(a.lo, b.lo), min(a.hi, b.hi)}; }
static inline bool has(I a, double v) { return a.lo <= v && v <= a.hi; }

struct C { I re, im; };
static inline C cpt(cd z) { return {ipt(z.real()), ipt(z.imag())}; }
static inline C cadd(C a, C b) { return {iadd(a.re, b.re), iadd(a.im, b.im)}; }
static inline C csub(C a, C b) { return {isub(a.re, b.re), isub(a.im, b.im)}; }
static inline C cmul(C a, C b) { return {isub(imul(a.re, b.re), imul(a.im, b.im)), iadd(imul(a.re, b.im), imul(a.im, b.re))}; }
static inline C cconjmul(C a, C b) { return {iadd(imul(a.re, b.re), imul(a.im, b.im)), isub(imul(a.re, b.im), imul(a.im, b.re))}; } // conj(a) b
static inline I cabs2(C a) { return iadd(isq(a.re), isq(a.im)); }

// ---- the system, generic in the scalar type ----
template <class T, class Mul, class Add, class Sub>
static void FJ(const T* x, T* F, T (*J)[n], T one, T zero, Mul mul, Add add, Sub sub, bool wantJ) {
    for (int k = 1; k <= n; k++) {
        T acc = zero; int nw = (k < n) ? n : 1;
        T rowJ[n]; for (int m = 0; m < n; m++) rowJ[m] = zero;
        for (int j = 0; j < nw; j++) {
            int w[n]; for (int r = 0; r < k; r++) w[r] = (j + r) % n;
            T pre[n + 1], suf[n + 1]; pre[0] = one; for (int r = 0; r < k; r++) pre[r + 1] = mul(pre[r], x[w[r]]);
            acc = add(acc, pre[k]);
            if (wantJ) {
                suf[k] = one; for (int r = k - 1; r >= 0; r--) suf[r] = mul(suf[r + 1], x[w[r]]);
                for (int r = 0; r < k; r++) rowJ[w[r]] = add(rowJ[w[r]], mul(pre[r], suf[r + 1]));
            }
        }
        F[k - 1] = (k < n) ? acc : sub(acc, one);
        if (wantJ) for (int m = 0; m < n; m++) J[k - 1][m] = rowJ[m];
    }
}
static void FJd(const cd* x, cd* F, cd (*J)[n], bool wantJ) {
    FJ<cd>(x, F, J, cd(1), cd(0), [](cd a, cd b) { return a * b; }, [](cd a, cd b) { return a + b; }, [](cd a, cd b) { return a - b; }, wantJ);
}
static void FJi(const C* x, C* F, C (*J)[n], bool wantJ) {
    FJ<C>(x, F, J, cpt(1.0), cpt(0.0), cmul, cadd, csub, wantJ);
}
static bool inverse(cd (*A)[n], cd (*Y)[n]) {
    cd M[n][2 * n];
    for (int i = 0; i < n; i++) for (int j = 0; j < 2 * n; j++) M[i][j] = j < n ? A[i][j] : cd(i == j - n ? 1 : 0);
    for (int c = 0; c < n; c++) {
        int p = c; for (int i = c + 1; i < n; i++) if (abs(M[i][c]) > abs(M[p][c])) p = i;
        if (abs(M[p][c]) == 0) return false;
        for (int j = 0; j < 2 * n; j++) swap(M[c][j], M[p][j]);
        cd f = 1.0 / M[c][c]; for (int j = 0; j < 2 * n; j++) M[c][j] *= f;
        for (int i = 0; i < n; i++) if (i != c && M[i][c] != 0.0) { cd g = M[i][c]; for (int j = 0; j < 2 * n; j++) M[i][j] -= g * M[c][j]; }
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Y[i][j] = M[i][n + j];
    return true;
}
static void krawczyk(const cd* xt, cd (*Y)[n], const C* X, C* K) {
    C xp[n], Fx[n], JX[n][n], dummy[n][n];
    for (int i = 0; i < n; i++) xp[i] = cpt(xt[i]);
    FJi(xp, Fx, dummy, false);
    FJi(X, dummy[0], JX, true);
    C D[n]; for (int j = 0; j < n; j++) D[j] = csub(X[j], xp[j]);
    for (int i = 0; i < n; i++) {
        C yf = cpt(0.0);
        for (int k = 0; k < n; k++) yf = cadd(yf, cmul(cpt(Y[i][k]), Fx[k]));
        C s = csub(xp[i], yf);
        for (int j = 0; j < n; j++) {
            C m = cpt(i == j ? 1.0 : 0.0);
            for (int k = 0; k < n; k++) m = csub(m, cmul(cpt(Y[i][k]), JX[k][j]));
            s = cadd(s, cmul(m, D[j]));
        }
        K[i] = s;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: c10krawczyk roots.bin outdir\n"); return 2; }
    FILE* f = fopen(argv[1], "rb"); int32_t N; if (fread(&N, 4, 1, f) != 1) return 3;
    vector<cd> R((size_t)N * n); if (fread(R.data(), 16, R.size(), f) != R.size()) return 3; fclose(f);
    string od = argv[2];
    {   // arithmetic environment: gradual underflow must be on (no FTZ/DAZ), round-to-nearest, IEEE binary64
        volatile double tiny = 2.2250738585072014e-308, half = tiny / 2;      // DBL_MIN / 2 must be a nonzero subnormal
        volatile double a = 1.0, b = 3.0; volatile double q = a / b;
        bool sub = (half > 0) && (half * 2 == tiny);
        bool rn = (q == 0.333333333333333314829616256247390992939472198486328125);
        printf("ENV: binary64 %d, gradual underflow %d, round-to-nearest %d, compiler %s\n", (int)(sizeof(double) == 8 && DBL_MANT_DIG == 53), (int)sub, (int)rn, __VERSION__);
        if (!sub || !rn) { printf("ENV: unsupported floating-point environment\n"); return 4; }
    }
    vector<int> status(N, 0);         // 1 certified
    vector<int> uni(N, -1);           // 1 unimodular (exact), 0 not unimodular, -1 undecided
    vector<double> rhoUsed(N, 0), widthS(N, 0);
    vector<C> Xs((size_t)N * n), Xb((size_t)N * n);
    const double rhos[] = {1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5};
    #pragma omp parallel for schedule(dynamic, 64)
    for (int t = 0; t < N; t++) {
        cd x[n]; for (int i = 0; i < n; i++) x[i] = R[(size_t)t * n + i];
        cd F[n], J[n][n], Y[n][n];
        for (int it = 0; it < 4; it++) {          // Newton polish with the exact Jacobian
            FJd(x, F, J, true); if (!inverse(J, Y)) break;
            cd dx[n]; for (int i = 0; i < n; i++) { dx[i] = 0; for (int k = 0; k < n; k++) dx[i] += Y[i][k] * F[k]; }
            for (int i = 0; i < n; i++) x[i] -= dx[i];
        }
        FJd(x, F, J, true); if (!inverse(J, Y)) continue;
        for (double rho : rhos) {
            C X[n], K[n];
            for (int i = 0; i < n; i++) { double r = rho * max(1.0, abs(x[i]));
                X[i] = {{dn(x[i].real() - r), up(x[i].real() + r)}, {dn(x[i].imag() - r), up(x[i].imag() + r)}}; }
            krawczyk(x, Y, X, K);
            bool ok = true;
            for (int i = 0; i < n && ok; i++) ok = inside_strict(K[i].re, X[i].re) && inside_strict(K[i].im, X[i].im);
            if (!ok) continue;
            C S[n]; for (int i = 0; i < n; i++) S[i] = {meet(K[i].re, X[i].re), meet(K[i].im, X[i].im)};   // root enclosure
            status[t] = 1; rhoUsed[t] = rho;
            double w = 0; for (int i = 0; i < n; i++) w = max(w, max(S[i].re.hi - S[i].re.lo, S[i].im.hi - S[i].im.lo));
            widthS[t] = w;
            for (int i = 0; i < n; i++) { Xs[(size_t)t * n + i] = S[i]; Xb[(size_t)t * n + i] = X[i]; }
            // unimodularity
            bool nonuni = false; for (int i = 0; i < n; i++) { I a = cabs2(S[i]); if (a.hi < 1.0 || a.lo > 1.0) nonuni = true; }
            bool sinside = true;
            for (int i = 0; i < n && sinside; i++) {
                I a = cabs2(S[i]); if (a.lo <= 0) { sinside = false; break; }
                I inv = irecip_pos(a); C s = {imul(S[i].re, inv), imul(S[i].im, inv)};   // 1/conj(x) = x/|x|^2
                sinside = inside(s.re, X[i].re) && inside(s.im, X[i].im);
            }
            uni[t] = sinside ? 1 : (nonuni ? 0 : -1);
            if (sinside && nonuni) uni[t] = -2;   // contradiction: would indicate a bug
            if (uni[t] == -1 && rho < 1e-5) continue;   // undecided: a larger box leaves room for s(Xs) inside X
            break;
        }
    }
    int cert = 0, nu = 0, nn = 0, und = 0, bad = 0; double maxrho = 0, maxw = 0;
    for (int t = 0; t < N; t++) { cert += status[t]; if (status[t]) { maxrho = max(maxrho, rhoUsed[t]); maxw = max(maxw, widthS[t]);
        nu += uni[t] == 1; nn += uni[t] == 0; und += uni[t] == -1; bad += uni[t] == -2; } }
    printf("KRAWCZYK: roots %d ; certified (unique root in box) %d ; largest box radius used %.0e ; widest enclosure %.2e\n", N, cert, maxrho, maxw);
    printf("UNIMODULAR: exact-unimodular %d ; certified non-unimodular %d ; undecided %d ; contradictions %d\n", nu, nn, und, bad);
    // distinctness of the enclosures: sweep on the EXACT interval endpoints of Re x_0 (no floating-point slack):
    // sorted by lower endpoint; once a later box starts above the current box's upper endpoint, so do all later ones
    vector<pair<double,int>> pr;
    for (int t = 0; t < N; t++) if (status[t]) pr.push_back({Xs[(size_t)t * n].re.lo, t});
    sort(pr.begin(), pr.end());
    long long cmp = 0, overl = 0;
    for (size_t a = 0; a < pr.size(); a++) {
        int s = pr[a].second; double hi = Xs[(size_t)s * n].re.hi;
        for (size_t b = a + 1; b < pr.size() && pr[b].first <= hi; b++) {
            cmp++; int t = pr[b].second; bool dis = false;
            for (int i = 0; i < n && !dis; i++) { C u = Xs[(size_t)s * n + i], v = Xs[(size_t)t * n + i]; dis = disjoint(u.re, v.re) || disjoint(u.im, v.im); }
            if (!dis) overl++;
        }
    }
    bool distinct_ok = (overl == 0) && ((int)pr.size() == cert);
    printf("DISTINCT: enclosures pairwise disjoint: %s (box tests %lld, overlapping %lld)\n", distinct_ok ? "YES" : "NO", cmp, overl);
    // rays from the exact-unimodular roots
    vector<int> U; for (int t = 0; t < N; t++) if (status[t] && uni[t] == 1) U.push_back(t);
    int M = U.size(); vector<C> Z((size_t)M * n);
    for (int a = 0; a < M; a++) { C z = cpt(1.0); Z[(size_t)a * n] = z;
        for (int j = 0; j + 1 < n; j++) { z = cmul(Xs[(size_t)U[a] * n + j], z); Z[(size_t)a * n + j + 1] = z; } }
    int W = (M + 63) / 64; vector<uint64_t> O((size_t)M * W, 0), Uu((size_t)M * W, 0);
    long long nO = 0, nU = 0, nOU = 0; double maxwO = 0, maxwU = 0, minExcl0 = INFINITY, minExcl10 = INFINITY;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:nO,nU,nOU) reduction(max:maxwO,maxwU) reduction(min:minExcl0,minExcl10)
    for (int a = 0; a < M; a++) for (int b = 0; b < M; b++) if (a != b) {
        C s = cpt(0.0); for (int j = 0; j < n; j++) s = cadd(s, cconjmul(Z[(size_t)a * n + j], Z[(size_t)b * n + j]));
        I v = cabs2(s);
        bool o = has(v, 0.0), u = has(v, 10.0);
        if (o) { O[(size_t)a * W + b / 64] |= 1ULL << (b % 64); nO++; maxwO = max(maxwO, v.hi - v.lo); } else minExcl0 = min(minExcl0, v.lo);
        if (u) { Uu[(size_t)a * W + b / 64] |= 1ULL << (b % 64); nU++; maxwU = max(maxwU, v.hi - v.lo); } else minExcl10 = min(minExcl10, min(fabs(v.lo - 10), fabs(v.hi - 10)));
        if (o && u) nOU++;
    }
    int maxdegU = 0, maxdegO = 0;
    for (int a = 0; a < M; a++) { int dO = 0, dU = 0; for (int w = 0; w < W; w++) { dO += __builtin_popcountll(O[(size_t)a * W + w]); dU += __builtin_popcountll(Uu[(size_t)a * W + w]); }
        maxdegU = max(maxdegU, dU); maxdegO = max(maxdegO, dO); }
    printf("GRAPHS: rays %d ; O+ ordered pairs %lld, U+ ordered pairs %lld, both %lld ; widest O+ enclosure %.1e, widest U+ enclosure %.1e (z-scale, MU value 10)\n",
           M, nO, nU, nOU, maxwO, maxwU);
    printf("  excluded pairs: min lower bound away from 0: %.3e ; min distance from 10: %.3e\n", minExcl0, minExcl10);
    printf("  max degree O+ %d ; max degree U+ %d\n", maxdegO, maxdegU);
    bool pass = (N == 34940) && (cert == N) && (und == 0) && (bad == 0) && distinct_ok && (nu == 3040) && (M == 3040) && (maxdegU <= 8) && (nOU == 0);
    printf("CONTRACT: N=34940 %d, all certified %d, unimodularity decided %d, disjoint %d, 3040 unimodular %d, max U+ degree <= 8 %d, O+ and U+ disjoint %d\n",
           N == 34940, cert == N, und == 0 && bad == 0, distinct_ok, nu == 3040, maxdegU <= 8, nOU == 0);
    printf(pass ? "C10-CERTIFY: PASS\n" : "C10-CERTIFY: FAIL\n");
    FILE* g2 = fopen((od + "/graph_plus.bin").c_str(), "wb"); int32_t MM = M; fwrite(&MM, 4, 1, g2); fwrite(O.data(), 8, O.size(), g2); fwrite(Uu.data(), 8, Uu.size(), g2); fclose(g2);
    FILE* g3 = fopen((od + "/rays_plus.txt").c_str(), "w");
    for (int a = 0; a < M; a++) { fprintf(g3, "%d", U[a]); for (int j = 0; j < n; j++) { C z = Z[(size_t)a * n + j]; fprintf(g3, " %.17g %.17g", 0.5 * (z.re.lo + z.re.hi), 0.5 * (z.im.lo + z.im.hi)); } fprintf(g3, "\n"); }
    fclose(g3);
    return pass ? 0 : 1;
}
