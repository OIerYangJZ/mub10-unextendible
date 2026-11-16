// laurent_mac.cpp -- symmetry-reduced Laurent (torus) Macaulay search for the AFFINE-chart certificate of a 2x5
// product triple, and export of an 18-variable cofactor certificate for verify_cert.py.
//
// Torus form: x in (F^*)^10 / scalars, Laurent monomials x^m with m in Z^10, sum m = 0.  Generators
//   f_g = sum_{k != l} zeta^{E_gk - E_gl} x_k / x_l                       (g = 0..19)
// "Root degree" of x^m: |m|_1 / 2.  Level n: look for 1 = sum_g c_g f_g with root degree(c_g) <= n-1.  This is the
// rank_cert.py system (R_{n,n}, c^n) with c = 1, but tested for MEMBERSHIP of the target, not for full rank.
// Symmetry: a group G of monomial substitutions (laurent_prep.py) permuting the f_g up to scalars.  A certificate may
// be averaged over G, so we search c_{g0} = sum_mu y_mu x^mu for generator-orbit representatives g0 only (mu over
// H_{g0}-orbit representatives), and test the G-averaged identity on G-orbit representatives of monomials.
// The G-averaged identity is  sum_g c_g f_g = 1  with  c_g = sum_{h: pi_h(g0) = g} lam_h sigma_h(c_{g0}).
// Solve: random sparse compression B = A S (R x (R+K)), dense elimination mod p, y = S z.
// Output (with --cert <prefix>): <prefix>_laurent.txt (Laurent cofactors) and <prefix>_cof.txt: 29 cofactor lines in
// the order of affine.ms (9 torus equations X_k*Y_k-1, then the 20 f rows), so that verify_cert.py checks
//   1 = sum_k t_k (X_k Y_k - 1) + sum_g h_g f_g   in F_p[X1..X9, Y1..Y9].
// Build: g++ -O3 -march=native -fopenmp -std=c++17 laurent_mac.cpp -o laurent_mac
// Usage: laurent_mac <prep.txt> <n> [--cert prefix] [--seed s] [--nnz k] [--extra K]
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32; typedef int64_t i64;
static const u64 P = 1073741741ULL;
static inline u64 powm(u64 a, u64 e) { u64 r = 1; a %= P; while (e) { if (e & 1) r = r * a % P; a = a * a % P; e >>= 1; } return r; }
static inline u64 invm(u64 a) { return powm(a, P - 2); }
static double T0;
static double now() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
#define LOG(...) do { fprintf(stderr, "[%7.1fs] ", now() - T0); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fflush(stderr); } while (0)

const int NV = 10;
typedef array<int8_t, NV> Mono;
static inline u64 key(const Mono& m) { u64 k = 0; for (int i = 0; i < NV; i++) k = (k << 6) | (u64)(m[i] + 32); return k; }
static inline Mono unkey(u64 k) { Mono m; for (int i = NV - 1; i >= 0; i--) { m[i] = (int8_t)((int)(k & 63) - 32); k >>= 6; } return m; }
struct GElt { int sig[NV]; int ph[NV]; };
static u64 zp[20];
static inline Mono act(const GElt& g, const Mono& m, int& e) {
    Mono r{}; int s = 0;
    for (int k = 0; k < NV; k++) { r[g.sig[k]] = m[k]; s += g.ph[k] * m[k]; }
    e = ((s % 20) + 20) % 20; return r;
}
static void enumerate(int N, vector<Mono>& out) {       // all m, sum m = 0, |m|_1 <= 2N
    Mono m{};
    function<void(int, int, int)> rec = [&](int i, int sum, int ab) {
        if (i == NV - 1) { int v = -sum; if (ab + abs(v) <= 2 * N) { m[i] = (int8_t)v; out.push_back(m); } return; }
        for (int v = -N; v <= N; v++) {
            if (ab + abs(v) > 2 * N) continue;
            m[i] = (int8_t)v; rec(i + 1, sum + v, ab + abs(v));
        }
    };
    rec(0, 0, 0);
}
static inline u64 splitmix(u64& s) { u64 z = (s += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }

int main(int argc, char** argv) {
    T0 = now();
    if (argc < 3) { fprintf(stderr, "usage: laurent_mac prep.txt n [--cert prefix] [--seed s] [--nnz k] [--extra K]\n"); return 2; }
    int n = atoi(argv[2]); string cert; u64 seed = 12345; int NNZ = 12, EXTRA = 40;
    for (int i = 3; i < argc; i++) {
        string a = argv[i];
        if (a == "--cert") cert = argv[++i]; else if (a == "--seed") seed = strtoull(argv[++i], 0, 10);
        else if (a == "--nnz") NNZ = atoi(argv[++i]); else if (a == "--extra") EXTRA = atoi(argv[++i]);
    }
    FILE* f = fopen(argv[1], "r"); if (!f) { perror("prep"); return 2; }
    u64 p, r; if (fscanf(f, "%lu %lu", &p, &r) != 2 || p != P) { fprintf(stderr, "bad p\n"); return 2; }
    for (int e = 0; e < 20; e++) zp[e] = powm(r, e);
    if (zp[10] != P - 1 || zp[4] == 1) { fprintf(stderr, "r is not a primitive 20th root\n"); return 2; }
    int E[20][NV]; for (int g = 0; g < 20; g++) for (int k = 0; k < NV; k++) if (fscanf(f, "%d", &E[g][k]) != 1) return 3;
    int ng; if (fscanf(f, "%d", &ng) != 1) return 3;
    vector<GElt> G(ng); for (auto& g : G) { for (int k = 0; k < NV; k++) if (fscanf(f, "%d", &g.sig[k]) != 1) return 3; for (int k = 0; k < NV; k++) if (fscanf(f, "%d", &g.ph[k]) != 1) return 3; }
    fclose(f);
    // generators
    vector<vector<pair<Mono, u64>>> F(20); vector<unordered_map<u64, u64>> Fm(20);
    for (int g = 0; g < 20; g++) for (int k = 0; k < NV; k++) for (int l = 0; l < NV; l++) if (k != l) {
        Mono m{}; m[k] = 1; m[l] = -1; u64 c = zp[((E[g][k] - E[g][l]) % 20 + 20) % 20];
        F[g].push_back({m, c}); Fm[g][key(m)] = c;
    }
    // generator permutation by each group element: sigma_h(f_g) = lam * f_{pi}
    vector<vector<int>> pi(ng, vector<int>(20, -1)); vector<vector<u64>> lam(ng, vector<u64>(20, 0));
    for (int h = 0; h < ng; h++) for (int g = 0; g < 20; g++) {
        vector<pair<u64, u64>> img;
        for (auto& t : F[g]) { int e; Mono m2 = act(G[h], t.first, e); img.push_back({key(m2), t.second * zp[e] % P}); }
        for (int g2 = 0; g2 < 20 && pi[h][g] < 0; g2++) {
            auto it = Fm[g2].find(img[0].first); if (it == Fm[g2].end()) continue;
            u64 L = img[0].second * invm(it->second) % P; bool ok = true;
            for (auto& t : img) { auto j = Fm[g2].find(t.first); if (j == Fm[g2].end() || j->second * L % P != t.second) { ok = false; break; } }
            if (ok) { pi[h][g] = g2; lam[h][g] = L; }
        }
    }
    {   // keep the subgroup of elements that permute the generators (up to scalars); the monomial Paulis form a group
        vector<GElt> G2; vector<vector<int>> pi2; vector<vector<u64>> lam2;
        for (int h = 0; h < ng; h++) { bool ok = true; for (int g = 0; g < 20; g++) if (pi[h][g] < 0) ok = false;
            if (ok) { G2.push_back(G[h]); pi2.push_back(pi[h]); lam2.push_back(lam[h]); } }
        if ((int)G2.size() < ng) LOG("dropped %d of %d group elements that do not permute the generators", ng - (int)G2.size(), ng);
        G = G2; pi = pi2; lam = lam2; ng = (int)G.size();
        // closure check (the averaging argument needs a group)
        set<vector<int>> S; for (auto& g : G) { vector<int> v(g.sig, g.sig + NV); v.insert(v.end(), g.ph, g.ph + NV); S.insert(v); }
        for (auto& a : G) for (auto& b : G) { GElt c; for (int k = 0; k < NV; k++) { c.sig[k] = a.sig[b.sig[k]]; c.ph[k] = (b.ph[k] + a.ph[b.sig[k]]) % 20; }
            // composition up to a global scalar: normalise phases so that ph[0] matches some element
            bool found = false;
            for (auto& g : G) { bool same = true; int d = ((c.ph[0] - g.ph[0]) % 20 + 20) % 20;
                for (int k = 0; k < NV && same; k++) same = g.sig[k] == c.sig[k] && ((c.ph[k] - g.ph[k] - d) % 20 + 20) % 20 == 0;
                if (same) { found = true; break; } }
            if (!found) { LOG("group not closed; abort"); return 4; } }
    }
    vector<int> reps; vector<int> seen(20, 0);
    for (int g = 0; g < 20; g++) if (!seen[g]) { reps.push_back(g); for (int h = 0; h < ng; h++) seen[pi[h][g]] = 1; }
    vector<vector<int>> Hs(reps.size());
    for (size_t i = 0; i < reps.size(); i++) for (int h = 0; h < ng; h++) if (pi[h][reps[i]] == reps[i]) Hs[i].push_back(h);
    { string s; for (size_t i = 0; i < reps.size(); i++) s += " g" + to_string(reps[i]) + "(|H|=" + to_string(Hs[i].size()) + ")";
      int nl = 0; for (int h = 0; h < ng; h++) for (int g = 0; g < 20; g++) if (lam[h][g] != 1) nl++;
      LOG("group order %d, generator orbit reps:%s ; scalars != 1: %d", ng, s.c_str(), nl); }
    // rows: G-orbit reps of R_n monomials with trivial stabiliser character
    vector<Mono> Rn; enumerate(n, Rn);
    size_t NR = Rn.size(); vector<u64> repk(NR), sval(NR);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < NR; i++) {
        u64 best = ~0ULL, s = 0;
        vector<pair<u64, int>> im(ng);
        for (int h = 0; h < ng; h++) { int e; Mono m2 = act(G[h], Rn[i], e); im[h] = {key(m2), e}; if (im[h].first < best) best = im[h].first; }
        for (int h = 0; h < ng; h++) if (im[h].first == best) s += zp[im[h].second];
        repk[i] = best; sval[i] = s % P;
    }
    unordered_map<u64, int> rowOf; unordered_map<u64, pair<int, u64>> info; info.reserve(NR * 2);
    int R = 0;
    for (size_t i = 0; i < NR; i++) {
        int row = -1;
        if (sval[i]) { auto it = rowOf.find(repk[i]); if (it == rowOf.end()) { row = R; rowOf[repk[i]] = R++; } else row = it->second; }
        info[key(Rn[i])] = {row, sval[i]};
    }
    Mono zero{}; int row0 = info[key(zero)].first;
    LOG("n=%d: |R_n| = %zu monomials, %d invariant rows (target row %d)", n, NR, R, row0);
    // columns
    vector<Mono> Rn1; enumerate(n - 1, Rn1);
    struct Col { int gi; u64 mk; vector<pair<int, u64>> v; };
    vector<Col> cols;
    for (size_t gi = 0; gi < reps.size(); gi++) {
        int g0 = reps[gi]; size_t before = cols.size();
        vector<Col> loc(Rn1.size());
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = 0; i < Rn1.size(); i++) {
            u64 k0 = key(Rn1[i]); bool isrep = true;
            for (int h : Hs[gi]) { int e; if (key(act(G[h], Rn1[i], e)) < k0) { isrep = false; break; } }
            if (!isrep) { loc[i].gi = -1; continue; }
            map<int, u64> acc;
            for (auto& t : F[g0]) {
                Mono m{}; for (int k = 0; k < NV; k++) m[k] = Rn1[i][k] + t.first[k];
                auto it = info.find(key(m));
                if (it == info.end()) { fprintf(stderr, "monomial outside R_n\n"); abort(); }
                if (it->second.first < 0) continue;
                acc[it->second.first] = (acc[it->second.first] + t.second * it->second.second) % P;
            }
            loc[i].gi = gi; loc[i].mk = k0;
            for (auto& kv : acc) if (kv.second) loc[i].v.push_back(kv);
            if (loc[i].v.empty()) loc[i].gi = -1;
        }
        for (auto& c : loc) if (c.gi >= 0) cols.push_back(move(c));
        LOG("generator orbit rep g%d: %zu nonzero columns", g0, cols.size() - before);
    }
    size_t C = cols.size(), nnzA = 0; for (auto& c : cols) nnzA += c.v.size();
    LOG("matrix A: %d rows x %zu cols, nnz %zu", R, C, nnzA);
    // compression B = A S, augmented with the target
    int W = min<i64>(R + EXTRA, (i64)C);
    int WW = W + 1;
    vector<u32> B((size_t)R * WW, 0);
    auto colslots = [&](size_t j, vector<pair<int, u64>>& sl) {
        sl.clear(); u64 s = seed * 0x100000001B3ULL ^ (j * 0x9E3779B97F4A7C15ULL);
        if ((i64)C <= W) { sl.push_back({(int)j, 1}); return; }
        for (int t = 0; t < NNZ; t++) { u64 a = splitmix(s), b = splitmix(s); sl.push_back({(int)(a % W), 1 + b % (P - 1)}); }
    };
    {
        vector<pair<int, u64>> sl;
        for (size_t j = 0; j < C; j++) {
            colslots(j, sl);
            for (auto& s : sl) for (auto& e : cols[j].v) {
                u32& x = B[(size_t)e.first * WW + s.first]; x = (u32)((x + e.second * s.second) % P);
            }
        }
    }
    B[(size_t)row0 * WW + W] = 1;
    LOG("compressed system %d x %d (+rhs), %.2f GB; eliminating", R, W, (double)B.size() * 4 / 1e9);
    // forward elimination
    vector<int> pivcol; int rk = 0;
    for (int c = 0; c < W && rk < R; c++) {
        int piv = -1; for (int i = rk; i < R; i++) if (B[(size_t)i * WW + c]) { piv = i; break; }
        if (piv < 0) continue;
        if (piv != rk) for (int j = 0; j < WW; j++) swap(B[(size_t)piv * WW + j], B[(size_t)rk * WW + j]);
        u32* Rr = &B[(size_t)rk * WW]; u64 iv = invm(Rr[c]);
        for (int j = c; j < WW; j++) Rr[j] = (u32)(Rr[j] * iv % P);
        #pragma omp parallel for schedule(static)
        for (int i = rk + 1; i < R; i++) {
            u32* Ri = &B[(size_t)i * WW]; u64 fct = Ri[c]; if (!fct) continue; u64 m = P - fct;
            for (int j = c; j < WW; j++) Ri[j] = (u32)((Ri[j] + m * Rr[j]) % P);
        }
        pivcol.push_back(c); rk++;
        if (rk % 2000 == 0) LOG("  pivots %d", rk);
    }
    bool consistent = true; for (int i = rk; i < R; i++) if (B[(size_t)i * WW + W]) { consistent = false; break; }
    LOG("RESULT n=%d: rank(A S) = %d of %d rows ; target %s", n, rk, R, consistent ? "IN THE IMAGE -> certificate exists" : "NOT in the image");
    printf("RESULT n=%d rows %d cols %zu rank %d target %s\n", n, R, C, rk, consistent ? "IN_IMAGE" : "NOT_IN_IMAGE");
    if (!consistent || cert.empty()) return consistent ? 0 : 1;
    // back substitution
    vector<u64> z(W, 0);
    for (int i = rk - 1; i >= 0; i--) {
        u32* Ri = &B[(size_t)i * WW]; int c = pivcol[i]; u64 s = Ri[W];
        for (int j = c + 1; j < W; j++) if (Ri[j] && z[j]) s = (s + (P - Ri[j]) * z[j]) % P;
        z[c] = s;                                    // pivot normalised to 1
    }
    vector<u32>().swap(B);
    // y = S z, base cofactors c_{g0}
    vector<unordered_map<u64, u64>> cb(reps.size());
    { vector<pair<int, u64>> sl;
      for (size_t j = 0; j < C; j++) { colslots(j, sl); u64 y = 0; for (auto& s : sl) y = (y + s.second * z[s.first]) % P; if (y) cb[cols[j].gi][cols[j].mk] = y; } }
    // full cofactors c_g = sum_{h: pi_h(g0) = g} lam_h sigma_h(c_{g0})
    vector<unordered_map<u64, u64>> cg(20);
    for (size_t gi = 0; gi < reps.size(); gi++) for (int h = 0; h < ng; h++) {
        int g = pi[h][reps[gi]]; u64 L = lam[h][reps[gi]];
        for (auto& kv : cb[gi]) { int e; Mono m2 = act(G[h], unkey(kv.first), e); u64& x = cg[g][key(m2)]; x = (x + kv.second * zp[e] % P * L) % P; }
    }
    size_t tot = 0; for (auto& c : cg) { for (auto it = c.begin(); it != c.end();) { if (!it->second) it = c.erase(it); else ++it; } tot += c.size(); }
    LOG("cofactors expanded: %zu Laurent terms in total", tot);
    // Laurent check
    { unordered_map<u64, u64> acc; acc.reserve(tot * 4);
      for (int g = 0; g < 20; g++) for (auto& kv : cg[g]) { Mono m = unkey(kv.first);
          for (auto& t : F[g]) { Mono m2; for (int k = 0; k < NV; k++) m2[k] = m[k] + t.first[k]; u64& x = acc[key(m2)]; x = (x + kv.second * t.second) % P; } }
      size_t bad = 0; for (auto& kv : acc) { u64 want = (kv.first == key(zero)) ? 1 : 0; if (kv.second != want) bad++; }
      if (!acc.count(key(zero))) bad++;
      LOG("Laurent check sum_g c_g f_g == 1: %s (%zu wrong coefficients)", bad ? "FAILED" : "OK", bad);
      if (bad) return 5; }
    // Laurent file
    { FILE* o = fopen((cert + "_laurent.txt").c_str(), "w");
      fprintf(o, "# Laurent cofactors c_g (g=0..19): lines 'g c m0 .. m9' meaning c * x^m, sum m = 0; p=%lu zeta20=%lu\n", P, r);
      for (int g = 0; g < 20; g++) { vector<pair<u64, u64>> v(cg[g].begin(), cg[g].end()); sort(v.begin(), v.end());
          for (auto& kv : v) { Mono m = unkey(kv.first); fprintf(o, "%d %lu", g, kv.second); for (int k = 0; k < NV; k++) fprintf(o, " %d", m[k]); fprintf(o, "\n"); } }
      fclose(o); }
    // 18-variable certificate: h_g = sum c x^m -> X^{m+} Y^{m-} (coordinates 1..9);  torus cofactors from overlaps.
    // product term N * u_k^{t_k} u_l^{t_l} (t <= 1) = N + t_k N (u_k - 1) + t_l N u_k^{t_k} (u_l - 1)
    // => 1 = sum_g h_g f_g - sum_j T_j (u_j - 1), so the cofactor of (X_j Y_j - 1) is -T_j.
    typedef array<int8_t, 18> M18;
    auto k18 = [](const M18& a) { u64 k = 0; for (int i = 0; i < 18; i++) k = (k << 3) | (u64)a[i]; return k; };   // exps <= 7
    vector<unordered_map<u64, u64>> T(10);
    for (int g = 0; g < 20; g++) for (auto& kv : cg[g]) {
        Mono m = unkey(kv.first);
        M18 base{}; for (int k = 1; k < NV; k++) { if (m[k] > 0) base[k - 1] = m[k]; else base[9 + k - 1] = -m[k]; }
        for (int k = 0; k < NV; k++) for (int l = 0; l < NV; l++) if (k != l) {
            u64 c = kv.second * zp[((E[g][k] - E[g][l]) % 20 + 20) % 20] % P;
            M18 a = base; if (k) a[k - 1]++; if (l) a[9 + l - 1]++;
            bool tk = k && m[k] < 0, tl = l && m[l] > 0;
            if (!tk && !tl) continue;
            M18 Nm = a; if (tk) { Nm[k - 1]--; Nm[9 + k - 1]--; } if (tl) { Nm[l - 1]--; Nm[9 + l - 1]--; }
            if (tk) { u64& x = T[k][k18(Nm)]; x = (x + c) % P; }
            if (tl) { M18 Q = Nm; if (tk) { Q[k - 1]++; Q[9 + k - 1]++; } u64& x = T[l][k18(Q)]; x = (x + c) % P; }
        }
    }
    auto poly18 = [&](FILE* o, const vector<pair<M18, u64>>& v) {
        if (v.empty()) { fprintf(o, "0\n"); return; }
        bool first = true;
        for (auto& t : v) {
            if (!first) fputc('+', o); first = false;
            fprintf(o, "%lu", t.second);
            for (int i = 0; i < 18; i++) if (t.first[i]) { fprintf(o, "*%c%d", i < 9 ? 'X' : 'Y', i % 9 + 1); if (t.first[i] > 1) fprintf(o, "^%d", t.first[i]); }
        }
        fputc('\n', o);
    };
    auto unk18 = [](u64 k) { M18 a; for (int i = 17; i >= 0; i--) { a[i] = (int8_t)(k & 7); k >>= 3; } return a; };
    FILE* o = fopen((cert + "_cof.txt").c_str(), "w"); size_t nterms = 0;
    for (int j = 1; j < NV; j++) {
        vector<pair<M18, u64>> v; for (auto& kv : T[j]) if (kv.second) v.push_back({unk18(kv.first), (P - kv.second) % P});
        sort(v.begin(), v.end()); nterms += v.size(); poly18(o, v);
    }
    for (int g = 0; g < 20; g++) {
        vector<pair<M18, u64>> v;
        for (auto& kv : cg[g]) { Mono m = unkey(kv.first); M18 a{}; for (int k = 1; k < NV; k++) { if (m[k] > 0) a[k - 1] = m[k]; else a[9 + k - 1] = -m[k]; } v.push_back({a, kv.second}); }
        sort(v.begin(), v.end()); nterms += v.size(); poly18(o, v);
    }
    fclose(o);
    LOG("wrote %s_cof.txt (29 cofactors, %zu terms) and %s_laurent.txt", cert.c_str(), nterms, cert.c_str());
    return 0;
}
