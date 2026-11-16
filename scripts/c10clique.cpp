// c10clique.cpp -- clique searches on the cyclic-10 ray catalogue (Astra-B round 8, request 1).
// Graphs on the rays unbiased to (I, F10):  O = orthogonality, U = mutual unbiasedness (|<r,s>|^2 = 1/10).
// Input (binary, from c10graph.py): int32 N, then N rows of O (W = ceil(N/64) uint64 words each), then N rows of U.
// Modes:
//   nbr10        test (6): for every ray r, max clique (capped at 10) of O restricted to N_U(r)
//   list10 out   enumerate all 10-cliques of O (= orthonormal bases among catalogue rays) -> text file, one per line
//   quad in      for every basis C listed in <in>: does O restricted to the common MU-neighbourhood of C contain a
//                10-clique?  (a yes = a quadruple containing (I, F10); prints it)
// Search: Bron-Kerbosch with Tomita pivoting on bitsets, early exit once the cap is reached.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 c10clique.cpp -o c10clique
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
using namespace std;
typedef uint64_t u64;
static int N, W;
static vector<u64> O, U;
static inline const u64* row(const vector<u64>& A, int i) { return &A[(size_t)i * W]; }
static inline int popc(const u64* s) { int c = 0; for (int k = 0; k < W; k++) c += __builtin_popcountll(s[k]); return c; }
static double now() { static auto t0 = chrono::steady_clock::now(); return chrono::duration<double>(chrono::steady_clock::now() - t0).count(); }

struct MaxClique {                      // max clique of O inside a candidate set, capped
    int cap, best = 0; vector<int> cur, bestset;
    void rec(vector<u64>& P, vector<u64>& X) {
        if (best >= cap) return;
        int pc = popc(P.data()), depth = (int)cur.size();
        if (pc == 0) { if (depth > best) { best = depth; bestset = cur; } return; }
        if (depth + pc <= best) return;
        int piv = -1, bc = -1;
        for (int w = 0; w < W; w++) { u64 m = P[w] | X[w]; while (m) { int b = __builtin_ctzll(m); m &= m - 1; int u = w * 64 + b;
            const u64* ru = row(O, u); int c = 0; for (int k = 0; k < W; k++) c += __builtin_popcountll(P[k] & ru[k]);
            if (c > bc) { bc = c; piv = u; } } }
        const u64* pr = row(O, piv); vector<u64> cand(W);
        for (int k = 0; k < W; k++) cand[k] = P[k] & ~pr[k];
        vector<u64> P2(W), X2(W);
        for (int w = 0; w < W; w++) { u64 m = cand[w]; while (m) { int b = __builtin_ctzll(m); m &= m - 1; int v = w * 64 + b;
            const u64* rv = row(O, v);
            for (int k = 0; k < W; k++) { P2[k] = P[k] & rv[k]; X2[k] = X[k] & rv[k]; }
            cur.push_back(v); { vector<u64> a = P2, x = X2; rec(a, x); } cur.pop_back();
            if (best >= cap) return;
            P[w] &= ~(1ULL << b); X[w] |= (1ULL << b); } }
    }
    int run(const vector<u64>& cand) { vector<u64> P = cand, X(W, 0); rec(P, X); return best; }
};

static void list_k(int k, vector<u64>& P, vector<int>& cur, vector<vector<int>>& out) {   // all k-cliques, increasing order
    if ((int)cur.size() == k) { out.push_back(cur); return; }
    if (popc(P.data()) < k - (int)cur.size()) return;
    vector<u64> P2(W);
    for (int w = 0; w < W; w++) { u64 m = P[w]; while (m) { int b = __builtin_ctzll(m); m &= m - 1; int v = w * 64 + b;
        const u64* rv = row(O, v);
        for (int q = 0; q < W; q++) P2[q] = (q < w) ? 0 : (P[q] & rv[q]);
        P2[w] &= (b == 63) ? 0ULL : (~0ULL << (b + 1));
        cur.push_back(v); { vector<u64> a = P2; list_k(k, a, cur, out); } cur.pop_back(); } }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: c10clique <graph.bin> nbr10 | list10 <out> | quad <in>\n"); return 2; }
    FILE* f = fopen(argv[1], "rb"); if (!f) return 3;
    int32_t n; if (fread(&n, 4, 1, f) != 1) return 3; N = n; W = (N + 63) / 64;
    O.resize((size_t)N * W); U.resize((size_t)N * W);
    if (fread(O.data(), 8, O.size(), f) != O.size() || fread(U.data(), 8, U.size(), f) != U.size()) return 3;
    fclose(f); now();
    string mode = argv[2];
    if (mode == "nbr10") {
        vector<int> res(N), sz(N); vector<vector<int>> wit(N);
        #pragma omp parallel for schedule(dynamic)
        for (int r = 0; r < N; r++) {
            vector<u64> P(row(U, r), row(U, r) + W);
            sz[r] = popc(P.data());
            MaxClique mc; mc.cap = 10; res[r] = mc.run(P); wit[r] = mc.bestset;
        }
        map<pair<int,int>, int> hist; int n10 = 0, first = -1;
        for (int r = 0; r < N; r++) { hist[{sz[r], res[r]}]++; if (res[r] >= 10) { n10++; if (first < 0) first = r; } }
        for (auto& kv : hist) printf("  |N_U(r)| = %4d, max clique of O[N_U(r)] (cap 10) = %2d : %d rays\n", kv.first.first, kv.first.second, kv.second);
        printf("NBR10: rays %d ; rays whose MU-neighbourhood contains an orthonormal basis: %d ; %.1fs\n", N, n10, now());
        if (first >= 0) { printf("  witness r=%d basis:", first); for (int v : wit[first]) printf(" %d", v); printf("\n"); }
        printf(n10 == 0 ? "TEST6: HOLDS (no quadruple containing the Fourier pair, given catalogue completeness)\n"
                        : "TEST6: FAILS (go to the two-basis search)\n");
    } else if (mode == "list10") {
        vector<vector<int>> all;
        vector<vector<vector<int>>> per(N);
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < N; v++) {          // cliques whose smallest vertex is v
            vector<u64> P(W); const u64* rv = row(O, v);
            for (int q = 0; q < W; q++) P[q] = (q < v / 64) ? 0 : rv[q];
            P[v / 64] &= ((v % 64) == 63) ? 0ULL : (~0ULL << (v % 64 + 1));
            vector<int> cur{v}; list_k(10, P, cur, per[v]);
        }
        FILE* g = fopen(argv[3], "w"); long long tot = 0; vector<int> deg(N, 0);
        for (int v = 0; v < N; v++) for (auto& c : per[v]) { tot++; for (int i = 0; i < 10; i++) { fprintf(g, "%d%c", c[i], i == 9 ? '\n' : ' '); deg[c[i]]++; } }
        fclose(g);
        int zero = 0, mx = 0; for (int v = 0; v < N; v++) { zero += deg[v] == 0; mx = max(mx, deg[v]); }
        printf("LIST10: orthonormal bases among catalogue rays: %lld ; rays in no basis: %d ; max bases through a ray: %d ; %.1fs\n", tot, zero, mx, now());
    } else if (mode == "quad") {
        vector<vector<int>> bases; FILE* g = fopen(argv[3], "r"); int a[10];
        while (true) { int ok = 1; for (int i = 0; i < 10; i++) if (fscanf(g, "%d", &a[i]) != 1) { ok = 0; break; } if (!ok) break; bases.push_back(vector<int>(a, a + 10)); }
        fclose(g);
        int nb = (int)bases.size(); vector<int> res(nb), csz(nb); vector<vector<int>> wit(nb);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nb; i++) {
            vector<u64> P(row(U, bases[i][0]), row(U, bases[i][0]) + W);
            for (int j = 1; j < 10; j++) { const u64* ru = row(U, bases[i][j]); for (int k = 0; k < W; k++) P[k] &= ru[k]; }
            csz[i] = popc(P.data());
            MaxClique mc; mc.cap = 10; res[i] = mc.run(P); wit[i] = mc.bestset;
        }
        map<pair<int,int>, int> hist; int found = 0;
        for (int i = 0; i < nb; i++) { hist[{csz[i], res[i]}]++; if (res[i] >= 10) {
            if (found < 5) { printf("  QUADRUPLE: basis %d {", i); for (int v : bases[i]) printf(" %d", v); printf(" } + {"); for (int v : wit[i]) printf(" %d", v); printf(" }\n"); }
            found++; } }
        for (auto& kv : hist) printf("  |common MU nbhd| = %4d, max clique (cap 10) = %2d : %d bases\n", kv.first.first, kv.first.second, kv.second);
        printf("QUAD: bases %d ; bases extendible to a quadruple: %d ; %.1fs\n", nb, found, now());
    }
    return 0;
}
