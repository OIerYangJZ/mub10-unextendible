// rankmodp.cpp -- rank of a dense matrix over F_p (p < 2^31) by Gaussian elimination, OpenMP over rows.
// Input (binary, little endian): int64 nrows, int64 ncols, int64 p, then nrows*ncols uint32 entries (row major, < p).
// Output: "RANK r" on stdout.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 rankmodp.cpp -o rankmodp
#include <bits/stdc++.h>
using namespace std;
typedef uint64_t u64; typedef uint32_t u32;
static u64 powmod(u64 a, u64 e, u64 p) { u64 r = 1; a %= p; while (e) { if (e & 1) r = r * a % p; a = a * a % p; e >>= 1; } return r; }
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: rankmodp <matrix.bin>\n"); return 2; }
    FILE* f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 2; }
    int64_t nr, nc, pp; if (fread(&nr, 8, 1, f) != 1 || fread(&nc, 8, 1, f) != 1 || fread(&pp, 8, 1, f) != 1) return 3;
    u64 p = (u64)pp;
    vector<u32> M((size_t)nr * nc);
    if (fread(M.data(), 4, M.size(), f) != M.size()) { fprintf(stderr, "short read\n"); return 3; }
    fclose(f);
    auto t0 = chrono::steady_clock::now();
    int64_t rk = 0;
    for (int64_t c = 0; c < nc && rk < nr; c++) {
        int64_t piv = -1;
        for (int64_t i = rk; i < nr; i++) if (M[(size_t)i * nc + c]) { piv = i; break; }
        if (piv < 0) continue;
        if (piv != rk) for (int64_t j = c; j < nc; j++) swap(M[(size_t)piv * nc + j], M[(size_t)rk * nc + j]);
        u32* R = &M[(size_t)rk * nc];
        u64 inv = powmod(R[c], p - 2, p);
        for (int64_t j = c; j < nc; j++) R[j] = (u32)((u64)R[j] * inv % p);
        #pragma omp parallel for schedule(static)
        for (int64_t i = rk + 1; i < nr; i++) {
            u32* Ri = &M[(size_t)i * nc];
            u64 fct = Ri[c];
            if (!fct) continue;
            u64 m = p - fct;
            for (int64_t j = c; j < nc; j++) Ri[j] = (u32)((Ri[j] + m * R[j]) % p);
        }
        rk++;
        if (rk % 500 == 0) { fprintf(stderr, "  pivots %lld  %.1fs\n", (long long)rk, chrono::duration<double>(chrono::steady_clock::now() - t0).count()); }
    }
    printf("RANK %lld of %lld x %lld  (%.1fs)\n", (long long)rk, (long long)nr, (long long)nc, chrono::duration<double>(chrono::steady_clock::now() - t0).count());
    return 0;
}
