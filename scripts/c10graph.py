#!/usr/bin/env python
"""Overlap graphs on the cyclic-10 ray catalogue (Astra-B round 8, request 1).
Reads data/cyclic10/rays.npz (psi: N x 10 dephased rays unbiased to I and F10; product: indices of the 40 chirps).
  * Gram |<r,s>|^2 by one BLAS matmul; classify pairs as O (=0), U (=1/10), or conflict; report the numerical gaps.
  * Symmetry orbits under the pair's symmetry group <shift, modulation, dilation by 3, conjugation, Fourier>, found by
    overlap matching (a closure check of the catalogue under these maps, and the orbit sizes for comparison with
    Fuehr-Rzeszotnik 20,20,200,200,200,400,400,400,400,800).
  * Writes the bitset graph file for c10clique (int32 N, N rows of O, N rows of U; W = ceil(N/64) uint64 words).
Usage: python c10graph.py <rays.npz> <graph.bin> [tol]
"""
import sys, time, json
import numpy as np

T0 = time.time()
def log(*a): print("[%6.1fs]" % (time.time() - T0), *a, flush=True)

def match(A, B):
    """index in B of each row of A (rays, up to phase); -1 if absent"""
    G = np.abs(A.conj() @ B.T) ** 2
    j = G.argmax(axis=1); ok = G[np.arange(len(A)), j] > 1 - 1e-9
    return np.where(ok, j, -1)

def main():
    src, out = sys.argv[1], sys.argv[2]; tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    d = np.load(src); psi = d["psi"]; prod = set(d["product"].tolist()); N = len(psi)
    log("rays %d, product rays %d" % (N, len(prod)))
    G = np.abs(psi.conj() @ psi.T) ** 2
    np.fill_diagonal(G, np.nan)
    off = G[~np.isnan(G)]
    isO = np.abs(G) < tol; isU = np.abs(G - 0.1) < tol
    np.fill_diagonal(isO, False); np.fill_diagonal(isU, False)
    other = ~(isO | isU); np.fill_diagonal(other, False)
    gO = np.nanmax(np.where(isO, G, np.nan)); gU = np.nanmax(np.where(isU, np.abs(G - 0.1), np.nan))
    rest = G[other]
    log("pairs: O %d, U %d, conflict %d (unordered)" % (isO.sum() // 2, isU.sum() // 2, other.sum() // 2))
    log("edge accuracy: max |g| on O %.1e ; max |g-0.1| on U %.1e" % (gO, gU))
    log("gap: min g over conflicts %.3e ; min |g-0.1| over conflicts %.3e ; max off-diagonal g %.6f" % (
        rest.min(), np.abs(rest - 0.1).min(), np.nanmax(off)))
    degO = isO.sum(1); degU = isU.sum(1)
    log("deg O: min %d max %d mean %.1f ; deg U: min %d max %d mean %.1f" % (degO.min(), degO.max(), degO.mean(), degU.min(), degU.max(), degU.mean()))
    # values taken by the overlaps (rounded) -- a fingerprint
    vals, cnt = np.unique(np.round(off, 6), return_counts=True)
    log("distinct overlap values (6 d.p.): %d ; most common: %s" % (len(vals), sorted(zip(cnt.tolist(), vals.tolist()), reverse=True)[:8]))
    # symmetry orbits
    n = np.arange(10); w = np.exp(2j * np.pi / 10)
    F = w ** np.outer(n, n) / np.sqrt(10)
    gens = {"shift": np.roll(psi, -1, axis=1), "mod": psi * w ** n, "dil3": psi[:, (3 * n) % 10], "conj": psi.conj(),
            "fourier": psi @ F.T}
    parent = list(range(N))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    closed = {}
    for name, img in gens.items():
        j = match(img, psi); closed[name] = int((j < 0).sum())
        for a, b in enumerate(j):
            if b >= 0:
                ra, rb = find(a), find(int(b))
                if ra != rb: parent[ra] = rb
    log("catalogue closure (rays mapped outside the catalogue): %s" % closed)
    roots = np.array([find(x) for x in range(N)])
    orb = {}
    for x, r in enumerate(roots): orb.setdefault(r, []).append(x)
    sizes = sorted(len(v) for v in orb.values())
    log("orbits under the pair symmetries: %d, sizes %s" % (len(orb), sizes))
    porb = sorted({int(roots[p]) for p in prod})
    log("orbits containing product rays: %s (sizes %s)" % (porb, [len(orb[r]) for r in porb]))
    # orbit representatives and orbit-wise degrees
    reps = sorted(orb.values(), key=len)
    info = [{"size": len(v), "rep": v[0], "degO": int(degO[v[0]]), "degU": int(degU[v[0]]), "product": v[0] in prod} for v in reps]
    for i in info: log("  orbit size %4d rep %4d degO %3d degU %4d product %s" % (i["size"], i["rep"], i["degO"], i["degU"], i["product"]))
    np.save(out.replace(".bin", "_orbit.npy"), roots)
    json.dump({"N": N, "tol": tol, "orbits": info, "closure": closed}, open(out.replace(".bin", "_info.json"), "w"), indent=1)
    W = (N + 63) // 64
    def pack(M):
        B = np.zeros((N, W * 64), dtype=bool); B[:, :N] = M
        return np.packbits(B, axis=1, bitorder="little").view("<u8").reshape(N, W)   # bit v%64 of word v//64
    with open(out, "wb") as f:
        f.write(np.int32(N).tobytes()); f.write(pack(isO).astype("<u8").tobytes()); f.write(pack(isU).astype("<u8").tobytes())
    # self-check of the bit layout on a few rows
    PO = pack(isO)
    for r in (0, N // 2, N - 1):
        bits = [v for v in range(N) if (int(PO[r, v // 64]) >> (v % 64)) & 1]
        assert bits == list(np.nonzero(isO[r])[0]), "bit layout"
    log("wrote %s (W=%d)" % (out, W))

if __name__ == "__main__":
    main()
