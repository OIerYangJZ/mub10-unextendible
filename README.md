[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24297026.svg)](https://doi.org/10.5281/zenodo.24297026)

# No complete set of mutually unbiased bases in dimension ten contains the Fourier pair or three product bases

Code, certificates and logs for the paper of the same title by Jinze Yang, Yile Liu, Yangyang Li and Xiu-Hao Deng.

- arXiv: [arXiv:2611.21188](https://arxiv.org/abs/2611.21188)
- DOI of this archive: [10.5281/zenodo.24297026](https://doi.org/10.5281/zenodo.24297026) (all versions); v1.0: [10.5281/zenodo.24297027](https://doi.org/10.5281/zenodo.24297027)

## Results
- **Theorem A.** Every triple of mutually unbiased product bases of C² ⊗ C⁵ is strongly unextendible: no vector is
  unbiased to all three bases.
  - Consequently, no set of four MUBs in dimension ten contains three product bases for one factorisation.
- **Theorem B.** The standard basis and the Fourier basis of C¹⁰ are contained in no set of four MUBs.
  - Exactly 3040 rays are unbiased to both bases.
  - Each of them is unbiased to at most eight of the others.
  - Every triple formed by the pair and a third MUB is strongly unextendible.

Both proofs are computer-assisted. Every step used in the proofs is either exact integer arithmetic or rigorous
interval arithmetic.

## Contents
| path | content |
|---|---|
| `paper/` | LaTeX source and PDF of the paper (revised version, v1.1) |
| `scripts/` | all programs (generation, certification, verification) |
| `logs/` | logs of the reference verification runs; `logs/bb7/` has the certifying completeness run (v1.1); `logs/bb/` has the earlier split runs and their merge |
| `data/` | the certificate archive, split into 90 MB parts (see below) |

### Certificate archive
`data/mub10-unextendible-data.tar.xz` is about 0.6 GB. It is split into parts because of GitHub's file-size limit.
Reassemble and check it:
```
cat data/mub10-unextendible-data.tar.xz.part-* > mub10-unextendible-data.tar.xz
shasum -a 256 -c data/SHA256SUMS.archive        # f2a0d82f55b5450cbf5ba2e6e20e8969d9f8a4644e4686fe51101078b50b93f5
tar -xJf mub10-unextendible-data.tar.xz
shasum -a 256 -c data/SHA256SUMS                # every extracted file
```
The archive contains:
- the 910 chart systems and their cofactor certificates (`runs/charts_all/oN/`);
- the eight rank witnesses (`runs/rank_witness/`);
- the approximate cyclic-10 roots (`data/cyclic10/roots.bin`);
- the certified ray data (`runs/c10cert/`);
- copies of the scripts and logs.

## Checking Theorem A
```
python3 scripts/verify_manifest.py        # Python 3 with numpy; build scripts/rankmodp from rankmodp.cpp first
```
The script:
- regenerates every chart system from the exact Gauss-sum tables;
- requires byte equality with the systems that were certified;
- re-expands all 910 cofactor identities with integer arithmetic;
- checks the orbit roster by an independent exact enumeration;
- re-ranks the rank witnesses.

Expected last line: `ALL ORBITS COVERED: YES` (reference: `logs/manifest_final4.txt`).

## Checking Theorem B
```
g++ -O2 -std=c++17 -fopenmp -ffp-contract=off -frounding-math scripts/c10krawczyk.cpp -o c10krawczyk
mkdir -p out && ./c10krawczyk data/cyclic10/roots.bin out                     # C10-CERTIFY: PASS
sage -python scripts/c10verify_arb.py data/cyclic10/roots.bin out/graph_plus.bin 12   # ARB-VERIFY: PASS
g++ -O3 -fopenmp -std=c++17 scripts/c10clique.cpp -o c10clique
./c10clique out/graph_plus.bin nbr10; ./c10clique out/graph_plus.bin list10 bases.txt; ./c10clique out/graph_plus.bin quad bases.txt
g++ -O2 -std=c++17 -fopenmp -ffp-contract=off -frounding-math scripts/c10bb7.cpp -o c10bb7
SUBLIST=0-255 OMP_NUM_THREADS=<cores> ./c10bb7 data/cyclic10/roots.bin       # C10-BB7-SUBLIST: PASS
```
The completeness step splits into 256 independent subproblems.
- Any subset can be run with `SUBLIST=a-b,c,...`; each finished subproblem prints `SUBDONE s ... fails 0`.
- `c10bb7.cpp` (v1.1) is `c10bb6.cpp` with fail-closed interval arithmetic: an operation with a non-finite operand
  returns the whole real line, so a floating-point overflow can never cause a false exclusion. It prints the number
  of overflowed enclosures at the end.
- The certifying run of `c10bb7` (`logs/bb7/`) was made in one piece on one workstation (Apple clang 21, 16 threads,
  48,482 s): `C10-BB7-SUBLIST: PASS`, 256/256 subproblems, 30,859,491,964 boxes, 0 fails, 0 overflowed enclosures.
  Its single catalogue of 3040 certified boxes is used by all subproblems.
- The earlier split runs of `c10bb6.cpp` (v1.0; sha256 7267865364b4e7aef6868b04fe6fbb2bb647513a55a6426ba54dc89196bcc8b1;
  server, cluster and workstation, about 2000 core-hours) are kept in `logs/bb/` with their build record
  (`logs/bb/BUILD_RECORD.md`); their per-subproblem node counts equal those of the `c10bb7` run.
  `python3 scripts/c10bb_merge.py logs/bb/<run>.out+logs/bb/<run>.err ...` reproduces their merge
  (`logs/bb/merge_final.txt`: `C10-BB-MERGE: COMPLETE`).

SHA-256 of the v1.1 additions:
```
14253c3d87ed9c6d223adcd3ee4bbffe27b130eb44f0f8d6db5ae02b861ac802  scripts/c10bb7.cpp
7d241f1a274c665975a231a523aa4c8c06923e0a19c0c13c6c5b04a6ee59ca39  logs/bb7/full.out
85eef63ed741384e43bf001cbebffd63ed2dcd75a01a507b77fb2b6621cc6212  logs/bb7/full.err
```

## Licences
Code: Apache License 2.0 (`LICENSE`). Data and certificates: CC BY 4.0 (`LICENSE-DATA`).
