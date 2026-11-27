# c10bb6 build record (reconstructed 2026-11-26)

Source sha256 7267865364b4e7aef6868b04fe6fbb2bb647513a55a6426ba54dc89196bcc8b1 and roots.bin sha256
151cda8f308d293c02c2093ac297e0f43e742fb3fa2cdd639ecbf3e8baf739be in every log below (checked).

Compile commands (verbatim, from the job scripts and terminal records of 2026-11-16):
- T1 cluster (phadcloud nodes, GCC 11.5.0 Red Hat), compiled INSIDE each Slurm job:
  `g++ -O2 -std=c++17 -fopenmp -ffp-contract=off -frounding-math c10bb6.cpp -o c10bb6_$SLURM_JOB_ID`
  (one binary per job; digests differ because __DATE__/__TIME__ are embedded and nodes differ in GCC build -5/-11/-14).
- school-server (amax, GCC 12.3.0): `g++ -O2 -std=c++17 -fopenmp -ffp-contract=off -frounding-math scripts/c10bb6.cpp -o scripts/c10bb6`;
  the binary is preserved: ~/mub10/scripts/c10bb6, sha256 6b978e80…6bf6a = logged BIN_SHA256.
- MacBook (Apple clang 21): `clang++ -O2 -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -ffp-contract=off -frounding-math c10bb6.cpp -o c10bb6`;
  the binary is preserved: runs/c10bb_mac/c10bb6 (sha256 e656531a… = logged BIN_SHA256).
No build used -ffast-math, -Ofast or -ffp-contract=fast; all are -O2 with contraction off (x86-64 SSE2 / ARM64: binary64).

| log | BIN sha256 (prefix) | host | compiler | zero-fail SUBDONE records |
|---|---|---|---|---|
| mac.out | e656531a434f7c26 | YangJinseys-MacBook-Pro-385.local | Apple LLVM 21.0.0 (clang-2100.4.1.1) | 7 |
| mac2.out | e656531a434f7c26 | YangJinseys-MacBook-Pro-385.local | Apple LLVM 21.0.0 (clang-2100.4.1.1) | 11 |
| v6_108-255_1682864.out | 6c9207c1d34031b6 | node138.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-5) | 41 |
| v6_44-107_1682865.out | 33f9396411d9ddf5 | node59.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 53 |
| v6n_B1_1685549.out | 55f27d98dce09a91 | node143.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 10 |
| v6n_B2_1685550.out | 757ad1e21a9bc364 | node78.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 10 |
| v6n_B3_1685551.out | 8707ef1a2094cdca | node164.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 10 |
| v6n_B4_1685552.out | 43705158fcfcd9b0 | node203.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 10 |
| v6n_D1_1686523.out | 286f54c7341bbe8f | node60.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 4 |
| v6n_D2_1686524.out | 14bb27de9e2a3699 | node78.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 4 |
| v6n_D3_1686525.out | 0ed6d496f3a3a3ab | node94.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 4 |
| v6n_D4_1686527.out | 4d4d8f6ee4f983a2 | node262.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 4 |
| v6n_D5_1686528.out | 81aaa9d066523f16 | node170.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 3 |
| v6n_G1_1687451.out | a6647da4f142adb8 | node253.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 3 |
| v6n_G10_1687460.out | 9ddd245e2f986dae | node271.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 1 |
| v6n_G2_1687452.out | 8cf7931568b3fd2c | node272.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 0 |
| v6n_G3_1687453.out | 8cf7931568b3fd2c | node239.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 3 |
| v6n_G4_1687454.out | e217c54f6ecb2773 | node240.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 3 |
| v6n_G5_1687455.out | c9ee5e1c41ffbf20 | node242.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 3 |
| v6n_G6_1687456.out | 868ddf4e1eda1546 | node243.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 2 |
| v6n_G7_1687457.out | f1927033058b6ef0 | node249.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 2 |
| v6n_G8_1687458.out | f1927033058b6ef0 | node250.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 1 |
| v6n_G9_1687459.out | c90f58b10c53e9ed | node251.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-11) | 0 |
| v6n_S2_1685497.out | 4a26f6e4ac323242 | node165.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 13 |
| v6r_A_1684035.out | 46268f6859dcf171 | node152.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 0 |
| v6r_A_1684044.out | 2f68fbb6e7123282 | node138.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-5) | 47 |
| v6r_B_1684069.out | 2ccebd47e2e24514 | node78.prod.phadcloud.local | 11.5.0 20240719 (Red Hat 11.5.0-14) | 0 |
| c10bb6_server.txt | 6b978e809925b1aa | amax | 12.3.0 | 32 |

Logs without a catalogue line (no binary started, no records): v6_44-107_1682862, v6_108-255_1682861.
The proof of Theorem B uses the single-build rerun of c10bb7 (logs/bb7/), so no cross-build identification of catalogues is needed.
