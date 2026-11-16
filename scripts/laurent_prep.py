#!/usr/bin/env python
"""Input file for laurent_mac (Laurent / torus Macaulay certificate search for the AFFINE chart of a product triple).
Writes: p r / the exact 20x10 exponent table E (exact_tables.py) / the monomial group: every Pauli
X2^a Z2^b (x) X5^c Z5^d (100 elements) whose action in U0-coordinates is monomial, as a substitution
z_i -> zeta20^{ph_i} z_{sig_i} (same convention as rank_cert.py).  The group is only used to FIND a certificate;
the certificate itself is checked against affine.ms by the independent expander verify_cert.py.
Usage: python laurent_prep.py <rep inf,inf,0,0,1,1> <out.txt>
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import product_triples_all as PA
import exact_tables as ET

rep, out = sys.argv[1], sys.argv[2]
p, r = 1073741741, 95182621
ell = [0 if x == "inf" else int(x) + 1 for x in rep.split(",")]
U = PA.triple(ell)
E = np.array(ET.exponent_table(rep), dtype=int)
C = np.vstack([U[1].conj().T @ U[0], U[2].conj().T @ U[0]]) * np.sqrt(10)
assert np.abs(C - np.exp(2j * np.pi * E / 20)).max() < 1e-12
mp = np.linalg.matrix_power
X2 = np.array([[0, 1], [1, 0]], complex); Z2 = np.diag([1, -1]).astype(complex)
X5 = np.roll(np.eye(5), 1, axis=0); Z5 = np.diag(np.exp(2j * np.pi * np.arange(5) / 5))
group = []
for a in range(2):
    for b in range(2):
        for c in range(5):
            for d in range(5):
                P = np.kron(mp(X2, a) @ mp(Z2, b), mp(X5, c) @ mp(Z5, d))
                M = U[0].conj().T @ P @ U[0]
                sig = np.argmax(np.abs(M), axis=1)
                if not all(abs(abs(M[i, sig[i]]) - 1) < 1e-9 for i in range(10)) or sorted(sig) != list(range(10)):
                    continue
                vals = np.array([M[i, sig[i]] for i in range(10)])
                ph = np.rint(np.angle(vals) / (2 * np.pi / 20)).astype(int) % 20
                assert np.abs(vals - np.exp(2j * np.pi * ph / 20)).max() < 1e-9
                group.append((list(sig), list(ph)))
with open(out, "w") as fh:
    fh.write("%d %d\n" % (p, r))
    for row in E:
        fh.write(" ".join(str(int(x)) for x in row) + "\n")
    fh.write("%d\n" % len(group))
    for sig, ph in group:
        fh.write(" ".join(str(int(x)) for x in sig) + " " + " ".join(str(int(x)) for x in ph) + "\n")
print("rep %s: %d monomial group elements written to %s" % (rep, len(group), out))
