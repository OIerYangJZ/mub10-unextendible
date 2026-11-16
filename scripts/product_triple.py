#!/usr/bin/env python
"""Astra-B round-4 request 1, numerical pilot: does the fully indirect 2x5 product triple
    U0 = [|0>(x)G_inf, |1>(x)G_0],  U1 = [|+>(x)G_1, |->(x)G_2],  U2 = [|+i>(x)G_3, |-i>(x)G_4]
(G_t(x,j) = w^{t x^2 + j x}/sqrt5, G_inf = I_5) admit a common unbiased unit vector?
Every vector unbiased to U0 is psi = U0 (1, e^{i theta_1..9}) / sqrt10.  Minimise
    F(theta) = sum_{b=1,2} sum_j (10 |<U_b e_j, psi>|^2 - 1)^2
from many random starts (BFGS with analytic gradient).  F = 0 <=> common unbiased vector.
Also reports the controls: pairwise unbiasedness of U0, U1, U2, and the qubit-marginal condition at the minimiser.
Usage: python product_triple.py [nstarts] [seed]
"""
import sys
import numpy as np
from scipy.optimize import minimize

w = np.exp(2j * np.pi / 5)
def G(t):
    if t is None:
        return np.eye(5, dtype=complex)
    return np.array([[w ** ((t * x * x + j * x) % 5) for j in range(5)] for x in range(5)]) / np.sqrt(5)

k0, k1 = np.array([1, 0], complex), np.array([0, 1], complex)
kp, km = (k0 + k1) / np.sqrt(2), (k0 - k1) / np.sqrt(2)
kpi, kmi = (k0 + 1j * k1) / np.sqrt(2), (k0 - 1j * k1) / np.sqrt(2)
def basis(q0, Ga, q1, Gb):
    return np.hstack([np.kron(q0[:, None], Ga), np.kron(q1[:, None], Gb)])
U0 = basis(k0, G(None), k1, G(0))
U1 = basis(kp, G(1), km, G(2))
U2 = basis(kpi, G(3), kmi, G(4))

def controls():
    out = []
    for name, U in (("U0", U0), ("U1", U1), ("U2", U2)):
        out.append("%s unitary err %.1e" % (name, np.abs(U.conj().T @ U - np.eye(10)).max()))
    for (a, A), (b, B) in ((("U0", U0), ("U1", U1)), (("U0", U0), ("U2", U2)), (("U1", U1), ("U2", U2))):
        out.append("|%s^* %s|^2 - 1/10 max %.1e" % (a, b, np.abs(np.abs(A.conj().T @ B) ** 2 - 0.1).max()))
    return out

M = np.vstack([U1.conj().T @ U0, U2.conj().T @ U0])   # rows: <U_b e_j| U0 ; psi = U0 x / sqrt10

def F_and_grad(th):
    x = np.concatenate([[1.0 + 0j], np.exp(1j * th)])
    amp = M @ x / np.sqrt(10)                            # 20 amplitudes
    p = np.abs(amp) ** 2
    r = 10 * p - 1
    F = (r ** 2).sum()
    # dF/dth_k = sum_j 2 r_j * 10 * d|amp_j|^2/dth_k ; d|amp|^2/dth_k = 2 Re(conj(amp) * M[:,k+1] * i e^{i th_k}/sqrt10)
    dx = 1j * np.exp(1j * th)
    D = (M[:, 1:] * dx[None, :]) / np.sqrt(10)          # d amp / d th
    dp = 2 * np.real(np.conj(amp)[:, None] * D)
    g = (2 * r * 10) @ dp
    return F, g

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    for c in controls():
        print(c)
    best = (np.inf, None)
    vals = []
    for s in range(n):
        th0 = rng.uniform(0, 2 * np.pi, 9)
        res = minimize(F_and_grad, th0, jac=True, method="BFGS", options={"gtol": 1e-12, "maxiter": 2000})
        vals.append(res.fun)
        if res.fun < best[0]:
            best = (res.fun, res.x)
    vals = np.sort(np.array(vals))
    print("starts %d ; min F %.6e ; 1%%/10%%/50%% quantiles %.4e %.4e %.4e" % (
        n, vals[0], vals[int(0.01 * n)], vals[int(0.1 * n)], vals[n // 2]))
    th = best[1]
    x = np.concatenate([[1.0 + 0j], np.exp(1j * th)])
    psi = U0 @ x / np.sqrt(10)
    rho = np.outer(psi, psi.conj()).reshape(2, 5, 2, 5)
    rq = np.einsum("ajbj->ab", rho)
    print("best: qubit marginal %s ; max |10 p - 1| over U1,U2: %.3e" % (
        np.array2string(rq, precision=4), np.abs(10 * np.abs(M @ x / np.sqrt(10)) ** 2 - 1).max()))
    print("distinct local minima (rounded 1e-6):", np.unique(np.round(vals, 6))[:12])

if __name__ == "__main__":
    main()
