"""Exact finite-sample certification probability for the strict fixed-sequence certificate.

Reference implementation of the closed form stated in Appendix A of the paper, taken from the
certificate's source work, cited in the paper, and used here unchanged. Vendored so this
repository is self-contained; it has no dependency on that work's code or data.

The procedure is strict fixed-sequence, so it certifies if and only if the first and most selective
grid threshold is rejected. That collapses the sequence to one hypergeometric event:

    m ~ Hypergeom(Npool, N1, n)                        calibration points inside S(lambda_1)
    k | m ~ Hypergeom(N1, K1, m)                       harmful ones among them
    reject iff hypergeom.cdf(k, N1, k*, m) <= delta,   k* = floor(alpha * N1) + 1

so Pr[certify] = sum_m P(m) * sum_{k : reject} P(k | m), a finite sum in (Npool, N1, K1, n, alpha,
delta) that needs no labels.
"""
import numpy as np
from scipy.stats import hypergeom

ALPHA, DELTA = 0.25, 0.1


def pr_certify(npool, n1, k1, n, alpha=ALPHA, delta=DELTA):
    """Exact Pr[certify] for strict fixed-sequence LTT with hypergeometric tests.

    npool: pool size; n1: size of the first grid selection; k1: harmful examples in it;
    n: calibration label budget; alpha: composition target; delta: confidence level.
    """
    ks = int(alpha * n1) + 1
    if ks > n1:
        return 1.0
    total = 0.0
    for m in range(0, min(n, n1) + 1):
        pm = hypergeom.pmf(m, npool, n1, n)
        if pm < 1e-12:
            continue
        kk = np.arange(0, min(m, k1) + 1)
        rej = hypergeom.cdf(kk, n1, ks, m) <= delta
        if not rej.any():
            continue
        total += pm * float(hypergeom.pmf(kk, n1, k1, m)[rej].sum())
    return float(total)
