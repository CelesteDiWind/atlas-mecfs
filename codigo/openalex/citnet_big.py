"""Versión por bloques de citnet para el corpus sistemático (miles de papers): nunca construye matrices n x n densas.
Mismo método que homologia/scripts/citnet.py (BiCM + proyección validada de Poisson con Benjamini-Hochberg + Curveball).
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import citnet as cn  # noqa: E402

fit_bicm, curveball, rows_of, from_rows = cn.fit_bicm, cn.curveball, cn.rows_of, cn.from_rows
BLOCK = 400


def _blocks(B, p, hcount, min_s=1):
    """Genera (i, j, s, mu) de todos los pares i<j con s >= min_s compartidos, por bloques de filas."""
    n = B.shape[0]
    Bc = B.tocsr()
    BT = Bc.T.tocsc().astype(np.float32)
    Bf = Bc.astype(np.float32)
    ph = (p * hcount).astype(np.float32)
    pt = p.T.astype(np.float32)
    for r0 in range(0, n, BLOCK):
        r1 = min(n, r0 + BLOCK)
        S = (Bf[r0:r1] @ BT).tocoo()
        keep = (S.col > S.row + r0) & (S.data >= min_s)
        i, j, s = S.row[keep] + r0, S.col[keep], S.data[keep].astype(np.int32)
        if len(i) == 0:
            continue
        mu_blk = ph[r0:r1] @ pt                      # bloque x n, solo se leen las posiciones necesarias
        yield i, j, s, mu_blk[i - r0, j].astype(np.float64)


def validated(B, p, hcount, active, q=cn.Q):
    """Pares validados (BH sobre todos los pares entre nodos activos). Devuelve (a, b, s, pv)."""
    na = int(active.sum())
    m_tests = na * (na - 1) // 2
    I, J, S, PV = [], [], [], []
    for i, j, s, mu in _blocks(B, p, hcount):
        ok = active[i] & active[j]
        i, j, s, mu = i[ok], j[ok], s[ok], mu[ok]
        pv = poisson.sf(s - 1, mu)
        k = pv <= q                                   # el umbral BH nunca supera q
        I.append(i[k]); J.append(j[k]); S.append(s[k]); PV.append(pv[k])
    a, b, s, pv = (np.concatenate(x) for x in (I, J, S, PV))
    order = np.argsort(pv, kind="stable")
    thr = q * np.arange(1, len(pv) + 1) / m_tests
    passed = np.where(pv[order] <= thr)[0]
    ok = np.zeros(len(pv), bool)
    if len(passed):
        ok[order[: passed.max() + 1]] = True
    return a[ok], b[ok], s[ok], pv[ok]


def top_m(B, p, hcount, active, m):
    """Los m pares más significativos (para las réplicas del nulo): mismo conteo de vínculos que la red real."""
    bestp, besti, bestj, bests = np.array([]), np.array([], int), np.array([], int), np.array([], int)
    for i, j, s, mu in _blocks(B, p, hcount):
        ok = active[i] & active[j]
        i, j, s, mu = i[ok], j[ok], s[ok], mu[ok]
        pv = poisson.sf(s - 1, mu)
        bestp, besti, bestj, bests = (np.concatenate([x, y]) for x, y in ((bestp, pv), (besti, i), (bestj, j), (bests, s)))
        if len(bestp) > 4 * m:
            o = np.argpartition(bestp, m)[:m]
            bestp, besti, bestj, bests = bestp[o], besti[o], bestj[o], bests[o]
    o = np.argsort(bestp, kind="stable")[:m]
    return besti[o], bestj[o], bests[o], bestp[o]
