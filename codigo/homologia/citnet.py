"""Utilidades de las redes de citas (pasos 14-15): BiCM, proyección validada, Curveball, filtraciones.

Redes bipartitas B (paper x columna): columnas = referencias (acoplamiento) o citantes (cocitación).
- BiCM (modelo de configuración bipartito, máxima entropía): p_ia = x_i y_a / (1 + x_i y_a) reproduce en media el grado
  de cada paper y de cada columna. Las columnas con el mismo grado comparten parámetro, así que se ajusta por clases.
- Proyección validada: un par (i, j) que comparte s columnas se conserva si P(Poisson(mu_ij) >= s) es significativo
  tras Benjamini-Hochberg, con mu_ij = sum_a p_ia p_ja (Tumminello 2011; Saracco 2017).
- Curveball (Strona 2014): recablea B conservando exactamente los dos grados; es el nulo de la red proyectada.
"""
import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.stats import poisson

Q = 0.05


def fit_bicm(B, iters=3000, tol=1e-10):
    """Devuelve (P_rowclass, row_x, col_classes): parámetros por fila y por clase de grado de columna."""
    k = np.asarray(B.sum(1)).ravel().astype(float)
    h = np.asarray(B.sum(0)).ravel().astype(float)
    hc, hcount = np.unique(h[h > 0], return_counts=True)  # clases de grado de columna
    x = np.where(k > 0, k / np.sqrt(k.sum() + 1), 0.0)
    y = hc / np.sqrt(k.sum() + 1)
    for _ in range(iters):
        xy = np.outer(x, y)
        den_x = (hcount * y / (1 + xy)).sum(1)
        xn = np.where(k > 0, k / np.maximum(den_x, 1e-300), 0.0)
        den_y = (x[:, None] / (1 + xy)).sum(0)
        yn = hc / np.maximum(den_y, 1e-300)
        if np.max(np.abs(xn - x) / (x + 1e-12)) < tol and np.max(np.abs(yn - y) / y) < tol:
            x, y = xn, yn
            break
        x, y = 0.5 * x + 0.5 * xn, 0.5 * y + 0.5 * yn  # amortiguado: converge de forma estable
    p = np.outer(x, y) / (1 + np.outer(x, y))       # n x clases
    err = np.abs((p * hcount).sum(1) - k).max()
    return p, hcount, err


def expected_shared(p, hcount):
    """mu_ij = sum_a p_ia p_ja agrupando por clase de columna."""
    return (p * hcount) @ p.T


def project(B):
    S = (B.astype(np.int32) @ B.T.astype(np.int32)).toarray()
    np.fill_diagonal(S, 0)
    return S


def validated(S, mu, active, q=Q):
    """Pares validados (i<j, ambos activos) -> (a, b, s, pvalor). BH sobre todos los pares entre nodos activos."""
    act = np.where(active)[0]
    m_tests = len(act) * (len(act) - 1) // 2
    iu = np.triu_indices(len(S), 1)
    s = S[iu]
    keep = (s >= 1) & active[iu[0]] & active[iu[1]]
    a, b, s = iu[0][keep], iu[1][keep], s[keep]
    pv = poisson.sf(s - 1, mu[a, b])
    order = np.argsort(pv)
    thr = q * np.arange(1, len(pv) + 1) / m_tests          # BH con el número total de pruebas
    ok = np.zeros(len(pv), bool)
    passed = np.where(pv[order] <= thr)[0]
    if len(passed):
        ok[order[: passed.max() + 1]] = True
    return a[ok], b[ok], s[ok], pv[ok]


def curveball(rows, rng, trades=None):
    """rows: lista de arrays (columnas de cada fila). Intercambios Curveball entre pares de filas."""
    rows = [set(r) for r in rows]
    nz = [i for i, r in enumerate(rows) if r]
    trades = trades or 10 * len(nz)
    for _ in range(trades):
        i, j = rng.choice(nz, 2, replace=False)
        a, b = rows[i], rows[j]
        ab, ba = a - b, b - a
        if not ab or not ba:
            continue
        pool = list(ab | ba)
        rng.shuffle(pool)
        na = len(ab)
        common = a & b
        rows[i] = common | set(pool[:na])
        rows[j] = common | set(pool[na:])
    return rows


def rows_of(B):
    B = B.tocsr()
    return [B.indices[B.indptr[i]:B.indptr[i + 1]] for i in range(B.shape[0])]


def from_rows(rows, shape):
    r = np.repeat(np.arange(len(rows)), [len(x) for x in rows])
    c = np.concatenate([np.fromiter(x, int) for x in rows if len(x)]) if any(len(x) for x in rows) else np.array([], int)
    return csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=shape)


def rank_filtration(a, b, pv, n):
    """Distancias para la filtración por fuerza: rango de significación normalizado a (0, 1] (más significativo
    primero). Matriz dispersa simétrica; los pares no validados no entran nunca."""
    order = np.argsort(pv, kind="stable")
    d = np.empty(len(pv))
    d[order] = (np.arange(len(pv)) + 1) / len(pv)
    M = csr_matrix((np.r_[d, d], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    return M


def time_filtration(a, b, tedge, tvert, nodes):
    """Filtración por fecha: vértices nacen en su fecha (diagonal) y aristas en la fecha del vínculo."""
    n = len(tvert)
    keep = np.isin(a, nodes) & np.isin(b, nodes)
    a, b, t = a[keep], b[keep], np.maximum(tedge[keep], np.maximum(tvert[a[keep]], tvert[b[keep]]))
    off = 1e-6  # ripser descarta entradas 0 explícitas: desplazamiento mínimo común a todo
    M = csr_matrix((np.r_[t, t, tvert[nodes]] - 1900 + off, (np.r_[a, b, nodes], np.r_[b, a, nodes])), shape=(n, n))
    return M


def h1_bars(M, maxdim=1):
    from ripser import ripser
    dg = ripser(M, distance_matrix=True, maxdim=maxdim)["dgms"]
    out = []
    for d in dg[1:]:
        d = d[np.isfinite(d[:, 1])]
        out.append(d[d[:, 1] > d[:, 0]])
    return dg[0], out
