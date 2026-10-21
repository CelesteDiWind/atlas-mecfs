"""Paso 8: topología de las redes de citas y comparación con la semántica (resultados previos 5 y 6).
Red: cocitación + acoplamiento (todas las aristas del atlas, >= 2), peso w = max(w_co, w_cp) en 0-1,
longitud de arista 1 - w + 0.05 (igual que la pasada previa, para comparar), geodésica en el componente gigante.
    - H0/H1 reales y nulo de recableado (modelo de configuración que conserva grados + permutación de pesos)
    - comparación semántica vs citas en los nodos compartidos, escalas normalizadas por la mediana:
      bottleneck y Wasserstein, con referencia = distancia al diagrama de redes recableadas
    - Mantel (Spearman, 999 permutaciones de nodos) y Mantel parcial controlando |diferencia de años|
    - comunidades de Louvain en la red de citas: ARI/NMI con los 8 clusters y con los modos semánticos
Salidas: out/citation_summary.json, out/h1_bars_citation.csv"""
import json
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from common import MAIN, N_NULL, SEED, finite, load_papers, out, rips, significance

N_MANTEL = 999
CIT_PERM = 300  # puntos de referencia para Rips en citas (real, nulos y comparación)


def weights(J, n):
    W = np.zeros((n, n))
    for key in ("co", "cp"):
        f = np.array(J["edges"][key]).reshape(-1, 4)
        a, b, w = f[:, 0].astype(int), f[:, 1].astype(int), f[:, 2]
        W[a, b] = np.maximum(W[a, b], w)
        W[b, a] = W[a, b]
    np.fill_diagonal(W, 0)
    return W


def geodesic(W):
    A = np.where(W > 0, 1 - W + 0.05, 0)
    return shortest_path(csr_matrix(A), directed=False)


def rewired(W, rng):
    """Modelo de configuración: se emparejan al azar los "cabos" de arista de cada nodo (conserva los grados salvo
    los pocos lazos y aristas repetidas que se descartan) y se reparten los pesos al azar. Sustituye al intercambio
    de aristas de networkx, que con ~60.000 aristas tardaba minutos por réplica."""
    iu = np.triu_indices(len(W), 1)
    w = W[iu][W[iu] > 0]
    deg = (W > 0).sum(1)
    stubs = rng.permutation(np.repeat(np.arange(len(W)), deg))
    a, b = stubs[0::2], stubs[1::2][: len(stubs) // 2]
    a = a[: len(b)]
    keep = a != b
    a, b = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    pairs = np.unique(a * len(W) + b)
    a, b = pairs // len(W), pairs % len(W)
    x = rng.permutation(w)[: len(a)]
    R = np.zeros_like(W)
    R[a, b] = R[b, a] = x
    return R


def norm_dgm(dg, scale):
    d = finite(dg)
    return d / scale


def mantel(A, B, rng, C=None):
    """Spearman entre triángulos superiores; p-valor por permutación de nodos. Si C, parcial (residuos de rangos)."""
    n = len(A)
    iu = np.triu_indices(n, 1)
    ra, rb = rankdata(A[iu]), rankdata(B[iu])
    if C is not None:
        rc = rankdata(C[iu])
        res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
        ra, rb = res(ra), res(rb)
    r0 = np.corrcoef(ra, rb)[0, 1]
    Ra = np.zeros((n, n))
    Ra[iu] = ra
    Ra = Ra + Ra.T
    cnt = 0
    for _ in range(N_MANTEL):
        p = rng.permutation(n)
        cnt += np.corrcoef(Ra[np.ix_(p, p)][iu], rb)[0, 1] >= r0
    return float(r0), (1 + cnt) / (1 + N_MANTEL)


def main():
    from persim import bottleneck, wasserstein
    J = load_papers()
    P = J["papers"]
    n = len(P)
    cl = np.array([p["c"] for p in P])
    yr = np.array([p["y"] or 0 for p in P], float)
    rng = np.random.default_rng(SEED)
    W = weights(J, n)
    ncomp, lab = connected_components(csr_matrix(W > 0))
    g = np.where(lab == np.bincount(lab).argmax())[0]
    Wg = W[np.ix_(g, g)]
    Dc = geodesic(Wg)
    # Rips completo en las redes nulas tarda ~3 min (miles de empates en las geodésicas aleatorias): se usan 300 puntos
    # de referencia por permutación voraz, igual en la red real, en los nulos y en el diagrama semántico comparado.
    dg_full = rips(Dc, 1)["dgms"]
    dg = rips(Dc, 1, n_perm=CIT_PERM)["dgms"]
    bars = finite(dg[1])
    bars = bars[bars[:, 0] > 0]
    pers, rat = bars[:, 1] - bars[:, 0], bars[:, 1] / bars[:, 0]

    npath = out("null_citation_config.npy")
    dpath = out("null_citation_dgms.npy")
    nulmax = list(np.load(npath)) if os.path.exists(npath) else []
    nuldg = list(np.load(dpath, allow_pickle=True)) if os.path.exists(dpath) else []
    while len(nulmax) < N_NULL:
        Dr = geodesic(rewired(Wg, rng))
        Dr[np.isinf(Dr)] = Dr[np.isfinite(Dr)].max() * 1.5
        R = rips(Dr, 1, n_perm=CIT_PERM)["dgms"]
        b = finite(R[1])
        b = b[b[:, 0] > 0]
        nulmax.append([finite(R[0])[:, 1].max(), (b[:, 1] - b[:, 0]).max(), (b[:, 1] / b[:, 0]).max(), len(b)])
        if len(nuldg) < 100:
            nuldg.append(norm_dgm(R[1], np.median(Dr)))
        if len(nulmax) % 10 == 0:
            np.save(npath, np.array(nulmax)); np.save(dpath, np.array(nuldg, dtype=object), allow_pickle=True)
            print("  nulo citas", len(nulmax), flush=True)
    nulmax = np.array(nulmax)
    tau, p = significance(rat, nulmax[:, 2])        # criterio principal: muerte/nacimiento
    taua, _ = significance(pers, nulmax[:, 1])      # secundario: persistencia aditiva
    pd.DataFrame({"birth": bars[:, 0], "death": bars[:, 1], "pers": pers, "ratio": rat, "p_rewire": p}).sort_values(
        "ratio", ascending=False).to_csv(out("h1_bars_citation.csv"), index=False)

    # comparación con la semántica en los mismos nodos
    summary = dict(n_giant=len(g), n_components=int(ncomp), n_isolated=int((W.sum(1) == 0).sum()),
                   h1_bars=len(pers), h1_bars_full_rips=len(finite(dg_full[1])), h1_tau_rewire=round(tau, 4),
                   h1_nsig=int((rat > tau).sum()), h1_tau_add=round(taua, 4), h1_nsig_add=int((pers > taua).sum()),
                   q_nbars=round(float((nulmax[:, 3] < len(bars)).mean()), 3),
                   h1_top5_ratio=[round(float(x), 4) for x in np.sort(rat)[::-1][:5]], n_landmarks=CIT_PERM)
    for sp in ("full", "k15"):
        Ds = np.load(out(f"D_{MAIN}_{sp}.npy")).astype(np.float64)[np.ix_(g, g)]
        ds = rips(Ds, 1, n_perm=CIT_PERM)["dgms"]
        a1, b1 = norm_dgm(ds[1], np.median(Ds)), norm_dgm(dg[1], np.median(Dc))
        a0, b0 = norm_dgm(ds[0], np.median(Ds)), norm_dgm(dg[0], np.median(Dc))
        bn1, ws1 = bottleneck(a1, b1), wasserstein(a1, b1)
        ref = np.array([[bottleneck(a1, x), wasserstein(a1, x)] for x in nuldg])
        summary[f"sem_{sp}"] = dict(bottleneck_h0=round(bottleneck(a0, b0), 4), bottleneck_h1=round(bn1, 4),
                                    wasserstein_h1=round(ws1, 4),
                                    ref_bottleneck_h1_median=round(float(np.median(ref[:, 0])), 4),
                                    ref_wasserstein_h1_median=round(float(np.median(ref[:, 1])), 4),
                                    frac_null_closer_bottleneck=round(float((ref[:, 0] <= bn1).mean()), 3),
                                    frac_null_closer_wasserstein=round(float((ref[:, 1] <= ws1).mean()), 3))
        r, pm = mantel(Ds, Dc, rng)
        has_y = yr[g] > 0
        Dy = np.abs(yr[g][:, None] - yr[g][None, :])
        sel = np.where(has_y)[0]
        rp, pp = mantel(Ds[np.ix_(sel, sel)], Dc[np.ix_(sel, sel)], rng, C=Dy[np.ix_(sel, sel)])
        summary[f"sem_{sp}"].update(mantel_rho=round(r, 4), mantel_p=pm, partial_mantel_rho_year=round(rp, 4), partial_mantel_p=pp)
        print(sp, summary[f"sem_{sp}"], flush=True)

    # comunidades de citas frente a clusters y modos semánticos (referencia independiente)
    G = nx.from_numpy_array(Wg)
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    lc = np.zeros(len(g), int)
    for k, c in enumerate(comms):
        lc[list(c)] = k
    modes = np.load(out("modes_labels.npy"), allow_pickle=True).item()
    summary["louvain"] = dict(n_comm=len(comms), ARI_clusters=round(ARI(cl[g], lc), 4), NMI_clusters=round(NMI(cl[g], lc), 4),
                              **{f"ARI_modes_{m}": round(ARI(v[g], lc), 4) for m, v in modes.items()})
    np.save(out("citation_communities.npy"), np.column_stack([g, lc]))
    with open(out("citation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
