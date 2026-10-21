"""Paso 14: proyecciones validadas y comparación acoplamiento / cocitación (fase 1, criterios sin homología).
Por red (acoplamiento; cocitación vista hoy; cocitación fechada = solo citantes publicados hasta el fin de la ventana):
    cobertura, vínculos validados, densidad, componente gigante, aislados, mezcla entre temas frente a permutación,
    alineamiento con el contenido (AUC: similitud SPECTER2 de pares unidos frente a pares al azar), Louvain vs temas,
    estabilidad (bootstrap de columnas: Jaccard de vínculos validados), sensibilidad al umbral q.
En red completa y en ventanas deslizantes de 3 años. Guarda los vínculos validados con su fecha para el paso 15.
Salidas: out/h14_comparison.csv, out/h14_windows.csv, out/val_<red>.npz"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score as ARI

import citnet as cn
from common import MAIN, SEED, load_emb, load_papers, out

WINDOWS = [(y, y + 2) for y in range(2017, 2023)]   # 2017-19 ... 2022-24
N_BOOT = 20
N_PERM = 200


def second_date(C, cdate, a, b):
    """Fecha del segundo citante común de cada par (cuándo el par queda cocitado dos veces)."""
    C = C.tocsr()
    out_ = np.empty(len(a))
    for k, (i, j) in enumerate(zip(a, b)):
        ci = C.indices[C.indptr[i]:C.indptr[i + 1]]
        cj = C.indices[C.indptr[j]:C.indptr[j + 1]]
        d = np.sort(cdate[np.intersect1d(ci, cj)])
        d = d[~np.isnan(d)]
        out_[k] = d[1] if len(d) > 1 else (d[0] if len(d) else np.nan)
    return out_


def net_stats(B, rows_sel, cl, sim, rng, q=cn.Q, boot=False):
    """Proyección validada sobre el subconjunto de filas `rows_sel` y sus métricas."""
    n = B.shape[0]
    Bs = B[rows_sel]
    Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
    k = np.asarray(Bs.sum(1)).ravel()
    p, hc, _ = cn.fit_bicm(Bs)
    S = cn.project(Bs)
    a, b, s, pv = cn.validated(S, cn.expected_shared(p, hc), k > 0, q=q)
    ga, gb = rows_sel[a], rows_sel[b]
    m = len(rows_sel)
    A = csr_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    ncomp, lab = connected_components(A + A.T)
    sizes = np.bincount(lab)
    deg = np.asarray((A + A.T).sum(1)).ravel()
    active = k > 0
    r = dict(papers=m, active=int(active.sum()), raw_pairs_ge2=int((np.triu(S, 1) >= 2).sum()), validated=len(a),
             density=round(2 * len(a) / max(1, active.sum() * (active.sum() - 1)), 5),
             in_network=int((deg > 0).sum()), giant=int(sizes.max()) if len(a) else 0,
             giant_frac=round(sizes.max() / m, 3) if len(a) else 0.0, isolated_active=int(((deg == 0) & active).sum()))
    if len(a) == 0:
        return r, (ga, gb, s, pv)
    cross = float((cl[ga] != cl[gb]).mean())
    perm = []
    for _ in range(N_PERM):
        c2 = cl.copy()
        c2[rows_sel] = rng.permutation(cl[rows_sel])
        perm.append((c2[ga] != c2[gb]).mean())
    sims = sim[ga, gb]
    ra, rb = rng.choice(rows_sel, 20000), rng.choice(rows_sel, 20000)
    keep = ra != rb
    rnd = sim[ra[keep], rb[keep]]
    auc = float((sims[:, None] > rnd[None, :2000]).mean())
    G = nx.Graph()
    G.add_weighted_edges_from(zip(ga, gb, -np.log10(np.maximum(pv, 1e-300))))
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    lc = {v: c for c, vs in enumerate(comms) for v in vs}
    nodes = sorted(lc)
    r.update(cross_topic=round(cross, 3), cross_topic_perm=round(float(np.mean(perm)), 3),
             cross_topic_ratio=round(cross / np.mean(perm), 3), sem_auc=round(auc, 3), louvain_n=len(comms),
             louvain_ari_topics=round(ARI(cl[nodes], [lc[v] for v in nodes]), 3))
    if boot:  # estabilidad: remuestreo de columnas con reemplazo
        E0 = set(zip(ga, gb))
        jac = []
        ncol = B.shape[1]
        for _ in range(N_BOOT):
            cols = rng.integers(0, ncol, ncol)
            Bb = B[:, cols][rows_sel]
            Bb = Bb[:, np.asarray(Bb.sum(0)).ravel() > 0]
            kb = np.asarray(Bb.sum(1)).ravel()
            pb, hb, _ = cn.fit_bicm(Bb)
            ab, bb, _, _ = cn.validated(cn.project(Bb), cn.expected_shared(pb, hb), kb > 0, q=q)
            E1 = set(zip(rows_sel[ab], rows_sel[bb]))
            jac.append(len(E0 & E1) / max(1, len(E0 | E1)))
        r["boot_jaccard"] = round(float(np.mean(jac)), 3)
    return r, (ga, gb, s, pv)


def main():
    J = load_papers()
    P = J["papers"]
    n = len(P)
    cl = np.array([p["c"] for p in P])
    E = load_emb(MAIN)
    sim = E @ E.T
    D = np.load(out("cit_dates.npz"))
    pdate, cdate = D["paper"], D["citer"]
    yr = np.floor(pdate)
    R = load_npz(out("bip_refs.npz")).astype(np.int32).tocsc()
    C = load_npz(out("bip_citers.npz")).astype(np.int32).tocsc()
    rng = np.random.default_rng(SEED)
    allrows = np.arange(n)
    rows = []

    # --- red completa
    for name, B in (("acoplamiento", R), ("cocitacion", C)):
        for q in (0.05, 0.2):
            r, (a, b, s, pv) = net_stats(B, allrows, cl, sim, rng, q=q, boot=(q == 0.05))
            r.update(red=name, q=q, ventana="todas")
            rows.append(r)
            print(name, q, r, flush=True)
            if q == 0.05:
                te = np.maximum(pdate[a], pdate[b]) if name == "acoplamiento" else second_date(C, cdate, a, b)
                np.savez(out(f"val_{name}.npz"), a=a, b=b, s=s, pv=pv, tedge=te)
    pd.DataFrame(rows).to_csv(out("h14_comparison.csv"), index=False)

    # --- ventanas deslizantes de 3 años
    wrows = []
    for y0, y1 in WINDOWS:
        sel = np.where((yr >= y0) & (yr <= y1))[0]
        Cdated = C.multiply(csr_matrix(np.nan_to_num(cdate, nan=9999) <= y1 + 1).astype(np.int32)).tocsc()
        for name, B in (("acoplamiento", R), ("cocitacion_hoy", C), ("cocitacion_fechada", Cdated)):
            r, _ = net_stats(B, sel, cl, sim, rng)
            r.update(red=name, ventana=f"{y0}-{y1}")
            wrows.append(r)
            print(r["ventana"], name, {k: r.get(k) for k in ("papers", "active", "validated", "giant_frac", "cross_topic_ratio", "sem_auc")}, flush=True)
    pd.DataFrame(wrows).to_csv(out("h14_windows.csv"), index=False)


if __name__ == "__main__":
    main()
