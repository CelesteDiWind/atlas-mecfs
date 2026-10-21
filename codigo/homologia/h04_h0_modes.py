"""Paso 4: H0, modos de densidad y mezcla entre clusters (resultados previos 1 y 2).
Salidas:
    out/modes_{modelo}.csv          picos de densidad (ToMATo sobre densidad kNN k=15) con prominencia y p-valor
    out/modes_labels.npy            etiquetas de modo por modelo (para comparar entre modelos y con citas)
    out/modes_agreement.csv         ARI/NMI entre modos, clusters del atlas y modos de otros modelos
    out/cluster_mixing.csv          fracción de vínculos kNN que salen de cada cluster frente a dos nulos
    out/h0_summary.csv              H0 (enlace simple) frente al nulo gaussiano
Por qué así: los 8 clusters salen de KMeans (k fijo por la paleta) sobre un UMAP de SPECTER2, así que compararlos
con la geometría de SPECTER2 es circular. Se añaden dos referencias: (a) el mismo procedimiento (UMAP 10D + KMeans 8)
aplicado a una nube gaussiana sin grupos, para saber qué mezcla produce "cortar un continuo en 8"; (b) modos de
densidad en modelos de embedding independientes (mpnet, bge-m3)."""
import os
import pickle

import numpy as np
import pandas as pd
from gudhi.clustering.tomato import Tomato
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

from common import MODELS, N_NULL, SEED, NullModel, load_emb, load_papers, max_pers, out, persistence, rips, space_distance

K_DENS = 15


def tomato_modes(D, k=K_DENS):
    """Densidad kNN (inversa de la distancia media a los k vecinos) sobre grafo kNN; devuelve el ajuste de ToMATo."""
    order = np.argsort(D, axis=1)[:, 1:k + 1]
    dens = -np.log(np.take_along_axis(D, order, axis=1).mean(1))
    t = Tomato(graph_type="manual", density_type="manual")
    t.fit([list(r) for r in order], weights=dens)
    return t, dens


def prominences(t):
    d = t.diagram_
    return np.sort(d[:, 0] - d[:, 1])[::-1] if len(d) else np.zeros(0)


def null_model(model, E):
    path = out(f"nullmodel_{model}_all.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    nm = NullModel(E)
    with open(path, "wb") as fh:
        pickle.dump(nm, fh)
    return nm


def cross_fraction(D, labels, k):
    """Por cluster: fracción de aristas del grafo kNN (dirigido) que acaban en otro cluster, y socio principal."""
    nb = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = []
    for c in np.unique(labels):
        m = labels == c
        tgt = labels[nb[m]].ravel()
        out_ = tgt[tgt != c]
        partner = np.bincount(out_, minlength=labels.max() + 1).argmax() if len(out_) else -1
        rows.append((c, m.sum(), (tgt != c).mean(), partner))
    return rows


def pipeline_partition(X, seed):
    """Mismo procedimiento que el atlas: UMAP 10D (coseno) + KMeans con k = 8."""
    import umap
    from sklearn.cluster import KMeans
    z = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=seed).fit_transform(X)
    return KMeans(8, n_init=10, random_state=seed).fit_predict(z)


def main():
    P = load_papers()["papers"]
    cl = np.array([p["c"] for p in P])
    rng = np.random.default_rng(SEED)
    mode_labels, agree, h0rows = {}, [], []
    for m in MODELS:
        if not os.path.exists(out(f"E_{m}.npy")):
            continue
        E = load_emb(m)
        D = np.load(out(f"D_{m}_full.npy")).astype(np.float64)
        t, dens = tomato_modes(D)
        prom = prominences(t)
        nm = null_model(m, E)
        npath = out(f"null_modes_{m}.npy")
        null_prom = list(np.load(npath)) if os.path.exists(npath) else []
        while len(null_prom) < N_NULL:  # nulo gaussiano: prominencia máxima de un modo (sin grupos reales)
            tn, _ = tomato_modes(space_distance(nm.sample("gauss", rng), "full"))
            pn = prominences(tn)
            null_prom.append(pn[0] if len(pn) else 0.0)
            if len(null_prom) % 20 == 0:
                np.save(npath, np.array(null_prom))
        null_prom = np.array(null_prom)
        tau = np.percentile(null_prom, 95)
        n_sig = int(max(1, (prom > tau).sum() + 1))  # + 1: el componente que nunca muere
        t.n_clusters_ = n_sig
        lab = np.array(t.labels_)
        mode_labels[m] = lab
        # tabla de picos: punto de densidad máxima de cada modo
        rows = []
        for j in np.unique(lab):
            idx = np.where(lab == j)[0]
            top = idx[np.argmax(dens[idx])]
            comp = pd.Series(cl[idx]).value_counts()
            rows.append(dict(mode=int(j), n=len(idx), peak_paper=int(top), title=P[top]["t"][:90], year=P[top]["y"],
                             main_cluster=int(comp.index[0]), purity=round(float(comp.iloc[0]) / len(idx), 3),
                             clusters={int(k): int(v) for k, v in comp.head(4).items()}))
        pr = pd.DataFrame(rows).sort_values("n", ascending=False)
        pr.to_csv(out(f"modes_{m}.csv"), index=False)
        agree.append(dict(model=m, ref="clusters_atlas", n_modes=n_sig, tau_prom=round(float(tau), 4),
                          top_prom=[round(float(x), 4) for x in prom[:10]], ARI=ARI(cl, lab), NMI=NMI(cl, lab)))
        print(f"{m}: {n_sig} modos significativos (prominencia > p95 nulo {tau:.4f}); ARI {ARI(cl, lab):.3f} NMI {NMI(cl, lab):.3f}", flush=True)
        # H0 del enlace simple frente al nulo gaussiano (barras finitas más largas)
        h0 = np.sort(persistence(rips(D, 0)["dgms"][0]))[::-1]
        h0n = []
        for _ in range(60):
            h0n.append(max_pers(rips(space_distance(nm.sample("gauss", rng), "full"), 0)["dgms"][0]))
        h0rows.append(dict(model=m, h0_top5=[round(float(x), 4) for x in h0[:5]], null_p95=round(float(np.percentile(h0n, 95)), 4),
                           n_over=int((h0 > np.percentile(h0n, 95)).sum())))
    np.save(out("modes_labels.npy"), mode_labels, allow_pickle=True)
    ms = list(mode_labels)
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            la, lb = mode_labels[ms[a]], mode_labels[ms[b]]
            agree.append(dict(model=ms[a], ref="modes_" + ms[b], n_modes=len(set(la)), ARI=ARI(la, lb), NMI=NMI(la, lb)))
    pd.DataFrame(agree).to_csv(out("modes_agreement.csv"), index=False)
    pd.DataFrame(h0rows).to_csv(out("h0_summary.csv"), index=False)

    # ---- mezcla entre clusters (resultado previo 1) en SPECTER2
    E = load_emb("specter2")
    D = np.load(out("D_specter2_full.npy")).astype(np.float64)
    nm = null_model("specter2", E)
    mix = []
    for k in (6, 15):
        obs = cross_fraction(D, cl, k)
        perm = np.array([[r[2] for r in cross_fraction(D, rng.permutation(cl), k)] for _ in range(200)])
        for (c, n, f, partner), pc in zip(obs, perm.T):
            mix.append(dict(k=k, cluster=int(c), n=int(n), cross_obs=round(f, 3), cross_perm_mean=round(pc.mean(), 3),
                            partner=int(partner)))
    # partición del mismo procedimiento sobre nubes sin grupos (lento: UMAP) -> rango de fracciones cruzadas
    ppath = out("pipeline_null_cross.npy")
    pn = list(np.load(ppath)) if os.path.exists(ppath) else []
    while len(pn) < 30:
        X = nm.sample("gauss", rng)
        lab = pipeline_partition(X, SEED + len(pn))
        Dx = space_distance(X, "full")
        pn.append(sorted(r[2] for r in cross_fraction(Dx, lab, 6)))
        np.save(ppath, np.array(pn))
        print("  nulo de partición", len(pn), flush=True)
    pn = np.array(pn)
    M = pd.DataFrame(mix)
    M.attrs["pipeline_null"] = pn
    M.to_csv(out("cluster_mixing.csv"), index=False)
    obs6 = np.sort(M[M.k == 6].cross_obs.values)
    print("mezcla k=6 observada (ordenada):", obs6, "\n  nulo de partición (media por rango):", pn.mean(0).round(3),
          "\n  permutación (media):", M[M.k == 6].cross_perm_mean.values)
    partners = M[M.k == 6].partner.value_counts()
    print("socio principal más frecuente:", partners.to_dict())


if __name__ == "__main__":
    main()
