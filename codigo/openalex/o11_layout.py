"""Paso o11: disposición del corpus sistemático para el visor (misma receta que scripts/04_layout.py).
UMAP 3D (visualización) y 10D (temas), 2D con inicio PCA del 3D, 8 temas con KMeans sobre el 10D, términos distintivos,
vecinos semánticos, PageRank de las citas directas internas y disposición por citas (acoplamiento validado + citas
directas, reducidas con SVD y mezcladas con el contenido). Después se separan los puntos solapados.
Salida: out/layout_oa.npz y out/clusters_oa.json (términos y títulos representativos para ponerles nombre)."""
import importlib.util
import json
import os
import sys

import networkx as nx
import numpy as np
import umap
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
spec = importlib.util.spec_from_file_location("layout04", os.path.join(ROOT, "scripts", "04_layout.py"))
L4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L4)
SEED = L4.SEED
# 11 veces más puntos en el mismo volumen: puntos más pequeños que en el atlas de 583 papers
SIZE_BASE, SIZE_CIT, GAP = 0.020, 0.0075, 0.004


def main():
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    n = len(W)
    E = normalize(np.load(os.path.join(OUT, "E_specter2.npy")).astype(np.float64))
    key = {w["key"]: i for i, w in enumerate(W)}

    xyz = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.15, metric="cosine", random_state=SEED).fit_transform(E)
    z10 = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(E)
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    init2 = PCA(2, random_state=SEED).fit_transform(xyz)
    init2 = 10 * init2 / np.abs(init2).max()
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5, metric="cosine", init=init2, random_state=SEED).fit_transform(E)
    xy = L4.SCALE_2D * (xy - xy.mean(0)) / xy.std(0)
    print("UMAP de contenido listo", flush=True)

    # ---------- redes: acoplamiento validado y citas directas internas
    V = np.load(os.path.join(OUT, "val_full.npz"))
    a, b, s, pv = V["a"], V["b"], V["s"], V["pv"]
    wgt = -np.log10(np.maximum(pv, 1e-300))
    di = [(i, key[r]) for i, w in enumerate(W) for r in set(w["refs"]) if r in key and key[r] != i]
    CP = csr_matrix((np.r_[wgt, wgt], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    DI = csr_matrix((np.ones(len(di)), ([x for x, _ in di], [y for _, y in di])), shape=(n, n))
    DI = DI + DI.T
    G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(di)
    pr = nx.pagerank(G, alpha=0.85)
    pagerank = np.array([pr[i] for i in range(n)])
    linked = (np.asarray(CP.sum(1)).ravel() + np.asarray(DI.sum(1)).ravel()) > 0
    print(f"vínculos validados {len(a)}, citas directas {len(di)}, papers con vínculos {int(linked.sum())}", flush=True)
    net = hstack([normalize(CP), 0.6 * normalize(DI)]).tocsr()
    red = normalize(TruncatedSVD(100, random_state=SEED).fit_transform(net))
    F = np.hstack([red, L4.SEM_W * E])
    xyz_c = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.15, metric="cosine", init=10 * xyz / np.abs(xyz).max(),
                      random_state=SEED).fit_transform(F)
    xyz_c = (xyz_c - xyz_c.mean(0)) / xyz_c.std(0)
    xy_c = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5, metric="cosine", init=10 * xy / np.abs(xy).max(),
                     random_state=SEED).fit_transform(F)
    xy_c = L4.SCALE_2D * (xy_c - xy_c.mean(0)) / xy_c.std(0)
    print("UMAP de citas listo", flush=True)

    # ---------- tamaños y separación
    cit = np.array([w["cited_by"] or 0 for w in W])
    sizes = SIZE_BASE + SIZE_CIT * np.log10(1 + cit)
    rank = np.argsort(np.argsort(pagerank)).astype(float)
    sizes_c = np.interp(rank, np.arange(n), np.sort(sizes))
    radii = np.maximum(sizes, sizes_c) / 2
    sep = {}
    for name, X in (("3d", xyz), ("2d", xy), ("3d_c", xyz_c), ("2d_c", xy_c)):
        before = len(L4.overlapping_pairs(X, radii, GAP)[0])
        moved, it = L4.separate(X, radii, GAP)
        shift = np.linalg.norm(moved - X, axis=1)
        print(f"{name}: solapes {before} -> {len(L4.overlapping_pairs(moved, radii, GAP)[0])} ({it} it.), "
              f"desplazamiento medio {shift.mean():.3f}, máx {shift.max():.3f}", flush=True)
        sep[name] = moved

    # ---------- temas, términos y vecinos
    # temas estables: los de la versión 2010-2025 (KMeans sobre el UMAP 10D, out/topic_labels_v1.json); los trabajos nuevos
    # van al centroide SPECTER2 más cercano de esos temas, para que nombres, síntesis y comparaciones sigan valiendo
    v1 = json.load(open(os.path.join(OUT, "topic_labels_v1.json")))
    known = np.array([v1.get(w["key"], -1) for w in W])
    cent_v1 = normalize(np.vstack([E[known == c].mean(0) for c in range(8)]))
    labels = np.where(known >= 0, known, np.argmax(E @ cent_v1.T, axis=1))
    print(f"temas: {int((known >= 0).sum())} de la versión anterior, {int((known < 0).sum())} nuevos asignados por cercanía; "
          f"silhouette en el 10D = {silhouette_score(z10, labels):.3f}", flush=True)
    docs = [(w["title"] or "") + " " + (w["abstract"] or "") + " " + " ".join(w.get("kw") or []) for w in W]
    tf = TfidfVectorizer(stop_words=list(set(L4.GENERIC) | set(TfidfVectorizer(stop_words="english").get_stop_words())),
                         ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True)
    X = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    info = {}
    cent = np.vstack([E[labels == c].mean(0) for c in range(8)])
    for c in range(8):
        m = np.asarray(X[labels == c].mean(0)).ravel() - 0.5 * np.asarray(X[labels != c].mean(0)).ravel()
        idx = np.where(labels == c)[0]
        near = idx[np.argsort(-(E[idx] @ cent[c]))[:8]]
        info[c] = dict(n=int(len(idx)), terms=vocab[np.argsort(-m)[:12]].tolist(), titles=[W[i]["title"][:110] for i in near],
                       years=[int(np.percentile([W[i]["year"] for i in idx], q)) for q in (10, 50, 90)])
    json.dump(info, open(os.path.join(OUT, "clusters_oa.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    nn = np.zeros((n, 6), int); nn_s = np.zeros((n, 6))
    for r0 in range(0, n, 500):
        S = E[r0:r0 + 500] @ E.T
        S[np.arange(len(S)), np.arange(r0, r0 + len(S))] = -1
        top = np.argsort(-S, axis=1)[:, :6]
        nn[r0:r0 + 500] = top
        nn_s[r0:r0 + 500] = np.take_along_axis(S, top, axis=1)
    np.savez(os.path.join(OUT, "layout_oa.npz"), xyz=sep["3d"], xy=sep["2d"], xyz_c=sep["3d_c"], xy_c=sep["2d_c"], sizes=sizes,
             sizes_c=sizes_c, labels=labels, nn=nn, nn_s=nn_s, pagerank=pagerank, linked=linked)
    print("listo")


if __name__ == "__main__":
    main()
