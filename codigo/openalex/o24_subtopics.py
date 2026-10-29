"""Paso o24: subtemas basados en datos del corpus OpenAlex de EM/SFC (segundo nivel de temas; 2026-10-24).
Los 8 temas siguen siendo los de siempre (layout_oa.npz, `labels`: KMeans sobre UMAP-10D de la versión 2010-2025). Dentro de cada
tema, los subtemas son KMeans sobre el mismo UMAP-10D (misma receta que o11_layout: coseno, 15 vecinos, distancia mínima 0) con k
entre 2 y 6 elegido por silhouette. Sale un subtema por trabajo, ordenado de mayor a menor dentro de su tema, y sus términos
distintivos (TF-IDF contra el resto del tema) y trabajos más centrales, para ponerles nombre a mano
(data/subtopic_labels_oa.json, [es, en] por tema y subtema).
Regla de tamaño mínimo: la silhouette premia los grupos diminutos (un subtema de 1 trabajo en el tema XMRV, otro de 12 en dolor), así
que solo se consideran los k con todos los subtemas de al menos MIN_SUB trabajos; si ninguno cumple, el tema no tiene subtemas.
Referencia metodológica (misma receta que atlas_complejidad/scripts/o04_choose_k.py): el número de temas NO se eligió con los
datos; ver metodologia/. Salidas: out/z10.npy, out/subtopics_oa.json"""
import json
import os
import sys

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
OUT = os.path.join(OA, "out")
SEED = 42
MIN_SUB = 30
GENERIC = ("chronic fatigue syndrome myalgic encephalomyelitis encephalopathy cfs mecfs me patients patient study studies "
           "results result conclusion conclusions methods background objective using used based analysis").split()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    E = normalize(np.load(os.path.join(OUT, "E_specter2.npy")).astype(np.float64))
    lab = np.load(os.path.join(OUT, "layout_oa.npz"))["labels"]
    zp = os.path.join(OUT, "z10.npy")
    if not os.path.exists(zp):
        np.save(zp, umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=SEED).fit_transform(E))
    Z = np.load(zp)
    docs = [(w["title"] or "") + " " + (w["abstract"] or "") for w in W]
    tf = TfidfVectorizer(stop_words=list(set(GENERIC) | set(TfidfVectorizer(stop_words="english").get_stop_words())),
                         ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True)
    X = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    sub = np.zeros(len(W), int)
    info = {}
    for c in range(int(lab.max()) + 1):
        idx = np.where(lab == c)[0]
        sil, ok = {}, {}
        for kk in range(2, 7):
            lab_k = KMeans(kk, n_init=5, random_state=SEED).fit_predict(Z[idx])
            sil[kk] = float(silhouette_score(Z[idx], lab_k, sample_size=min(3000, len(idx)), random_state=0))
            ok[kk] = int(np.bincount(lab_k).min()) >= MIN_SUB
        valid = [kk for kk in sil if ok[kk]]
        if not valid:
            sub[idx] = 0
            info[str(c)] = dict(n=int(len(idx)), k=1, silhouette={str(a): round(b, 3) for a, b in sil.items()}, valid=[], subtopics={})
            print(f"tema {c}: n={len(idx)} sin subtemas (ningún k tiene todos los grupos >= {MIN_SUB}; silhouette {sil})", flush=True)
            continue
        k = max(valid, key=sil.get)
        r = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z[idx])
        order = np.argsort(-np.bincount(r, minlength=k))
        sub[idx] = np.argsort(order)[r]
        rest = np.setdiff1d(idx, idx)
        subs = {}
        for s in range(k):
            m = idx[sub[idx] == s]
            other = idx[sub[idx] != s]
            d = np.asarray(X[m].mean(0)).ravel() - 0.5 * np.asarray(X[other].mean(0)).ravel()
            cen = E[m].mean(0)
            near = m[np.argsort(-(E[m] @ cen))[:6]]
            subs[str(s)] = dict(n=int(len(m)), terms=vocab[np.argsort(-d)[:10]].tolist(), titles=[W[i]["title"][:110] for i in near],
                                years=[int(np.percentile([W[i]["year"] for i in m], q)) for q in (10, 50, 90)])
        info[str(c)] = dict(n=int(len(idx)), k=int(k), silhouette={str(a): round(b, 3) for a, b in sil.items()}, valid=valid, subtopics=subs)
        print(f"tema {c}: n={len(idx)} k={k} silhouette={sil[k]:.3f} tamaños {[subs[str(s)]['n'] for s in range(k)]}", flush=True)
    json.dump(dict(sub=sub.tolist(), topics=info), open(os.path.join(OUT, "subtopics_oa.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("subtemas en total:", sum(v["k"] for v in info.values() if v["subtopics"]))


if __name__ == "__main__":
    main()
