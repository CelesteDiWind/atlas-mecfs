"""Paso 4: UMAP 3D + clusters temáticos + vecinos semánticos -> data/map.json
Etiquetas de clusters: si existe data/cluster_labels.json ({"0": "nombre", ...}) se usan; si no,
se imprimen los términos y títulos representativos de cada cluster para nombrarlos."""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import umap
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
SEED = 42
N_TOPICS = 8
# diámetro de cada punto en unidades del mapa (el visor lo dibuja tal cual): base + extra por citas
SIZE_BASE, SIZE_CIT = 0.05, 0.02
GAP = 0.012  # separación mínima entre bordes de dos puntos
SCALE_2D = 1.6  # el plano tiene menos sitio que el volumen: se abre un poco para no apelotonar


def point_sizes(papers):
    return np.array([SIZE_BASE + SIZE_CIT * np.log10(1 + (p["cited_by"] or 0)) for p in papers])


def overlapping_pairs(X, radii, gap):
    pairs = cKDTree(X).query_pairs(2 * radii.max() + gap, output_type="ndarray")
    if not len(pairs):
        return pairs, np.zeros(0), np.zeros(0)
    i, j = pairs.T
    dist = np.linalg.norm(X[j] - X[i], axis=1)
    need = radii[i] + radii[j] + gap
    m = dist < need
    return pairs[m], dist[m], need[m]


def separate(xyz, radii, gap, iters=500):
    """Empuja los pares que se solapan lo justo para que sus bordes queden a >= gap.
    Gauss-Seidel: cada par se corrige con las posiciones ya actualizadas (empujar todos a la vez oscila
    en los grupos densos)."""
    rng = np.random.default_rng(SEED)
    X = xyz.astype(np.float64)  # UMAP devuelve float32: sin margen, el redondeo deja pares "casi" separados
    for it in range(iters):
        pairs, _, _ = overlapping_pairs(X, radii, gap)
        if not len(pairs):
            break
        for a, b in pairs[rng.permutation(len(pairs))]:
            d = X[b] - X[a]
            dist = np.linalg.norm(d)
            need = radii[a] + radii[b] + gap
            if dist >= need:
                continue
            u = d / dist if dist > 1e-9 else rng.normal(size=X.shape[1])  # idénticos: dirección aleatoria
            u /= np.linalg.norm(u)
            push = (need * 1.002 - dist) / 2 * u
            X[a] -= push
            X[b] += push
    return X, it


GENERIC = ("chronic fatigue syndrome myalgic encephalomyelitis encephalomyelitis cfs mecfs me patients "
           "syndrome fatigue chronic study studies results conclusion conclusions methods background "
           "disease illness using also may associated compared however significant significantly found "
           "including within among controls healthy participants data").split()


def clean_title(p):
    t = re.sub(r"\s*\((?:z-library|[^)]*1lib)[^)]*\)", "", p["title"])
    t = re.sub(r"\s*\(([^)]+)\)\s*$", r" — \1", t) if "z-lib" in p["title"] else t
    if re.fullmatch(r"[a-z]+_?\d+", t) or "_" in t:  # nombres de archivo crudos: usar 1a línea del PDF
        lines = [l.strip() for l in p.get("_text", "").splitlines() if len(l.strip()) > 25]
        t = lines[0] if lines and "_" not in t else t.replace("_", " ")
    return t.strip()


SEM_W = 0.35  # peso del contenido en el layout por citas: solo desempata/ubica papers con pocos vínculos


def load_networks(n):
    """Redes de citas de 02b_citations.py (None si no se ha ejecutado)."""
    path = os.path.join(DATA, "citations.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        net = json.load(fh)
    assert net["n"] == n, "citations.json no corresponde a papers.json: vuelve a ejecutar 02b_citations.py"
    return net


def network_features(net, n):
    """Fila de vínculos de cada paper (cocitación, acoplamiento, citas directas), normalizada por bloque."""
    co, cp, di = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for a, b, _, w in net["cocitation"]:
        co[a, b] = co[b, a] = w
    for a, b, _, w in net["coupling"]:
        cp[a, b] = cp[b, a] = w
    for a, b in net["direct"]:
        di[a, b] = di[b, a] = 1
    linked = (co.sum(1) + cp.sum(1) + di.sum(1)) > 0
    return np.hstack([normalize(co), normalize(cp), 0.6 * normalize(di)]), linked


def centrality_sizes(pagerank, sizes):
    """Tamaños por centralidad con la misma distribución que los tamaños por citas (empates -> mismo tamaño)."""
    rank = rankdata(pagerank, method="average") - 1
    return np.interp(rank, np.arange(len(sizes)), np.sort(sizes))


def all_edges(rows):
    """Todas las aristas de una red, de la más fuerte a la más débil, aplanadas como
    [a, b, peso relativo, recuento, ...]: el visor dibuja las k primeras según el control deslizante."""
    rows = sorted(rows, key=lambda r: -r[3])
    wmax = max((r[3] for r in rows), default=1)
    return [v for a, b, c, w in rows for v in (a, b, round(w / wmax, 3), c)]


def neighbor_lists(net, n, cited_by):
    """Por paper: cocitados y con referencias compartidas (top 8, con su recuento) y citas directas internas."""
    co, cp = defaultdict(list), defaultdict(list)
    for a, b, c, w in net["cocitation"]:
        co[a].append((b, c, w)); co[b].append((a, c, w))
    for a, b, s, w in net["coupling"]:
        cp[a].append((b, s, w)); cp[b].append((a, s, w))
    cites, cited_in = defaultdict(list), defaultdict(list)
    for a, b in net["direct"]:
        cites[a].append(b); cited_in[b].append(a)
    top = lambda L: [[j, c] for j, c, _ in sorted(L, key=lambda r: (-r[1], -r[2]))[:8]]
    by_cites = lambda L: sorted(L, key=lambda j: -cited_by[j])
    return [{"co": top(co[i]), "cp": top(cp[i]), "ci": by_cites(cites[i]), "cb": by_cites(cited_in[i])}
            for i in range(n)]


def main():
    with open(os.path.join(DATA, "papers.json"), encoding="utf-8") as fh:
        papers = json.load(fh)
    E = normalize(np.load(os.path.join(DATA, "embeddings.npy")))

    # posiciones 3D para visualizar y espacio 10D para agrupar
    xyz = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.15, metric="cosine",
                    random_state=SEED).fit_transform(E)
    z10 = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine",
                    random_state=SEED).fit_transform(E)
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)

    # vista 2D: UMAP propio en 2D, inicializado con la proyección PCA del 3D para que ambas vistas se parezcan
    init2 = PCA(2, random_state=SEED).fit_transform(xyz)
    init2 = 10 * init2 / np.abs(init2).max()
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5, metric="cosine", init=init2,
                   random_state=SEED).fit_transform(E)
    xy = SCALE_2D * (xy - xy.mean(0)) / xy.std(0)

    # redes de citas: tamaño por centralidad y layout por citas (iniciado desde el semántico para que se parezcan)
    n = len(papers)
    sizes = point_sizes(papers)
    net = load_networks(n)
    if net:
        sizes_c = centrality_sizes(np.array(net["pagerank"]), sizes)
        feat, linked = network_features(net, n)
        F = np.hstack([feat, SEM_W * E])
        print(f"papers con vínculos de citas: {linked.sum()} de {n}")
        xyz_c = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.15, metric="cosine",
                          init=10 * xyz / np.abs(xyz).max(), random_state=SEED).fit_transform(F)
        xyz_c = (xyz_c - xyz_c.mean(0)) / xyz_c.std(0)
        xy_c = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.5, metric="cosine",
                         init=10 * xy / np.abs(xy).max(), random_state=SEED).fit_transform(F)
        xy_c = SCALE_2D * (xy_c - xy_c.mean(0)) / xy_c.std(0)
    else:
        print("sin data/citations.json: se omiten las redes de citas (ejecuta 02b_citations.py)")
        sizes_c = sizes

    # distancia mínima entre papers para que no se tapen (en 2D garantiza que no se tapen en pantalla);
    # con el mayor de los dos tamaños, para que valga tanto con "tamaño = citas" como con "= centralidad"
    radii = np.maximum(sizes, sizes_c) / 2

    def apply_separation(X, name):
        before = len(overlapping_pairs(X, radii, GAP)[0])
        moved, n_it = separate(X, radii, GAP)
        shift = np.linalg.norm(moved - X, axis=1)
        after = len(overlapping_pairs(moved, radii, GAP)[0])
        print(f"{name} solapamientos: {before} pares -> {after} ({n_it} iteraciones); "
              f"desplazamiento medio {shift.mean():.3f}, máx {shift.max():.3f}")
        return moved

    xyz = apply_separation(xyz, "3D")
    xy = apply_separation(xy, "2D")
    if net:
        xyz_c = apply_separation(xyz_c, "3D citas")
        xy_c = apply_separation(xy_c, "2D citas")

    # 8 temas: uno por color de la paleta categórica del visor (más clases no se distinguen por color)
    k = N_TOPICS
    labels = KMeans(k, n_init=20, random_state=SEED).fit_predict(z10)
    print(f"k={k} silhouette={silhouette_score(z10, labels):.3f}")

    # términos distintivos por cluster (c-TF-IDF simple)
    docs = [p["title"] + " " + p["abstract"] + " " + " ".join(p["keywords"]) for p in papers]
    tf = TfidfVectorizer(stop_words=list(set(GENERIC) | set(TfidfVectorizer(stop_words="english").get_stop_words())),
                         ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True)
    X = tf.fit_transform(docs)
    vocab = np.array(tf.get_feature_names_out())
    terms = {}
    for c in range(k):
        m = np.asarray(X[labels == c].mean(0)).ravel() - 0.5 * np.asarray(X[labels != c].mean(0)).ravel()
        terms[c] = vocab[np.argsort(-m)[:10]].tolist()

    labels_path = os.path.join(DATA, "cluster_labels.json")
    names = {}
    if os.path.exists(labels_path):
        with open(labels_path, encoding="utf-8") as fh:
            names = json.load(fh)
    else:
        cent = np.vstack([E[labels == c].mean(0) for c in range(k)])
        for c in range(k):
            idx = np.where(labels == c)[0]
            near = idx[np.argsort(-(E[idx] @ cent[c]))[:6]]
            print(f"\n## {c} ({len(idx)}) {', '.join(terms[c])}")
            for i in near:
                print("   -", papers[i]["title"][:100])

    # vecinos semánticos (coseno en el espacio original de 768 dims)
    S = E @ E.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, axis=1)[:, :6]

    lists = neighbor_lists(net, n, [p["cited_by"] or 0 for p in papers]) if net else None
    has_data = set(net["has_data"]) if net else set()

    out_papers = []
    for i, p in enumerate(papers):
        extra = {}
        if net:
            extra = {
                "pc": [round(float(v), 4) for v in xyz_c[i]],
                "pc2": [round(float(v), 4) for v in xy_c[i]],
                "sc": round(float(sizes_c[i]), 4),
                "hd": int(i in has_data),
                **lists[i],
            }
        out_papers.append({
            "id": i,
            "t": clean_title(p),
            "y": p["year"],
            "a": p["authors"][:8] + (["et al."] if p["n_authors"] > 8 else []),
            "v": p["venue"],
            "doi": p["doi"],
            "url": p["url"],
            "oa": p["oa_url"],
            "cit": p["cited_by"],
            "abs": p["abstract"],
            "abs_src": p["abstract_source"],
            "kw": p["keywords"][:6],
            "c": int(labels[i]),
            "p": [round(float(v), 4) for v in xyz[i]],
            "p2": [round(float(v), 4) for v in xy[i]],
            "s": round(float(sizes[i]), 4),
            "nn": [int(j) for j in nn[i]],
            "nn_s": [round(float(S[i, j]), 3) for j in nn[i]],
            "f": [f.split("/")[-1] for f in p["files"]],
            **extra,
        })
    names_en = {}
    en_path = os.path.join(DATA, "cluster_labels_en.json")
    if os.path.exists(en_path):
        with open(en_path, encoding="utf-8") as fh:
            names_en = json.load(fh)
    clusters = [{"id": c, "name": names.get(str(c), f"Cluster {c}"),
                 "name_en": names_en.get(str(c), names.get(str(c), f"Cluster {c}")), "terms": terms[c],
                 "n": int((labels == c).sum())} for c in range(k)]
    with open(os.path.join(DATA, "map.json"), "w", encoding="utf-8") as fh:
        out = {"clusters": clusters, "papers": out_papers}
        if net:  # todas las aristas de cada red, de la más fuerte a la más débil (el visor elige cuántas dibujar)
            out["edges"] = {
                "co": all_edges(net["cocitation"]),
                "cp": all_edges(net["coupling"]),
                "di": [v for a, b in net["direct"] for v in (a, b, 1, 1)],
            }
            print("aristas:", {k: len(v) // 4 for k, v in out["edges"].items()})
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("\nmap.json:", len(out_papers), "papers,", k, "clusters", "(con nombres)" if names else "(sin nombres)")


if __name__ == "__main__":
    sys.exit(main())
