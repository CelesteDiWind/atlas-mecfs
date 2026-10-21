"""Utilidades comunes del análisis de homología persistente (rutas, distancias, nulos, persistencia, figuras).
Cada paso guarda sus salidas en homologia/out/ y, si ya existen, las reutiliza (pipeline reanudable)."""
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
HOM = os.path.dirname(HERE)
ROOT = os.path.dirname(HOM)
OUT = os.path.join(HOM, "out")
FIGS = os.path.join(HOM, "figs")
RES = os.path.join(HOM, "resultados")
PREV = os.path.join(HOM, "previo", "resultados")
for d in (OUT, FIGS, RES):
    os.makedirs(d, exist_ok=True)

SEED = 20261018
N_NULL = 200        # réplicas por modelo nulo (el plan pedía >= 100)
N_SUB = 50          # réplicas de submuestreo para estabilidad
SUB_FRAC = 0.8
KS = (6, 15, 30)    # resoluciones de los grafos kNN
MODELS = ("specter2", "mpnet", "bge-m3")
MAIN = "specter2"


def out(name):
    return os.path.join(OUT, name)


def load_papers():
    with open(out("papers.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_emb(model):
    return normalize(np.load(out(f"E_{model}.npy")).astype(np.float64))


# ---------------------------------------------------------------- distancias
def chord(E):
    """Distancia de cuerda en la esfera unidad, sqrt(2 - 2 cos): métrica y monótona con el coseno."""
    D = np.sqrt(np.maximum(2 - 2 * np.clip(E @ E.T, -1, 1), 0))
    np.fill_diagonal(D, 0)
    return (D + D.T) / 2


def knn_geodesic(D, k):
    """Geodésica en el grafo kNN (unión simétrica) + árbol de expansión mínima completo, para que sea conexo
    sin inventar distancias infinitas."""
    n = len(D)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    A = np.zeros_like(D)
    rows = np.repeat(np.arange(n), k)
    A[rows, idx.ravel()] = D[rows, idx.ravel()]
    A = np.maximum(A, A.T)
    mst = minimum_spanning_tree(csr_matrix(D)).toarray()
    A = np.where(A > 0, A, np.maximum(mst, mst.T))
    G = shortest_path(csr_matrix(A), directed=False)
    return (G + G.T) / 2


def space_distance(E, space):
    """space = 'full' (cuerda completa) o 'k6' / 'k15' / 'k30' (geodésica kNN)."""
    D = chord(E)
    return D if space == "full" else knn_geodesic(D, int(space[1:]))


# ---------------------------------------------------------------- modelos nulos
class NullModel:
    """Tres nulos para embeddings normalizados:
    gauss   : gaussiana con la misma media y covarianza (conserva la concentración de distancias, destruye todo lo demás)
    colperm : permutación independiente de cada coordenada (conserva marginales, destruye correlaciones)
    gmm     : mezcla gaussiana ajustada en PCA-50 (+ residuo gaussiano); n.º de componentes por BIC.
              Conserva los grupos y destruye los ciclos: es el nulo relevante para H1."""

    def __init__(self, E, seed=SEED, pca_dim=50, max_comp=14):
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        self.E = E
        self.mu = E.mean(0)
        self.cov = np.cov(E.T)
        self.L = np.linalg.cholesky(self.cov + 1e-10 * np.eye(E.shape[1]))
        pca_dim = min(pca_dim, max(5, len(E) // 4))  # subconjuntos pequeños (filtración por año): no más de n/4 dimensiones
        self.pca = PCA(pca_dim, random_state=seed).fit(E)
        Z = self.pca.transform(E)
        resid = E - self.pca.inverse_transform(Z)
        self.Lr = np.linalg.cholesky(np.cov(resid.T) + 1e-10 * np.eye(E.shape[1]))
        bics = []
        for c in range(1, max_comp + 1):
            g = GaussianMixture(c, covariance_type="diag", random_state=seed, n_init=3).fit(Z)
            bics.append(g.bic(Z))
        self.n_comp = int(np.argmin(bics)) + 1
        self.bics = bics
        self.gmm = GaussianMixture(self.n_comp, covariance_type="diag", random_state=seed, n_init=3).fit(Z)

    def sample(self, kind, rng, n=None):
        n = n or len(self.E)
        d = self.E.shape[1]
        if kind == "gauss":
            X = self.mu + rng.standard_normal((n, d)) @ self.L.T
        elif kind == "colperm":
            src = self.E if n == len(self.E) else self.E[rng.choice(len(self.E), n, replace=False)]
            X = np.column_stack([rng.permutation(src[:, j]) for j in range(d)])
        elif kind == "gmm":
            self.gmm.random_state = int(rng.integers(1 << 31))
            Z, _ = self.gmm.sample(n)
            X = self.pca.inverse_transform(Z) + rng.standard_normal((n, d)) @ self.Lr.T
            X = X[rng.permutation(n)]
        else:
            raise ValueError(kind)
        return normalize(X)


# ---------------------------------------------------------------- persistencia
def stable_hash(*parts):
    """Semilla estable entre ejecuciones (hash() de Python cambia en cada proceso)."""
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def rips(D, maxdim=1, cocycles=False, n_perm=None):
    from ripser import ripser
    return ripser(D, distance_matrix=True, maxdim=maxdim, do_cocycles=cocycles, n_perm=n_perm)


def finite(dg):
    return dg[np.isfinite(dg[:, 1])]


def persistence(dg):
    d = finite(dg)
    return d[:, 1] - d[:, 0]


def max_pers(dg):
    p = persistence(dg)
    return float(p.max()) if len(p) else 0.0


def ratio(dg):
    """Persistencia multiplicativa muerte/nacimiento (Bobrowski, Kahle y Skraba, 2017): invariante a la escala.
    Es el criterio principal: el nulo no reproduce la escala local del corpus (vecindarios muy cerrados), y la
    persistencia aditiva compara magnitudes que no son comparables."""
    d = finite(dg)
    d = d[d[:, 0] > 0]
    return d[:, 1] / d[:, 0]


STATS = ("pers", "ratio", "nbars", "total")  # columnas por dimensión en los archivos de nulos


def dgm_stats(dg):
    p, r = persistence(dg), ratio(dg)
    return [float(p.max()) if len(p) else 0.0, float(r.max()) if len(r) else 1.0, float(len(p)), float(p.sum())]


def col(dims, dim, stat):
    return dims.index(dim) * len(STATS) + STATS.index(stat)


def null_maxima(E, space, kind, null, reps=N_NULL, dims=(1,), tag=None, n=None, seed=SEED, h2_perm=200):
    """Por réplica y dimensión: persistencia máxima, cociente máximo, n.º de barras y persistencia total
    (columnas según STATS; usar col()). Reanudable: guarda cada 10 réplicas."""
    path = out(f"null2_{tag or ''}{space}_{kind}.npy")
    done = np.load(path) if os.path.exists(path) else np.zeros((0, len(dims) * len(STATS)))
    rng = np.random.default_rng([seed, stable_hash(space, kind, tag or "", n or 0)])
    for _ in range(len(done)):  # avanza el generador para que la reanudación sea reproducible
        null.sample(kind, rng, n)
    rows = list(done)
    for r in range(len(done), reps):
        X = null.sample(kind, rng, n)
        D = space_distance(X, space)
        row = []
        for dim in dims:
            dg = rips(D, 1)["dgms"][dim] if dim <= 1 else rips(D, 2, n_perm=h2_perm)["dgms"][2]
            row += dgm_stats(dg)
        rows.append(row)
        if (r + 1) % 10 == 0 or r + 1 == reps:
            np.save(path, np.array(rows))
    return np.array(rows)


def significance(pers, null_max):
    """p-valor de cada barra frente al máximo del nulo (control del error por diagrama) y umbral p95."""
    null_max = np.asarray(null_max)
    tau = float(np.percentile(null_max, 95))
    p = np.array([(1 + (null_max >= x).sum()) / (1 + len(null_max)) for x in pers])
    return tau, p


# ---------------------------------------------------------------- figuras (oscuras, verticales, 300 DPI)
DARK = {"bg": "#0D1117", "panel": "#161B22", "fg": "#E6EDF3", "muted": "#8B949E", "grid": "#30363D",
        "accent": "#58A6FF", "warn": "#F2CC60", "bad": "#FF7B72", "good": "#3FB950"}
CLUSTER_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2ea043", "#9085e9", "#e66767"]


def dark_fig(nrows=1, height=10.0, width=5.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": DARK["bg"], "axes.facecolor": DARK["panel"], "savefig.facecolor": DARK["bg"],
        "text.color": DARK["fg"], "axes.labelcolor": DARK["fg"], "xtick.color": DARK["muted"], "ytick.color": DARK["muted"],
        "axes.edgecolor": DARK["grid"], "grid.color": DARK["grid"], "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "legend.frameon": False,
    })
    fig, axes = plt.subplots(nrows, 1, figsize=(width, height))
    return fig, axes


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
