"""Paso 2: embeddings de sensibilidad -> out/E_mpnet.npy, out/E_bge-m3.npy, out/embedding_validation.csv
SPECTER2 (el del atlas) ya está en out/E_specter2.npy (título + abstract). Aquí se calculan dos modelos de
sentence-transformers con título + palabras clave + abstract (o título + palabras clave si no hay abstract) y se
validan todos contra el atlas: solapamiento de los 6 vecinos con `nn` y pureza de cluster en los 10 vecinos."""
import os

import numpy as np
import pandas as pd

from common import HOM, MODELS, chord, load_emb, load_papers, out

# bge-m3 se descarga en homologia/models/ (en Windows sin modo desarrollador la caché de HF falla al crear enlaces simbólicos)
_LOCAL_BGE = os.path.join(HOM, "models", "bge-m3")
ST_MODELS = {"mpnet": "sentence-transformers/all-mpnet-base-v2",
             "bge-m3": _LOCAL_BGE if os.path.exists(os.path.join(_LOCAL_BGE, "pytorch_model.bin")) else "BAAI/bge-m3"}


def texts(P):
    return [p["t"] + ". " + "; ".join(p["kw"] or []) + ". " + (p["abs"] or "") for p in P]


def validate(E, P):
    D = chord(E)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    cl = np.array([p["c"] for p in P])
    has_abs = np.array([bool(p["abs"]) for p in P])
    ov = np.array([len(set(order[i, :6]) & set(P[i]["nn"])) / 6 for i in range(len(P))])
    pur = np.array([np.mean(cl[order[i, :10]] == cl[i]) for i in range(len(P))])
    iu = np.triu_indices(len(P), 1)
    cos = 1 - chord(E)[iu] ** 2 / 2
    ev = np.linalg.eigvalsh(np.cov(E.T))[::-1]
    return dict(nn6_overlap=ov.mean(), nn6_overlap_with_abstract=ov[has_abs].mean(), nn6_overlap_no_abstract=ov[~has_abs].mean(),
                purity10=pur.mean(), cos_median=np.median(cos), cos_p5=np.percentile(cos, 5), cos_p95=np.percentile(cos, 95),
                effective_dim=ev.sum() ** 2 / (ev ** 2).sum())


def main():
    P = load_papers()["papers"]
    for name, hub in ST_MODELS.items():
        path = out(f"E_{name}.npy")
        if os.path.exists(path):
            continue
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(hub, device="cpu")
        model.max_seq_length = 512
        E = model.encode(texts(P), batch_size=8, show_progress_bar=True, normalize_embeddings=True)
        np.save(path, E)
        print(name, E.shape, flush=True)
    rows = [{"model": m, **validate(load_emb(m), P)} for m in MODELS]
    V = pd.DataFrame(rows)
    V.to_csv(out("embedding_validation.csv"), index=False)
    print(V.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
