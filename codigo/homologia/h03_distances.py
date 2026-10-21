"""Paso 3: matrices de distancia -> out/D_{modelo}_{espacio}.npy y out/subsets.json
Espacios: 'full' = distancia de cuerda completa (583 x 583, sin grafo intermedio); 'k6'/'k15'/'k30' = geodésica en el
grafo kNN (+ árbol de expansión mínima para que sea conexo). Subconjuntos de sensibilidad:
    all          : los 583
    clean        : sin papers fuera de tema y sin el segundo de cada par duplicado
    with_abstract: solo papers con abstract (su embedding no viene del texto inicial del PDF)
Para los subconjuntos, las geodésicas se recalculan dentro del subconjunto (no se heredan del grafo completo)."""
import json
import os

import numpy as np

from common import KS, MODELS, load_emb, load_papers, out, space_distance


def subsets(P, flags):
    n = len(P)
    drop_dup = {b for a, b in flags["duplicates"]}
    off = {int(k) for k in flags["off_topic"]}
    return {
        "all": list(range(n)),
        "clean": [i for i in range(n) if i not in drop_dup and i not in off],
        "with_abstract": [i for i in range(n) if P[i]["abs"]],
    }


def main():
    P = load_papers()["papers"]
    with open(out("flags.json"), encoding="utf-8") as fh:
        S = subsets(P, json.load(fh))
    with open(out("subsets.json"), "w") as fh:
        json.dump(S, fh)
    print({k: len(v) for k, v in S.items()})
    for m in MODELS:
        if not os.path.exists(out(f"E_{m}.npy")):
            print("falta", m, "(ejecuta h02)"); continue
        E = load_emb(m)
        for sp in ("full",) + tuple(f"k{k}" for k in KS):
            path = out(f"D_{m}_{sp}.npy")
            if not os.path.exists(path):
                np.save(path, space_distance(E, sp).astype(np.float32))
        D = np.load(out(f"D_{m}_full.npy"))
        print(m, "listo | mediana cuerda %.3f" % np.median(D[np.triu_indices(len(D), 1)]))


if __name__ == "__main__":
    main()
