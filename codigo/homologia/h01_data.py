"""Paso 1: datos de partida -> out/papers.json, out/E_specter2.npy, out/flags.json
Lee el atlas desde data/map.json (el mismo objeto DATA que va dentro de atlas_mecfs.html) y los embeddings SPECTER2
originales (data/embeddings.npy). No modifica el atlas: deja una copia en homologia/atlas_copia.html.
Marca, para los análisis de sensibilidad: papers sin abstract, duplicados (coseno >= 0,985 o señalados en el
análisis de contenido) y papers que no tratan de EM/SFC según ese análisis."""
import json
import os
import shutil

import numpy as np

from common import HOM, ROOT, chord, out

# Señalados por los agentes del análisis de contenido (data/analysis): no tratan de EM/SFC o son ajenos al corpus.
OFF_TOPIC = {
    348: "aneurismas intracraneales", 350: "ictus en población rural", 579: "carta sobre nominaciones al Nobel",
    425: "ascorbato y ceftriaxona contra S. aureus", 293: "fármacos y mitocondria (sin EM/SFC)",
    349: "Parkinson tras COVID-19", 324: "editorial (fibrosis pulmonar / Parkinson)", 247: "fatiga en Sjögren",
    85: "vacuna de la hepatitis B y síndrome ASIA", 100: "fibromialgia", 251: "fibromialgia",
    537: "dolor musculoesquelético crónico", 246: "fibromialgia", 250: "fibromialgia / dolor", 542: "músculo en fibromialgia",
    224: "fibromialgia / dolor", 498: "neuropatía de fibra fina", 459: "gaslighting en psicoterapia", 145: "fatiga en trastornos del hierro",
}
DUPLICATES_FLAGGED = [(4, 347), (179, 313)]


def main():
    with open(os.path.join(ROOT, "data", "map.json"), encoding="utf-8") as fh:
        M = json.load(fh)
    shutil.copyfile(os.path.join(ROOT, "atlas_mecfs.html"), os.path.join(HOM, "atlas_copia.html"))
    P = M["papers"]
    keep = ("id", "t", "y", "a", "v", "doi", "cit", "abs", "kw", "c", "p", "p2", "nn", "nn_s", "pc", "pc2")
    papers = [{k: p.get(k) for k in keep} for p in P]
    E = np.load(os.path.join(ROOT, "data", "embeddings.npy"))
    np.save(out("E_specter2.npy"), E)

    D = chord(E / np.linalg.norm(E, axis=1, keepdims=True))
    cos = 1 - D ** 2 / 2
    iu = np.triu_indices(len(P), 1)
    near = [(int(a), int(b)) for a, b in zip(*iu) if cos[a, b] >= 0.985]
    dup = sorted({tuple(sorted(x)) for x in near + DUPLICATES_FLAGGED})
    flags = {
        "no_abstract": [p["id"] for p in P if not p["abs"]],
        "no_year": [p["id"] for p in P if not p["y"]],
        "duplicates": dup,
        "off_topic": {str(k): v for k, v in sorted(OFF_TOPIC.items())},
    }
    with open(out("papers.json"), "w", encoding="utf-8") as fh:
        json.dump({"clusters": M["clusters"], "papers": papers, "edges": M.get("edges")}, fh, ensure_ascii=False)
    with open(out("flags.json"), "w", encoding="utf-8") as fh:
        json.dump(flags, fh, ensure_ascii=False, indent=1)
    print(f"{len(P)} papers | sin abstract {len(flags['no_abstract'])} | sin año {len(flags['no_year'])} | "
          f"pares duplicados {len(dup)} {dup} | fuera de tema {len(OFF_TOPIC)}")


if __name__ == "__main__":
    main()
