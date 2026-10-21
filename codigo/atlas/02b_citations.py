"""Paso 2b: redes de citas de la colección -> data/citations.json
    - citas directas: A cita a B, ambos en la colección (de los referenced_works ya cacheados)
    - acoplamiento bibliográfico: A y B comparten referencias
    - cocitación: un tercer trabajo (de cualquier parte) cita a A y a B a la vez
      (descarga de OpenAlex los trabajos que citan a la colección; respuestas cacheadas en data/cache/)
Índices = posición en data/papers.json (el mismo orden que usa 04_layout.py)."""
import importlib.util
import json
import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
spec = importlib.util.spec_from_file_location("enrich", os.path.join(os.path.dirname(__file__), "02_enrich.py"))
enrich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrich)

BATCH = 50  # IDs por filtro cites:A|B|... (OpenAlex admite hasta 100)


def short(url):
    return url.rsplit("/", 1)[-1]


def citing_works(ids):
    """Trabajos que citan a alguno de `ids`: {id_citante: set(ids de la colección que cita)}."""
    out, calls = {}, 0
    for b in range(0, len(ids), BATCH):
        batch = ids[b:b + BATCH]
        # paginación por número de página: la paginación por cursor exige API key
        page = 1
        while True:
            d = enrich.cached_get("https://api.openalex.org/works", {
                "filter": "cites:" + "|".join(batch), "select": "id,referenced_works",
                "per_page": 200, "page": page})
            calls += 1
            if not d:
                print(f"  sin respuesta (lote {b // BATCH + 1}, página {page}); se reintentará en la próxima ejecución")
                break
            results = d.get("results", [])
            for w in results:
                refs = {short(r) for r in w.get("referenced_works") or []}
                out.setdefault(short(w["id"]), set()).update(refs)
            if len(results) < 200 or page * 200 >= d.get("meta", {}).get("count", 0):
                break
            page += 1
        print(f"  lote {b // BATCH + 1}/{(len(ids) + BATCH - 1) // BATCH}: {len(out)} trabajos citantes, {calls} consultas")
    return out


def pagerank(n, edges, d=0.85, iters=100):
    """PageRank sobre aristas dirigidas (citante -> citado)."""
    out_deg = np.zeros(n)
    for a, _ in edges:
        out_deg[a] += 1
    r = np.full(n, 1 / n)
    for _ in range(iters):
        nr = np.full(n, (1 - d) / n)
        for a, b in edges:
            nr[b] += d * r[a] / out_deg[a]
        nr += d * r[out_deg == 0].sum() / n  # nodos sin salida reparten uniforme
        if np.abs(nr - r).sum() < 1e-10:
            break
        r = nr
    return r


def main():
    with open(os.path.join(DATA, "papers.json"), encoding="utf-8") as fh:
        papers = json.load(fh)
    idx = {p["key"]: i for i, p in enumerate(papers) if p["matched_by"] != "none"}
    n = len(papers)

    # referencias de cada paper (registro OpenAlex ya cacheado al enriquecer)
    refs = {}
    for p in papers:
        if p["key"] in idx and p["doi"]:
            w = enrich.openalex_by_doi(p["doi"])
            if w and short(w["id"]) == p["key"]:
                refs[idx[p["key"]]] = {short(r) for r in w.get("referenced_works") or []}
    print(f"papers con referencias: {sum(1 for r in refs.values() if r)} de {len(idx)}")

    # 1) citas directas dentro de la colección
    direct = sorted({(i, idx[k]) for i, rs in refs.items() for k in rs if k in idx and idx[k] != i})
    print("citas directas internas:", len(direct))

    # 2) acoplamiento bibliográfico
    by_ref = defaultdict(list)
    for i, rs in refs.items():
        for k in rs:
            by_ref[k].append(i)
    shared = Counter()
    for owners in by_ref.values():
        for a, b in combinations(sorted(owners), 2):
            shared[(a, b)] += 1
    nref = {i: len(r) for i, r in refs.items()}
    coupling = [(a, b, s, s / np.sqrt(nref[a] * nref[b])) for (a, b), s in shared.items() if s >= 2]
    print("pares con >= 2 referencias compartidas:", len(coupling))

    # 3) cocitación
    print("descargando trabajos que citan a la colección…")
    citing = citing_works(sorted(idx))
    cocount = Counter()
    ncited = Counter()
    for cites in citing.values():
        mine = sorted(idx[k] for k in cites if k in idx)
        for a in mine:
            ncited[a] += 1
        for a, b in combinations(mine, 2):
            cocount[(a, b)] += 1
    cocit = [(a, b, c, c / np.sqrt(ncited[a] * ncited[b])) for (a, b), c in cocount.items() if c >= 2]
    print(f"trabajos citantes: {len(citing)} | pares cocitados >= 2 veces: {len(cocit)}")

    pr = pagerank(n, direct)
    with open(os.path.join(DATA, "citations.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "n": n,
            "direct": direct,
            "coupling": [(a, b, s, round(float(w), 4)) for a, b, s, w in coupling],
            "cocitation": [(a, b, c, round(float(w), 4)) for a, b, c, w in cocit],
            "pagerank": [float(x) for x in pr],
            "has_data": sorted(set(refs) | {i for i in range(n) if ncited[i]}),
        }, fh)
    top = np.argsort(-pr)[:5]
    print("más centrales (PageRank interno):")
    for i in top:
        print(f"   {pr[i]:.4f}  {papers[i]['title'][:90]}")


if __name__ == "__main__":
    main()
