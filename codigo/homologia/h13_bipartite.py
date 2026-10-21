"""Paso 13: redes bipartitas de las citas, con fechas -> out/bip_refs.npz, out/bip_citers.npz, out/cit_dates.npz
    - paper x referencia (acoplamiento): referenced_works de cada paper (registros de OpenAlex ya cacheados)
    - citante x paper (cocitación): trabajos de cualquier parte que citan a la colección, ahora con publication_date
      (nueva descarga: la de 02b pidió solo id y referenced_works; las respuestas se cachean en data/cache/)
    - fechas: publication_date exacta de cada paper (años decimales); sin fecha -> NaN
Índices de paper = posición en data/map.json (el mismo orden que homologia/out/papers.json).
Comprobación: las proyecciones con >= 2 elementos compartidos reproducen data/citations.json."""
import importlib.util
import json
import os
from datetime import date

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from common import ROOT, out

DATA = os.path.join(ROOT, "data")
spec = importlib.util.spec_from_file_location("enrich", os.path.join(ROOT, "scripts", "02_enrich.py"))
enrich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrich)
BATCH = 50


def short(url):
    return url.rsplit("/", 1)[-1]


def decimal_year(s):
    """'2021-06-15' -> 2021.45 (año decimal); None -> NaN."""
    if not s:
        return np.nan
    y, m, d = (int(x) for x in s[:10].split("-"))
    t0, t1 = date(y, 1, 1), date(y + 1, 1, 1)
    return y + (date(y, m, d) - t0).days / (t1 - t0).days


def citing_with_dates(ids):
    """{id citante: (fecha decimal, set(ids de la colección que cita))}; paginación por página (sin API key)."""
    res, missing = {}, 0
    for b in range(0, len(ids), BATCH):
        batch, page = ids[b:b + BATCH], 1
        while True:
            d = enrich.cached_get("https://api.openalex.org/works", {
                "filter": "cites:" + "|".join(batch), "select": "id,publication_date,referenced_works",
                "per_page": 200, "page": page})
            if not d:
                missing += 1
                break
            rs = d.get("results", [])
            for w in rs:
                k = short(w["id"])
                t, refs = res.get(k, (decimal_year(w.get("publication_date")), set()))
                refs.update(short(r) for r in w.get("referenced_works") or [])
                res[k] = (t, refs)
            if len(rs) < 200 or page * 200 >= d.get("meta", {}).get("count", 0):
                break
            page += 1
        print(f"  lote {b // BATCH + 1}/{(len(ids) + BATCH - 1) // BATCH}: {len(res)} citantes", flush=True)
    if missing:
        print(f"  AVISO: {missing} consultas sin respuesta; relanzar el paso para completarlas")
    return res, missing


def main():
    papers = json.load(open(os.path.join(DATA, "papers.json"), encoding="utf-8"))
    n = len(papers)
    idx = {p["key"]: i for i, p in enumerate(papers) if p["matched_by"] != "none"}

    # referencias y fecha exacta de cada paper
    refs, pdate = {}, np.full(n, np.nan)
    for p in papers:
        i = idx.get(p["key"])
        if i is None or not p["doi"]:
            continue
        w = enrich.openalex_by_doi(p["doi"])
        if w and short(w["id"]) == p["key"]:
            refs[i] = {short(r) for r in w.get("referenced_works") or []}
            pdate[i] = decimal_year(w.get("publication_date"))
    for i, p in enumerate(papers):  # sin registro: mitad del año declarado
        if np.isnan(pdate[i]) and p.get("year"):
            pdate[i] = p["year"] + 0.5
    ref_ids = sorted({r for rs in refs.values() for r in rs})
    col = {r: j for j, r in enumerate(ref_ids)}
    rows = [i for i, rs in refs.items() for r in rs]
    cols = [col[r] for i, rs in refs.items() for r in rs]
    R = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, len(ref_ids)))
    save_npz(out("bip_refs.npz"), R)
    print(f"paper x referencia: {R.shape}, {R.nnz} enlaces; papers con referencias: {(R.sum(1) > 0).sum()}")

    # citantes con fecha
    citing, missing = citing_with_dates(sorted(idx))
    cit_ids = sorted(citing)
    rows, cols = [], []
    for j, k in enumerate(cit_ids):
        for key in citing[k][1]:
            if key in idx:
                rows.append(idx[key]); cols.append(j)
    C = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, len(cit_ids)))
    cdate = np.array([citing[k][0] for k in cit_ids])
    save_npz(out("bip_citers.npz"), C)
    np.savez(out("cit_dates.npz"), paper=pdate, citer=cdate)
    print(f"paper x citante: {C.shape}, {C.nnz} enlaces; citantes sin fecha: {np.isnan(cdate).sum()}")

    # comprobación frente a data/citations.json
    ref = json.load(open(os.path.join(DATA, "citations.json"), encoding="utf-8"))
    for name, B, key in (("acoplamiento", R, "coupling"), ("cocitación", C, "cocitation")):
        S = (B.astype(np.int32) @ B.T.astype(np.int32)).toarray()
        np.fill_diagonal(S, 0)
        mine = int((np.triu(S, 1) >= 2).sum())
        print(f"{name}: pares >= 2 = {mine} (citations.json: {len(ref[key])})")
    json.dump({"n_refs": len(ref_ids), "n_citers": len(cit_ids), "missing_queries": missing,
               "papers_with_refs": int((R.sum(1) > 0).sum()), "papers_cited": int((C.sum(1) > 0).sum()),
               "papers_dated": int((~np.isnan(pdate)).sum())}, open(out("h13_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
