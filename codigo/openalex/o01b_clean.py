"""Paso o01b: limpieza del corpus descargado (works_raw.json + E_raw.npy) -> works.json + E_specter2.npy
Quita: archivos suplementarios indexados como artículos ("MOESM…", "Additional file…", "Figure S…") y duplicados con
el mismo DOI (se queda el registro con más información: referencias, resumen, citas).
Marca: retractados (filtro is_retracted de OpenAlex + títulos "RETRACTED/Retracted/WITHDRAWN") -> campo `retracted`.
No se quita nada por relevancia: eso lo etiqueta o18-o19 (campo `rel`), y el visor lo filtra.
Salidas: out/works.json, out/E_specter2.npy, out/o01b_summary.json"""
import importlib.util
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
spec = importlib.util.spec_from_file_location("enrich", os.path.join(ROOT, "scripts", "02_enrich.py"))
enrich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrich)
spec2 = importlib.util.spec_from_file_location("o01", os.path.join(HERE, "o01_download.py"))
o01 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(o01)
SUPPL = re.compile(r"^\s*(MOESM\d+|Additional file|Supplementary (material|file|table|figure|data)|Figure S?\d|Table S?\d|Data[_ ]Sheet|Data from)", re.I)
RETR_TITLE = re.compile(r"^\s*(RETRACTED|Retracted|WITHDRAWN|Retraction note)\b")


def retracted_ids():
    ids, page = set(), 1
    years = f"{min(o01.YEARS)}-{max(o01.YEARS)}"
    while True:
        d = enrich.cached_get("https://api.openalex.org/works", {
            "filter": f"title_and_abstract.search:{o01.QUERY},publication_year:{years},type:article|review,is_retracted:true",
            "select": "id", "per_page": 200, "page": page})
        if not d:
            raise SystemExit("sin respuesta de OpenAlex (¿cuota agotada?): relanza más tarde")
        ids.update(o01.short(w["id"]) for w in d["results"])
        if page * 200 >= d["meta"]["count"] or not d["results"]:
            return ids
        page += 1


def main():
    W = json.load(open(os.path.join(OUT, "works_raw.json"), encoding="utf-8"))
    E = np.load(os.path.join(OUT, "E_raw.npy"))
    assert len(E) == len(W), "E_raw.npy no corresponde a works_raw.json: relanza o02"
    drop = {}
    for i, w in enumerate(W):
        if SUPPL.match(w["title"] or ""):
            drop[i] = "archivo suplementario"
    by_doi = defaultdict(list)
    for i, w in enumerate(W):
        if w["doi"] and i not in drop:
            by_doi[w["doi"].lower()].append(i)
    for doi, ii in by_doi.items():
        if len(ii) > 1:
            best = max(ii, key=lambda i: (len(W[i]["refs"]) > 0, len(W[i]["abstract"] or ""), W[i]["cited_by"] or 0, -i))
            for i in ii:
                if i != best:
                    drop[i] = f"duplicado de {W[best]['key']} (mismo DOI)"
    retr = retracted_ids()
    keep = [i for i in range(len(W)) if i not in drop]
    out = []
    for i in keep:
        w = dict(W[i])
        w["retracted"] = w["key"] in retr or bool(RETR_TITLE.match(w["title"] or ""))
        out.append(w)
    json.dump(out, open(os.path.join(OUT, "works.json"), "w", encoding="utf-8"), ensure_ascii=False)
    np.save(os.path.join(OUT, "E_specter2.npy"), E[keep])
    s = dict(raw=len(W), kept=len(out), dropped={W[i]["key"]: dict(title=W[i]["title"][:120], reason=r) for i, r in drop.items()},
             n_suppl=sum(r == "archivo suplementario" for r in drop.values()), n_dup=sum(r.startswith("duplicado") for r in drop.values()),
             retracted=[dict(key=w["key"], year=w["year"], title=w["title"][:120]) for w in out if w["retracted"]])
    json.dump(s, open(os.path.join(OUT, "o01b_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(W)} -> {len(out)} | suplementarios {s['n_suppl']} | duplicados {s['n_dup']} | retractados {len(s['retracted'])}")


if __name__ == "__main__":
    main()
