"""Paso o12: exporta el corpus sistemático para el visor -> viewer/oa/oa_map.json, viewer/oa/oa_abs_<año>.json,
viewer/oa/oa_meta.json y out/manual_oa_map.json (índice del corpus manual -> índice OpenAlex, para el botón "ver en el otro corpus").
Mismo contrato que data/map.json (clusters, papers, edges), con estas diferencias:
  - sin cocitación (co vacío): haría falta descargar los trabajos que citan a cada paper
  - cp = acoplamiento validado (proyección de Poisson con BH, q < 0.05), ordenado por significación
  - los resúmenes van aparte, un archivo por año, que el visor descarga al abrir un paper de ese año
  - mi = índice del mismo paper en el corpus manual (si está)
Se puede relanzar tras completar works_extra.json (revista, palabras clave y enlace abierto)."""
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
VIEW = os.path.join(ROOT, "viewer", "oa")
os.makedirs(VIEW, exist_ok=True)
TAG = re.compile(r"<[^>]+>")
REL_CODE = {"central": 0, "secundaria": 1, "periferica": 2, "modelo_animal": 3, "otro_concepto": 4, "incierto": 5}


def clean(t):
    return re.sub(r"\s+", " ", TAG.sub("", (t or "").replace("�", ""))).strip()


VAGUE = {"based", "self", "organized", "organization", "model", "models", "theory", "science", "sciences", "time", "new", "adaptive"}


def label_terms(terms, k=5):
    """Términos para la etiqueta de un subtema sin repeticiones: una frase de dos palabras desplaza a las palabras sueltas
    que contiene, y se omiten las poco informativas."""
    out = []
    for t in terms:
        w = set(t.split())
        if any(w <= set(x.split()) for x in out):
            continue
        if len(w) == 1 and t in VAGUE:
            continue
        out = [x for x in out if not (len(x.split()) == 1 and x in w)] + [t]
    return out[:k]


def r4(x):
    return [round(float(v), 4) for v in x]


def main():
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    n = len(W)
    L = np.load(os.path.join(OUT, "layout_oa.npz"))
    V = np.load(os.path.join(OUT, "val_full.npz"))
    ex_path = os.path.join(OUT, "works_extra.json")
    extra = json.load(open(ex_path, encoding="utf-8")) if os.path.exists(ex_path) else {}
    key = {w["key"]: i for i, w in enumerate(W)}
    rel_path = os.path.join(OUT, "relevance.json")
    rel = json.load(open(rel_path, encoding="utf-8")) if os.path.exists(rel_path) else {}
    papers = json.load(open(os.path.join(ROOT, "data", "papers.json"), encoding="utf-8"))
    manual_of = {p["key"]: i for i, p in enumerate(papers) if p["matched_by"] != "none"}
    cit = [w["cited_by"] or 0 for w in W]

    # redes: acoplamiento validado (de la más a la menos significativa) y citas directas internas
    a, b, s, pv = V["a"], V["b"], V["s"], V["pv"]
    order = np.argsort(pv, kind="stable")
    a, b, s, pv = a[order], b[order], s[order], pv[order]
    m = len(a)
    wrel = 1 - np.arange(m) / m                                   # peso por rango: 1 = el más significativo
    cp = defaultdict(list)
    for i, (x, y, c) in enumerate(zip(a.tolist(), b.tolist(), s.tolist())):
        cp[x].append((y, c)); cp[y].append((x, c))                # ya ordenados por significación
    cites, cited = defaultdict(list), defaultdict(list)
    di = []
    for i, w in enumerate(W):
        for r in set(w["refs"]):
            j = key.get(r)
            if j is not None and j != i:
                cites[i].append(j); cited[j].append(i); di.append((i, j))
    by_cit = lambda L_: sorted(L_, key=lambda j: -cit[j])

    names = json.load(open(os.path.join(ROOT, "data", "cluster_labels_oa.json"), encoding="utf-8"))
    names_en = json.load(open(os.path.join(ROOT, "data", "cluster_labels_oa_en.json"), encoding="utf-8"))
    info = json.load(open(os.path.join(OUT, "clusters_oa.json"), encoding="utf-8"))
    # subtemas (segundo nivel; o24_subtopics.py) con nombres a mano en data/subtopic_labels_oa.json: {tema: {subtema: [es, en]}}
    st_path = os.path.join(OUT, "subtopics_oa.json")
    sub_info = json.load(open(st_path, encoding="utf-8")) if os.path.exists(st_path) else None
    sub_names = json.load(open(os.path.join(ROOT, "data", "subtopic_labels_oa.json"), encoding="utf-8")) if sub_info else {}
    clusters = [dict(id=c, name=names[str(c)], name_en=names_en[str(c)], terms=info[str(c)]["terms"], n=info[str(c)]["n"]) for c in range(8)]
    if sub_info:
        for c in clusters:
            ts = sub_info["topics"][str(c["id"])]["subtopics"]
            if ts:
                c["subs"] = [dict(id=int(k), n=v["n"], terms=label_terms(v["terms"]), name=sub_names[str(c["id"])][k][0],
                                  name_en=sub_names[str(c["id"])][k][1]) for k, v in ts.items()]

    out_papers, abs_by_year = [], defaultdict(dict)
    for i, w in enumerate(W):
        e = extra.get(w["key"], {})
        au = w["authors"] or []
        doi = (w["doi"] or "").replace("https://doi.org/", "")
        p = {"id": i, "t": clean(w["title"]), "y": w["year"], "a": [clean(x) for x in au[:8]] + (["et al."] if len(au) > 8 else []), "v": clean(e.get("v", "")),
             "doi": doi, "oa": e.get("oa", ""), "cit": cit[i], "kw": [clean(k) for k in e.get("kw", [])], "c": int(L["labels"][i]),
             **({"st": int(sub_info["sub"][i])} if sub_info and sub_info["topics"][str(int(L["labels"][i]))]["subtopics"] else {}),
             "p": r4(L["xyz"][i]), "p2": r4(L["xy"][i][:2]), "s": round(float(L["sizes"][i]), 4),
             "nn": [int(j) for j in L["nn"][i]], "nn_s": [round(float(v), 3) for v in L["nn_s"][i]],
             "pc": r4(L["xyz_c"][i]), "pc2": r4(L["xy_c"][i][:2]), "sc": round(float(L["sizes_c"][i]), 4),
             "hd": int(bool(w["refs"])), "co": [], "cp": [[j, c] for j, c in cp[i][:8]],
             "ci": by_cit(cites[i]), "cb": by_cit(cited[i])}
        if w["key"] in manual_of:
            p["mi"] = manual_of[w["key"]]
        # relevancia para la EM/SFC (o18-o19; 0 = central, se omite) y retractado (o01b)
        r = REL_CODE.get((rel.get(w["key"]) or {}).get("rel", "central"), 0)
        if r:
            p["r"] = r
        if w.get("retracted"):
            p["rt"] = 1
        out_papers.append(p)
        if w["abstract"]:
            abs_by_year[w["year"]][str(i)] = clean(w["abstract"])
    edges = {"co": [],
             "cp": [v for x, y, c, wt in zip(a.tolist(), b.tolist(), s.tolist(), wrel.tolist()) for v in (x, y, round(wt, 3), c)],
             "di": [v for x, y in di for v in (x, y, 1, 1)]}
    blob = json.dumps({"clusters": clusters, "papers": out_papers, "edges": edges}, ensure_ascii=False, separators=(",", ":"))
    open(os.path.join(VIEW, "oa_map.json"), "w", encoding="utf-8").write(blob.replace("</", "<\\/"))
    for y, d in abs_by_year.items():
        open(os.path.join(VIEW, f"oa_abs_{y}.json"), "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
    json.dump({str(manual_of[w["key"]]): i for i, w in enumerate(W) if w["key"] in manual_of},
              open(os.path.join(OUT, "manual_oa_map.json"), "w"))
    meta = dict(n=n, manual_in=sum("mi" in p for p in out_papers), relevance={k: sum(p.get("r", 0) == v for p in out_papers) for k, v in REL_CODE.items()},
                retracted=sum(p.get("rt", 0) for p in out_papers), edges_cp=m, edges_di=len(di), extra_covered=sum(w["key"] in extra for w in W),
                years=sorted(abs_by_year), map_mb=round(len(blob.encode()) / 1e6, 2),
                abs_mb={str(y): round(len(json.dumps(d, ensure_ascii=False).encode()) / 1e6, 2) for y, d in sorted(abs_by_year.items())})
    json.dump(meta, open(os.path.join(VIEW, "oa_meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
