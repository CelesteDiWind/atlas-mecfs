"""Paso o01: corpus sistemático de OpenAlex (sin selección manual) -> openalex/out/works_raw.json (o01b lo limpia en works.json)
Consulta: título o resumen con "myalgic encephalomyelitis" | "chronic fatigue syndrome" | "myalgic encephalopathy" |
"ME/CFS"; artículos y revisiones; 2010-2025. Un bloque de páginas por año (paginación por número de página, sin API
key; respuestas cacheadas en data/cache/).
Marcas: core = la EM/SFC aparece en el título; manual = está en el corpus seleccionado a mano (data/papers.json)."""
import importlib.util
import json
import os
import re
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("enrich", os.path.join(ROOT, "scripts", "02_enrich.py"))
enrich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrich)

QUERY = '"myalgic encephalomyelitis"|"chronic fatigue syndrome"|"myalgic encephalopathy"|"ME/CFS"'
YEARS = range(2010, 2027)  # 2026 añadido el 2026-10-21 (año en curso)
SELECT = "id,doi,title,publication_date,publication_year,type,language,authorships,referenced_works,abstract_inverted_index,primary_topic,cited_by_count"
CORE = re.compile(r"myalgic encephalo|chronic fatigue syndrome|ME/CFS|CFS/ME|\bCFS\b|\bME\b", re.I)


def short(u):
    return u.rsplit("/", 1)[-1] if u else None


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def decimal_year(s):
    if not s:
        return None
    y, m, d = (int(x) for x in s[:10].split("-"))
    return y + (date(y, m, d) - date(y, 1, 1)).days / (date(y + 1, 1, 1) - date(y, 1, 1)).days


def main():
    manual = {p["key"] for p in json.load(open(os.path.join(ROOT, "data", "papers.json"), encoding="utf-8")) if p["matched_by"] != "none"}
    works, calls, missing = {}, 0, []
    for y in YEARS:
        page, got = 1, 0
        while True:
            d = enrich.cached_get("https://api.openalex.org/works", {
                "filter": f"title_and_abstract.search:{QUERY},publication_year:{y},type:article|review",
                "select": SELECT, "per_page": 200, "page": page})
            calls += 1
            if not d:
                missing.append((y, page))
                break
            for w in d["results"]:
                k = short(w["id"])
                au = w.get("authorships") or []
                works[k] = dict(key=k, doi=w.get("doi"), title=w.get("title") or "", date=w.get("publication_date"),
                                t=decimal_year(w.get("publication_date")), year=w.get("publication_year"), type=w.get("type"),
                                lang=w.get("language"), cited_by=w.get("cited_by_count"),
                                authors=[a["author"].get("display_name") for a in au if a.get("author")],
                                author_ids=[short(a["author"].get("id")) for a in au if a.get("author")],
                                institutions=sorted({short(i.get("id")) for a in au for i in (a.get("institutions") or []) if i.get("id")}),
                                countries=sorted({c for a in au for c in (a.get("countries") or [])}),
                                refs=[short(r) for r in w.get("referenced_works") or []],
                                abstract=abstract(w.get("abstract_inverted_index")),
                                oa_topic=(w.get("primary_topic") or {}).get("display_name"))
                got += 1
            n = d["meta"]["count"]
            if page * 200 >= n or not d["results"]:
                break
            page += 1
        print(f"{y}: {got} de {n}", flush=True)
    W = sorted(works.values(), key=lambda w: (w["t"] if w["t"] is not None else (w["year"] or 0) + 0.5, w["key"]))
    for w in W:
        w["core"] = bool(CORE.search(w["title"]))
        w["manual"] = w["key"] in manual
    json.dump(W, open(os.path.join(OUT, "works_raw.json"), "w", encoding="utf-8"), ensure_ascii=False)
    s = dict(n=len(W), calls=calls, missing=missing, core=sum(w["core"] for w in W),
             with_refs=sum(bool(w["refs"]) for w in W), with_abstract=sum(len(w["abstract"]) > 200 for w in W),
             manual_in=sum(w["manual"] for w in W), manual_total=len(manual),
             english=sum(w["lang"] == "en" for w in W))
    json.dump(s, open(os.path.join(OUT, "o01_summary.json"), "w"), indent=1)
    print(s)


if __name__ == "__main__":
    main()
