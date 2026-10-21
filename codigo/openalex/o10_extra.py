"""Paso o10: campos que faltan para la ficha del visor (revista, palabras clave, enlace de acceso abierto).
Mismas consultas que o01 con otro `select` (41 llamadas por páginas, sin API key; cacheadas) -> out/works_extra.json
{id: {v, oa, kw}}."""
import importlib.util
import json
import os

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
SELECT = "id,primary_location,open_access,keywords"


def main():
    extra, missing = {}, []
    for y in o01.YEARS:
        page = 1
        while True:
            d = enrich.cached_get("https://api.openalex.org/works", {
                "filter": f"title_and_abstract.search:{o01.QUERY},publication_year:{y},type:article|review",
                "select": SELECT, "per_page": 200, "page": page})
            if not d:
                missing.append((y, page))   # cuota diaria agotada: se completa al relanzar
                print(f"  AVISO: sin respuesta en {y} página {page}", flush=True)
                break
            for w in d["results"]:
                loc = w.get("primary_location") or {}
                src = loc.get("source") or {}
                oa = w.get("open_access") or {}
                extra[o01.short(w["id"])] = dict(
                    v=src.get("display_name") or "", oa=(oa.get("oa_url") or "") if oa.get("is_oa") else "",
                    kw=[k["display_name"] for k in (w.get("keywords") or [])][:6])
            if page * 200 >= d["meta"]["count"] or not d["results"]:
                break
            page += 1
        print(y, len(extra), flush=True)
    json.dump(extra, open(os.path.join(OUT, "works_extra.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("páginas sin descargar:", missing)
    W = json.load(open(os.path.join(OUT, "works_raw.json"), encoding="utf-8"))
    print("cubiertos:", sum(w["key"] in extra for w in W), "de", len(W), "| con revista:", sum(bool(extra.get(w["key"], {}).get("v")) for w in W))


if __name__ == "__main__":
    main()
