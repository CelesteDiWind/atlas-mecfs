"""Paso o21: año de publicación de las referencias externas al corpus (para la edad de lo citado e índice de Price, o22).
Las referencias dentro del corpus ya tienen año (works.json). Las externas se piden a OpenAlex de 100 en 100
(filtro openalex:W1|W2|…, sin API key; cacheadas en data/cache/), de la más citada a la menos, para que la cobertura
ponderada por citas suba rápido aunque se agote la cuota diaria. Relanzar completa lo que falte.
Salida: out/ref_years.json {id: año}"""
import collections
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
B = 100


def main():
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    keys = {w["key"] for w in W}
    ext = collections.Counter(r for w in W for r in w["refs"] if r not in keys)
    ids = sorted(ext, key=lambda k: (-ext[k], k))
    path = os.path.join(OUT, "ref_years.json")
    got = json.load(open(path)) if os.path.exists(path) else {}
    fails = 0
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        d = enrich.cached_get("https://api.openalex.org/works", {
            "filter": "openalex:" + "|".join(chunk), "select": "id,publication_year", "per_page": B})
        if not d:
            fails += 1
            if fails >= 3:   # cuota agotada: se guarda lo que hay
                print("sin respuesta 3 veces seguidas; se para en el bloque", i // B, flush=True)
                break
            continue
        fails = 0
        for w in d["results"]:
            if w.get("publication_year"):
                got[w["id"].rsplit("/", 1)[-1]] = w["publication_year"]
        if (i // B) % 50 == 0:
            json.dump(got, open(path, "w"))
            cov = sum(ext[k] for k in got) / sum(ext.values())
            print(f"bloque {i // B}/{len(ids) // B + 1}: {len(got)} con año; cobertura ponderada {cov:.3f}", flush=True)
    json.dump(got, open(path, "w"))
    cov = sum(ext[k] for k in got if k in ext) / sum(ext.values())
    print(f"fin: {len(got)} de {len(ids)} referencias externas con año; cobertura ponderada {cov:.3f}")


if __name__ == "__main__":
    main()
