"""Paso 5: inserta data/map.json (y el análisis por tema, si existe) en viewer/template.html ->
    atlas_mecfs.html             página autónoma para abrir en el navegador
    viewer/atlas_publish.html    mismo contenido sin <head> (para publicarlo como Artifact)
Análisis: data/analysis/out/*.json (un análisis por paper) y data/analysis/synth/c*.json (síntesis por tema),
generados a partir de data/analysis/inputs/ (paso 6). Huecos (capa «Mostrar huecos»): homologia/out/."""
import csv
import glob
import json
import os
import re
import shutil

ROOT = os.path.join(os.path.dirname(__file__), "..")
ARTIFACT_URL = "https://claude.ai/artifact/EksRuPPNTzvL81J6A98dRF"  # la página se enlaza a sí misma (mapa <-> análisis)
ANALYSIS_FIELDS = ("tipo", "tema", "diseno", "hallazgos", "conclusion", "triangulacion", "conceptos")
# tipos de estudio (el filtro del análisis agrupa por ellos): la traducción al inglés va en data/analysis/en/
TIPO_EN = {
    "Estudio original": "Original study", "Revisión narrativa": "Narrative review",
    "Revisión sistemática / metaanálisis": "Systematic review / meta-analysis",
    "Hipótesis / modelo teórico": "Hypothesis / theoretical model", "Otro": "Other", "Ensayo clínico": "Clinical trial",
    "Caso clínico": "Case report", "Estudio cualitativo": "Qualitative study", "Libro / tesis": "Book / thesis",
    "Guía / documento clínico": "Guideline / clinical document",
}


def load_analysis():
    """{"papers": {id: análisis + fuentes disponibles}, "clusters": {c: síntesis}} o None si no hay nada."""
    base = os.path.join(ROOT, "data", "analysis")
    sources = {}
    for f in glob.glob(os.path.join(base, "inputs", "*.json")):
        with open(f, encoding="utf-8") as fh:
            for p in json.load(fh)["papers"]:
                sources[p["id"]] = {
                    "fk": bool(p["keywords"]),
                    "fa": "inicio" if p["abstract_es_texto_inicial"] else bool(p["abstract"]),
                    "fc": p["fuente_conclusiones"],
                }
    papers = {}
    for f in sorted(glob.glob(os.path.join(base, "out", "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                rows = json.load(fh)["analisis"]
        except (ValueError, KeyError) as e:
            print(f"  aviso: {os.path.basename(f)} no se pudo leer ({e}); se omite")
            continue
        for r in rows:
            papers[int(r["id"])] = {**{k: r.get(k) for k in ANALYSIS_FIELDS}, **sources.get(int(r["id"]), {})}
    clusters = {}
    for f in sorted(glob.glob(os.path.join(base, "synth", "c*.json"))):
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        clusters[int(s["cluster"])] = s
    if not papers and not clusters:
        return None
    print(f"análisis: {len(papers)} papers, síntesis de {len(clusters)} temas")
    load_english(papers, clusters)
    return {"papers": papers, "clusters": clusters}


def load_english(papers, clusters):
    """Versión en inglés (traducción del texto en español, no un análisis nuevo): se guarda en el campo `en` de cada
    paper y de cada síntesis, y el visor la usa al elegir EN. Fuentes: data/analysis/en/out/p*.jsonl (papers) y
    data/analysis/en/synth_out/c*.json (síntesis; los números de paper se toman del original)."""
    base = os.path.join(ROOT, "data", "analysis", "en")
    n = 0
    for f in sorted(glob.glob(os.path.join(base, "out", "p*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                p = papers.get(int(r["id"]))
                if p is None:
                    continue
                en = {k: r[k] for k in ANALYSIS_FIELDS if k != "tipo"}
                en["tipo"] = TIPO_EN.get(p.get("tipo"), p.get("tipo"))
                p["en"] = en
                n += 1
    m = 0
    for f in sorted(glob.glob(os.path.join(base, "synth_out", "*.json"))):
        with open(f, encoding="utf-8") as fh:
            t = json.load(fh)
        s = clusters.get(int(t["cluster"]))
        if s is None:
            continue

        def with_papers(items, orig):  # textos traducidos + los números de paper del original (mismo orden)
            out = []
            for new, old in zip(items, orig):
                row = dict(new) if isinstance(new, dict) else {"texto": new}
                if isinstance(old, dict) and old.get("papers"):
                    row["papers"] = old["papers"]
                out.append(row)
            return out
        en = {"resumen": t["resumen"], "sintesis": t["sintesis"], "lineas": with_papers(t["lineas"], s["lineas"])}
        for k in ("consensos", "debates", "lagunas"):
            en[k] = with_papers(t[k], s[k])
        s["en"] = en
        m += 1
    print(f"  traducción al inglés: {n} papers, {m} síntesis")


def load_topology():
    """Huecos (ciclos H1 significativos en la cuerda completa de SPECTER2) de homologia/out (pasos 6 y 10):
    {"cycles": [{"nodes": [ids en el orden del lazo], "reappear": fracción de submuestras, "ntopics": n}]} o None."""
    base = os.path.join(ROOT, "homologia", "out")
    try:
        with open(os.path.join(base, "cycle_members.json"), encoding="utf-8") as fh:
            members = json.load(fh)
        with open(os.path.join(base, "cycles.csv"), encoding="utf-8") as fh:
            rows = {r["cycle_id"]: r for r in csv.DictReader(fh)}
    except FileNotFoundError:
        return None
    ids = sorted((k for k in members if k.startswith("full-")), key=lambda k: int(k.split("-")[1]))
    cycles = [{"nodes": members[k]["nodes"], "reappear": float(rows[k]["reappear"]), "ntopics": int(rows[k]["n_clusters"])}
              for k in ids]
    print(f"huecos: {len(cycles)}")
    return {"cycles": cycles} if cycles else None


SITE_README = """# Atlas EM/SFC

Mapa interactivo de la literatura sobre encefalomielitis miálgica / síndrome de fatiga crónica (EM/SFC), 2010–2026:
una selección manual de 583 artículos y un corpus sistemático de 7169 trabajos de OpenAlex, con análisis por tema,
análisis del campo (estructura, diálogo entre temas, evolución temporal) y síntesis por tema.

Interactive map of the ME/CFS literature, 2010–2026: a hand-picked selection of 583 papers and a systematic corpus of
7,169 OpenAlex works, with topic analyses, a field analysis (structure, dialogue between topics, change over time) and
topic syntheses.

Los cálculos y textos se hicieron con ayuda de Claude (IA); pueden contener errores. Datos bibliográficos: OpenAlex (CC0).
Calculations and texts were produced with the help of Claude (AI) and may contain errors. Bibliographic data: OpenAlex (CC0).
"""

STANDALONE_HEAD = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body{margin:0;font-size:14px}[hidden]{display:none!important}</style>
</head>
<body>
"""


def load_oa(data):
    """Corpus OpenAlex completo (homologia/openalex/, pasos o11-o12): enlaza cada paper manual con el suyo en OpenAlex (`oi`)
    y devuelve la configuración de carga (los archivos van aparte: el visor los descarga al activar el corpus) o None."""
    base = os.path.join(ROOT, "homologia", "openalex")
    try:
        with open(os.path.join(ROOT, "viewer", "oa", "oa_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(base, "out", "manual_oa_map.json"), encoding="utf-8") as fh:
            links = json.load(fh)
    except FileNotFoundError:
        return None
    for i, p in enumerate(data["papers"]):
        if str(i) in links:
            p["oi"] = links[str(i)]
    print(f"corpus OpenAlex: {meta['n']} trabajos, {len(links)} enlazados con el corpus manual")
    return {"map": "oa_map.json", "abs": "oa_abs_", "n": meta["n"]}


def load_sys():
    """Resultados del análisis sistemático (homologia/openalex/out/sys_data.json, paso o13) o None."""
    try:
        with open(os.path.join(ROOT, "homologia", "openalex", "out", "sys_data.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def load_evo():
    """Análisis temporales del corpus OpenAlex (homologia/openalex/out/o22_temporal.json, paso o22) o None."""
    try:
        with open(os.path.join(ROOT, "homologia", "openalex", "out", "o22_temporal.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def load_oa_synth():
    """Síntesis por tema del corpus OpenAlex (homologia/openalex/out/synth/t<c>.json y t<c>_en.json, escritas a partir de
    out/synth_inputs/, pasos o16-o17). Devuelve {clusters, papers (solo los citados), analysis} o None si no hay ninguna."""
    base = os.path.join(ROOT, "homologia", "openalex", "out")
    try:
        with open(os.path.join(ROOT, "viewer", "oa", "oa_map.json"), encoding="utf-8") as fh:
            oa = json.load(fh)
    except FileNotFoundError:
        return None
    P, C = oa["papers"], oa["clusters"]
    synth, cited, sample = {}, set(), {}
    for c in range(len(C)):
        try:
            with open(os.path.join(base, "synth", f"t{c}.json"), encoding="utf-8") as fh:
                es = json.load(fh)
            with open(os.path.join(base, "synth", f"t{c}_en.json"), encoding="utf-8") as fh:
                en = json.load(fh)
            with open(os.path.join(base, "synth_inputs", f"t{c}.json"), encoding="utf-8") as fh:
                sample[c] = len(json.load(fh)["papers"])
        except FileNotFoundError:
            continue
        es["en"] = {k: en[k] for k in ("resumen", "sintesis", "lineas", "consensos", "debates", "lagunas")}
        synth[str(c)] = es
        for doc in (es, en):
            texts = [doc["resumen"], *doc["sintesis"]] + [x if isinstance(x, str) else x["texto"]
                                                          for k in ("lineas", "consensos", "debates", "lagunas") for x in doc[k]]
            for t in texts:
                cited.update(int(i) for m in re.finditer(r"\[((?:#\d+[\s,;]*)+)\]", t) for i in re.findall(r"\d+", m.group(1)))
            for k in ("lineas", "consensos", "debates", "lagunas"):
                for x in doc[k]:
                    if isinstance(x, dict):
                        cited.update(x.get("papers", []))
    if not synth:
        return None
    years = {}
    for p in P:
        years.setdefault(p["c"], []).append(p["y"])
    clusters = [{"id": c["id"], "name": c["name"], "name_en": c["name_en"], "n": c["n"], "y0": min(years[c["id"]]),
                 "y1": max(years[c["id"]]), "sample": sample.get(c["id"], 0)} for c in C]
    papers = {str(i): {"t": P[i]["t"], "y": P[i]["y"], "a": P[i]["a"][:2], "v": P[i]["v"], "cit": P[i]["cit"], "c": P[i]["c"]}
              for i in sorted(cited) if 0 <= i < len(P)}
    print(f"síntesis OpenAlex: {len(synth)} temas, {len(papers)} papers citados")
    return {"clusters": clusters, "papers": papers, "analysis": {"clusters": synth, "papers": {}}}


def main():
    with open(os.path.join(ROOT, "data", "map.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    for p in data["papers"]:
        p["abs"] = re.sub(r"^\s*(abstract|summary|resumen)[:.\s]+", "", p["abs"] or "", flags=re.I).strip()
    analysis = load_analysis()
    if analysis:
        data["analysis"] = analysis
    topo = load_topology()
    if topo:
        data["topo"] = topo
    oa_cfg = load_oa(data)
    sys_data = load_sys()
    with open(os.path.join(ROOT, "viewer", "template.html"), encoding="utf-8") as fh:
        template = fh.read()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    sys_blob = json.dumps(sys_data, ensure_ascii=False, separators=(",", ":")) if sys_data else "null"
    oa_synth = load_oa_synth() if oa_cfg else None
    synth_blob = json.dumps(oa_synth, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") if oa_synth else "null"
    evo = load_evo()
    evo_blob = json.dumps(evo, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") if evo else "null"

    def fill(cfg, url=ARTIFACT_URL):
        # sin corpus OpenAlex (versión portable) tampoco van sus síntesis: sus citas abren papers de ese mapa
        return (template.replace("__MAP_DATA__", blob, 1).replace("__ARTIFACT_URL__", url, 1)
                .replace("__OA_CFG__", json.dumps(cfg) if cfg else "null", 1).replace("__SYS_DATA__", sys_blob, 1)
                .replace("__OA_SYNTH__", synth_blob if cfg else "null", 1).replace("__EVO_DATA__", evo_blob, 1))

    # publicable: con el corpus OpenAlex (sus archivos se publican aparte, ver viewer/oa/)
    pub = fill(oa_cfg)
    with open(os.path.join(ROOT, "viewer", "atlas_publish.html"), "w", encoding="utf-8") as fh:
        fh.write(pub)
    # prueba local: la misma página con <head> y los archivos del corpus a su lado (python -m http.server en viewer/test)
    test_dir = os.path.join(ROOT, "viewer", "test")
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "index.html"), "w", encoding="utf-8") as fh:
        collector = ("<script>window.__errs=[];addEventListener('error',e=>__errs.push(e.message+' @'+(e.lineno||'')));"
                     "addEventListener('unhandledrejection',e=>__errs.push('promesa: '+(e.reason&&e.reason.message||e.reason)));</script>\n")
        fh.write(STANDALONE_HEAD.replace("</head>", collector + "</head>") + pub + "\n</body>\n</html>\n")
    oa_dir = os.path.join(ROOT, "viewer", "oa")
    if oa_cfg:
        for name in os.listdir(oa_dir):
            if name.endswith(".json"):
                shutil.copyfile(os.path.join(oa_dir, name), os.path.join(test_dir, name))
    # sitio para GitHub Pages (carpeta site/, se sube tal cual a un repositorio aparte): misma página con <head> y los
    # archivos del corpus al lado; sin URL fija, así que se enlaza a sí misma por la dirección donde esté alojada
    site = os.path.join(ROOT, "site")
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(STANDALONE_HEAD + fill(oa_cfg, url="") + "\n</body>\n</html>\n")
    if oa_cfg:
        for name in os.listdir(oa_dir):
            if name.endswith(".json"):
                shutil.copyfile(os.path.join(oa_dir, name), os.path.join(site, name))
    open(os.path.join(site, ".nojekyll"), "w").close()   # GitHub Pages: servir los archivos tal cual
    readme = os.path.join(site, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(SITE_README)
    # versión portable: Three.js incrustado para que funcione sin conexión; sin corpus OpenAlex (no hay dónde cargarlo)
    html = fill(None)
    for url, name in [("https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js", "three.min.js"),
                      ("https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js",
                       "OrbitControls.js")]:
        with open(os.path.join(ROOT, "viewer", "vendor", name), encoding="utf-8") as fh:
            lib = fh.read()
        tag = f'<script src="{url}"></script>'
        assert tag in html, tag
        html = html.replace(tag, "<script>\n" + lib.replace("</script", "<\\/script") + "\n</script>", 1)
    out = os.path.join(ROOT, "atlas_mecfs.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(STANDALONE_HEAD + html + "\n</body>\n</html>\n")
    print(f"{out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
