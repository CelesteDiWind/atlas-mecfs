"""Paso 6: material para el análisis de contenido por tema -> data/analysis/inputs/*.json
Por paper: keywords, abstract y conclusiones. Las conclusiones salen del PDF completo:
    1) la sección "Conclusion(s)" (la última que aparezca), hasta la siguiente sección de cierre
    2) si no hay, la parte "Conclusion(s):" de un abstract estructurado
    3) si no, el final de la discusión (lo que precede a las referencias)
Los temas grandes se parten en lotes de MAX_CHUNK papers (cada lote lo analiza un agente)."""
import json
import os
import re
import unicodedata

import pymupdf as fitz

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "analysis", "inputs")
MAX_CHUNK = 60
fitz.TOOLS.mupdf_display_errors(False)

HEAD = re.compile(r"(?im)^[ \t]*(?:\d+(?:\.\d+)?\.?[ \t]*)?(conclusions?|concluding remarks|summary and conclusions?|conclusiones?)[ \t]*:?[ \t]*$")
END = re.compile(r"(?im)^[ \t]*(?:\d+\.?[ \t]*)?(references|bibliography|referencias|acknowledg|funding|financial support|author contributions|"
                 r"conflicts? of interest|competing interests|declaration|data availability|supplementary|abbreviations|ethics)")
ABS_CONC = re.compile(r"(?i)\b(conclusions?|interpretation|significance)\s*[:.]\s*")


def clean(s, n):
    s = re.sub(r"-\n(?=[a-z])", "", s)  # palabras partidas a final de línea
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]


def full_text(path):
    try:
        doc = fitz.open(path)
        txt = "".join(pg.get_text() for pg in doc)
        doc.close()
    except Exception:
        return ""
    return re.sub(r"[​-‏⁠﻿­]", "", unicodedata.normalize("NFKC", txt))


def conclusions(txt, abstract):
    """(texto, fuente) con fuente en {"pdf", "abstract", "discusion", ""}."""
    if txt:
        heads = list(HEAD.finditer(txt))
        if heads:
            rest = txt[heads[-1].end():heads[-1].end() + 5000]
            stop = END.search(rest)
            body = clean(rest[:stop.start()] if stop else rest, 2200)
            if len(body) > 150:
                return body, "pdf"
    m = None
    for m in ABS_CONC.finditer(abstract or ""):
        pass
    if m and len(abstract) - m.end() > 80:
        return clean(abstract[m.end():], 1200), "abstract"
    if txt:
        ends = list(END.finditer(txt))
        ref = next((e for e in reversed(ends) if e.group(1).lower().startswith(("ref", "bibl"))), None)
        if ref and ref.start() > 3000:
            return clean(txt[max(0, ref.start() - 1800):ref.start()], 1800), "discusion"
    return "", ""


def main():
    with open(os.path.join(DATA, "papers.json"), encoding="utf-8") as fh:
        papers = json.load(fh)
    with open(os.path.join(DATA, "map.json"), encoding="utf-8") as fh:
        m = json.load(fh)
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    by_cluster, stats = {}, {"pdf": 0, "abstract": 0, "discusion": 0, "": 0}
    for i, (p, mp) in enumerate(zip(papers, m["papers"])):
        txt = full_text(os.path.join(DATA, "raw", p["files"][0]))
        conc, src = conclusions(txt, p["abstract"])
        stats[src] += 1
        abstract = p["abstract"] or ""
        if len(abstract) < 200:  # sin abstract: el comienzo del documento como contexto
            abstract = (abstract + " " + clean(p.get("_text", ""), 1500)).strip()
        by_cluster.setdefault(mp["c"], []).append({
            "id": i, "titulo": mp["t"], "año": mp["y"], "keywords": p["keywords"][:8],
            "abstract": clean(abstract, 2600), "abstract_es_texto_inicial": not p["abstract"],
            "conclusiones": conc, "fuente_conclusiones": src,
        })
        if i % 100 == 0:
            print(f"{i}/{len(papers)}", flush=True)

    chunks = []
    for c in sorted(by_cluster):
        items = by_cluster[c]
        n = -(-len(items) // MAX_CHUNK)
        size = -(-len(items) // n)
        for k in range(n):
            name = f"c{c}_{k}"
            part = items[k * size:(k + 1) * size]
            with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"cluster": c, "tema": m["clusters"][c]["name"], "papers": part}, fh, ensure_ascii=False, indent=1)
            chunks.append((name, len(part)))
    print("conclusiones por fuente:", stats)
    print("lotes:", chunks)


if __name__ == "__main__":
    main()
