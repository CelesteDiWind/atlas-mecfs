"""Paso o18: relevancia de cada trabajo para la EM/SFC. Reglas para los casos claros y lotes para que la IA clasifique
los dudosos -> out/relevance_rules.json, out/relevance_inputs/b<k>.jsonl (un trabajo por línea)
Regla: EM/SFC en el título (formas largas o siglas CFS/ME, comprobadas a mano: siempre significan EM/SFC) -> "central".
A la IA van: los que solo la mencionan en el resumen o en ninguna parte, y los del tema de medicina china aunque la
nombren en el título (ahí "síndrome de fatiga crónica" designa a veces modelos animales o la «subsalud»).
Etiquetas: central · secundaria · modelo_animal · otro_concepto · periferica · incierto (ver o19)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
OUT = os.path.join(OA, "out")
DST = os.path.join(OUT, "relevance_inputs")
os.makedirs(DST, exist_ok=True)
BATCH = 250
ACR = re.compile(r"\bME\b|\bCFS\b|\bSEID\b")   # siglas: solo en mayúsculas ("me" en minúscula es otra cosa)
ME_I = re.compile(r"myalgic encephalo|chronic fatigue syndrome|ME/CFS|CFS/ME|\bCFS\b|\bSEID\b|systemic exertion intolerance", re.I)
SUPPL = re.compile(r"^\s*(MOESM\d+|Additional file|Supplementary (material|file|table|figure|data)|Figure S?\d|Table S?\d|Data[_ ]Sheet|Data from)", re.I)


def snippets(text, k=3, width=220):
    sents = re.split(r"(?<=[.;!?])\s+", re.sub(r"\s+", " ", text or ""))
    hit = [s for s in sents if ME_I.search(s)]
    return [s[:width * 2] for s in hit[:k]]


def main():
    W = json.load(open(os.path.join(OUT, "works_raw.json"), encoding="utf-8"))
    ex = json.load(open(os.path.join(OUT, "works_extra.json"), encoding="utf-8"))
    v1 = json.load(open(os.path.join(OUT, "topic_labels_v1.json")))   # temas de la versión 2010-2025
    rules, todo = {}, []
    for w in W:
        if SUPPL.match(w["title"] or ""):
            continue
        title_hit = bool(ME_I.search(w["title"] or "") or ACR.search(w["title"] or ""))
        tcm = v1.get(w["key"]) == 3
        if title_hit and not tcm:
            rules[w["key"]] = "central"
            continue
        ab = w["abstract"] or ""
        item = dict(key=w["key"], titulo=w["title"], año=w["year"], revista=(ex.get(w["key"]) or {}).get("v") or None,
                    menciones_en_resumen=len(ME_I.findall(ab)), em_sfc_en_titulo=title_hit)
        sn = snippets(ab)
        if sn:
            item["frases_con_mencion"] = sn
        elif ab:
            item["inicio_resumen"] = ab[:450]
        else:
            item["sin_resumen"] = True
        todo.append(item)
    json.dump(rules, open(os.path.join(OUT, "relevance_rules.json"), "w"))
    nb = 0
    for b in range(0, len(todo), BATCH):
        with open(os.path.join(DST, f"b{nb}.jsonl"), "w", encoding="utf-8") as fh:
            for it in todo[b:b + BATCH]:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        nb += 1
    print(f"por regla: {len(rules)} | para la IA: {len(todo)} en {nb} lotes")


if __name__ == "__main__":
    main()
