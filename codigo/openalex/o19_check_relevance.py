"""Paso o19: valida la clasificación de relevancia de un lote (out/relevance/b<k>.jsonl) frente a su entrada, o junta
todos los lotes con las reglas (sin argumento) -> out/relevance.json {key: {rel, motivo}}.
Uso: python o19_check_relevance.py <k>   |   python o19_check_relevance.py"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
LABELS = {"central", "secundaria", "modelo_animal", "otro_concepto", "periferica", "incierto"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def check(k):
    inp = {x["key"] for x in load_jsonl(os.path.join(OUT, "relevance_inputs", f"b{k}.jsonl"))}
    path = os.path.join(OUT, "relevance", f"b{k}.jsonl")
    if not os.path.exists(path):
        return [f"falta relevance/b{k}.jsonl"], None
    errs, got = [], {}
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            x = json.loads(line)
        except Exception as e:
            errs.append(f"línea {n}: JSON inválido ({e})"); continue
        if x.get("key") not in inp:
            errs.append(f"línea {n}: key {x.get('key')} no está en la entrada")
        if x.get("rel") not in LABELS:
            errs.append(f"línea {n}: etiqueta «{x.get('rel')}» no válida")
        if not x.get("motivo") or len(x["motivo"].split()) > 20:
            errs.append(f"línea {n}: falta el motivo o pasa de 20 palabras")
        got[x.get("key")] = x
    miss = inp - set(got)
    if miss:
        errs.append(f"faltan {len(miss)} trabajos: {sorted(miss)[:5]}…")
    return errs, got


if __name__ == "__main__":
    if len(sys.argv) > 1:
        e, got = check(int(sys.argv[1]))
        print(f"lote {sys.argv[1]}: " + ("OK" if not e else f"{len(e)} problemas"))
        for x in e[:30]:
            print("  -", x)
        if got and not e:
            print("  etiquetas:", dict(Counter(x["rel"] for x in got.values())))
    else:
        rel = {k: {"rel": v, "motivo": "EM/SFC en el título"} for k, v in json.load(open(os.path.join(OUT, "relevance_rules.json"))).items()}
        k = 0
        while os.path.exists(os.path.join(OUT, "relevance_inputs", f"b{k}.jsonl")):
            e, got = check(k)
            if e:
                raise SystemExit(f"lote {k} con problemas: {e[:3]}")
            rel.update({key: {"rel": x["rel"], "motivo": x["motivo"]} for key, x in got.items()})
            k += 1
        json.dump(rel, open(os.path.join(OUT, "relevance.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print(len(rel), "trabajos:", dict(Counter(x["rel"] for x in rel.values())))
