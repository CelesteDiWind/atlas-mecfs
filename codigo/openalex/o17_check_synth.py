"""Paso o17: valida la síntesis de un tema (out/synth/t<c>.json y t<c>_en.json) frente a su entrada.
Uso: python o17_check_synth.py <c>   (o sin argumento: todos los temas). Imprime OK o la lista de problemas."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
REF = re.compile(r"\[((?:#\d+[\s,;]*)+)\]")
LIMITS = {"sintesis": (3, 5), "lineas": (4, 6), "consensos": (3, 5), "debates": (3, 5), "lagunas": (3, 5)}


def cites(text):
    return [int(x) for m in REF.finditer(text or "") for x in re.findall(r"\d+", m.group(1))]


def items(S, key):
    return [x if isinstance(x, dict) else {"texto": x} for x in S.get(key, [])]


def check(c):
    errs = []
    inp = json.load(open(os.path.join(OUT, "synth_inputs", f"t{c}.json"), encoding="utf-8"))
    ok_ids = {p["id"] for p in inp["papers"]}
    docs = {}
    for lang, name in (("es", f"t{c}.json"), ("en", f"t{c}_en.json")):
        path = os.path.join(OUT, "synth", name)
        if not os.path.exists(path):
            errs.append(f"falta {name}"); continue
        try:
            S = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            errs.append(f"{name}: JSON inválido ({e})"); continue
        docs[lang] = S
        if S.get("cluster") != c:
            errs.append(f"{name}: cluster debe ser {c}")
        if not isinstance(S.get("resumen"), str) or len(S["resumen"]) < 80:
            errs.append(f"{name}: resumen ausente o demasiado corto")
        for key, (lo, hi) in LIMITS.items():
            L = S.get(key)
            if not isinstance(L, list) or not lo <= len(L) <= hi:
                errs.append(f"{name}: {key} debe tener entre {lo} y {hi} elementos (tiene {len(L) if isinstance(L, list) else 'ninguno'})")
        texts = [S.get("resumen", "")] + list(S.get("sintesis", []))
        for key in ("lineas", "consensos", "debates", "lagunas"):
            for it in items(S, key):
                texts.append(it.get("texto", ""))
                if key == "lineas" and not it.get("titulo"):
                    errs.append(f"{name}: una línea sin título")
                bad = [i for i in it.get("papers", []) if i not in ok_ids]
                if bad:
                    errs.append(f"{name}: {key} cita en 'papers' ids que no están en la muestra: {bad}")
                if not cites(it.get("texto", "")):
                    errs.append(f"{name}: un elemento de {key} no tiene ninguna cita [#id]: «{it.get('texto', '')[:60]}…»")
        for t in texts:
            bad = [i for i in cites(t) if i not in ok_ids]
            if bad:
                errs.append(f"{name}: cita ids que no están en la muestra: {bad} en «{t[:60]}…»")
        if "—" in " ".join(texts):
            errs.append(f"{name}: contiene rayas (—); reescribe esas frases sin incisos")
    if len(docs) == 2:
        es, en = docs["es"], docs["en"]
        for key in ("sintesis", "lineas", "consensos", "debates", "lagunas"):
            a, b = es.get(key, []), en.get(key, [])
            if len(a) != len(b):
                errs.append(f"{key}: el inglés tiene {len(b)} elementos y el español {len(a)}")
                continue
            for k, (x, y) in enumerate(zip(a, b)):
                tx = x if isinstance(x, str) else x.get("texto", ""); ty = y if isinstance(y, str) else y.get("texto", "")
                if sorted(set(cites(tx))) != sorted(set(cites(ty))):
                    errs.append(f"{key}[{k}]: las citas del inglés no coinciden con las del español")
                if isinstance(x, dict) and isinstance(y, dict) and sorted(x.get("papers", [])) != sorted(y.get("papers", [])):
                    errs.append(f"{key}[{k}]: 'papers' distinto entre español e inglés")
    return errs


if __name__ == "__main__":
    topics = [int(sys.argv[1])] if len(sys.argv) > 1 else range(8)
    for c in topics:
        e = check(c)
        print(f"tema {c}: " + ("OK" if not e else f"{len(e)} problemas"))
        for x in e:
            print("  -", x)
