"""Paso o16: entradas para las síntesis por tema del corpus OpenAlex -> out/synth_inputs/t<c>.json
Por tema, una muestra de ~80 papers con resumen (los más centrales, los más citados, los recientes y los más
representativos de cada subgrupo interno) y el contexto cuantitativo: tamaño, años, revistas, palabras clave, diálogo
con otros temas (o14) y tendencias. Las síntesis las escriben subagentes a partir de estos archivos."""
import json
import os
import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
DST = os.path.join(OUT, "synth_inputs")
os.makedirs(DST, exist_ok=True)
SEED = 20261021
N_SAMPLE = 80
SUPPL = re.compile(r"^(MOESM\d+|Additional file|Supplementary|Figure \d|Table \d|Data from)", re.I)


def main():
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    ex = json.load(open(os.path.join(OUT, "works_extra.json"), encoding="utf-8"))
    m = json.load(open(os.path.join(ROOT, "viewer", "oa", "oa_map.json"), encoding="utf-8"))
    P, C = m["papers"], m["clusters"]
    D = json.load(open(os.path.join(OUT, "o14_dialogue.json"), encoding="utf-8"))
    E = normalize(np.load(os.path.join(OUT, "E_specter2.npy")).astype(np.float64))
    lab = np.array([p["c"] for p in P]); cit = np.array([p["cit"] for p in P]); yr = np.array([p["y"] for p in P])
    rng = np.random.default_rng(SEED)
    names = [c["name"] for c in C]
    for c in range(len(C)):
        idx = np.where(lab == c)[0]
        # solo trabajos sobre EM/SFC: centrales, secundarios y modelos animales (sin menciones periféricas, otro concepto,
        # inciertos ni retractados)
        cand = np.array([i for i in idx if len(W[i]["abstract"]) >= 300 and not SUPPL.match(P[i]["t"])
                         and P[i].get("r", 0) in (0, 1, 3) and not P[i].get("rt")])
        cen = normalize(E[idx].mean(0)[None])[0]
        central = cand[np.argsort(-(E[cand] @ cen))]
        chosen, why = [], {}

        def take(ids, k, reason):
            n = 0
            for i in ids:
                if n >= k or len(chosen) >= N_SAMPLE:
                    break
                if i in why:
                    why[i].append(reason); continue
                chosen.append(int(i)); why[i] = [reason]; n += 1
        take(central, 20, "central")
        take(cand[np.argsort(-cit[cand])], 15, "muy citado")
        rec = cand[yr[cand] >= 2024]
        take(rec[np.argsort(-(E[rec] @ cen))], 12, "reciente")
        # cobertura de los subgrupos internos: el más representativo de cada uno, por turnos
        k = min(8, max(2, len(cand) // 25))
        sub = KMeans(k, n_init=10, random_state=SEED).fit(E[cand])
        orders = []
        for s in range(k):
            ii = cand[sub.labels_ == s]
            cs = normalize(sub.cluster_centers_[s][None])[0]
            orders.append(list(ii[np.argsort(-(E[ii] @ cs))]))
        while len(chosen) < min(N_SAMPLE, len(cand)) and any(orders):
            for s in range(k):
                while orders[s] and orders[s][0] in why:
                    orders[s].pop(0)
                if orders[s] and len(chosen) < N_SAMPLE:
                    i = orders[s].pop(0); chosen.append(int(i)); why[i] = [f"subgrupo {s + 1}"]
        papers = []
        for i in sorted(chosen, key=lambda i: (yr[i], -cit[i])):
            w = W[i]; au = w["authors"] or []
            papers.append(dict(id=int(i), titulo=P[i]["t"], año=int(yr[i]), revista=P[i]["v"] or None,
                               autores=(au[0] + (" et al." if len(au) > 1 else "")) if au else None, citas=int(cit[i]),
                               en_seleccion_manual="mi" in P[i], motivo=why[i],
                               resumen=re.sub(r"\s+", " ", w["abstract"]).strip()[:1400]))
        ven = Counter(P[i]["v"] for i in idx if P[i]["v"]).most_common(10)
        kws = Counter(k_ for i in idx for k_ in (P[i]["kw"] or [])).most_common(15)
        oe = D["cp_ext"]["oe"][c]; q = D["cp_ext"]["q"][c]
        dialog = sorted([dict(tema=names[j], oe_acoplamiento=round(oe[j], 2), significativo=q[j] < 0.05,
                              oe_citas_que_hace=round(D["di_ext"]["oe"][c][j], 2), oe_citas_que_recibe=round(D["di_ext"]["oe"][j][c], 2))
                         for j in range(len(C)) if j != c], key=lambda r: -r["oe_acoplamiento"])
        trends = [dict(con=names[t["b"] if t["a"] == c else t["a"]], red="acoplamiento" if t["net"] == "cp" else "citas directas",
                       desde=f'{t["first"]} ({t["first_w"]})', hasta=f'{t["last"]} ({t["last_w"]})')
                  for t in D["trends"] if c in (t["a"], t["b"])][:4]
        ctx = dict(
            tema=names[c], tema_en=C[c]["name_en"], id=c, n_trabajos=int(len(idx)), n_con_resumen=int(len(cand)),
            n_en_seleccion_manual=int(sum("mi" in P[i] for i in idx)), n_muestra=len(papers),
            relevancia=dict(Counter({0: "central", 1: "secundaria", 2: "periferica", 3: "modelo_animal", 4: "otro_concepto", 5: "incierto"}[P[i].get("r", 0)] for i in idx)),
            nota_relevancia="Etiquetas de relevancia para la EM/SFC de todos los trabajos del tema; la muestra solo incluye centrales, secundarios y modelos animales.",
            años={int(y): int(n) for y, n in sorted(Counter(yr[idx]).items())},
            revistas_principales=[dict(revista=v, n=n) for v, n in ven],
            palabras_clave_principales=[dict(kw=k_, n=n) for k_, n in kws],
            terminos_distintivos=C[c]["terms"],
            apertura=dict(indice_EI_acoplamiento=D["cp"]["ei"][c], esperado_por_azar=D["cp"]["ei_null"][c],
                          nota="−1 = solo vínculos dentro del tema, +1 = solo con otros temas"),
            balance_citas=dict(observado=D["flow"]["balance"][c], esperado=D["flow"]["balance_null"][c],
                               nota="log2(citas recibidas de otros temas / citas hechas a otros temas); por encima de lo esperado = más citado de lo que cita"),
            dialogo_con_otros_temas=dialog, tendencias=trends,
            nota_dialogo="O/E calculado solo sobre los vínculos que salen de cada tema: 1 = lo que daría el azar; >1 dialogan más; <1 menos.",
        )
        json.dump(dict(contexto=ctx, papers=papers), open(os.path.join(DST, f"t{c}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(c, names[c], "n", len(idx), "candidatos", len(cand), "muestra", len(papers),
              "tokens aprox", int(sum(len(p["resumen"]) + len(p["titulo"]) for p in papers) / 4))


if __name__ == "__main__":
    main()
