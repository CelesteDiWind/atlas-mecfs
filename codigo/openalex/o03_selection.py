"""Paso o03: ¿cómo se seleccionó el corpus manual? Comparación con el corpus sistemático de OpenAlex.
Para cada paper sistemático: ¿está en el corpus manual? Según año, equipo productivo (h17), citas, tipo y si la EM/SFC
está en el título. Regresión logística con todo a la vez. Además: por qué faltan papers manuales en la consulta.
Salidas: openalex/out/o03_summary.json"""
import json
import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
TEAMS = {"NL_Visser": ["Frans C. Visser", "C. M. C. van Campen", "Peter C. Rowe"],
         "AU_Griffith": ["Sonya Marshall-Gradisnik", "Donald Staines", "Natalie Eaton-Fitch", "Hélène Cabanas",
                         "Leighton R. Barnden", "Kiran Thapaliya"],
         "EUROMENE": ["Derek Pheby", "Modra Murovska", "Diāna Arāja", "Zaiga Nora-Krūkle", "Luís Nacul", "Eliana Lacerda",
                      "Caroline C. Kingdon"],
         "Berlin": ["Carmen Scheibenbogen", "Helma Freitag"], "DePaul": ["Leonard A. Jason", "Joseph H. Cotler"],
         "ES": ["José Alegre", "Jesús Castro-Marrero"]}
CORE = re.compile(r"myalgic encephalo|chronic fatigue syndrome|ME/CFS|CFS/ME|\bCFS\b|\bME\b")  # siglas en mayúsculas


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower().replace("‐", "-"))


def main():
    W = pd.DataFrame(json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8")))
    W["core"] = W.title.map(lambda t: bool(CORE.search(t or "")))
    # equipos por apellido + inicial del nombre (robusto a guiones y puntos)
    keyset = {g: {norm(a.split()[0][0] + a.split()[-1]) for a in v} for g, v in TEAMS.items()}
    akeys = W.authors.map(lambda L: {norm((a.split()[0][0] if a and a.split() else "") + (a.split()[-1] if a and a.split() else "")) for a in L})
    for g, ks in keyset.items():
        W[g] = akeys.map(lambda s, ks=ks: int(bool(s & ks)))
    W["equipo"] = W[list(TEAMS)].max(1)
    W["review"] = (W.type == "review").astype(int)
    W["log_cit"] = np.log1p(W.cited_by.fillna(0))
    W["manual"] = W.manual.astype(int)
    S = {}
    yr = W.groupby("year").agg(n=("key", "size"), manual=("manual", "sum"), equipo=("equipo", "mean"))
    yr["cobertura"] = (yr.manual / yr.n).round(3)
    yr["equipo_en_manual"] = W[W.manual == 1].groupby("year").equipo.mean().round(3)
    yr["equipo"] = yr.equipo.round(3)
    print(yr.to_string(), flush=True)
    S["por_año"] = yr.reset_index().to_dict("records")
    w = W[(W.year >= 2017) & (W.year <= 2024)].copy()
    cov = {g: dict(n=int(w[g].sum()), cobertura=round(float(w.manual[w[g] == 1].mean()), 3)) for g in list(TEAMS) + ["equipo"]}
    cov["resto"] = dict(n=int((w.equipo == 0).sum()), cobertura=round(float(w.manual[w.equipo == 0].mean()), 3))
    cov["core"] = round(float(w.manual[w.core].mean()), 3)
    cov["no_core"] = round(float(w.manual[~w.core].mean()), 3)
    print("cobertura 2017-24:", cov, flush=True)
    S["cobertura_2017_24"] = cov
    m = smf.logit("manual ~ equipo + log_cit + review + core + C(year)", w.assign(core=w.core.astype(int))).fit(disp=0)
    S["logit"] = {t: dict(OR=round(float(np.exp(m.params[t])), 3), ci=[round(float(x), 3) for x in np.exp(m.conf_int().loc[t])],
                          p=round(float(m.pvalues[t]), 4)) for t in ["equipo", "log_cit", "review", "core"]}
    print("logit:", S["logit"], flush=True)
    # equipo x periodo: ¿la sobrerrepresentación de 2020-21 viene de la selección?
    w["per"] = pd.cut(w.year, [2016, 2019, 2021, 2024], labels=["2017-19", "2020-21", "2022-24"])
    tab = w.groupby(["per", "equipo"], observed=True).manual.agg(["size", "mean"]).round(3)
    print(tab.to_string(), flush=True)
    S["cobertura_equipo_periodo"] = {f"{a}|{b}": dict(n=int(r["size"]), cobertura=float(r["mean"])) for (a, b), r in tab.iterrows()}
    S["equipo_fraccion_sistematico"] = W.groupby("year").equipo.mean().round(3).to_dict()
    # papers manuales que la consulta no recoge
    P = json.load(open(os.path.join(ROOT, "data", "papers.json"), encoding="utf-8"))
    keys = set(W.key)
    miss = [p for p in P if p["matched_by"] != "none" and p["key"] not in keys]
    S["manual_fuera"] = dict(n=len(miss), por_año=pd.Series([p["year"] for p in miss]).value_counts().sort_index().to_dict(),
                             por_tipo=pd.Series([p["type"] for p in miss]).value_counts().to_dict(),
                             ejemplos=[p["title"][:100] for p in miss[:12]])
    print("manuales fuera de la consulta:", S["manual_fuera"], flush=True)
    json.dump(S, open(os.path.join(OUT, "o03_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
