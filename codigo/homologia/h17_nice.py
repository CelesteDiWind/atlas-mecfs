"""Paso 17: ¿por qué los papers de 2020-21 citan más el canon de criterios diagnósticos (h16d)?
Hipótesis NICE (borrador 11-11-2020, guía final 29-10-2021) y sus predicciones, fijadas antes de mirar:
    P1 momento: el aumento empieza después del borrador (con retraso de publicación); un 2020 ya alto la contradice
    P2 compañía: los papers que citan el canon citan también NICE o IOM 2015 más que en otros periodos
    P3 contenido: su texto menciona NICE, IOM/SEID o la terapia de ejercicio gradual / TCC más que en otros periodos
Alternativas:
    A1 composición: más estudios con pacientes que citan los criterios en Métodos para seleccionar la muestra
    A2 grupo: un equipo muy productivo que cita los tres criterios de forma rutinaria
Canon = Fukuda 1994, Consenso Canadiense 2003, ICC 2011 (las 3 referencias más citadas del corpus).
Texto: data/extracted.json (los primeros ~12.000 caracteres de cada PDF: resumen, introducción y métodos).
Salidas: out/h17_papers.csv, out/h17_summary.json"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.sparse import load_npz

from common import ROOT, SEED, out

CANON = {"fukuda": "W2098287790", "icc": "W2155596333", "ccc": "W2008087869"}
REFSETS = {"iom": ["W4230833759", "W1994696630"], "nice_ref": ["W2048862903", "W3111546247"],
           "pace": ["W2136415723", "W2134945464", "W2144067520", "W2151761397", "W2155523469"],
           "oxford": ["W1538919177"], "reeves": ["W1517511965", "W2171701188", "W1596126302"]}
DRAFT = 2020 + 315 / 366   # 11-11-2020
TXT = {"nice_txt": r"\bNICE\b", "iom_txt": r"Institute of Medicine|\bIOM\b|National Academ|\bSEID\b|systemic exertion intolerance",
       "get_txt": r"graded exercise|\bGET\b|cognitive behaviou?ral therap|\bCBT\b",
       "covid_txt": r"COVID|SARS-CoV-2|long COVID|post-acute sequelae"}
CRIT = {"fukuda": r"Fukuda|CDC[- ]1994|1994 CDC|1994 (case|research) definition",
        "ccc": r"Canadian Consensus|Carruthers|\bCCC\b|Canadian (criteria|case definition)",
        "icc": r"International Consensus|\bICC\b"}
SEL = r"\b(met|meet|meeting|fulfil\w*|satisf\w*|diagnosed|according to|inclusion|recruited|eligib\w*|classified|selected)\b"
PART = r"patients|participants|subjects|cases|individuals|cohort|people"


def period(t):
    if t < 2020:
        return "a_2018-19"
    if t < DRAFT:
        return "b_2020_pre"
    if t < 2022:
        return "c_post_2021"
    return "d_2022-23"


def main():
    papers = json.load(open(os.path.join(ROOT, "data", "papers.json"), encoding="utf-8"))
    ext = {x["file"]: x["text"] or "" for x in json.load(open(os.path.join(ROOT, "data", "extracted.json"), encoding="utf-8"))}
    an = {}
    for f in glob.glob(os.path.join(ROOT, "data", "analysis", "out", "*.json")):
        for x in json.load(open(f, encoding="utf-8"))["analisis"]:
            an[int(x["id"])] = x["tipo"]
    ref_ids = json.load(open(out("ref_ids.json")))
    col = {r: j for j, r in enumerate(ref_ids)}
    R = load_npz(out("bip_refs.npz")).tocsr()
    tv = np.load(out("cit_dates.npz"))["paper"]
    k = np.asarray(R.sum(1)).ravel()

    rows = []
    for i, p in enumerate(papers):
        cols = set(R.indices[R.indptr[i]:R.indptr[i + 1]].tolist())
        r = dict(i=i, t=tv[i], year=int(np.floor(tv[i])) if np.isfinite(tv[i]) else -1, n_refs=int(k[i]), tipo=an.get(i, "Otro"),
                 authors=json.dumps(p["authors"] or [], ensure_ascii=False))
        for nm, w in CANON.items():
            r[nm] = int(col[w] in cols) if w in col else 0
        r["canon_n"] = r["fukuda"] + r["icc"] + r["ccc"]
        for nm, ws in REFSETS.items():
            r[nm] = int(any(col.get(w, -1) in cols for w in ws))
        txt = " ".join(ext.get(f, "") for f in p["files"])
        txt = re.sub(r"-\n(?=\w)", "", txt).replace("\n", " ")
        for nm, rx in TXT.items():
            r[nm] = int(bool(re.search(rx, txt)))
        sents = re.split(r"(?<=[.;])\s+", txt)
        r["crit_txt_n"] = sum(bool(re.search(rx, txt)) for rx in CRIT.values())
        anyc = "|".join(CRIT.values())
        r["selection_use"] = int(any(re.search(anyc, s) and re.search(SEL, s, re.I) and re.search(PART, s, re.I) for s in sents))
        r["text_len"] = len(txt)
        rows.append(r)
    df = pd.DataFrame(rows)
    df = df[(df.n_refs > 0) & (df.t >= 2018) & (df.t < 2024)].copy()
    df["period"] = df.t.map(period)
    df["canon_any"] = (df.canon_n > 0).astype(int)
    df["canon_all3"] = (df.canon_n == 3).astype(int)
    df["empirico"] = df.tipo.isin(["Estudio original", "Ensayo clínico"]).astype(int)
    summ = {}

    # ---------- P1 momento: por semestre y por periodo
    df["sem"] = df.year.astype(str) + np.where(df.t - df.year < 0.5, "a", "b")
    by_sem = df.groupby("sem").agg(n=("i", "size"), canon_n=("canon_n", "mean"), canon_any=("canon_any", "mean"),
                                   all3=("canon_all3", "mean"), selection=("selection_use", "mean")).round(3)
    print(by_sem.to_string(), flush=True)
    summ["semestres"] = by_sem.reset_index().to_dict("records")
    cols_p = ["canon_n", "canon_any", "canon_all3", "iom", "nice_ref", "pace", "nice_txt", "iom_txt", "get_txt",
              "selection_use", "empirico", "covid_txt"]
    by_p = df.groupby("period")[cols_p].mean().round(3)
    by_p["n"] = df.groupby("period").size()
    print(by_p.T.to_string(), flush=True)
    summ["periodos"] = by_p.reset_index().to_dict("records")

    # ---------- P2/P3: entre los que citan el canon, ¿citan o mencionan NICE/IOM/GET más en 2020-21?
    sub = df[df.canon_any == 1]
    p23 = sub.groupby("period")[["iom", "nice_ref", "pace", "nice_txt", "iom_txt", "get_txt"]].mean().round(3)
    print("entre los que citan el canon:\n", p23.T.to_string(), flush=True)
    summ["canon_citers"] = p23.reset_index().to_dict("records")

    # ---------- A2: autores productivos
    df["alist"] = df.authors.map(json.loads)
    cnt = Counter(a for L in df.alist for a in set(L))
    prolific = [a for a, c in cnt.most_common() if c >= 6]
    aut = []
    for a in prolific:
        m = df.alist.map(lambda L: a in L)
        aut.append(dict(author=a, n=int(m.sum()), n_2020_21=int((m & df.period.isin(["b_2020_pre", "c_post_2021"])).sum()),
                        canon_n=round(float(df.canon_n[m].mean()), 2), all3=round(float(df.canon_all3[m].mean()), 2)))
    aut = pd.DataFrame(aut)
    print(aut.to_string(), flush=True)
    summ["autores"] = aut.to_dict("records")
    # grupo = componente de coautoría de los autores productivos con canon alto
    hi = set(aut.author[(aut.all3 >= 0.5)])
    df["grupo"] = df.alist.map(lambda L: int(bool(hi & set(L))))
    gp = df.groupby(["period", "grupo"]).agg(n=("i", "size"), canon_n=("canon_n", "mean"), all3=("canon_all3", "mean")).round(3)
    print(gp.to_string(), flush=True)
    summ["grupo_autores"] = sorted(hi)
    summ["grupo_periodo"] = gp.reset_index().to_dict("records")

    # ---------- modelos: efecto del periodo antes y después de ajustar
    df["p2021"] = df.period.isin(["b_2020_pre", "c_post_2021"]).astype(int)
    df["tipo_g"] = df.tipo.where(df.tipo.isin(["Estudio original", "Revisión narrativa", "Revisión sistemática / metaanálisis"]), "Otro")
    df["log_refs"] = np.log(df.n_refs)
    models = {"crudo": "canon_n ~ C(period)",
              "ajustado": "canon_n ~ C(period) + selection_use + C(tipo_g) + log_refs",
              "ajustado_grupo": "canon_n ~ C(period) + selection_use + C(tipo_g) + log_refs + grupo"}
    summ["modelos"] = {}
    for nm, f in models.items():
        m = smf.poisson(f, df).fit(disp=0)
        res = {t: dict(rr=round(float(np.exp(m.params[t])), 3), ci=[round(float(x), 3) for x in np.exp(m.conf_int().loc[t])],
                       p=round(float(m.pvalues[t]), 4)) for t in m.params.index if t != "Intercept"}
        summ["modelos"][nm] = res
        print(nm, {t: (v["rr"], v["ci"], v["p"]) for t, v in res.items()}, flush=True)

    # ---------- ¿queda el pico de cohesión sin el grupo o sin los papers que usan criterios para seleccionar?
    Rf = R.astype(np.float64)
    rng = np.random.default_rng(SEED + 17)

    def cohesion(idx, N=20, reps=300):
        if len(idx) < 2 * N:
            return None
        v = []
        for _ in range(reps):
            p = rng.permutation(idx)
            a, b = np.asarray(Rf[p[:N]].sum(0)).ravel(), np.asarray(Rf[p[N:2 * N]].sum(0)).ravel()
            v.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        return round(float(np.mean(v)), 3)
    coh = {}
    for y in range(2018, 2024):
        d = df[df.year == y]
        coh[y] = dict(todos=cohesion(d.i.values), sin_grupo=cohesion(d.i[d.grupo == 0].values),
                      sin_seleccion=cohesion(d.i[d.selection_use == 0].values), solo_seleccion=cohesion(d.i[d.selection_use == 1].values))
    print("cohesión:", coh, flush=True)
    summ["cohesion"] = coh

    # ---------- A2 ampliada: los equipos más productivos (autores con >= 6 papers y sus coautores habituales)
    teams = {"NL_Visser": ["Frans C. Visser", "C. M. C. van Campen", "Peter C. Rowe"],
             "AU_Griffith": ["Sonya Marshall‐Gradisnik", "Donald Staines", "Natalie Eaton-Fitch", "Hélène Cabanas",
                             "Leighton R Barnden", "Kiran Thapaliya"],
             "EUROMENE": ["Derek Pheby", "Modra Murovska", "Diāna Arāja", "Zaiga Nora-Krūkle", "Luís Nacul", "Eliana Lacerda",
                          "Caroline C. Kingdon"],
             "Berlin": ["Carmen Scheibenbogen", "Helma Freitag"], "DePaul": ["Leonard A. Jason", "Joseph H Cotler"],
             "ES": ["José Alegre", "Jesús Castro‐Marrero"]}
    allt = {a for v in teams.values() for a in v}
    for g, a in teams.items():
        df[g] = df.alist.map(lambda L, a=set(a): int(bool(a & set(L))))
    df["equipos"] = df.alist.map(lambda L: int(bool(allt & set(L))))
    summ["equipos_por_año"] = df.groupby("year")[list(teams) + ["equipos"]].sum().to_dict("index")
    summ["equipos_fraccion"] = df.groupby("year").equipos.mean().round(3).to_dict()
    summ["canon_por_equipo"] = df.groupby(["period", "equipos"]).canon_n.mean().round(2).unstack().to_dict("index")

    def cohesion_ci(idx, N=15, reps=400):
        if len(idx) < 2 * N:
            return None
        v = []
        for _ in range(reps):
            p = rng.permutation(idx)
            a, b = np.asarray(Rf[p[:N]].sum(0)).ravel(), np.asarray(Rf[p[N:2 * N]].sum(0)).ravel()
            v.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        return [round(float(np.mean(v)), 3), round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    summ["cohesion_equipos"] = {y: dict(todos=cohesion_ci(df.i[df.year == y].values),
                                        sin_equipos=cohesion_ci(df.i[(df.year == y) & (df.equipos == 0)].values))
                                for y in range(2018, 2024)}
    print("equipos:", summ["equipos_fraccion"], "\ncohesión con/sin equipos:", summ["cohesion_equipos"], flush=True)
    df.drop(columns=["alist"]).to_csv(out("h17_papers.csv"), index=False)
    json.dump(summ, open(out("h17_summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
