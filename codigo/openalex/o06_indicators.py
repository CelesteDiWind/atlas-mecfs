"""Paso o06: indicadores por ventana de 3 años y por año en el corpus sistemático (sustituye a h16/h16c/h16d/h17).
Temas: cada trabajo se asigna al centroide SPECTER2 más cercano de los 8 temas del corpus manual (comparable con el atlas).
Por ventana (red de acoplamiento validada):
    - mezcla entre temas: fracción de vínculos entre temas distintos / la esperada si los temas se permutan (100 réplicas)
    - alineamiento contenido-citas: Spearman entre similitud SPECTER2 y exceso estandarizado de referencias compartidas
    - COVID persistente: papers, y mezcla de sus vínculos con los demás frente a permutación
    - componente gigante, densidad, continuidad con la ventana anterior (Jaccard de vínculos entre papers compartidos)
    Control de tamaño: las mismas métricas en submuestras de N_SUB papers activos (20 réplicas).
Por año: cohesión de la literatura citada (coseno de perfiles de referencias entre dos mitades de N papers), citas al
canon (Fukuda 1994, CCC 2003, ICC 2011), fracción de COVID, de trabajos con EM/SFC en el título, y de los 6 equipos.
Salidas: out/o06_windows.csv, out/o06_years.csv, out/topics.npy"""
import json
import os
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.preprocessing import normalize

import citnet_big as cb
from o04_network import OUT, WINDOWS, build
from o03_selection import TEAMS, norm

import citnet as cn  # noqa: E402 (ruta añadida por citnet_big)

HERE = os.path.dirname(os.path.abspath(__file__))
HOM = os.path.dirname(os.path.dirname(HERE))
SEED = 20261019
N_SUB = 700
REPS_SUB = 20
CANON = {"W2098287790", "W2155596333", "W2008087869"}
COVID = re.compile(r"covid|sars-cov-2|post-acute sequelae|pasc\b|long[- ]haul", re.I)
CORE = re.compile(r"myalgic encephalo|chronic fatigue syndrome|ME/CFS|CFS/ME|\bCFS\b|\bME\b")


def topics(E):
    J = json.load(open(os.path.join(HOM, "out", "papers.json"), encoding="utf-8"))
    Em = normalize(np.load(os.path.join(HOM, "out", "E_specter2.npy")).astype(np.float64))
    lab = np.array([int(p["c"]) for p in J["papers"]])
    cent = normalize(np.vstack([Em[lab == c].mean(0) for c in sorted(set(lab))]))
    return np.argmax(normalize(E) @ cent.T, axis=1), J["clusters"]


def window_metrics(R, E, sel, tp, covid, rng, reps=100):
    Bs = R[sel]
    Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
    k = np.asarray(Bs.sum(1)).ravel()
    act = k > 0
    p, hc, _ = cn.fit_bicm(Bs)
    a, b, s, pv = cb.validated(Bs, p, hc, act)
    r = dict(papers=int(len(sel)), active=int(act.sum()), validated=int(len(a)))
    if len(a) < 20:
        return r, (a, b)
    ga, gb = sel[a], sel[b]
    perm = []
    for _ in range(reps):
        t2 = tp.copy()
        t2[sel] = rng.permutation(tp[sel])
        perm.append((t2[ga] != t2[gb]).mean())
    r["mixing_ratio"] = float((tp[ga] != tp[gb]).mean() / np.mean(perm))
    # alineamiento contenido-citas
    ia = np.where(act)[0]
    S = (Bs[ia] @ Bs[ia].T).toarray().astype(np.float64)
    mu = (p[ia] * hc) @ p[ia].T
    iu = np.triu_indices(len(ia), 1)
    z = (S[iu] - mu[iu]) / np.sqrt(mu[iu] + 1e-12)
    sim = (E[sel[ia]] @ E[sel[ia]].T)[iu]
    r["mantel_rho"] = float(spearmanr(sim, z)[0])
    cv = covid[sel]
    r["covid_papers"] = int(cv.sum())
    if cv.sum() >= 8:
        obs = (covid[ga] != covid[gb]).mean()
        pc = []
        for _ in range(reps):
            c2 = covid.copy()
            c2[sel] = rng.permutation(covid[sel])
            pc.append((c2[ga] != c2[gb]).mean())
        r["covid_mix_ratio"] = float(obs / max(1e-9, np.mean(pc)))
    G = nx.Graph()
    G.add_edges_from(zip(a.tolist(), b.tolist()))
    r["giant_frac"] = float(max(len(c) for c in nx.connected_components(G)) / max(1, act.sum()))
    r["density"] = float(2 * len(a) / max(1, act.sum() * (act.sum() - 1)))
    return r, (ga, gb)


def main():
    W, R, t = build()
    E = normalize(np.load(os.path.join(OUT, "E_specter2.npy")).astype(np.float64))
    tp, clusters = topics(E)
    np.save(os.path.join(OUT, "topics.npy"), tp)
    yr = np.floor(t)
    k_all = np.asarray(R.sum(1)).ravel()
    txt = [(w["title"] or "") + " " + (w["abstract"] or "") for w in W]
    covid = np.array([bool(COVID.search(x)) for x in txt])
    core = np.array([bool(CORE.search(w["title"] or "")) for w in W])
    rng = np.random.default_rng(SEED)

    # ---------- ventanas
    rows, edges = [], {}
    for y0, y1 in WINDOWS:
        sel = np.where((yr >= y0) & (yr <= y1))[0]
        r, e = window_metrics(R, E, sel, tp, covid, rng)
        r["ventana"] = f"{y0}-{y1}"
        edges[y0] = (set(zip(e[0].tolist(), e[1].tolist())), set(sel.tolist()))
        # control de tamaño
        act = sel[k_all[sel] > 0]
        sub = []
        for _ in range(REPS_SUB):
            ss = np.sort(rng.choice(act, min(N_SUB, len(act)), replace=False))
            sub.append(window_metrics(R, E, ss, tp, covid, rng, reps=30)[0])
        for key in ("mixing_ratio", "mantel_rho", "covid_mix_ratio", "giant_frac"):
            v = [x[key] for x in sub if key in x]
            if v:
                r[f"{key}_sub"] = float(np.mean(v))
                r[f"{key}_sub_sd"] = float(np.std(v))
        rows.append(r)
        print(r, flush=True)
    for i in range(1, len(WINDOWS)):
        (E0, S0), (E1, S1) = edges[WINDOWS[i - 1][0]], edges[WINDOWS[i][0]]
        sh = S0 & S1
        e0 = {e for e in E0 if e[0] in sh and e[1] in sh}
        e1 = {e for e in E1 if e[0] in sh and e[1] in sh}
        rows[i]["continuity_jaccard"] = len(e0 & e1) / max(1, len(e0 | e1))
        rows[i]["continuity_shared_papers"] = len(sh)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "o06_windows.csv"), index=False)

    # ---------- años
    keyset = {g: {norm(a.split()[0][0] + a.split()[-1]) for a in v} for g, v in TEAMS.items()}
    akeys = [{norm((a.split()[0][0] if a and a.split() else "") + (a.split()[-1] if a and a.split() else "")) for a in w["authors"]} for w in W]
    team = np.array([any(s & ks for ks in keyset.values()) for s in akeys])
    canon = np.array([len(CANON & set(w["refs"])) for w in W])
    N = 24
    Rf = R.astype(np.float64)

    def cohesion(idx, reps=300):
        if len(idx) < 2 * N:
            return np.nan
        v = []
        for _ in range(reps):
            p = rng.permutation(idx)
            a, b = np.asarray(Rf[p[:N]].sum(0)).ravel(), np.asarray(Rf[p[N:2 * N]].sum(0)).ravel()
            v.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        return float(np.mean(v))
    yrows = []
    for y in range(2010, 2027):
        m = (yr == y)
        a = m & (k_all > 0)
        yrows.append(dict(year=y, n=int(m.sum()), con_refs=int(a.sum()), cohesion=cohesion(np.where(a)[0]),
                          cohesion_core=cohesion(np.where(a & core)[0]), cohesion_sin_covid=cohesion(np.where(a & ~covid)[0]),
                          cohesion_sin_equipos=cohesion(np.where(a & ~team)[0]),
                          canon_medio=float(canon[a].mean()), covid=float(covid[m].mean()), core=float(core[m].mean()),
                          equipos=float(team[m].mean()), refs_mediana=float(np.median(k_all[a]))))
        print(yrows[-1], flush=True)
    pd.DataFrame(yrows).to_csv(os.path.join(OUT, "o06_years.csv"), index=False)


if __name__ == "__main__":
    main()
