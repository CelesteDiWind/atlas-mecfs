"""Paso 16d: ¿qué hay detrás de la caída de continuidad 2020-22 -> 2021-23 (h16c)?
1. Vínculos perdidos y ganados entre los papers compartidos por las dos ventanas, por pares de temas.
2. Perfiles de referencias por año de publicación (suma de las filas de la bipartita, normalizada), con cohortes del
   mismo tamaño (N papers): coseno entre años y "cohesión" de cada año = coseno entre dos mitades disjuntas del año.
   Controles de la cohesión: sin papers de COVID persistente; cada paper recortado a K referencias al azar.
Salidas: out/h16d_cohorts.json"""
import json
from collections import Counter

import numpy as np
from scipy.sparse import load_npz

from common import SEED, load_papers, out
from h16_convergence import COVID
from h16c_continuity import window_net

N = 24        # tamaño de cohorte (2024 tiene 24 papers con referencias)
K = 30        # recorte de referencias por paper
REPS = 400
YEARS = list(range(2018, 2025))


def main():
    J = load_papers()
    P = J["papers"]
    cl = np.array([int(p["c"]) for p in P])
    names = {c["id"]: c["name"] for c in J["clusters"]}
    cv = np.array([bool(COVID.search(p["t"] + " " + (p.get("abs") or "") + " " + " ".join(p.get("kw") or []))) for p in P])
    yr = np.floor(np.load(out("cit_dates.npz"))["paper"])
    R = load_npz(out("bip_refs.npz")).astype(np.float64).tocsr()
    k = np.asarray(R.sum(1)).ravel()
    act = k > 0
    rng = np.random.default_rng(SEED + 164)
    res = {}

    # 1. vínculos perdidos y ganados
    s0 = np.where((yr >= 2020) & (yr <= 2022))[0]
    s1 = np.where((yr >= 2021) & (yr <= 2023))[0]
    E0, _ = window_net(R.astype(np.int32), s0, SEED)
    E1, _ = window_net(R.astype(np.int32), s1, SEED)
    sh = set(s0.tolist()) & set(s1.tolist())
    e0 = {e for e in E0 if e[0] in sh and e[1] in sh}
    e1 = {e for e in E1 if e[0] in sh and e[1] in sh}
    tp = lambda E: [[names[a], names[b], c] for (a, b), c in Counter(tuple(sorted((int(cl[x]), int(cl[y])))) for x, y in E).most_common()]
    res["transicion_2020_22_2021_23"] = dict(comunes=len(e0 & e1), perdidos=len(e0 - e1), nuevos=len(e1 - e0),
                                             nuevos_por_temas=tp(e1 - e0), perdidos_por_temas=tp(e0 - e1))
    print({x: v for x, v in res["transicion_2020_22_2021_23"].items() if "temas" not in x}, flush=True)

    # 2. perfiles de referencias por año
    Rt = R.tolil()
    for i, r in enumerate(R.tolil().rows):
        if len(r) > K:
            for c in rng.choice(r, len(r) - K, replace=False):
                Rt[i, c] = 0
    Rt = Rt.tocsr()

    def prof(M, idx):
        v = np.asarray(M[idx].sum(0)).ravel()
        return v / np.linalg.norm(v)

    def halves(M, pool):
        if len(pool) < 2 * N:
            return None
        v = []
        for _ in range(REPS):
            p = rng.permutation(pool)
            v.append(float(prof(M, p[:N]) @ prof(M, p[N:2 * N])))
        return [round(float(np.mean(v)), 3), round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]

    pool = {y: np.where((yr == y) & act)[0] for y in YEARS}
    cross = {}
    for i, y0 in enumerate(YEARS):
        for y1 in YEARS[i + 1:]:
            cross[f"{y0}-{y1}"] = round(float(np.mean([prof(R, rng.choice(pool[y0], N, False)) @ prof(R, rng.choice(pool[y1], N, False))
                                                       for _ in range(REPS)])), 3)
    coh = {}
    for y in YEARS:
        coh[y] = dict(n=int(len(pool[y])), refs_mediana=int(np.median(k[pool[y]])), cohesion=halves(R, pool[y]),
                      cohesion_sin_covid=halves(R, np.where((yr == y) & act & ~cv)[0]),
                      cohesion_recorte=halves(Rt, np.where((yr == y) & (k >= K))[0]))
        print(y, coh[y], flush=True)
    norm = {}
    for key, v in cross.items():
        y0, y1 = map(int, key.split("-"))
        c0, c1 = coh[y0]["cohesion"], coh[y1]["cohesion"]
        if c0 and c1:
            norm[key] = round(v / np.sqrt(c0[0] * c1[0]), 3)
    # 3. ¿cuánto del pico de 2020-21 se debe a las referencias más citadas por todo el corpus?
    f = np.asarray(R[act].sum(0)).ravel()
    drop_top = {}
    for d in (3, 10, 30):
        M = R.tocsc()[:, np.argsort(-f)[d:]].tocsr()
        ok = np.asarray(M.sum(1)).ravel() > 0
        drop_top[d] = {y: (halves(M, np.where((yr == y) & ok)[0]) or [None])[0] for y in YEARS}
        print("sin las", d, "más citadas:", drop_top[d], flush=True)
    # referencias sobrerrepresentadas en 2020-21 frente a 2018-19 + 2022-23 (índices de columna de bip_refs)
    A = (yr >= 2020) & (yr <= 2021) & act
    B = (((yr >= 2018) & (yr <= 2019)) | ((yr >= 2022) & (yr <= 2023))) & act
    fa, fb = np.asarray(R[A].sum(0)).ravel() / A.sum(), np.asarray(R[B].sum(0)).ravel() / B.sum()
    over = [dict(col=int(j), frac_2020_21=round(float(fa[j]), 3), frac_resto=round(float(fb[j]), 3)) for j in np.argsort(-(fa - fb))[:15]]
    res.update(coseno_entre_años=cross, cohesion=coh, coseno_normalizado=norm, cohesion_sin_top=drop_top, sobrerrepresentadas_2020_21=over,
               nota_top3="las 3 más citadas son Fukuda 1994, Consenso Canadiense 2003 e ICC 2011 (títulos consultados en OpenAlex)")
    print("consecutivos normalizados:", {x: v for x, v in norm.items() if int(x[5:]) - int(x[:4]) == 1}, flush=True)
    json.dump(res, open(out("h16d_cohorts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
