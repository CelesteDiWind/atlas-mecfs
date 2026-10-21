"""Paso 16b: complementos de h16.
- Fusiones con umbrales de tamaño 2 y 3 (con 5 solo hay 3 fusiones: el gigante absorbe piezas pequeñas) y nulo de fechas.
- Lagunas más largas de la filtración por fecha (barras finitas; en h16 las infinitas tapaban el orden).
- Base de la cocitación retrospectiva: entre los pares sin ninguna referencia compartida, ¿qué fracción es entre temas?
Salidas: out/h16b_merges_size{2,3}.csv, out/h16_gaps.csv (rehecho), out/h16b_summary.json"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import citnet as cn
from common import MAIN, SEED, load_emb, load_papers, out
from h16_convergence import N_NULL, gap_cycle, merges

YEARS = list(range(2010, 2025))


def main():
    J = load_papers()
    P = J["papers"]
    cl = np.array([int(p["c"]) for p in P])
    names = {c["id"]: c["name"] for c in J["clusters"]}
    E = load_emb(MAIN)
    sim = E @ E.T
    tv = np.load(out("cit_dates.npz"))["paper"]
    rng = np.random.default_rng(SEED + 161)
    V = np.load(out("val_acoplamiento.npz"))
    a, b, te = V["a"], V["b"], V["tedge"]
    nodes = np.where(~np.isnan(tv))[0]
    summ = {}

    def perm(within):
        t2 = tv.copy()
        if within:
            for c in np.unique(cl[nodes]):
                idx = nodes[cl[nodes] == c]
                t2[idx] = rng.permutation(tv[idx])
        else:
            t2[nodes] = rng.permutation(tv[nodes])
        return t2

    # ---------- fusiones con umbrales menores
    for ms in (2, 3):
        ev = pd.DataFrame(merges(a, b, te, tv, nodes, cl, min_size=ms))
        ev["year"] = np.floor(ev.t).astype(int)
        ev["title"] = [P[i]["t"] for i in ev.paper]
        ev["topic_a_name"] = ev.topic_a.map(names)
        ev["topic_b_name"] = ev.topic_b.map(names)
        ev["sem_sim"] = [round(float(sim[i, j]), 3) for i, j in zip(ev.paper, ev.other)]
        ev.to_csv(out(f"h16b_merges_size{ms}.csv"), index=False)
        obs_y = Counter(ev.year[ev.cross])
        obs_all = Counter(ev.year)
        res = {}
        for within in (False, True):
            tot, cross, byy, byall = [], [], [], []
            for _ in range(N_NULL):
                t2 = perm(within)
                e2 = merges(a, b, np.maximum(t2[a], t2[b]), t2, nodes, cl, min_size=ms)
                tot.append(len(e2))
                cross.append(sum(x["cross"] for x in e2))
                cy = Counter(int(x["t"]) for x in e2 if x["cross"])
                ca_ = Counter(int(x["t"]) for x in e2)
                byy.append([cy.get(y, 0) for y in YEARS])
                byall.append([ca_.get(y, 0) for y in YEARS])
            byy, byall, tot, cross = map(np.array, (byy, byall, tot, cross))
            nc = int(ev.cross.sum())
            res["tema" if within else "corpus"] = dict(
                merges_obs=int(len(ev)), merges_median=float(np.median(tot)),
                p_merges_ge=round(float((tot >= len(ev)).mean()), 3), p_merges_le=round(float((tot <= len(ev)).mean()), 3),
                cross_obs=nc, cross_median=float(np.median(cross)),
                p_cross_ge=round(float((cross >= nc).mean()), 3), p_cross_le=round(float((cross <= nc).mean()), 3),
                share_cross_obs=round(float(ev.cross.mean()), 3),
                share_cross_null=round(float(np.median(cross / np.maximum(tot, 1))), 3),
                by_year=[dict(year=y, merges=obs_all.get(y, 0), merges_null=float(np.median(byall[:, k])),
                              cross=obs_y.get(y, 0), cross_null=float(np.median(byy[:, k])),
                              cross_p95=float(np.quantile(byy[:, k], 0.95)),
                              p_ge=round(float((byy[:, k] >= obs_y.get(y, 0)).mean()), 3)) for k, y in enumerate(YEARS)])
        summ[f"fusiones_size{ms}"] = res
        print(ms, {k: {x: v[x] for x in v if x != "by_year"} for k, v in res.items()}, flush=True)

    # ---------- lagunas
    from ripser import ripser
    M = cn.time_filtration(a, b, te, tv, nodes)
    rr = ripser(M, distance_matrix=True, maxdim=1, do_cocycles=True)
    dg, cocy = rr["dgms"][1], rr["cocycles"][1]
    fin = np.where(np.isfinite(dg[:, 1]) & (dg[:, 1] > dg[:, 0]))[0]
    print("H1 infinitas:", int((~np.isfinite(dg[:, 1])).sum()), "finitas:", len(fin), flush=True)
    pers = dg[fin, 1] - dg[fin, 0]
    nulls = {k: np.load(out(f"h15_null_acoplamiento_fecha_{k}.npy"))[:, 0] for k in ("fechas", "fechas_tema", "curveball")}
    bridges = pd.read_csv(out("bridges.csv"))
    brank = {int(i): k + 1 for k, i in enumerate(bridges.id)}
    ri, rj = rng.choice(nodes, 20000), rng.choice(nodes, 20000)
    sem_rand = float(sim[ri[ri != rj], rj[ri != rj]].mean())
    rows = []
    for k in fin[np.argsort(-pers)][:10]:
        birth, death = dg[k, 0] + 1900, dg[k, 1] + 1900
        cyc = [int(x) for x in (gap_cycle(a, b, te - 1900 + 1e-6, cocy[k], dg[k, 0]) or [])]
        ce = np.abs(te - death) < 1e-4
        cand = sorted({int(x if tv[x] >= tv[y] else y) for x, y in zip(a[ce], b[ce])})
        # el que la cierra: el vínculo de esa fecha que toca el anillo (si lo hay)
        touch = [int(x if tv[x] >= tv[y] else y) for x, y in zip(a[ce], b[ce]) if x in cyc or y in cyc]
        closers = sorted(set(touch)) or cand
        msim = float(np.mean([sim[i, j] for x, i in enumerate(cyc) for j in cyc[x + 1:]])) if len(cyc) > 2 else np.nan
        csim = float(np.mean([sim[c, i] for c in closers for i in cyc if c != i])) if cyc else np.nan
        yrs = dg[k, 1] - dg[k, 0]
        rows.append(dict(birth=round(birth, 2), death=round(death, 2), years=round(yrs, 2),
                         **{f"p_{n_}": round((1 + (v >= yrs).sum()) / (1 + len(v)), 3) for n_, v in nulls.items()},
                         n_papers=len(cyc), papers=json.dumps(cyc), topics=json.dumps(dict(Counter(int(cl[i]) for i in cyc))),
                         n_topics=len(set(cl[cyc])) if cyc else 0,
                         sem_sim_in_cycle=round(msim, 3), sem_sim_random=round(sem_rand, 3), closer=json.dumps(closers),
                         closer_sim_to_cycle=round(csim, 3), closer_topic=json.dumps([int(cl[i]) for i in closers]),
                         closer_bridge_rank=json.dumps([brank.get(i) for i in closers]),
                         closer_title=" | ".join(f"{P[i]['t']} ({P[i]['y']})" for i in closers),
                         titles=" | ".join(f"[{cl[i]}] {P[i]['t']} ({P[i]['y']})" for i in cyc)))
    pd.DataFrame(rows).to_csv(out("h16_gaps.csv"), index=False)
    summ["lagunas"] = rows

    # ---------- base de la cocitación retrospectiva
    R = load_npz(out("bip_refs.npz")).astype(np.int32)
    S = cn.project(R)
    act = np.asarray(R.sum(1)).ravel() > 0
    iu = np.triu_indices(len(S), 1)
    z = (S[iu] == 0) & act[iu[0]] & act[iu[1]]
    Vc = np.load(out("val_cocitacion.npz"))
    ca, cb = Vc["a"], Vc["b"]
    cocited = set(zip(ca.tolist(), cb.tolist()))
    zi, zj = iu[0][z], iu[1][z]
    notco = np.array([(x, y) not in cocited for x, y in zip(zi.tolist(), zj.tolist())])
    none = (S[ca, cb] == 0) & act[ca] & act[cb]
    summ["retro_base"] = dict(
        cocit_no_refs_both_active=int(none.sum()),
        cross_share_cocit_no_refs=round(float((cl[ca[none]] != cl[cb[none]]).mean()), 3),
        cross_share_all_pairs_no_refs=round(float((cl[zi[notco]] != cl[zj[notco]]).mean()), 3),
        cross_share_cocit_with_refs=round(float((cl[ca[~none]] != cl[cb[~none]]).mean()), 3),
        sem_sim_cocit_no_refs=round(float(sim[ca[none], cb[none]].mean()), 3),
        sem_sim_cocit_with_refs=round(float(sim[ca[~none], cb[~none]].mean()), 3),
        sem_sim_pairs_no_refs=round(float(sim[zi[notco], zj[notco]].mean()), 3))
    print(summ["retro_base"], flush=True)
    json.dump(summ, open(out("h16b_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
