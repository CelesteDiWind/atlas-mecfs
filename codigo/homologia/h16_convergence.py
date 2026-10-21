"""Paso 16: quiebres y convergencia en la red de acoplamiento validada (fases 2-4 reorientadas tras el piloto h15).
1. Fusiones en el tiempo (H0 de la filtración por fecha). Los papers entran en su fecha y los vínculos validados en la
   del paper más reciente del par. Cada vez que un vínculo une dos componentes de >= MIN_SIZE papers se registra una
   fusión: fecha, paper que la produce, tamaños y temas de cada lado. Convergencia = fusión entre componentes de tema
   dominante distinto. Nulo: fechas permutadas (todo el corpus y dentro de cada tema), 500 réplicas.
2. Indicadores por ventana de 3 años, cada uno con un control del mismo tamaño (submuestras de N_SUB papers):
   continuidad de comunidades entre ventanas consecutivas (ARI en los papers compartidos, frente a la consistencia de
   Louvain consigo mismo), mezcla entre temas (observada / permutación), alineamiento contenido-citas (Spearman entre
   similitud SPECTER2 y exceso de referencias compartidas, Mantel por permutación de nodos), y papers de COVID
   persistente: cuántos hay y si su literatura se une a la del resto (vínculos COVID-no COVID frente a permutación).
3. Las lagunas más largas de la filtración por fecha (no significativas en h15, se describen como tales): papers del
   anillo, temas, similitud de contenido dentro del anillo frente al azar, paper que la cierra y si es puente semántico.
4. Cocitación como contraste retrospectivo: pares validados en cocitación sin ninguna referencia compartida
   (el campo los une después aunque no partieran de la misma literatura), por pares de temas.
Salidas: out/h16_merges.csv, out/h16_windows.csv, out/h16_gaps.csv, out/h16_retro_pairs.csv, out/h16_summary.json"""
import json
import re
from collections import Counter, deque

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score as ARI

import citnet as cn
from common import MAIN, SEED, load_emb, load_papers, out
from h14_compare import WINDOWS

MIN_SIZE = 5       # una fusión cuenta si cada lado tiene al menos 5 papers
N_NULL = 500
N_SUB = 140        # tamaño de las submuestras de control (la ventana más pequeña tiene 145 papers)
N_SUBREP = 30
COVID = re.compile(r"covid|sars-cov-2|post-acute sequelae|pasc\b|long[- ]haul", re.I)


# ------------------------------------------------------------------ 1. fusiones en el tiempo
def merges(a, b, t_edge, tvert, nodes, cl, min_size=MIN_SIZE):
    """Kruskal cronológico con unión-búsqueda. Devuelve la lista de fusiones de componentes grandes."""
    parent = {}
    comp = {}  # raíz -> Counter de temas
    events = []
    order = np.argsort(t_edge, kind="stable")
    born = set()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for v in nodes:
        parent[v] = v
        comp[v] = Counter({cl[v]: 1})
    for e in order:
        i, j, t = a[e], b[e], t_edge[e]
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        si, sj = sum(comp[ri].values()), sum(comp[rj].values())
        if si >= min_size and sj >= min_size:
            ti, tj = comp[ri].most_common(1)[0][0], comp[rj].most_common(1)[0][0]
            newer = i if tvert[i] >= tvert[j] else j
            events.append(dict(t=float(t), paper=int(newer), other=int(j if newer == i else i), size_a=int(max(si, sj)),
                               size_b=int(min(si, sj)), topic_a=int(ti if si >= sj else tj), topic_b=int(tj if si >= sj else ti),
                               cross=bool(ti != tj)))
        if si < sj:
            ri, rj = rj, ri
        parent[rj] = ri
        comp[ri] += comp.pop(rj)
    return events


def giant_curve(a, b, t_edge, tvert, nodes, years):
    """Fracción de papers ya publicados que están en la componente más grande, al final de cada año."""
    res = []
    for y in years:
        live = nodes[tvert[nodes] < y + 1]
        sel = t_edge < y + 1
        if len(live) == 0:
            res.append(0.0)
            continue
        idx = {v: k for k, v in enumerate(live)}
        ea = [idx[x] for x in a[sel] if x in idx]
        eb = [idx[x] for x in b[sel] if x in idx]
        G = nx.Graph()
        G.add_nodes_from(range(len(live)))
        G.add_edges_from(zip(ea, eb))
        res.append(max(len(c) for c in nx.connected_components(G)) / len(live))
    return res


# ------------------------------------------------------------------ 3. representantes de lagunas
def gap_cycle(a, b, t_edge, cocycle, birth):
    """Ciclo más corto (en número de vínculos) que pasa por una arista del cociclo, en la red justo tras nacer."""
    live = t_edge <= birth + 1e-4
    adj = {}
    for i, j in zip(a[live], b[live]):
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    co = {(min(int(i), int(j)), max(int(i), int(j))) for i, j, v in cocycle if int(v) % 2}
    best = None
    for i, j in co:
        if j not in adj.get(i, ()):  # la arista debe existir ya
            continue
        prev, q = {i: None}, deque([i])
        while q:
            x = q.popleft()
            if x == j:
                break
            for y in adj.get(x, ()):
                if (min(x, y), max(x, y)) in co or y in prev:
                    continue
                prev[y] = x
                q.append(y)
        if j in prev:
            path, x = [], j
            while x is not None:
                path.append(x)
                x = prev[x]
            if best is None or len(path) < len(best):
                best = path[::-1]
    return best


def main():
    J = load_papers()
    P = J["papers"]
    n = len(P)
    cl = np.array([p["c"] for p in P])
    names = {c["id"]: c["name"] for c in J["clusters"]}
    E = load_emb(MAIN)
    sim = E @ E.T
    D = np.load(out("cit_dates.npz"))
    tv = D["paper"]
    yr = np.floor(tv)
    rng = np.random.default_rng(SEED + 16)
    R = load_npz(out("bip_refs.npz")).astype(np.int32)
    V = np.load(out("val_acoplamiento.npz"))
    a, b, pv, te = V["a"], V["b"], V["pv"], V["tedge"]
    nodes = np.where(~np.isnan(tv))[0]
    keep = np.isin(a, nodes) & np.isin(b, nodes)
    a, b, pv, te = a[keep], b[keep], pv[keep], te[keep]
    summary = {}

    # ---------- 1. fusiones
    ev = merges(a, b, te, tv, nodes, cl)
    df = pd.DataFrame(ev)
    if len(df):
        df["year"] = np.floor(df.t).astype(int)
        df["title"] = [P[i]["t"] for i in df.paper]
        df["topic_a_name"] = df.topic_a.map(names)
        df["topic_b_name"] = df.topic_b.map(names)
    df.to_csv(out("h16_merges.csv"), index=False)
    years = list(range(2010, 2025))

    def perm_dates(within):
        t2 = tv.copy()
        if within:
            for c in np.unique(cl[nodes]):
                idx = nodes[cl[nodes] == c]
                t2[idx] = rng.permutation(tv[idx])
        else:
            t2[nodes] = rng.permutation(tv[nodes])
        return t2
    obs_cross = int(df.cross.sum()) if len(df) else 0
    by_year_obs = Counter(df.year[df.cross]) if len(df) else Counter()
    nullres = {}
    for within in (False, True):
        tot, cross, byy = [], [], []
        for _ in range(N_NULL):
            t2 = perm_dates(within)
            e2 = merges(a, b, np.maximum(t2[a], t2[b]), t2, nodes, cl)
            tot.append(len(e2))
            cross.append(sum(x["cross"] for x in e2))
            c = Counter(int(np.floor(x["t"])) for x in e2 if x["cross"])
            byy.append([c.get(y, 0) for y in years])
        byy = np.array(byy)
        nullres["tema" if within else "corpus"] = dict(
            merges_median=float(np.median(tot)), cross_median=float(np.median(cross)),
            p_cross_ge=round(float((np.array(cross) >= obs_cross).mean()), 3),
            p_cross_le=round(float((np.array(cross) <= obs_cross).mean()), 3),
            by_year=[dict(year=y, obs=by_year_obs.get(y, 0), null_median=float(np.median(byy[:, k])),
                          null_p95=float(np.quantile(byy[:, k], 0.95)),
                          p_ge=round(float((byy[:, k] >= by_year_obs.get(y, 0)).mean()), 3)) for k, y in enumerate(years)])
    summary["fusiones"] = dict(n=int(len(df)), cross=obs_cross, nulos=nullres)
    print("fusiones:", len(df), "entre temas:", obs_cross, {k: (v["merges_median"], v["cross_median"], v["p_cross_ge"]) for k, v in nullres.items()}, flush=True)
    gc = giant_curve(a, b, te, tv, nodes, years)
    gnull = np.array([giant_curve(a, b, np.maximum(t2[a], t2[b]), t2, nodes, years) for t2 in (perm_dates(False) for _ in range(100))])
    summary["giant_curve"] = [dict(year=y, obs=round(g, 3), null_median=round(float(np.median(gnull[:, k])), 3),
                                   null_p05=round(float(np.quantile(gnull[:, k], 0.05)), 3),
                                   null_p95=round(float(np.quantile(gnull[:, k], 0.95)), 3)) for k, (y, g) in enumerate(zip(years, gc))]

    # ---------- 2. indicadores por ventana
    covid = np.array([bool(COVID.search(p["t"] + " " + (p.get("abs") or "") + " " + " ".join(p.get("kw") or []))) for p in P])

    def window_metrics(sel, seed):
        Bs = R[sel]
        Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
        ks = np.asarray(Bs.sum(1)).ravel()
        p_, hc, _ = cn.fit_bicm(Bs)
        S = cn.project(Bs)
        mu = cn.expected_shared(p_, hc)
        wa, wb, _, wpv = cn.validated(S, mu, ks > 0)
        ga, gb = sel[wa], sel[wb]
        r = dict(validated=len(wa))
        if len(wa) == 0:
            return r, None
        lr = np.random.default_rng(seed)
        perm = [(lambda c2: (c2[ga] != c2[gb]).mean())(cl_perm(sel, lr)) for _ in range(100)]
        r["mixing_ratio"] = float((cl[ga] != cl[gb]).mean() / np.mean(perm))
        iu = np.triu_indices(len(sel), 1)
        z = (S[iu] - mu[iu]) / np.sqrt(mu[iu] + 1e-12)
        ok = ks[iu[0]] * ks[iu[1]] > 0
        r["mantel_rho"] = float(spearmanr(sim[np.ix_(sel, sel)][iu][ok], z[ok])[0])
        cv = covid[sel]
        if cv.sum() >= 5 and (~cv).sum() >= 5:
            obs = float((covid[ga] != covid[gb]).mean())
            pc = [(lambda c: (c[wa] != c[wb]).mean())(lr.permutation(cv)) for _ in range(100)]
            r["covid_n"] = int(cv.sum())
            r["covid_mix_ratio"] = obs / max(1e-9, float(np.mean(pc)))
        G = nx.Graph()
        G.add_weighted_edges_from(zip(ga, gb, -np.log10(np.maximum(wpv, 1e-300))))
        return r, G

    def cl_perm(sel, lr):
        c2 = cl.copy()
        c2[sel] = lr.permutation(cl[sel])
        return c2

    wrows, parts = [], {}
    for k, (y0, y1) in enumerate(WINDOWS):
        sel = np.where((yr >= y0) & (yr <= y1))[0]
        r, G = window_metrics(sel, SEED + k)
        comms = [nx.community.louvain_communities(G, weight="weight", seed=s) for s in range(10)]
        labs = [{v: c for c, vs in enumerate(cm) for v in vs} for cm in comms]
        parts[(y0, y1)] = labs
        nodes_g = sorted(labs[0])
        r["louvain_self_ari"] = float(np.mean([ARI([labs[0][v] for v in nodes_g], [L[v] for v in nodes_g]) for L in labs[1:]]))
        sub = []
        for rep in range(N_SUBREP):
            ss = np.sort(rng.choice(sel, min(N_SUB, len(sel)), replace=False))
            rs, _ = window_metrics(ss, SEED + 1000 * k + rep)
            sub.append(rs)
        for key in ("mixing_ratio", "mantel_rho", "covid_mix_ratio"):
            vals = [x[key] for x in sub if key in x]
            if vals:
                r[f"{key}_sub_mean"] = float(np.mean(vals))
                r[f"{key}_sub_sd"] = float(np.std(vals))
        r.update(ventana=f"{y0}-{y1}", papers=int(len(sel)), covid_papers=int(covid[sel].sum()))
        wrows.append(r)
        print(r, flush=True)
    for k in range(len(WINDOWS) - 1):
        L0, L1 = parts[WINDOWS[k]], parts[WINDOWS[k + 1]]
        common = sorted(set(L0[0]) & set(L1[0]))
        wrows[k + 1]["continuity_ari"] = float(np.mean([ARI([x[v] for v in common], [y[v] for v in common]) for x in L0[:5] for y in L1[:5]])) if len(common) > 10 else None
        wrows[k + 1]["continuity_common"] = len(common)
    pd.DataFrame(wrows).to_csv(out("h16_windows.csv"), index=False)
    summary["ventanas"] = wrows

    # ---------- 3. lagunas más largas
    from ripser import ripser
    M = cn.time_filtration(a, b, te, tv, nodes)
    rr = ripser(M, distance_matrix=True, maxdim=1, do_cocycles=True)
    dg, cocy = rr["dgms"][1], rr["cocycles"][1]
    pers = dg[:, 1] - dg[:, 0]
    bars = pd.read_csv(out("h15_bars_acoplamiento_fecha.csv"))
    grows = []
    sem_rand = sim[rng.choice(nodes, 5000), rng.choice(nodes, 5000)].mean()
    bridges = pd.read_csv(out("bridges.csv"))
    brank = {int(i): k + 1 for k, i in enumerate(bridges.id)}
    for k in np.argsort(-pers)[:8]:
        if not np.isfinite(dg[k, 1]) or pers[k] <= 0:
            continue
        birth, death = dg[k, 0] + 1900, dg[k, 1] + 1900
        cyc = gap_cycle(a, b, te - 1900 + 1e-6, cocy[k], dg[k, 0])
        ce = np.abs(te - death) < 1e-4
        closers = sorted({int(x if tv[x] >= tv[y] else y) for x, y in zip(a[ce], b[ce])})
        members = cyc or []
        msim = float(np.mean([sim[i, j] for x, i in enumerate(members) for j in members[x + 1:]])) if len(members) > 2 else np.nan
        p_row = bars.iloc[(bars.years - pers[k]).abs().argsort()[:1]].p_fechas.values[0]
        grows.append(dict(birth=round(birth, 2), death=round(death, 2), years=round(pers[k], 2), p_fechas=round(float(p_row), 3),
                          n_papers=len(members), papers=json.dumps(members), topics=json.dumps(dict(Counter(int(cl[i]) for i in members))),
                          sem_sim_in_cycle=round(msim, 3), sem_sim_random=round(float(sem_rand), 3),
                          closer=json.dumps(closers), closer_title=" | ".join(P[i]["t"] for i in closers),
                          closer_topic=json.dumps([int(cl[i]) for i in closers]),
                          closer_bridge_rank=json.dumps([brank.get(i) for i in closers]),
                          titles=" | ".join(f"{P[i]['t']} ({P[i]['y']})" for i in members)))
    pd.DataFrame(grows).to_csv(out("h16_gaps.csv"), index=False)
    summary["lagunas"] = grows

    # ---------- 4. cocitación sin acoplamiento
    Vc = np.load(out("val_cocitacion.npz"))
    S_ref = cn.project(R)
    ca, cb = Vc["a"], Vc["b"]
    none = S_ref[ca, cb] == 0
    cp = set(zip(V["a"], V["b"]))
    not_val = np.array([(x, y) not in cp for x, y in zip(ca, cb)])
    tp = Counter((int(min(cl[x], cl[y])), int(max(cl[x], cl[y]))) for x, y in zip(ca[none], cb[none]))
    tpall = Counter((int(min(cl[x], cl[y])), int(max(cl[x], cl[y]))) for x, y in zip(ca, cb))
    retro = pd.DataFrame([dict(topic_a=names[t[0]], topic_b=names[t[1]], same_topic=t[0] == t[1], pairs_no_shared_refs=c,
                               pairs_cocited=tpall[t], share=round(c / tpall[t], 3)) for t, c in tp.items()]).sort_values("pairs_no_shared_refs", ascending=False)
    retro.to_csv(out("h16_retro_pairs.csv"), index=False)
    summary["retro"] = dict(cocit_validated=int(len(ca)), no_shared_refs=int(none.sum()), not_in_coupling_backbone=int(not_val.sum()),
                            cross_topic_share_no_refs=round(float((cl[ca[none]] != cl[cb[none]]).mean()), 3),
                            cross_topic_share_all=round(float((cl[ca] != cl[cb]).mean()), 3))
    print(summary["retro"])
    json.dump(summary, open(out("h16_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
