"""Paso 6: representantes de los ciclos H1 significativos (resultado previo 4).
Método (sin depender de empates): para cada barra significativa se toma el cociclo de ripser y, en el complejo a
escala t = nacimiento + 25 % de la persistencia, el ciclo más corto que cruza el cociclo una sola vez
(arista del cociclo (u, v) + camino mínimo u -> v que no use aristas del cociclo). Ese ciclo empareja con el
cociclo (producto = 1 mod 2), así que es no trivial y pertenece a la clase dual. Antes se rompen los empates con una
perturbación de 1e-9. La composición por cluster lleva incertidumbre: frecuencia de cada paper en los representantes
de las 50 submuestras del 80 % (barra emparejada por bottleneck).
Salidas: out/cycles.csv, out/cycle_members.json"""
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from common import MAIN, N_SUB, SEED, SUB_FRAC, finite, load_emb, load_papers, out, rips, space_distance, stable_hash


def jitter(D, rng):
    J = rng.uniform(0, 1e-9, D.shape)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return D + J


def representative(D, cocycle, birth, death):
    t = birth + 0.25 * (death - birth)
    A = np.where(D <= t, D, 0.0)
    np.fill_diagonal(A, 0)
    co = {(int(min(i, j)), int(max(i, j))) for i, j, v in cocycle if int(v) % 2 and D[int(i), int(j)] <= t}
    if not co:
        return None
    B = A.copy()
    for i, j in co:
        B[i, j] = B[j, i] = 0
    G = csr_matrix(B)
    best, best_cycle = np.inf, None
    sources = sorted({i for i, _ in co})
    dist, pred = dijkstra(G, directed=False, indices=sources, return_predecessors=True)
    pos = {s: k for k, s in enumerate(sources)}
    for i, j in co:
        L = D[i, j] + dist[pos[i], j]
        if L < best:
            path, x = [j], j
            while x != i and pred[pos[i], x] >= 0:
                x = pred[pos[i], x]
                path.append(x)
            if path[-1] == i:
                best, best_cycle = L, path[::-1]
    return best_cycle


def cycles_for(E, sp, sig_bars, rng):
    D = jitter(space_distance(E, sp), rng)
    R = rips(D, 1, cocycles=True)
    dg, cocs = R["dgms"][1], R["cocycles"][1]
    out_ = []
    for b, d in sig_bars:
        k = int(np.argmin(np.abs(dg[:, 0] - b) + np.abs(dg[:, 1] - d)))
        out_.append(representative(D, cocs[k], dg[k, 0], dg[k, 1]))
    return out_


def main():
    P = load_papers()["papers"]
    cl = np.array([p["c"] for p in P])
    E = load_emb(MAIN)
    n = len(E)
    rows, members = [], {}
    for sp in ("full", "k15"):
        B = pd.read_csv(out(f"h1_bars_{MAIN}_all_{sp}.csv"))
        sig = B[B.p_gmm < 0.05].reset_index(drop=True)
        if not len(sig):
            print(sp, "sin ciclos significativos frente al nulo gmm"); continue
        rng = np.random.default_rng([SEED, stable_hash("cyc", sp)])
        reps = cycles_for(E, sp, sig[["birth", "death"]].values, rng)
        # frecuencia de pertenencia en submuestras (barra emparejada por bottleneck con la original)
        from persim import bottleneck
        freq = [dict() for _ in range(len(sig))]
        for r in range(N_SUB):
            idx = np.sort(rng.choice(n, int(SUB_FRAC * n), replace=False))
            Es = E[idx]
            dgs = finite(rips(space_distance(Es, sp), 1)["dgms"][1])
            if not len(dgs):
                continue
            _, M = bottleneck(sig[["birth", "death"]].values, dgs, matching=True)
            pairs = {}
            for i, j, *_ in np.atleast_2d(np.asarray(M)):
                i, j = int(i), int(j)
                if 0 <= i < len(sig) and 0 <= j < len(dgs) and dgs[j, 1] - dgs[j, 0] >= 0.5 * sig.pers[i]:
                    pairs[i] = dgs[j]
            if not pairs:
                continue
            got = cycles_for(Es, sp, [pairs[i] for i in sorted(pairs)], rng)
            for i, cyc in zip(sorted(pairs), got):
                for v in cyc or []:
                    g = int(idx[v])
                    freq[i][g] = freq[i].get(g, 0) + 1
        for i, (bar, cyc) in enumerate(zip(sig.itertuples(), reps)):
            cid = f"{sp}-{i + 1}"
            nodes = [int(v) for v in (cyc or [])]
            comp = pd.Series(cl[nodes]).value_counts() if nodes else pd.Series(dtype=int)
            stab = {k: round(v / N_SUB, 2) for k, v in sorted(freq[i].items(), key=lambda kv: -kv[1])}
            members[cid] = {"nodes": nodes, "freq_subsample": stab}
            rows.append(dict(cycle_id=cid, space=sp, birth=round(bar.birth, 4), death=round(bar.death, 4),
                             pers=round(bar.pers, 4), p_gmm=round(bar.p_gmm, 4), reappear=getattr(bar, "reappear", np.nan),
                             n_nodes=len(nodes), n_clusters=int(comp.size), clusters={int(k): int(v) for k, v in comp.items()},
                             core_nodes=[k for k, v in stab.items() if v >= 0.5][:20],
                             titles=[P[v]["t"][:70] for v in nodes[:8]], years=[P[v]["y"] for v in nodes]))
        print(sp, f"{len(sig)} ciclos significativos (gmm):")
        for r in rows:
            if r["space"] == sp:
                print(f"   {r['cycle_id']}: pers {r['pers']} (p={r['p_gmm']}), reaparece {r['reappear']}, "
                      f"{r['n_nodes']} papers, clusters {r['clusters']}")
    pd.DataFrame(rows).to_csv(out("cycles.csv"), index=False)
    with open(out("cycle_members.json"), "w") as fh:
        json.dump(members, fh)


if __name__ == "__main__":
    main()
