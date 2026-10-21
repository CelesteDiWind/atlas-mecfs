"""Paso o14: ¿qué temas dialogan entre sí? (corpus OpenAlex completo; la selección manual, como comparación)
Diálogo = citas entre papers de temas distintos (los temas salen del contenido; las citas son una señal independiente).
Redes: citas directas (A cita a B, con dirección) y acoplamiento validado (A y B comparten literatura, simétrico).
Nulo: etiquetas de tema permutadas entre papers del mismo año y del mismo quintil de grado (conserva la red, el tamaño
de cada tema, su antigüedad y lo conectados que están sus papers). 1000 réplicas; BH sobre las celdas.
Métricas
  1. O/E por par de temas (observado / media del nulo) en las dos redes
  2. asimetría del flujo de citas directas por par, y balance citado/citante de cada tema
  3. apertura de cada tema: índice E–I (externos − internos) / (externos + internos), frente al nulo
  4. contenido frente a citas: similitud SPECTER2 media entre temas frente a log2 O/E (residuos = sorpresas)
  5. evolución: O/E por ventana de 3 años (acoplamiento validado por ventana de o04; citas directas por año del citante)
  6. papers puente: coeficiente de participación de cada paper (reparto de sus vínculos entre temas)
Salidas: out/o14_dialogue.json (todo lo que necesitan las figuras y el visor), out/o14_bridges.csv, out/o14_windows.csv"""
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
SEED = 20261020
N_PERM = 1000
N_PERM_WIN = 300
K = 8
WINDOWS = [(y, y + 2) for y in range(2010, 2025)]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def strata(years, deg):
    """Estratos año × quintil de grado (los papers sin vínculos, aparte)."""
    s = np.zeros(len(years), int)
    for y in np.unique(years):
        m = years == y
        d = deg[m]
        b = np.zeros(m.sum(), int)
        pos = d > 0
        if pos.sum() >= 5:
            b[pos] = 1 + np.searchsorted(np.quantile(d[pos], [0.2, 0.4, 0.6, 0.8]), d[pos], side="right")
        s[m] = y * 10 + b
    return s


def permuter(s, rng):
    groups = [np.where(s == g)[0] for g in np.unique(s)]

    def perm(lab):
        out = lab.copy()
        for g in groups:
            if len(g) > 1:
                out[g] = lab[rng.permutation(g)]
        return out
    return perm


def directed_counts(a, b, lab):
    return np.bincount(lab[a] * K + lab[b], minlength=K * K).reshape(K, K).astype(float)


def sym_counts(a, b, lab):
    M = directed_counts(a, b, lab)
    return M + M.T - np.diag(np.diag(M))     # cada vínculo cuenta una vez en (A,B) y en (B,A); en la diagonal, una vez


def ei(M, directed):
    """Índice E–I por tema a partir de la matriz de recuentos."""
    I = np.diag(M)
    if directed:
        X = M.sum(1) + M.sum(0) - 2 * I
    else:
        X = M.sum(1) - I
    return (X - I) / np.maximum(X + I, 1)


def quasi_indep(M, directed, iters=500):
    """Cuasi-independencia: recuentos esperados entre temas distintos dados solo sus márgenes fuera de la diagonal
    (ajuste proporcional iterativo con la diagonal excluida). Responde: cuando un tema cita fuera, ¿a quién?"""
    O = M.astype(float).copy(); np.fill_diagonal(O, 0)
    F = np.ones_like(O); np.fill_diagonal(F, 0)
    for _ in range(iters):
        F *= (O.sum(1) / np.maximum(F.sum(1), 1e-12))[:, None]
        F *= (O.sum(0) / np.maximum(F.sum(0), 1e-12))[None, :]
        if not directed:
            F = (F + F.T) / 2
    z = (O - F) / np.sqrt(np.maximum(F, 1e-9))
    return O / np.maximum(F, 1e-9), F, z


def oe_with_null(a, b, lab, perm, n, directed):
    f = directed_counts if directed else sym_counts
    obs = f(a, b, lab)
    null = np.stack([f(a, b, perm(lab)) for _ in range(n)])
    mu = null.mean(0)
    p = (1 + (np.abs(null - mu) >= np.abs(obs - mu)[None]).sum(0)) / (1 + n)
    return obs, null, mu, p


def analyse(P, edges, E, names, rng, n_perm, label, SKIP=None):
    n = len(P)
    lab = np.array([p["c"] for p in P]); yr = np.array([p["y"] or 0 for p in P])
    f = edges["di"]; da, db = np.array(f[0::4], int), np.array(f[1::4], int)
    f = edges["cp"]; ca, cb = np.array(f[0::4], int), np.array(f[1::4], int)
    deg = np.bincount(np.r_[da, db, ca, cb], minlength=n)
    perm = permuter(strata(yr, deg), rng)
    R = {"label": label, "names": names, "n_topic": np.bincount(lab, minlength=K).tolist()}
    for key, a, b, directed in (("di", da, db, True), ("cp", ca, cb, False)):
        obs, null, mu, p = oe_with_null(a, b, lab, perm, n_perm, directed)
        oe = obs / np.maximum(mu, 1e-9)
        cells = [(i, j) for i in range(K) for j in range(K)] if directed else [(i, j) for i in range(K) for j in range(i, K)]
        q = bh([p[i, j] for i, j in cells])
        Q = np.ones((K, K))
        for (i, j), v in zip(cells, q):
            Q[i, j] = v
            if not directed:
                Q[j, i] = v
        eio = ei(obs, directed); ein = np.stack([ei(m, directed) for m in null])
        from scipy.stats import norm
        qx, Fx, zx = quasi_indep(obs, directed)
        cells_x = [(i, j) for i in range(K) for j in range(K) if i != j and (directed or i < j)]
        qq = bh([2 * norm.sf(abs(zx[i, j])) for i, j in cells_x]); Qx = np.ones((K, K))
        for (i, j), v in zip(cells_x, qq):
            Qx[i, j] = v
            if not directed:
                Qx[j, i] = v
        R[key + "_ext"] = dict(oe=qx.round(3).tolist(), exp=Fx.round(2).tolist(), z=zx.round(2).tolist(), q=Qx.round(4).tolist())
        R[key] = dict(obs=obs.astype(int).tolist(), exp=mu.round(2).tolist(), oe=oe.round(3).tolist(), q=Q.round(4).tolist(),
                      ei=eio.round(3).tolist(), ei_null=ein.mean(0).round(3).tolist(),
                      ei_p=((1 + (np.abs(ein - ein.mean(0)) >= np.abs(eio - ein.mean(0))[None]).sum(0)) / (1 + n_perm)).round(4).tolist(),
                      n_edges=int(len(a)))
        if directed:
            # asimetría por par a partir de O/E, y balance citado/citante (fuera del propio tema) por tema
            asym = (oe - oe.T) / np.maximum(oe + oe.T, 1e-9)
            nasym = np.stack([(lambda o: (o - o.T) / np.maximum(o + o.T, 1e-9))(m / np.maximum(mu, 1e-9)) for m in null])
            ap = (1 + (np.abs(nasym) >= np.abs(asym)[None]).sum(0)) / (1 + n_perm)
            def balance(M):
                ext_in = M.sum(0) - np.diag(M); ext_out = M.sum(1) - np.diag(M)
                return np.log2(np.maximum(ext_in, 1) / np.maximum(ext_out, 1))
            bo = balance(obs); bn = np.stack([balance(m) for m in null])
            R["flow"] = dict(asym=asym.round(3).tolist(), asym_p=ap.round(4).tolist(),
                             balance=bo.round(3).tolist(), balance_null=bn.mean(0).round(3).tolist(),
                             balance_p=((1 + (np.abs(bn - bn.mean(0)) >= np.abs(bo - bn.mean(0))[None]).sum(0)) / (1 + n_perm)).round(4).tolist())
    # contenido frente a citas (pares de temas distintos)
    if E is not None:
        Em = normalize(E)
        idx = [np.where(lab == t)[0] for t in range(K)]
        samp = [rng.choice(ix, min(400, len(ix)), replace=False) for ix in idx]
        rows = []
        for i in range(K):
            for j in range(i + 1, K):
                sim = float((Em[samp[i]] @ Em[samp[j]].T).mean())
                d_obs = R["di"]["obs"][i][j] + R["di"]["obs"][j][i]; d_exp = R["di"]["exp"][i][j] + R["di"]["exp"][j][i]
                rows.append(dict(a=i, b=j, sim=round(sim, 4), oe_di=round(d_obs / max(d_exp, 1e-9), 3), oe_cp=R["cp"]["oe"][i][j]))
        df = pd.DataFrame(rows)
        for key in ("oe_di", "oe_cp"):
            sub = df[(df.a != SKIP) & (df.b != SKIP)] if SKIP is not None else df
            y2 = np.log2(np.maximum(sub[key], 1e-3)); X2 = np.c_[np.ones(len(sub)), sub.sim]
            b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            df[f"res2_{key}"] = np.where((df.a != SKIP) & (df.b != SKIP), (np.log2(np.maximum(df[key], 1e-3)) - (b2[0] + b2[1] * df.sim)).round(3), np.nan)
            R[f"fit2_{key}"] = dict(intercept=round(float(b2[0]), 3), slope=round(float(b2[1]), 3), r=round(float(np.corrcoef(sub.sim, y2)[0, 1]), 3),
                                    rho=round(float(spearmanr(sub.sim, y2)[0]), 3), excluded=SKIP)
            y = np.log2(np.maximum(df[key], 1e-3)); X = np.c_[np.ones(len(df)), df.sim]
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            df[f"res_{key}"] = (y - X @ beta).round(3)
            R[f"fit_{key}"] = dict(intercept=round(float(beta[0]), 3), slope=round(float(beta[1]), 3),
                                    r=round(float(np.corrcoef(df.sim, y)[0, 1]), 3), rho=round(float(spearmanr(df.sim, y)[0]), 3))
        R["content"] = df.to_dict("records")
    return R, lab, yr, (da, db, ca, cb), perm


def main():
    rng = np.random.default_rng(SEED)
    m = json.load(open(os.path.join(ROOT, "viewer", "oa", "oa_map.json"), encoding="utf-8"))
    P, C = m["papers"], m["clusters"]
    names = [dict(es=c["name"], en=c["name_en"]) for c in C]
    E = np.load(os.path.join(OUT, "E_specter2.npy")).astype(np.float64)
    xmrv = next(c["id"] for c in C if "XMRV" in c["name"])  # tema que solo existe en 2010-12: fuera del ajuste contenido-citas
    R, lab, yr, (da, db, ca, cb), perm = analyse(P, m["edges"], E, names, rng, N_PERM, "oa", SKIP=xmrv)
    print("O/E citas directas (filas citan, columnas citadas):\n", np.array(R["di"]["oe"]).round(2), flush=True)
    print("O/E acoplamiento:\n", np.array(R["cp"]["oe"]).round(2), flush=True)

    # ---------- evolución por ventanas
    n = len(P)
    wrows = []
    for y0, y1 in WINDOWS:
        sel = (yr >= y0) & (yr <= y1)
        V = np.load(os.path.join(OUT, f"val_win_{y0}.npz"))
        wa, wb = V["a"], V["b"]
        dm = sel[da] & (yr[db] <= y1)                       # citas hechas por papers de la ventana
        a_d, b_d = da[dm], db[dm]
        deg = np.bincount(np.r_[wa, wb, a_d, b_d], minlength=n)
        s = strata(yr, deg)
        pw = permuter(s, rng)
        present = np.bincount(lab[sel], minlength=K)
        for key, a, b, directed in (("cp", wa, wb, False), ("di", a_d, b_d, True)):
            obs, null, mu, p = oe_with_null(a, b, lab, pw, N_PERM_WIN, directed)
            oe = obs / np.maximum(mu, 1e-9)
            for i in range(K):
                for j in range(i, K):
                    o_ = obs[i, j] + (obs[j, i] if directed and i != j else 0)
                    e_ = mu[i, j] + (mu[j, i] if directed and i != j else 0)
                    wrows.append(dict(net=key, ventana=f"{y0}-{y1}", a=i, b=j, obs=int(o_), exp=round(float(e_), 2),
                                      oe=round(float(o_ / e_), 3) if e_ > 0 else None, n_a=int(present[i]), n_b=int(present[j])))
        print("ventana", y0, flush=True)
    W = pd.DataFrame(wrows)
    W.to_csv(os.path.join(OUT, "o14_windows.csv"), index=False)
    trends = []
    for (net, i, j), g in W[W.a != W.b].groupby(["net", "a", "b"]):
        g = g[(g.n_a >= 30) & (g.n_b >= 30) & (g.exp >= 5)]
        if len(g) >= 6:
            x = np.arange(len(g)); yv = np.log2(np.maximum(g.oe.values.astype(float), 1e-3))
            trends.append(dict(net=net, a=int(i), b=int(j), n=len(g), rho=round(float(spearmanr(x, yv)[0]), 3),
                               slope=round(float(np.polyfit(x, yv, 1)[0]), 4), first=round(float(g.oe.iloc[0]), 3), last=round(float(g.oe.iloc[-1]), 3),
                               first_w=g.ventana.iloc[0], last_w=g.ventana.iloc[-1]))
    R["trends"] = sorted(trends, key=lambda t: -abs(t["slope"]))
    R["windows"] = W.to_dict("records")

    # ---------- papers puente: coeficiente de participación
    A = csr_matrix((np.ones(len(da) * 2 + len(ca) * 2), (np.r_[da, db, ca, cb], np.r_[db, da, cb, ca])), shape=(n, n))
    A.data[:] = 1
    k = np.asarray(A.sum(1)).ravel()
    onehot = lambda L: csr_matrix((np.ones(n), (np.arange(n), L)), shape=(n, K))
    kt = (A @ onehot(lab)).toarray()
    part = 1 - ((kt / np.maximum(k, 1)[:, None]) ** 2).sum(1)
    outside = 1 - kt[np.arange(n), lab] / np.maximum(k, 1)
    ok = k >= 8
    nullP = []
    for _ in range(200):
        L2 = perm(lab); kt2 = (A @ onehot(L2)).toarray()
        p2 = 1 - ((kt2 / np.maximum(k, 1)[:, None]) ** 2).sum(1)
        nullP.append([p2[ok & (L2 == t)].mean() for t in range(K)])
    nullP = np.array(nullP)
    R["participation"] = [dict(topic=t, mean=round(float(part[ok & (lab == t)].mean()), 3), null=round(float(nullP[:, t].mean()), 3),
                               n=int((ok & (lab == t)).sum())) for t in range(K)]
    rows = []
    for i in np.where(k >= 15)[0]:
        share = kt[i] / k[i]
        top = np.argsort(-share)[:3]
        rows.append(dict(id=int(i), title=P[i]["t"], year=P[i]["y"], topic=int(lab[i]), k=int(k[i]), participation=round(float(part[i]), 3),
                         outside=round(float(outside[i]), 3), mix=json.dumps({int(t): round(float(share[t]), 2) for t in top if share[t] > 0})))
    B = pd.DataFrame(rows).sort_values(["participation", "k"], ascending=False)
    B.to_csv(os.path.join(OUT, "o14_bridges.csv"), index=False)
    R["bridges"] = B.head(25).to_dict("records")
    print(B.head(10)[["year", "topic", "k", "participation", "title"]].to_string(), flush=True)

    # ---------- selección manual, como comparación (temas y red propios)
    mm = json.load(open(os.path.join(ROOT, "data", "map.json"), encoding="utf-8"))
    Vm = np.load(os.path.join(os.path.dirname(OA), "out", "val_acoplamiento.npz"))
    em = {"di": mm["edges"]["di"], "cp": [v for x, y in zip(Vm["a"].tolist(), Vm["b"].tolist()) for v in (x, y, 1, 1)]}
    Em = np.load(os.path.join(ROOT, "data", "embeddings.npy")).astype(np.float64)
    Rm, *_ = analyse(mm["papers"], em, Em, [dict(es=c["name"], en=c.get("name_en", c["name"])) for c in mm["clusters"]], rng, N_PERM, "manual")
    R["manual"] = Rm
    json.dump(R, open(os.path.join(OUT, "o14_dialogue.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
    print("listo")


if __name__ == "__main__":
    main()
