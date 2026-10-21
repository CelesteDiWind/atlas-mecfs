"""Paso 15: homología de las redes de citas validadas (fase 1, piloto con nulos completos).
Filtraciones:
    - fuerza: los vínculos entran por orden de significación (rango normalizado 0-1); H1 = anillo de papers unidos
      de dos en dos por vínculos fuertes sin vínculos fuertes que lo crucen
    - fecha: los papers entran en su fecha de publicación y los vínculos en la suya (acoplamiento: el paper más
      reciente del par; cocitación: el segundo citante común); H1 = laguna que se abre y que un paper posterior cierra
Nulos (200 réplicas, criterio: persistencia frente al máximo de cada réplica):
    - curveball: recableado de la bipartita conservando los dos grados, reproyección y los m pares más significativos
      (m = vínculos validados de la red real): misma densidad y los triángulos que crea toda proyección
    - pesos: mismo grafo, orden de significación permutado (solo filtración por fuerza)
    - fechas: mismo grafo, fechas permutadas entre papers, en todo el corpus y dentro de cada tema (solo por fecha)
Además: filtración por fuerza en cada ventana de 3 años con nulo curveball (100 réplicas).
Reanudable: out/h15_null_<clave>.npy. Salidas: out/h15_bars_<red>_<filtración>.csv, out/h15_summary.json"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import poisson

import citnet as cn
from common import SEED, load_papers, out
from h14_compare import WINDOWS

N_NULL = 200
N_NULL_WIN = 100


def top_m_pairs(S, mu, active, m):
    """Los m pares más significativos (Poisson) de una proyección: sirve para reales y nulos por igual."""
    iu = np.triu_indices(len(S), 1)
    s = S[iu]
    keep = (s >= 1) & active[iu[0]] & active[iu[1]]
    a, b, s = iu[0][keep], iu[1][keep], s[keep]
    pv = poisson.sf(s - 1, mu[a, b])
    o = np.argsort(pv, kind="stable")[:m]
    return a[o], b[o], s[o], pv[o]


def bars_of(M):
    _, (h1,) = cn.h1_bars(M)
    return h1


def maxpers(h1):
    return float((h1[:, 1] - h1[:, 0]).max()) if len(h1) else 0.0


def run_null(key, fn, reps):
    path = out(f"h15_null_{key}.npy")
    res = list(np.load(path)) if os.path.exists(path) else []
    t0 = time.time()
    while len(res) < reps:
        res.append(fn())
        if len(res) % 20 == 0 or len(res) == reps:
            np.save(path, np.array(res))
            print(f"    {key}: {len(res)}/{reps} ({time.time() - t0:.0f} s)", flush=True)
    return np.array(res)


def summarize(h1, null, label):
    pers = h1[:, 1] - h1[:, 0] if len(h1) else np.array([])
    thr = float(np.quantile(null[:, 0], 0.95))
    pval = np.array([(1 + (null[:, 0] >= p).sum()) / (1 + len(null)) for p in pers])
    return dict(filtracion=label, n_bars=int(len(h1)), max_pers=round(float(pers.max()), 4) if len(pers) else 0,
                thr95=round(thr, 4), n_sig=int((pval < 0.05).sum()), min_p=round(float(pval.min()), 4) if len(pval) else 1.0,
                null_nbars_median=float(np.median(null[:, 1])), q_nbars=round(float((null[:, 1] < len(h1)).mean()), 3)), pval


def main():
    J = load_papers()
    P = J["papers"]
    n = len(P)
    cl = np.array([p["c"] for p in P])
    D = np.load(out("cit_dates.npz"))
    pdate, cdate = D["paper"], D["citer"]
    yr = np.floor(pdate)
    rng = np.random.default_rng(SEED + 15)
    summary = {}
    nets = {"acoplamiento": load_npz(out("bip_refs.npz")).astype(np.int32),
            "cocitacion": load_npz(out("bip_citers.npz")).astype(np.int32)}
    for name, B in nets.items():
        V = np.load(out(f"val_{name}.npz"))
        a, b, s, pv, te = V["a"], V["b"], V["s"], V["pv"], V["tedge"]
        m = len(a)
        k = np.asarray(B.sum(1)).ravel()
        p, hc, _ = cn.fit_bicm(B)
        mu = cn.expected_shared(p, hc)
        rows0 = cn.rows_of(B)

        def curveball_net():
            Bn = cn.from_rows(cn.curveball(rows0, rng), B.shape)
            return top_m_pairs(cn.project(Bn), mu, k > 0, m), Bn

        # mezcla del Curveball: fracción de enlaces cambiados en una réplica
        Bt = cn.from_rows(cn.curveball(rows0, rng), B.shape)
        mix = 1 - B.multiply(Bt).sum() / B.sum()
        print(f"{name}: m={m}; curveball cambia {mix:.1%} de los enlaces", flush=True)

        # ---------- filtración por fuerza
        h1 = bars_of(cn.rank_filtration(a, b, pv, n))

        def null_cb_rank():
            (na, nb, _, npv), _ = curveball_net()
            h = bars_of(cn.rank_filtration(na, nb, npv, n))
            return [maxpers(h), len(h)]

        def null_w_rank():
            h = bars_of(cn.rank_filtration(a, b, rng.permutation(pv), n))
            return [maxpers(h), len(h)]

        res = {}
        for nk, fn in (("curveball", null_cb_rank), ("pesos", null_w_rank)):
            nul = run_null(f"{name}_fuerza_{nk}", fn, N_NULL)
            res[nk], pval = summarize(h1, nul, "fuerza")
            if nk == "curveball":
                pd.DataFrame({"birth": h1[:, 0], "death": h1[:, 1], "pers": h1[:, 1] - h1[:, 0], "p_curveball": pval}).sort_values(
                    "pers", ascending=False).to_csv(out(f"h15_bars_{name}_fuerza.csv"), index=False)
        summary[f"{name}_fuerza"] = res
        print(name, "fuerza", res, flush=True)

        # ---------- filtración por fecha
        nodes = np.where(~np.isnan(pdate))[0]
        tv = pdate.copy()
        h1t = bars_of(cn.time_filtration(a, b, te, tv, nodes))

        def null_dates(within):
            t2 = tv.copy()
            if within:
                for c in np.unique(cl[nodes]):
                    idx = nodes[cl[nodes] == c]
                    t2[idx] = rng.permutation(tv[idx])
            else:
                t2[nodes] = rng.permutation(tv[nodes])
            te2 = np.maximum(t2[a], t2[b]) if name == "acoplamiento" else te
            h = bars_of(cn.time_filtration(a, b, te2, t2, nodes))
            return [maxpers(h), len(h)]

        def null_cb_time():
            (na, nb, _, _), Bn = curveball_net()
            if name == "acoplamiento":
                te2 = np.maximum(tv[na], tv[nb])
            else:
                from h14_compare import second_date
                te2 = second_date(Bn.tocsc(), cdate, na, nb)
            h = bars_of(cn.time_filtration(na, nb, te2, tv, nodes))
            return [maxpers(h), len(h)]

        res = {}
        for nk, fn in (("fechas", lambda: null_dates(False)), ("fechas_tema", lambda: null_dates(True)), ("curveball", null_cb_time)):
            nul = run_null(f"{name}_fecha_{nk}", fn, N_NULL)
            res[nk], pval = summarize(h1t, nul, "fecha")
            if nk == "fechas":
                pd.DataFrame({"birth": h1t[:, 0] + 1900, "death": h1t[:, 1] + 1900, "years": h1t[:, 1] - h1t[:, 0],
                              "p_fechas": pval}).sort_values("years", ascending=False).to_csv(out(f"h15_bars_{name}_fecha.csv"), index=False)
        summary[f"{name}_fecha"] = res
        print(name, "fecha", res, flush=True)

        # ---------- ventanas (filtración por fuerza, nulo curveball)
        wres = []
        for y0, y1 in WINDOWS:
            sel = np.where((yr >= y0) & (yr <= y1))[0]
            Bs = B[sel]
            Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
            ks = np.asarray(Bs.sum(1)).ravel()
            ps, hs, _ = cn.fit_bicm(Bs)
            mus = cn.expected_shared(ps, hs)
            wa, wb, _, wpv = cn.validated(cn.project(Bs), mus, ks > 0)
            if len(wa) < 10:
                wres.append(dict(ventana=f"{y0}-{y1}", validated=int(len(wa))))
                continue
            hw = bars_of(cn.rank_filtration(wa, wb, wpv, len(sel)))
            rws = cn.rows_of(Bs)

            def null_win():
                Bn = cn.from_rows(cn.curveball(rws, rng), Bs.shape)
                na, nb, _, npv = top_m_pairs(cn.project(Bn), mus, ks > 0, len(wa))
                h = bars_of(cn.rank_filtration(na, nb, npv, len(sel)))
                return [maxpers(h), len(h)]

            nul = run_null(f"{name}_win{y0}", null_win, N_NULL_WIN)
            r, _ = summarize(hw, nul, "fuerza")
            r.update(ventana=f"{y0}-{y1}", validated=int(len(wa)))
            wres.append(r)
            print("   ", name, r, flush=True)
        summary[f"{name}_ventanas"] = wres
        json.dump(summary, open(out("h15_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
