"""Paso o08: el episodio XMRV (2010-11) como momento de convergencia y su fin.
- Bloque XMRV = las 30 referencias más sobrerrepresentadas en 2010-11 frente a 2013-16 (o07), que son todas de XMRV.
- Test de permutación: la caída de cohesión 2010-11 -> 2012-13 frente a permutar los años entre esos papers.
- ¿Desaparece el pico de cohesión al quitar el bloque XMRV de las referencias?
- Red de la ventana 2010-12: vínculos que dependen de papers que citan el bloque XMRV, y la red sin ellos.
- Cruce de temas del bloque: ¿qué temas (centroides del atlas) cita XMRV? ¿Se mezcla virología con clínica?
Salidas: out/o08_summary.json"""
import json
import os

import numpy as np

import citnet_big as cb
from o04_network import OUT, build

import citnet as cn  # noqa: E402

SEED = 20261019
N = 24
REPS = 2000


def main():
    W, R, t = build()
    yr = np.floor(t)
    tp = np.load(os.path.join(OUT, "topics.npy"))
    rng = np.random.default_rng(SEED + 8)
    k = np.asarray(R.sum(1)).ravel()
    A = (yr >= 2010) & (yr <= 2011) & (k > 0)
    B = (yr >= 2013) & (yr <= 2016) & (k > 0)
    fa, fb = np.asarray(R[A].sum(0)).ravel() / A.sum(), np.asarray(R[B].sum(0)).ravel() / B.sum()
    xm = np.argsort(-(fa - fb))[:30]
    keep = np.ones(R.shape[1], bool)
    keep[xm] = False
    Rf = R.astype(np.float64).tocsr()
    Rx = Rf[:, np.where(keep)[0]]
    nx = np.asarray(Rf[:, xm].sum(1)).ravel()     # cuántas referencias del bloque cita cada paper
    S = dict(papers_citan_bloque_ge3=int((nx >= 3).sum()), por_año={int(y): int(((yr == y) & (nx >= 3)).sum()) for y in range(2008, 2016)})

    def coh(M, idx, reps=400):
        v = []
        for _ in range(reps):
            p = rng.permutation(idx)
            a, b = np.asarray(M[p[:N]].sum(0)).ravel(), np.asarray(M[p[N:2 * N]].sum(0)).ravel()
            v.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return float(np.mean(v))

    g1 = np.where((yr >= 2010) & (yr <= 2011) & (k > 0))[0]
    g2 = np.where((yr >= 2012) & (yr <= 2013) & (k > 0))[0]
    obs = coh(Rf, g1) - coh(Rf, g2)
    pool = np.concatenate([g1, g2])
    null = []
    for _ in range(300):
        pp = rng.permutation(pool)
        null.append(coh(Rf, pp[:len(g1)], 50) - coh(Rf, pp[len(g1):], 50))
    S["cohesion"] = dict(c_2010_11=round(coh(Rf, g1), 3), c_2012_13=round(coh(Rf, g2), 3), caida=round(obs, 3),
                         nulo_media=round(float(np.mean(null)), 3), nulo_p95=round(float(np.quantile(null, 0.95)), 3),
                         p=round(float((1 + sum(x >= obs for x in null)) / 301), 4),
                         sin_bloque_2010_11=round(coh(Rx, g1), 3), sin_bloque_2012_13=round(coh(Rx, g2), 3))
    print(S["cohesion"], flush=True)

    # red 2010-12 con y sin los papers del bloque
    sel = np.where((yr >= 2010) & (yr <= 2012))[0]

    def net(sel):
        Bs = R[sel]
        Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
        kk = np.asarray(Bs.sum(1)).ravel()
        p, hc, _ = cn.fit_bicm(Bs)
        a, b, s, pv = cb.validated(Bs, p, hc, kk > 0)
        return sel[a], sel[b]
    a, b = net(sel)
    inx = nx >= 3
    both, one = int((inx[a] & inx[b]).sum()), int((inx[a] ^ inx[b]).sum())
    a2, b2 = net(sel[~inx[sel]])
    S["red_2010_12"] = dict(vinculos=len(a), ambos_bloque=both, uno_bloque=one, ninguno=int(len(a) - both - one),
                            papers_bloque=int(inx[sel].sum()), vinculos_sin_papers_bloque=len(a2))
    # mezcla de temas del bloque: temas de los papers que lo citan
    S["temas_bloque"] = {int(c): int(((tp == c) & inx).sum()) for c in np.unique(tp[inx])}
    S["temas_todos_2010_12"] = {int(c): int(((tp == c) & (yr >= 2010) & (yr <= 2012)).sum()) for c in np.unique(tp)}
    print(S["red_2010_12"], S["temas_bloque"], flush=True)
    json.dump(S, open(os.path.join(OUT, "o08_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
