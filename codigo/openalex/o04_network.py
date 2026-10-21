"""Paso o04: red de acoplamiento validada del corpus sistemático (completa y por ventanas de 3 años).
Bipartita paper x referencia (referenced_works de OpenAlex). Índices = posición en out/works.json.
Salidas: out/bip_refs.npz, out/val_full.npz (a, b, s, pv, tedge), out/val_win_<y0>.npz, out/o04_summary.json"""
import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix, save_npz

import citnet_big as cb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
WINDOWS = [(y, y + 2) for y in range(2010, 2025)]   # 2010-12 ... 2024-26 (2026 incompleto: hasta octubre)


def build():
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    refs = sorted({r for w in W for r in w["refs"]})
    col = {r: j for j, r in enumerate(refs)}
    rows = [i for i, w in enumerate(W) for r in w["refs"]]
    cols = [col[r] for w in W for r in w["refs"]]
    R = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(W), len(refs)))
    t = np.array([w["t"] if w["t"] is not None else np.nan for w in W])
    return W, R, t


def net(R, sel):
    """Red validada sobre la submuestra `sel` (índices globales). Devuelve índices globales."""
    Bs = R[sel]
    Bs = Bs[:, np.asarray(Bs.sum(0)).ravel() > 0]
    k = np.asarray(Bs.sum(1)).ravel()
    p, hc, err = cb.fit_bicm(Bs)
    a, b, s, pv = cb.validated(Bs, p, hc, k > 0)
    return sel[a], sel[b], s, pv, int((k > 0).sum()), float(err)


def main():
    W, R, t = build()
    save_npz(os.path.join(OUT, "bip_refs.npz"), R)
    yr = np.floor(t)
    k = np.asarray(R.sum(1)).ravel()
    summ = dict(n=len(W), with_refs=int((k > 0).sum()), n_refs=R.shape[1], links=int(R.nnz), windows=[])
    t0 = time.time()
    sel = np.where(~np.isnan(t))[0]
    a, b, s, pv, na, err = net(R, sel)
    np.savez(os.path.join(OUT, "val_full.npz"), a=a, b=b, s=s, pv=pv, tedge=np.maximum(t[a], t[b]))
    summ["full"] = dict(active=na, validated=int(len(a)), bicm_err=err, secs=round(time.time() - t0))
    print("completa:", summ["full"], flush=True)
    for y0, y1 in WINDOWS:
        sel = np.where((yr >= y0) & (yr <= y1))[0]
        a, b, s, pv, na, err = net(R, sel)
        np.savez(os.path.join(OUT, f"val_win_{y0}.npz"), a=a, b=b, s=s, pv=pv, tedge=np.maximum(t[a], t[b]))
        summ["windows"].append(dict(ventana=f"{y0}-{y1}", papers=int(len(sel)), active=na, validated=int(len(a)), bicm_err=err))
        print(summ["windows"][-1], flush=True)
    json.dump(summ, open(os.path.join(OUT, "o04_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
