"""Paso o09: figura explicativa del corpus sistemático (vertical, fondo oscuro, 300 DPI) -> homologia/figs/o09_sistematico.png"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from common import CLUSTER_COLORS, DARK, dark_fig, save_fig  # noqa: E402

C1, C2, C3, C4 = CLUSTER_COLORS[0], CLUSTER_COLORS[1], CLUSTER_COLORS[2], CLUSTER_COLORS[3]


def main():
    y = pd.read_csv(os.path.join(OUT, "o06_years.csv"))
    w = pd.read_csv(os.path.join(OUT, "o06_windows.csv"))
    ph = json.load(open(os.path.join(OUT, "o05_summary.json")))
    ep = json.load(open(os.path.join(OUT, "o07_summary.json")))
    fig, ax = dark_fig(4, height=15.5, width=5.8)
    mid = w.ventana.str[:4].astype(int) + 1

    a = ax[0]
    a.plot(y.year, y.cohesion, color=C1, lw=2, marker="o", ms=4, label="todos")
    a.plot(y.year, y.cohesion_core, color=C3, lw=1.6, ls="--", label="EM/SFC en el título")
    a.axvspan(2009.5, 2011.5, color=C2, alpha=0.18)
    a.text(2009.7, 0.05, "XMRV", color=C2, fontsize=8)
    a.axvspan(2019.5, 2026.5, color=C4, alpha=0.10)
    a.text(2019.7, 0.05, "COVID persistente", color=C4, fontsize=8)
    a.set_ylim(0, 0.5)
    a.set_title("1. Cohesión de la literatura citada, por año")
    a.set_ylabel("coseno de perfiles de referencias")
    a.legend(loc="upper right", fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1]
    a.plot(mid, w.mixing_ratio, color=C1, lw=2, marker="o", ms=4, label="mezcla entre temas")
    a.fill_between(mid, w.mixing_ratio_sub - 2 * w.mixing_ratio_sub_sd, w.mixing_ratio_sub + 2 * w.mixing_ratio_sub_sd,
                   color=C1, alpha=0.25, label="control, mismo tamaño (±2 sd)")
    a.set_ylabel("vínculos entre temas / azar", color=C1)
    b = a.twinx()
    b.plot(mid, w.mantel_rho, color=C2, lw=2, marker="s", ms=4)
    b.fill_between(mid, w.mantel_rho_sub - 2 * w.mantel_rho_sub_sd, w.mantel_rho_sub + 2 * w.mantel_rho_sub_sd, color=C2, alpha=0.25)
    b.set_ylabel("contenido ↔ citas (Spearman)", color=C2)
    b.tick_params(axis="y", colors=C2)
    a.tick_params(axis="y", colors=C1)
    a.set_title("2. Convergencia gradual (ventanas de 3 años)")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.3)

    a = ax[2]
    c = w.dropna(subset=["continuity_jaccard"])
    a.plot(c.ventana.str[:4].astype(int) + 1.5, c.continuity_jaccard, color=C1, lw=2, marker="o", ms=4, label="con todos los papers")
    a.axvline(2020.5, color=C4, lw=1, ls=":")
    a.annotate("entran los papers de COVID\npersistente: aparecen vínculos nuevos\nentre papers antiguos", xy=(2020.5, 0.75), xytext=(2011.3, 0.62),
               color=C4, fontsize=7.5, arrowprops=dict(arrowstyle="->", color=C4))
    nc = [(int(k.split("->")[1][:4]) + 1.5, v["jaccard_sin_covid"]) for k, v in ep.items() if "->" in k]  # de o07
    a.scatter([x for x, _ in nc], [y for _, y in nc], color=C3, zorder=5, label="sin papers de COVID")
    a.set_ylim(0.55, 1.02)
    a.set_title("3. Continuidad entre ventanas consecutivas")
    a.set_ylabel("Jaccard de vínculos (papers compartidos)")
    a.legend(loc="lower left", fontsize=8)
    a.grid(alpha=0.3)

    a = ax[3]
    keys = [k for k in ph if k.startswith("win")]
    xs = np.arange(len(keys))
    real = [ph[k]["n_bars"] for k in keys]
    nul = [ph[k]["null_nbars_median"] for k in keys]
    a.bar(xs - 0.2, nul, 0.4, color=DARK["muted"], label="red recableada al azar (mediana)")
    a.bar(xs + 0.2, real, 0.4, color=C1, label="red real")
    a.set_xticks(xs)
    a.set_xticklabels([ph[k]["ventana"][2:4] + "–" + ph[k]["ventana"][7:9] for k in keys], rotation=60, fontsize=7)
    a.set_title("4. Huecos (H1): ninguno significativo")
    a.set_ylabel("anillos")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.3, axis="y")
    save_fig(fig, "o09_sistematico.png")


if __name__ == "__main__":
    main()
