"""Paso o15: figuras del diálogo entre temas (o14) -> homologia/figs/dialogo_*.png (fondo oscuro, 300 DPI, verticales).
Colores de tema = los del atlas en tema oscuro."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "homologia", "scripts"))
from common import DARK, FIGS, dark_fig  # noqa: E402

REF = "#4a5361"  # barras de referencia (esperado por azar): visibles sobre el panel oscuro


def save_fig(fig, name):
    """Como common.save_fig, pero el recorte se ajusta a todo lo dibujado: ningún título ni etiqueta queda fuera."""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

TC = ["#0095d9", "#ed8840", "#1fc08f", "#e8b923", "#e58bbe", "#7ccbf2", "#9ba5ff", "#fe576f"]
SHORT = ["COVID / diagnóstico", "XMRV", "TCC / PACE", "Medicina china", "Ejercicio / autonómico", "Dolor / fibromialgia",
         "Inmunidad / mitocondria", "Revisiones / fatiga"]


def heat(ax, M, Q, title, directed, vmax=3.0, diag=True):
    import matplotlib.colors as mcolors
    L = np.log2(np.maximum(np.array(M, float), 1e-3))
    if not diag:
        L = L.copy(); np.fill_diagonal(L, np.nan)
    cmap = mcolors.LinearSegmentedColormap.from_list("d", ["#3b6fd8", "#1b2533", "#e0603a"])
    cmap.set_bad(DARK["panel"])
    im = ax.imshow(L, cmap=cmap, vmin=-vmax, vmax=vmax)
    for i in range(8):
        for j in range(8):
            if not diag and i == j:
                continue
            v = M[i][j]; sig = Q[i][j] < 0.05
            ax.text(j, i, f"{v:.1f}" if v < 9.95 else f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                    color=DARK["fg"] if sig else DARK["muted"], fontweight="bold" if sig else "normal")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(SHORT, rotation=55, ha="right", fontsize=6.5); ax.set_yticklabels(SHORT, fontsize=6.5)
    for t, c in zip(ax.get_xticklabels(), TC): t.set_color(c)
    for t, c in zip(ax.get_yticklabels(), TC): t.set_color(c)
    ax.set_title(title, fontsize=9)
    if directed:
        ax.set_ylabel("cita →", fontsize=7); ax.set_xlabel("citado", fontsize=7)
    return im


def main():
    R = json.load(open(os.path.join(OUT, "o14_dialogue.json"), encoding="utf-8"))
    m = json.load(open(os.path.join(ROOT, "viewer", "oa", "oa_map.json"), encoding="utf-8"))
    P = m["papers"]

    # 1. O/E total (con la diagonal: cuánto se cita cada tema a sí mismo)
    fig, ax = dark_fig(2, height=14.5, width=5.6)
    heat(ax[0], R["cp"]["oe"], R["cp"]["q"], "Acoplamiento: O/E frente al azar (negrita: q < 0,05)", False, vmax=3.3)
    im = heat(ax[1], R["di"]["oe"], R["di"]["q"], "Citas directas: O/E frente al azar", True, vmax=3.3)
    cb = fig.colorbar(im, cax=ax[1].inset_axes([0.0, -0.50, 1.0, 0.035]), orientation="horizontal", ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.ax.set_xticklabels(["⅛", "¼", "½", "1", "2", "4", "8"]); cb.ax.tick_params(colors=DARK["muted"], labelsize=7)
    cb.set_label("observado / esperado (escala log)", color=DARK["muted"], fontsize=7)
    fig.suptitle("Cada tema dialoga sobre todo consigo mismo", color=DARK["fg"], fontsize=10.5, fontweight="bold")
    save_fig(fig, "dialogo_1_total.png")

    # 2. solo vínculos entre temas (cuasi-independencia)
    fig, ax = dark_fig(2, height=14.5, width=5.6)
    heat(ax[0], R["cp_ext"]["oe"], R["cp_ext"]["q"], "Acoplamiento entre temas: ¿con quién comparten literatura?", False, vmax=1.6, diag=False)
    im = heat(ax[1], R["di_ext"]["oe"], R["di_ext"]["q"], "Citas directas entre temas: ¿a quién citan fuera de su tema?", True, vmax=1.6, diag=False)
    cb = fig.colorbar(im, cax=ax[1].inset_axes([0.0, -0.50, 1.0, 0.035]), orientation="horizontal", ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cb.ax.set_xticklabels(["0,35", "0,5", "0,7", "1", "1,4", "2", "2,8"]); cb.ax.tick_params(colors=DARK["muted"], labelsize=7)
    cb.set_label("observado / esperado si cada tema repartiera sus vínculos externos al azar", color=DARK["muted"], fontsize=7)
    fig.suptitle("Cuando salen de su tema, ¿con quién hablan?", color=DARK["fg"], fontsize=10.5, fontweight="bold")
    save_fig(fig, "dialogo_2_entre_temas.png")

    # 3. red de temas: posición = centro del tema en el mapa 2D de contenido
    xy = np.array([p["p2"] for p in P]); lab = np.array([p["c"] for p in P])
    cen = np.array([xy[lab == t].mean(0) for t in range(8)])
    oe = np.array(R["cp_ext"]["oe"]); q = np.array(R["cp_ext"]["q"]); n = np.array(R["n_topic"])
    fig, ax = dark_fig(1, height=7.8, width=5.6)
    for i in range(8):
        for j in range(i + 1, 8):
            if q[i, j] >= 0.05:
                continue
            v = np.log2(oe[i, j])
            if v > 0:
                ax.plot(*zip(cen[i], cen[j]), color="#e0603a", lw=1 + 7 * v, alpha=0.75, zorder=1, solid_capstyle="round")
            elif v < -0.75:
                ax.plot(*zip(cen[i], cen[j]), color="#3b6fd8", lw=1.4, ls=(0, (3, 3)), alpha=0.9, zorder=1)
    ax.scatter(cen[:, 0], cen[:, 1], s=n / 2.2, c=TC, zorder=2, edgecolors=DARK["bg"], linewidths=1.5)
    for t in range(8):
        ax.annotate(SHORT[t], cen[t], xytext=(0, -np.sqrt(n[t] / 2.2) / 2 - 9), textcoords="offset points", ha="center",
                    fontsize=7.2, color=TC[t], zorder=3, bbox=dict(boxstyle="round,pad=0.25", fc=DARK["panel"], ec="none", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    pad = 0.8
    ax.set_xlim(cen[:, 0].min() - pad, cen[:, 0].max() + pad); ax.set_ylim(cen[:, 1].min() - pad, cen[:, 1].max() + pad)
    ax.set_title("Red de diálogo por acoplamiento (entre temas)\nnaranja: más de lo esperado · azul discontinuo: mucho menos (O/E < 0,6)\n"
                 "posición = centro del tema en el mapa de contenido · tamaño = n.º de trabajos", fontsize=8)
    save_fig(fig, "dialogo_3_red.png")

    # 4. apertura y balance
    fig, ax = dark_fig(2, height=10, width=5.6)
    order = np.argsort(R["cp"]["ei"])
    y = np.arange(8)
    ax[0].barh(y, np.array(R["cp"]["ei_null"])[order], color=REF, height=0.75, label="esperado por azar")
    ax[0].barh(y, np.array(R["cp"]["ei"])[order], color=[TC[t] for t in order], height=0.42, label="observado")
    ax[0].set_yticks(y); ax[0].set_yticklabels([SHORT[t] for t in order], fontsize=7.5)
    ax[0].axvline(0, color=DARK["muted"], lw=0.8); ax[0].set_xlim(-1, 1)
    ax[0].set_title("Apertura (índice E–I del acoplamiento)\n−1 = solo dentro del tema · +1 = solo con otros temas", fontsize=8.5)
    ax[0].legend(fontsize=7, loc="upper left")
    bal = np.array(R["flow"]["balance"]) - np.array(R["flow"]["balance_null"])
    order = np.argsort(bal)
    ax[1].barh(y, bal[order], color=[TC[t] for t in order], height=0.6)
    ax[1].set_yticks(y); ax[1].set_yticklabels([SHORT[t] for t in order], fontsize=7.5)
    ax[1].axvline(0, color=DARK["muted"], lw=0.8)
    ax[1].set_title("¿Fuente o receptor?\ncitas recibidas de otros temas / citas hechas a otros temas\n"
                    "(log₂, menos lo esperado por azar · > 0: lo citan más de lo que cita)", fontsize=8.5)
    save_fig(fig, "dialogo_4_apertura_balance.png")

    # 5. contenido frente a citas
    fig, ax = dark_fig(1, height=7.4, width=5.6)
    xm = R["fit2_oe_cp"]["excluded"]
    rows = [r for r in R["content"] if xm not in (r["a"], r["b"])]
    ranked = sorted(rows, key=lambda r: r["res2_oe_cp"])
    neg, pos = ranked[:3], ranked[-3:][::-1]
    for r in R["content"]:
        skip = xm in (r["a"], r["b"])
        col = DARK["muted"] if skip else "#e0603a" if r in pos else "#3b6fd8" if r in neg else DARK["fg"]
        ax.scatter(r["sim"], np.log2(r["oe_cp"]), s=30, color=col, alpha=0.45 if skip else 0.95, zorder=2)
    # etiquetas de los 3 pares con más y menos diálogo del que predice su contenido, en columnas a los lados
    for k, r in enumerate(pos):
        ax.annotate(f"+ {SHORT[r['a']]} – {SHORT[r['b']]}", (r["sim"], np.log2(r["oe_cp"])), xytext=(0.03, 0.95 - 0.06 * k),
                    textcoords="axes fraction", fontsize=6.6, color="#e0603a", arrowprops=dict(arrowstyle="-", color="#e0603a", lw=0.6))
    for k, r in enumerate(neg):
        ax.annotate(f"− {SHORT[r['a']]} – {SHORT[r['b']]}", (r["sim"], np.log2(r["oe_cp"])), xytext=(0.40, 0.30 - 0.06 * k),
                    textcoords="axes fraction", fontsize=6.6, color="#7aa2f7", arrowprops=dict(arrowstyle="-", color="#3b6fd8", lw=0.6))
    f = R["fit2_oe_cp"]; xs = np.linspace(min(r["sim"] for r in rows), max(r["sim"] for r in rows), 10)
    ax.plot(xs, f["intercept"] + f["slope"] * xs, color=DARK["muted"], lw=1, ls="--", zorder=1)
    ax.set_xlabel("similitud de contenido entre los dos temas (coseno SPECTER2 medio)", fontsize=7.5)
    ax.set_ylabel("log₂ O/E del acoplamiento", fontsize=7.5)
    ax.set_title(f"El contenido predice buena parte del diálogo (r = {f['r']:.2f})".replace(".", ",") + "\n"
                 "naranja: más diálogo del que predice su contenido · azul: menos\ngris: pares con XMRV (fuera del ajuste)", fontsize=8)
    ax.grid(alpha=0.25)
    save_fig(fig, "dialogo_5_contenido.png")

    # 6. evolución de los pares que más cambian
    W = R["windows"]
    tr = [t for t in R["trends"] if t["net"] == "cp"][:6]
    fig, ax = dark_fig(3, height=11, width=5.6)
    fig.axes[0].figure.clf()
    axs = fig.subplots(3, 2, sharex=True)
    for k, t in enumerate(tr):
        a_ = axs.flat[k]
        rows = [w for w in W if w["net"] == "cp" and w["a"] == t["a"] and w["b"] == t["b"] and w["oe"] is not None
                and w["n_a"] >= 30 and w["n_b"] >= 30 and w["exp"] >= 5]
        x = [int(w["ventana"][:4]) + 1 for w in rows]; yv = [w["oe"] for w in rows]
        a_.plot(x, yv, color=TC[t["a"]], lw=2, marker="o", ms=2.5)
        a_.axhline(1, color=DARK["muted"], lw=0.7, ls=":")
        a_.set_title(f"{SHORT[t['a']]}\n– {SHORT[t['b']]}", fontsize=7)
        a_.tick_params(labelsize=6.5); a_.set_facecolor(DARK["panel"]); a_.grid(alpha=0.25)
        for s_ in a_.spines.values(): s_.set_color(DARK["grid"])
    fig.suptitle("Pares cuyo acoplamiento más cambia (O/E por ventana de 3 años)", color=DARK["fg"], fontsize=9.5, fontweight="bold")
    save_fig(fig, "dialogo_6_evolucion.png")

    # 7. participación
    fig, ax = dark_fig(1, height=5.2, width=5.6)
    pr = R["participation"]; order = np.argsort([x["mean"] for x in pr]); y = np.arange(8)
    ax.barh(y, [pr[t]["null"] for t in order], color=REF, height=0.75, label="esperado por azar")
    ax.barh(y, [pr[t]["mean"] for t in order], color=[TC[t] for t in order], height=0.42, label="observado")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in order], fontsize=7.5); ax.set_xlim(0, 1)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Papers puente: participación media por tema\n0 = todos sus vínculos en un tema · 0,875 = repartidos\npor igual entre los 8",
                 fontsize=8.5)
    save_fig(fig, "dialogo_7_puentes.png")
    print("ok")


if __name__ == "__main__":
    main()
