"""Paso 12: gráficos explicativos de la red de citas (cocitación + acoplamiento) -> figs/explicativas_citas/*.png
Mismo formato y maquetación que h11_explain.py (se reutiliza su clase Page); cada título es la respuesta a una pregunta.
Usa lo ya calculado en h08 (out/citation_summary.json, h1_bars_citation.csv, null_citation_config.npy,
citation_communities.npy); lo único nuevo que calcula es la mezcla de temas entre vecinos de citas (permutación)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

import h11_explain as X
from common import CLUSTER_COLORS, DARK, FIGS, SEED, load_papers, out
from h08_citations import weights
from h11_explain import M, CW, W, Page, axes_in, dec, pct, text_h, topic_list, wrap

X.DST = os.path.join(FIGS, "explicativas_citas")
os.makedirs(X.DST, exist_ok=True)
K = 6  # vecinos de citas por paper (igual que los 6 vecinos semánticos del gráfico 01 de las explicativas)


def main():
    J = load_papers()
    P, CL = J["papers"], J["clusters"]
    n = len(P)
    cl = np.array([p["c"] for p in P])
    names = {c["id"]: c["name"] for c in CL}
    xy = np.array([p["p2"] for p in P])
    cit = np.array([p["cit"] or 0 for p in P])
    CS = json.load(open(out("citation_summary.json")))
    Wt = weights(J, n)
    A = Wt > 0
    deg = A.sum(1)
    _, labc = connected_components(csr_matrix(A))
    g = np.where(labc == np.bincount(labc).argmax())[0]
    iso = np.where(deg == 0)[0]
    rng = np.random.default_rng(SEED)
    asp = np.ptp(xy[:, 1]) / np.ptp(xy[:, 0])
    map_h = CW * asp

    def base_map(ax):
        ax.set_xticks([]); ax.set_yticks([])
        [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(xy[:, 0].min() - 0.2, xy[:, 0].max() + 0.2)
        ax.set_ylim(xy[:, 1].min() - 0.2, xy[:, 1].max() + 0.2)

    # ---------------------------------------------------------------- 0. resumen
    cards = [
        ("¿Quedan papers fuera de la red de citas?",
         f"Pocos: {len(iso)} de {n} no tienen ninguna cita en común con otro. Los otros {len(g)} forman un solo bloque conectado.",
         "#3FB950", "CONECTADA"),
        ("¿Las citas separan los mismos temas?",
         f"Solo en parte: la red se parte en {CS['louvain']['n_comm']} comunidades que coinciden poco con los 8 temas (ARI {dec(CS['louvain']['ARI_clusters'])}).",
         "#F0883E", "EN PARTE"),
        ("¿La red de citas tiene huecos propios?",
         "No: sus bucles no destacan sobre los de una red de citas al azar con los mismos vínculos, y tiene incluso menos bucles que ella.",
         "#FF7B72", "REFUTADO"),
        ("¿Contenido y citas cuentan lo mismo?",
         f"Se parecen (ρ = {dec(CS['sem_full']['mantel_rho'])}, no explicado por el año), pero su forma global es distinta.",
         "#7EE787", "CONFIRMADO EN PARTE"),
    ]
    p = Page("00_resumen_citas.png")
    p.header("Qué dice la red de citas",
             "Red que une dos papers si comparten citas o referencias (cocitación y acoplamiento bibliográfico, mínimo 2).")
    for k, (q, a, colr, lab) in enumerate(cards, 1):
        xq = M + 0.5
        qw = wrap(q, 9.5 * 1.12, CW - 0.5 - 1.45)
        aw = wrap(a, 8, CW - 0.5 - 0.2)
        h = 0.16 + text_h(qw, 9.5, 1.25) + 0.08 + text_h(aw, 8) + 0.16

        def card(fig, top, H, k=k, colr=colr, lab=lab, qw=qw, aw=aw, h=h):
            fig.patches.append(FancyBboxPatch((M / W, 1 - (top + h) / H), CW / W, h / H, transform=fig.transFigure,
                                              boxstyle="round,pad=0,rounding_size=0.012", facecolor=DARK["panel"],
                                              edgecolor=DARK["grid"], lw=0.6))
            fig.text((M + 0.17) / W, 1 - (top + 0.12) / H, str(k), fontsize=16, fontweight="bold", color=colr, va="top")
            fig.text(xq / W, 1 - (top + 0.16) / H, qw, fontsize=9.5, fontweight="bold", va="top", linespacing=1.25)
            fig.text((M + CW - 0.15) / W, 1 - (top + 0.17) / H, lab, fontsize=7, fontweight="bold", va="top", ha="right", color=colr)
            fig.text(xq / W, 1 - (top + 0.16 + text_h(qw, 9.5, 1.25) + 0.08) / H, aw, fontsize=8, va="top",
                     color=DARK["muted"], linespacing=1.35)
        p.add(h + 0.1, card)
    p.footer("Los números salen de h08_citations.py. Los gráficos de contenido (huecos, montañas, tiempo) están en la carpeta "
             "explicativas; el 06 de esa carpeta compara contenido y citas par a par.")
    p.render()

    # ---------------------------------------------------------------- 1. la red sobre el mapa
    iu = np.triu_indices(n, 1)
    w_ = Wt[iu]
    top = np.argsort(-w_)[:2500]
    ea, eb = iu[0][top], iu[1][top]
    same = cl[ea] == cl[eb]
    p = Page("01_la_red_en_el_mapa.png")
    p.header("Una red densa y conectada,\ncon pocos papers sueltos",
             "El mismo mapa 2D del atlas. Cada línea es uno de los 2.500 vínculos de citas más fuertes: blanca si une papers "
             "del mismo tema, naranja si cruza de un tema a otro.")

    def netmap(fig, top_, H):
        ax = axes_in(fig, H, M, top_, CW, map_h)
        base_map(ax)
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c=[CLUSTER_COLORS[c] for c in cl], lw=0, zorder=3)
        for msk, colr, al in ((same, "#F0F6FC", 0.10), (~same, "#F0883E", 0.18)):
            seg = np.stack([xy[ea[msk]], xy[eb[msk]]], axis=1)
            ax.add_collection(LineCollection(seg, colors=colr, linewidths=0.35, alpha=al, zorder=1))
        if len(iso):
            ax.scatter(xy[iso, 0], xy[iso, 1], s=38, facecolor="none", edgecolor=DARK["fg"], lw=0.9, zorder=4)
    p.add(map_h + 0.15, netmap)
    topic_list(p, list(range(len(CL))), names)
    tot = int(A[iu].sum())
    cross_all = float((cl[iu[0]][A[iu]] != cl[iu[1]][A[iu]]).mean())
    p.footer(f"Cómo leerlo: {len(g)} papers forman un solo bloque; los {len(iso)} círculos blancos no comparten citas con ninguno. "
             f"De los {f"{tot:,}".replace(",", ".")} vínculos de la red, el {pct(cross_all)} une papers de temas distintos, "
             "así que los temas del atlas no son compartimentos cerrados en las citas.")
    p.render()

    # ---------------------------------------------------------------- 2. mezcla de temas entre vecinos de citas
    Wg = Wt.copy()
    nn = np.argsort(-Wg, axis=1)[:, :K]
    ok = np.take_along_axis(Wg, nn, 1) > 0

    def cross_by(labels):
        other = (labels[nn] != labels[:, None]) & ok
        return np.array([other[labels == c].sum() / max(ok[labels == c].sum(), 1) for c in range(len(CL))])
    obs = cross_by(cl)
    perm = np.array([cross_by(rng.permutation(cl)) for _ in range(200)])
    pm, plo, phi = perm.mean(0), perm.min(0), perm.max(0)
    order = np.argsort(obs)
    p = Page("02_temas_en_las_citas.png")
    p.header("Los temas se agrupan en las citas,\npero con fronteras muy abiertas",
             f"Para cada tema: qué parte de los {K} papers con más citas en común con cada paper pertenece a OTRO tema. "
             "Menos = tema más cerrado.")
    LW = 2.05
    labels = [wrap(names[c], 8, LW - 0.1) for c in order]
    row_h = max(text_h(t, 8) for t in labels) + 0.16
    ax_h = row_h * (len(order) + 1.2)

    def bars(fig, top_, H):
        ax = axes_in(fig, H, M + LW, top_, CW - LW - 0.05, ax_h)
        y = np.arange(len(order))
        ax.axvspan(phi.min(), phi.max(), color=DARK["warn"], alpha=0.13, lw=0)
        ax.barh(y, obs[order], color=[CLUSTER_COLORS[c] for c in order], height=0.6)
        ax.scatter(pm[order], y, marker="|", s=240, color=DARK["muted"], lw=2)
        for yi, v in zip(y, obs[order]):
            ax.text(v + 0.015, yi, pct(v), va="center", fontsize=8.5, fontweight="bold")
        ax.set_yticks(y, labels, fontsize=8, linespacing=1.25)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.set_xlabel("vecinos de citas que son de otro tema")
        ax.set_ylim(-0.6, len(y) + 0.6)
        ax.text(pm.mean(), len(y) - 0.1, "temas\nal azar", ha="center", va="bottom", fontsize=7.5, color=DARK["muted"])
    p.add(ax_h + 0.55, bars)
    n_far = int((obs < plo).sum())
    p.footer(f"Cómo leerlo: la marca gris es lo que saldría si los temas se repartieran al azar; la franja amarilla, el rango de "
             f"200 repartos al azar. {n_far} de los {len(CL)} temas quedan por debajo de la franja: se citan entre sí más de lo "
             "que daría el azar. Aun así, en los temas más abiertos la mayor parte de sus vecinos de citas son de otro tema.")
    p.render()

    # ---------------------------------------------------------------- 3. comunidades de Louvain
    CC = np.load(out("citation_communities.npy"))
    comm = np.full(n, -1)
    comm[CC[:, 0]] = CC[:, 1]
    sizes = np.bincount(CC[:, 1])
    corder = np.argsort(-sizes)
    ccol = ["#58A6FF", "#F2CC60", "#FF7B72", "#A371F7", "#3FB950", "#F0883E", "#8B949E"]
    cmap = {int(c): ccol[k % len(ccol)] for k, c in enumerate(corder)}
    p = Page("03_comunidades_de_citas.png")
    p.header(f"La red se parte en {len(sizes)} comunidades\nque coinciden poco con los temas",
             "Arriba, el mapa del atlas coloreado por la comunidad de citas de cada paper (algoritmo de Louvain). Abajo, de "
             "qué temas está hecha cada comunidad.")

    def commmap(fig, top_, H):
        ax = axes_in(fig, H, M, top_, CW, map_h)
        base_map(ax)
        cols = ["#30363D" if c < 0 else cmap[int(c)] for c in comm]
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=cols, lw=0)
    p.add(map_h + 0.15, commmap)
    BW = CW - 0.9
    for k, c in enumerate(corder, 1):
        members = np.where(comm == c)[0]
        frac = np.bincount(cl[members], minlength=len(CL)) / len(members)
        big = [t for t in np.argsort(-frac) if frac[t] >= 0.1]
        txt = wrap("Mayor parte: " + "; ".join(f"{names[t]} ({pct(frac[t])})" for t in big[:3]), 7, CW - 0.35)

        def row(fig, top_, H, k=k, c=c, members=members, frac=frac, txt=txt):
            fig.text(M / W, 1 - top_ / H, str(k), fontsize=10, fontweight="bold", color=cmap[int(c)], va="top")
            fig.text((M + 0.3) / W, 1 - (top_ + 0.02) / H, f"{len(members)} papers", fontsize=8, fontweight="bold", va="top")
            ax = axes_in(fig, H, M + 0.3, top_ + 0.22, BW, 0.2)
            left = 0
            for t in range(len(CL)):
                ax.barh(0, frac[t], left=left, color=CLUSTER_COLORS[t], height=1, lw=0)
                left += frac[t]
            ax.set_xlim(0, 1); ax.axis("off")
            fig.text((M + 0.3) / W, 1 - (top_ + 0.48) / H, txt, fontsize=7, va="top", color=DARK["muted"], linespacing=1.35)
        p.add(0.48 + text_h(txt, 7) + 0.16, row)
    topic_list(p, list(range(len(CL))), names)
    p.footer(f"Cómo leerlo: cada barra es una comunidad y sus colores, los temas de sus papers. Si las comunidades de citas "
             f"fueran los temas del atlas, cada barra tendría un solo color. La coincidencia es débil (ARI {dec(CS['louvain']['ARI_clusters'])}; "
             "1 sería idéntica) y con las montañas de densidad semánticas casi nula: las citas agrupan la literatura "
             "con un criterio distinto al del contenido (qué criterio es, no se ha analizado).")
    p.render()

    # ---------------------------------------------------------------- 4. huecos frente al azar
    nul = np.load(out("null_citation_config.npy"))
    bars_df = pd.read_csv(out("h1_bars_citation.csv"))
    tau, real_n = CS["h1_tau_rewire"], CS["h1_bars"]
    p = Page("04_huecos_en_citas.png")
    p.header("Sin huecos propios:\nla red de citas es más «rellena» que el azar",
             "Un hueco es un bucle de papers vinculados en cadena sin un paper central que los una. Se compara la red real con "
             f"{len(nul)} redes al azar que conservan cuántas citas tiene cada paper.")

    def hist(fig, top_, H):
        ax = axes_in(fig, H, M + 0.1, top_ + 0.5, CW - 0.2, 2.6)
        ax.hist(nul[:, 3], bins=24, color=DARK["muted"], alpha=0.85)
        ymax = ax.get_ylim()[1]
        ax.axvline(real_n, color=DARK["warn"], lw=2.5)
        ax.text(real_n + (nul[:, 3].max() - nul[:, 3].min()) * 0.12, ymax * 0.82, f"red real\n{real_n} bucles", color=DARK["warn"],
                fontsize=9, fontweight="bold", va="top")
        ax.text(np.median(nul[:, 3]), ymax * 1.04, f"{len(nul)} redes al azar\n({int(nul[:, 3].min())}-{int(nul[:, 3].max())} bucles)",
                ha="center", va="bottom", fontsize=8, color=DARK["muted"])
        ax.set_xlabel("número de bucles"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(min(real_n, nul[:, 3].min()) - 30, nul[:, 3].max() + 30)
    p.add(2.6 + 0.95, hist)

    p.text("¿Algún bucle destaca? Cada punto es un bucle de la red real", 10, weight="bold", gap=0.06)

    def diagram(fig, top_, H):
        ax = axes_in(fig, H, M + 0.5, top_ + 0.05, CW - 0.6, 3.0)
        b, d = bars_df.birth.values, bars_df.death.values
        hi = bars_df.ratio.values > tau
        ax.scatter(b[~hi], d[~hi], s=7, color=DARK["muted"], alpha=0.7, lw=0)
        t5 = bars_df.sort_values("ratio", ascending=False).head(5)
        ax.scatter(t5.birth, t5.death, s=22, color=DARK["warn"], lw=0, zorder=3, label="los 5 más marcados")
        xs = np.array([0, b.max() * 1.05])
        ax.plot(xs, xs, color=DARK["grid"], lw=1)
        ax.plot(xs, xs * tau, color=DARK["bad"], lw=1.4, ls="--")
        ax.text(xs[1] * 0.55, xs[1] * 0.55 * tau + 0.08, "umbral del azar", color=DARK["bad"], fontsize=8, ha="right", va="bottom", rotation=0)
        ax.set_xlim(0, xs[1]); ax.set_ylim(0, max(d.max(), xs[1] * tau) * 1.05)
        ax.set_xlabel("nace (distancia a la que aparece el bucle)")
        ax.set_ylabel("muere (distancia a la que se rellena)")
        ax.legend(fontsize=7.5, loc="upper left")
    p.add(3.0 + 0.65, diagram)
    p.footer(f"Cómo leerlo: arriba, la red real (línea amarilla) tiene menos bucles que ninguna de las redes al azar: no hay "
             f"agujeros que el azar no produzca. Abajo, un bucle «importante» viviría muy por encima de la diagonal; el umbral "
             f"rojo (muerte = {dec(tau)} × nacimiento) lo marca el mejor bucle de las redes al azar. Ninguno de los "
             f"{real_n} bucles reales lo supera (el mejor llega a {dec(CS['h1_top5_ratio'][0])}). Se usan 300 papers de "
             "referencia por red por coste de cálculo.")
    p.render()

    # ---------------------------------------------------------------- 5. papers más centrales en las citas
    Wn = np.where(A, 1 - Wt + 0.05, 0)
    wdeg = Wt.sum(1)
    order5 = np.argsort(-wdeg)[:10]
    # fracción de su fuerza de citas que va a otros temas
    ws_other = np.array([Wt[i][cl != cl[i]].sum() / max(Wt[i].sum(), 1e-9) for i in range(n)])
    mx = wdeg[order5].max()
    pages = {}
    for rank, i in enumerate(order5, 1):
        half = "a" if rank <= 5 else "b"
        if half not in pages:
            lo = 1 if half == "a" else 6
            pages[half] = Page(f"05{half}_papers_centrales_{lo}_a_{lo + 4}.png")
            pages[half].header(f"Los papers más conectados por citas ({lo}-{lo + 4})",
                               "Fuerza de citas = suma de los vínculos del paper con todos los demás (cocitación y acoplamiento). "
                               "La barra es la fuerza relativa al primero; el porcentaje, la parte de esa fuerza que va a papers de otro tema.")
        pg = pages[half]
        t = wrap(f"{P[i]['t']} ({P[i]['y'] or 's/a'})", 8, CW - 0.32)
        meta = wrap(f"{names[cl[i]]} · {cit[i]} citas · {int(deg[i])} vínculos", 7, CW - 0.32)
        h = text_h(t, 8) + text_h(meta, 7) + 0.05 + 0.2 + 0.2

        def item(fig, top_, H, i=i, rank=rank, t=t, meta=meta):
            colr = CLUSTER_COLORS[cl[i]]
            fig.text(M / W, 1 - top_ / H, str(rank), fontsize=10, fontweight="bold", color=colr, va="top")
            x = M + 0.32
            fig.text(x / W, 1 - top_ / H, t, fontsize=8, va="top", linespacing=1.35)
            y = top_ + text_h(t, 8)
            fig.text(x / W, 1 - y / H, meta, fontsize=7, va="top", color=colr, linespacing=1.35)
            y += text_h(meta, 7) + 0.05
            bw = CW - 0.32 - 0.7
            for k, (v, lab_) in enumerate(((wdeg[i] / mx, "fuerza de citas"), (ws_other[i], "a otros temas"))):
                by = y + k * 0.12
                fig.patches.append(plt.Rectangle((x / W, 1 - (by + 0.07) / H), bw / W, 0.07 / H, transform=fig.transFigure,
                                                 facecolor=DARK["panel"], edgecolor="none"))
                fig.patches.append(plt.Rectangle((x / W, 1 - (by + 0.07) / H), bw * v / W, 0.07 / H, transform=fig.transFigure,
                                                 facecolor=colr if k == 0 else DARK["muted"], edgecolor="none"))
                fig.text((x + bw + 0.05) / W, 1 - (by - 0.02) / H, pct(v) + " a otros temas" if k else lab_,
                         fontsize=6.3, va="top", color=DARK["muted"])
        pg.add(h, item)
    for pg in pages.values():
        pg.footer("Cómo leerlo: son los papers que más comparten citas o referencias con el resto de la literatura; suelen ser "
                  "revisiones o trabajos de referencia. Un porcentaje alto a otros temas indica un paper que une áreas, no solo "
                  "uno central en la suya.")
        pg.render()

    print("OK ->", X.DST)


if __name__ == "__main__":
    main()
