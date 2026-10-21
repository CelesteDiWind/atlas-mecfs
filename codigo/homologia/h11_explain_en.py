"""Paso 11 (inglés): los mismos gráficos explicativos que h11_explain.py, en inglés -> figs/explicativas_en/*.png
Generado a partir de h11_explain.py; si cambia el original, regenerar este archivo.
Cada gráfico responde una pregunta: el título es la respuesta y el subtítulo explica qué se mide.
Maquetación: cada página se arma con bloques cuya altura se calcula a partir del texto que contienen, así que los
nombres de temas y los títulos de papers se muestran enteros (se reparten en varias líneas, nunca se recortan).
Al guardar se comprueba que ningún texto se salga de la figura ni se pise con otro.
Formato: fondo oscuro #0D1117, vertical (pensado para el móvil), 300 DPI."""
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import CLUSTER_COLORS, DARK, FIGS, MAIN, RES, chord, load_emb, load_papers, out

# nombres de los temas en inglés (data/map.json); por si homologia/out/papers.json no los trae
EN_NAMES = {c["id"]: c["name_en"] for c in json.load(open(os.path.join(os.path.dirname(os.path.dirname(FIGS)), "data", "map.json"), encoding="utf-8"))["clusters"]}

DST = os.path.join(FIGS, "explicativas_en")
os.makedirs(DST, exist_ok=True)
W = 6.0          # ancho de la figura (pulgadas)
M = 0.38         # margen lateral
CW = W - 2 * M   # ancho útil
CHAR = 0.55      # ancho medio de un carácter en em (DejaVu Sans), para repartir líneas
VERDICT = {"confirmado": ("#3FB950", "CONFIRMED"), "confirmado (en parte)": ("#7EE787", "PARTLY CONFIRMED"),
           "refutado (en parte)": ("#F0883E", "PARTLY REFUTED"), "refutado": ("#FF7B72", "REFUTED"),
           "inconclusivo": ("#8B949E", "INCONCLUSIVE")}

plt.rcParams.update({
    "figure.facecolor": DARK["bg"], "axes.facecolor": DARK["bg"], "savefig.facecolor": DARK["bg"],
    "text.color": DARK["fg"], "axes.labelcolor": DARK["fg"], "xtick.color": DARK["muted"], "ytick.color": DARK["muted"],
    "axes.edgecolor": DARK["grid"], "font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


# ---------------------------------------------------------------- maquetación
def wrap(text, fs, width_in):
    """Reparte el texto en líneas que caben en width_in pulgadas con letra de fs puntos (sin recortar nada)."""
    n = max(12, int(width_in * 72 / (fs * CHAR)))
    return "\n".join(textwrap.fill(" ".join(p.split()), n) if p.strip() else "" for p in text.split("\n"))


def text_h(text, fs, ls=1.35):
    return (text.count("\n") + 1) * fs * ls / 72


class Page:
    """Página vertical hecha de bloques; la altura total sale de la suma de los bloques."""

    def __init__(self, name):
        self.name, self.blocks = name, []

    def add(self, h, draw):
        self.blocks.append((h, draw))
        return self

    def space(self, h):
        return self.add(h, lambda fig, top, H: None)

    def text(self, text, fs=9, color=None, weight="normal", x=M, width=CW, gap=0.08, ls=1.35):
        t = wrap(text, fs * (1.12 if weight == "bold" else 1.0), width)  # la negrita es más ancha
        return self.add(text_h(t, fs, ls) + gap, lambda fig, top, H: fig.text(
            x / W, 1 - top / H, t, fontsize=fs, color=color or DARK["fg"], fontweight=weight, va="top", ha="left", linespacing=ls))

    def header(self, title, subtitle):
        self.space(0.28)
        self.text(title, 14, weight="bold", gap=0.12, ls=1.15)
        self.text(subtitle, 9, DARK["muted"], gap=0.3, ls=1.4)

    def footer(self, text):
        self.space(0.18)
        self.text(text, 7.5, DARK["muted"], gap=0.22, ls=1.4)

    def render(self):
        H = sum(h for h, _ in self.blocks)
        fig = plt.figure(figsize=(W, H))
        top = 0.0
        for h, draw in self.blocks:
            draw(fig, top, H)
            top += h
        problems = check(fig)
        fig.savefig(os.path.join(DST, self.name), dpi=300)
        plt.close(fig)
        print(f"  {self.name}: {W:.0f} x {H:.1f} in", "OK" if not problems else problems)


def axes_in(fig, H, left, top, width, height):
    """Ejes situados en pulgadas desde la esquina superior izquierda."""
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def check(fig):
    """Textos fuera de la figura o que se pisan entre sí (en píxeles, con 2 px de tolerancia)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    # etiquetas de marca que Matplotlib genera fuera del rango visible no se dibujan: no cuentan
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    outside = [t.get_text()[:40] for t, b in boxes if b.x0 < fb.x0 - 2 or b.x1 > fb.x1 + 2 or b.y0 < fb.y0 - 2 or b.y1 > fb.y1 + 2]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 2 and min(a.y1, b.y1) - max(a.y0, b.y0) > 2:
                overlaps.append((boxes[i][0].get_text()[:30], boxes[j][0].get_text()[:30]))
    return {"fuera": outside, "solapes": overlaps} if outside or overlaps else None


def topic_list(page, topics, names, fs=7.5, cols=2):
    """Leyenda de temas con el nombre completo (columnas; cada nombre se reparte en líneas si no cabe)."""
    colw = CW / cols
    rows = [topics[i:i + cols] for i in range(0, len(topics), cols)]
    for row in rows:
        wrapped = [wrap(names[c], fs, colw - 0.25) for c in row]
        h = max(text_h(t, fs) for t in wrapped) + 0.07

        def draw(fig, top, H, row=row, wrapped=wrapped):
            for k, (c, t) in enumerate(zip(row, wrapped)):
                x = M + k * colw
                fig.text(x / W, 1 - top / H, "●", fontsize=fs + 2, color=CLUSTER_COLORS[c], va="top")
                fig.text((x + 0.2) / W, 1 - (top + 0.01) / H, t, fontsize=fs, va="top", color=DARK["fg"], linespacing=1.35)
        page.add(h, draw)


def pct(v):
    return f"{v:.0%}"


def dec(v, n=2):
    return f"{v:.{n}f}"


# ---------------------------------------------------------------- gráficos
def main():
    J = load_papers()
    P, CL = J["papers"], J["clusters"]
    cl = np.array([p["c"] for p in P])
    names = {c["id"]: c.get("name_en") or EN_NAMES[c["id"]] for c in CL}
    xy = np.array([p["p2"] for p in P])
    S = pd.read_csv(out("h1_summary.csv"))
    S1 = S[(S.subset == "all") & (S.dim == 1)]

    # ---------------------------------------------------------------- 0. resumen de veredictos
    T = pd.read_csv(os.path.join(RES, "comparacion_con_resultados_previos.csv"))
    plain = {
        1: ("Are the 8 topics separate from each other?", "Partly: 4 topics are compact (12–17% of neighbours from another topic) and 4 have blurry borders (31–45%). None is an island."),
        2: ("Are there 9 density “mountains”?", "No, there are 3: one large central body and two small groups on exercise and post-exertional malaise."),
        3: ("Are there holes in the neighbour network?", "One at most, and a doubtful one. The earlier 4 do not survive a stricter test."),
        4: ("Are there large holes linking topics?", "No. There are 6 small holes (5–7 papers), stable, and present in all three models."),
        5: ("Does the citation network have holes of its own?", "No: it does no better than a random citation network."),
        6: ("Do content and citations tell the same story?", "Partly: they are similar (ρ = 0.36, twice the earlier estimate), but their shape differs."),
        7: ("Did the structure change over the years?", "No: what looked like a change is explained by the number of papers."),
    }
    p = Page("00_verdicts_summary.png")
    p.header("What was confirmed and what was not", "The 7 findings of the first pass, tested with the atlas's real embeddings.")
    for r in T.itertuples():
        colr, lab = VERDICT.get(r.veredicto, ("#8B949E", r.veredicto.upper()))
        q, a = plain[int(r.n)]
        xq = M + 0.5
        qw = wrap(q, 9.5 * 1.12, CW - 0.5 - 1.45)
        aw = wrap(a, 8, CW - 0.5 - 0.2)
        h = 0.16 + text_h(qw, 9.5, 1.25) + 0.08 + text_h(aw, 8) + 0.16

        def card(fig, top, H, n=int(r.n), colr=colr, lab=lab, qw=qw, aw=aw, h=h):
            fig.patches.append(FancyBboxPatch((M / W, 1 - (top + h) / H), CW / W, h / H, transform=fig.transFigure,
                                              boxstyle="round,pad=0,rounding_size=0.012", facecolor=DARK["panel"],
                                              edgecolor=DARK["grid"], lw=0.6))
            fig.text((M + 0.17) / W, 1 - (top + 0.12) / H, str(n), fontsize=16, fontweight="bold", color=colr, va="top")
            fig.text(xq / W, 1 - (top + 0.16) / H, qw, fontsize=9.5, fontweight="bold", va="top", linespacing=1.25)
            fig.text((M + CW - 0.15) / W, 1 - (top + 0.17) / H, lab, fontsize=7, fontweight="bold", va="top", ha="right", color=colr)
            fig.text(xq / W, 1 - (top + 0.16 + text_h(qw, 9.5, 1.25) + 0.08) / H, aw, fontsize=8, va="top",
                     color=DARK["muted"], linespacing=1.35)
        p.add(h + 0.1, card)
    p.footer("Green: confirmed · orange and red: does not hold · grey: cannot be decided. "
             "Details in resultados/comparacion_con_resultados_previos.csv.")
    p.render()

    # ---------------------------------------------------------------- 1. ¿temas separados?
    MX = pd.read_csv(out("cluster_mixing.csv"))
    m6 = MX[MX.k == 6].sort_values("cross_obs")
    pn = np.load(out("pipeline_null_cross.npy"))
    below = m6[m6.cross_obs < pn.min()]
    p = Page("01_separate_topics.png")
    p.header("Four compact topics,\nfour with blurry borders",
             "For each topic: the share of each paper's 6 most similar neighbours that belong to ANOTHER topic. "
             "Lower = more compact topic.")
    LW = 2.05  # columna de nombres de tema
    labels = [wrap(names[c], 8, LW - 0.1) for c in m6.cluster]
    row_h = max(text_h(t, 8) for t in labels) + 0.16
    ax_h = row_h * (len(m6) + 1.2)

    def bars(fig, top, H):
        ax = axes_in(fig, H, M + LW, top, CW - LW - 0.05, ax_h)
        y = np.arange(len(m6))
        ax.axvspan(pn.min(), pn.max(), color=DARK["warn"], alpha=0.13, lw=0)
        ax.barh(y, m6.cross_obs, color=[CLUSTER_COLORS[c] for c in m6.cluster], height=0.6)
        ax.scatter(m6.cross_perm_mean, y, marker="|", s=240, color=DARK["muted"], lw=2)
        for yi, v in zip(y, m6.cross_obs):
            ax.text(v + 0.015, yi, pct(v), va="center", fontsize=8.5, fontweight="bold")
        ax.set_yticks(y, labels, fontsize=8, linespacing=1.25)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("neighbours from another topic")
        ax.set_ylim(-0.6, len(y) + 0.6)
        ax.text((pn.min() + pn.max()) / 2, len(y) - 0.1, "if topics were\narbitrary cuts", ha="center",
                va="bottom", fontsize=7.5, color=DARK["warn"])
        ax.text(m6.cross_perm_mean.mean(), len(y) - 0.1, "random\ntopics", ha="center", va="bottom", fontsize=7.5, color=DARK["muted"])
    p.add(ax_h + 0.55, bars)
    p.footer(f"How to read it: {len(below)} topics ({'; '.join(names[c] for c in below.cluster)}) sit well below the yellow "
             "band: they are genuinely compact. The others fall within what cutting a cloud with no groups at random would "
             "give: their borders are blurry. None is isolated (none reaches 0%), and all are far from random topics "
             "(grey marks).")
    p.render()

    # ---------------------------------------------------------------- 2. montañas de densidad (mapa)
    lab = np.load(out("modes_labels.npy"), allow_pickle=True).item()[MAIN]
    md = pd.read_csv(out(f"modes_{MAIN}.csv")).sort_values("n", ascending=False).reset_index(drop=True)
    mode_cols = ["#8B949E", "#F2CC60", "#FF7B72"]
    asp = np.ptp(xy[:, 1]) / np.ptp(xy[:, 0])
    map_h = CW * asp

    def scatter_map(ax, colors):
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=colors, lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(xy[:, 0].min() - 0.2, xy[:, 0].max() + 0.2)
        ax.set_ylim(xy[:, 1].min() - 0.2, xy[:, 1].max() + 0.2)

    p = Page("02_density_mountains.png")
    p.header("One large continent and two small islands",
             "Top: the atlas's 2D map coloured by the 8 topics. Bottom: the same map coloured by the density "
             "“mountains” the analysis finds (areas where papers cluster).")
    p.text("The 8 topics (as the atlas draws them)", 10, weight="bold", gap=0.06)
    p.add(map_h + 0.12, lambda fig, top, H: scatter_map(axes_in(fig, H, M, top, CW, map_h), [CLUSTER_COLORS[c] for c in cl]))
    topic_list(p, list(range(8)), names)
    p.space(0.25)
    p.text("The 3 density mountains (what the data say)", 10, weight="bold", gap=0.06)
    mcolor = {int(r["mode"]): mode_cols[k] for k, r in md.iterrows()}  # r.mode sería el método de pandas

    def mode_map(fig, top, H):
        ax = axes_in(fig, H, M, top, CW, map_h)
        scatter_map(ax, [mcolor[int(v)] for v in lab])
        for k, r in md.iterrows():
            i = int(r.peak_paper)
            ax.scatter(*xy[i], s=110, facecolor="none", edgecolor=mode_cols[k], lw=1.8)
            ax.annotate(str(k + 1), xy[i], xytext=(7, 5), textcoords="offset points", fontsize=9, fontweight="bold",
                        color=mode_cols[k])
    p.add(map_h + 0.15, mode_map)
    for k, r in md.iterrows():
        txt = wrap(f"{int(r.n)} papers. Peak: “{P[int(r.peak_paper)]['t']}” ({int(r.year)})", 7.5, CW - 0.3)

        def item(fig, top, H, k=k, txt=txt):
            fig.text(M / W, 1 - top / H, str(k + 1), fontsize=9, fontweight="bold", color=mode_cols[k], va="top")
            fig.text((M + 0.25) / W, 1 - (top + 0.01) / H, txt, fontsize=7.5, va="top", linespacing=1.35)
        p.add(text_h(txt, 7.5) + 0.08, item)
    p.footer("How to read it: almost the whole corpus (534 papers) forms a single dense body; the 8 topics are slices of "
             "that body, not separate hills. Only two groups, on exercise and post-exertional malaise, stand out as mountains "
             "of their own. The first pass counted 9 mountains; against a cloud with no groups only these 3 survive, in all "
             "three models.")
    p.render()

    # ---------------------------------------------------------------- 3. huecos por escala y modelo
    p = Page("03_holes_by_scale.png")
    p.header("There are small holes,\nnot large ones",
             "A “hole” (cycle) is a chain of papers, each similar to the next, that loops back to the start with no central "
             "topic joining them. Only those that beat 200 random simulations are counted.")
    models = ["specter2", "mpnet", "bge-m3"]
    scales = [("full", "Fine scale\n(direct similarity)"), ("k15", "Medium scale\n(neighbourhoods of 15)"),
              ("k30", "Broad scale\n(neighbourhoods of 30)")]
    mc = {"specter2": DARK["accent"], "mpnet": "#A371F7", "bge-m3": "#3FB950"}

    def scale_bars(fig, top, H):
        ax = axes_in(fig, H, M + 0.45, top, CW - 0.5, 3.6)
        x = np.arange(len(scales))
        w = 0.26
        for k, m in enumerate(models):
            vals = [int(S1[(S1.model == m) & (S1.space == sp)].nsig_gmm.iloc[0]) for sp, _ in scales]
            bb = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=mc[m], label=m + (" (the atlas model)" if m == MAIN else ""))
            for b_, v in zip(bb, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=8.5, fontweight="bold")
        ax.axhline(17, color=DARK["bad"], ls="--", lw=1)
        ax.text(-0.45, 17.4, "first pass (LSA proxy): 17 holes", ha="left", fontsize=7.5, color=DARK["bad"])
        ax.set_xticks(x, [s for _, s in scales], fontsize=8.5)
        ax.set_ylabel("holes that beat chance")
        ax.set_ylim(0, 20)
        ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1, 0.55))
    p.add(3.6 + 0.65, scale_bars)
    p.footer("How to read it: the three language models agree. There are a few holes up close (fine scale) and none when "
             "you look at the big picture (medium and broad scales). The structure does not depend on the model chosen.")
    p.render()

    # ---------------------------------------------------------------- 4. menos huecos que el azar
    nul = np.load(out(f"null2_{MAIN}_all_k15_gmm.npy"))[:, 2]
    real = int(S1[(S1.model == MAIN) & (S1.space == "k15")].n_bars.iloc[0])
    p = Page("04_fewer_holes_than_chance.png")
    p.header("The corpus has fewer holes\nthan chance",
             "Total number of small loops in the neighbourhood map (medium scale), in the real corpus and in 200 simulated "
             "clouds with the same groups and the same spread.")

    def hist(fig, top, H):
        ax = axes_in(fig, H, M + 0.1, top + 0.5, CW - 0.2, 3.0)
        ax.hist(nul, bins=24, color=DARK["muted"], alpha=0.85)
        ymax = ax.get_ylim()[1]
        ax.axvline(real, color=DARK["warn"], lw=2.5)
        ax.text(real + 18, ymax * 0.82, f"real corpus\n{real} loops", color=DARK["warn"], fontsize=9, fontweight="bold", va="top")
        ax.text(np.median(nul), ymax * 1.04, f"200 simulations\n({int(nul.min())}–{int(nul.max())} loops)", ha="center",
                va="bottom", fontsize=8, color=DARK["muted"])
        ax.set_xlabel("number of loops"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlim(min(real, nul.min()) - 80, nul.max() + 60)
    p.add(3.0 + 1.05, hist)
    p.footer("How to read it: the real corpus (yellow line) has half as many loops as any simulation. The literature is more "
             "“filled in” than chance: papers form a continuous fabric with no large thematic holes. The same happens in all "
             "three models and at every scale.")
    p.render()

    # ---------------------------------------------------------------- 5. los 6 ciclos (dos páginas de tres)
    cyc = json.load(open(out("cycle_members.json")))
    C = pd.read_csv(out("cycles.csv")).set_index("cycle_id")
    ids = [k for k in cyc if k.startswith("full")]
    RING = 1.5
    LX = M + RING + 0.25           # columna de títulos
    LWID = M + CW - LX
    for part, chunk in (("a", ids[:3]), ("b", ids[3:])):
        first = ids.index(chunk[0]) + 1
        p = Page(f"05{part}_holes_{first}_to_{first + len(chunk) - 1}.png")
        p.header(f"The 6 holes ({first}–{first + len(chunk) - 1}): short chains of papers that close a circle",
                 "Each circle is a hole: every paper is similar to the next, but no central paper is similar to all of "
                 "them. The numbers on the circle match the list; the colour is the topic.")
        for cid in chunk:
            k = ids.index(cid) + 1
            nodes = cyc[cid]["nodes"]
            ntop = len(set(cl[nodes]))
            head = wrap(f"Hole {k} · {len(nodes)} papers · {ntop} topic{'s' if ntop > 1 else ''} · reappears in "
                        f"{pct(C.loc[cid, 'reappear'])} of subsamples", 9 * 1.12, CW)
            items = []
            for i, v in enumerate(nodes):
                t = wrap(f"{P[v]['t']} ({P[v]['y'] or 'n.d.'})", 7, LWID - 0.22)
                tp = wrap(names[cl[v]], 6.5, LWID - 0.22)
                items.append((i + 1, v, t, tp, text_h(t, 7) + text_h(tp, 6.5) + 0.07))
            list_h = sum(it[-1] for it in items)
            hh = text_h(head, 9) + 0.1
            block_h = hh + max(RING, list_h) + 0.3

            def block(fig, top, H, nodes=nodes, items=items, head=head, hh=hh):
                fig.text(M / W, 1 - top / H, head, fontsize=9, fontweight="bold", va="top", linespacing=1.3)
                ax = axes_in(fig, H, M, top + hh, RING, RING)
                ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(nodes), endpoint=False)
                px, py = np.cos(ang), np.sin(ang)
                ax.plot(np.r_[px, px[0]], np.r_[py, py[0]], color=DARK["grid"], lw=1.3, zorder=1)
                for i, (a_, b_, v) in enumerate(zip(px, py, nodes)):
                    ax.scatter(a_, b_, s=230, color=CLUSTER_COLORS[cl[v]], zorder=2, edgecolor=DARK["bg"], lw=1)
                    ax.text(a_, b_, str(i + 1), ha="center", va="center", fontsize=6.5, fontweight="bold", color=DARK["bg"], zorder=3)
                ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
                y = top + hh
                for num, v, t, tp, h in items:
                    fig.text(LX / W, 1 - y / H, str(num), fontsize=7.5, fontweight="bold", color=CLUSTER_COLORS[cl[v]], va="top")
                    fig.text((LX + 0.22) / W, 1 - y / H, t, fontsize=7, va="top", linespacing=1.35)
                    fig.text((LX + 0.22) / W, 1 - (y + text_h(t, 7)) / H, tp, fontsize=6.5, va="top",
                             color=CLUSTER_COLORS[cl[v]], linespacing=1.35)
                    y += h
            p.add(block_h, block)
        p.footer("How to read it: hole 2 lies entirely within one topic (orthostatic intolerance and POTS) and hole 1 almost does "
                 "(6 of its 7 papers are on clinical care, society and stigma); the other four cross 3 or 4 topics. All are "
                 "small: the large cycles between topics that the first pass described do not appear.")
        p.render()

    # ---------------------------------------------------------------- 6. contenido vs citas
    from h08_citations import geodesic, weights
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    Wt = weights(J, len(P))
    _, labc = connected_components(csr_matrix(Wt > 0))
    g = np.where(labc == np.bincount(labc).argmax())[0]
    Dc = geodesic(Wt[np.ix_(g, g)])
    Ds = chord(load_emb(MAIN))[np.ix_(g, g)]
    iu = np.triu_indices(len(g), 1)
    CS = json.load(open(out("citation_summary.json")))
    p = Page("06_content_vs_citations.png")
    p.header("Content and citations are alike,\nbut not the same",
             "Each point is a pair of papers: further left, more similar in content; further down, closer in the citation "
             "network. If they told the same story, the points would form a diagonal.")

    def hexes(fig, top, H):
        ax = axes_in(fig, H, M + 0.5, top, CW - 0.55, 3.6)
        ax.hexbin(Ds[iu], Dc[iu], gridsize=45, cmap="magma", bins="log", mincnt=1, linewidths=0)
        ax.set_xlabel("content distance (lower = more similar)")
        ax.set_ylabel("distance in the citation network")
        ax.text(0.03, 0.97, f"correlation ρ = {dec(CS['sem_full']['mantel_rho'])}\n(the first pass estimated 0.17)",
                transform=ax.transAxes, va="top", fontsize=9, fontweight="bold", color=DARK["warn"])
    p.add(3.6 + 0.6, hexes)
    p.footer("How to read it: there is a trend (similar papers tend to be close in citations), twice as strong as estimated "
             "and not explained by publication year. But the cloud is wide: many similar papers do not cite each other, and "
             "the “shape” of the citation network resembles that of the content no more than a random network does.")
    p.render()

    # ---------------------------------------------------------------- 7. tiempo
    Y = pd.read_csv(out("year_filtration.csv"))
    yf = Y[Y.space == "full"].sort_values("year")
    p = Page("07_time.png")
    p.header("The structure did not change over the years:\nthe corpus just grew",
             "Holes that beat chance in the corpus accumulated up to each year, against what a random sample of the corpus "
             "with the same number of papers gives.")

    def years(fig, top, H):
        ax = axes_in(fig, H, M + 0.45, top + 0.45, CW - 0.5, 3.2)
        ax.fill_between(yf.year, 0, yf.control_p95, color=DARK["accent"], alpha=0.18, lw=0, label="expected from size alone (95%)")
        ax.plot(yf.year, yf.control_mean, color=DARK["accent"], ls="--", lw=1, label="expected on average")
        ax.plot(yf.year, yf.n_sig, "o-", color=DARK["warn"], lw=2, ms=4, label="real corpus")
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(yf.year[::2], [f"{int(n)}" for n in yf.n[::2]], fontsize=7)
        ax2.set_xlabel("cumulative papers", fontsize=8, color=DARK["muted"])
        ax2.spines["top"].set_visible(True)
        ax.set_xlabel("year"); ax.set_ylabel("holes that beat chance")
        ax.set_ylim(0, max(yf.control_p95.max(), yf.n_sig.max()) + 2.5)
        ax.legend(fontsize=7.5, loc="upper left")
    p.add(3.2 + 1.0, years)
    p.footer("How to read it: the yellow line never leaves the blue band. Holes increase because there are more papers, not "
             "because the literature changed shape. The first pass's “2019 peak” does not appear when recomputed by year.")
    p.render()

    # ---------------------------------------------------------------- 8. papers puente (lista con títulos completos)
    B = pd.read_csv(out("bridges.csv"))
    for col in ("h1_bottleneck_change", "sig_persistence_lost", "cross_cluster_betweenness"):
        B[col + "_pct"] = B[col].fillna(0).rank(pct=True)
    top10 = B.head(10)
    comps = [("h1_bottleneck_change_pct", "topology change when removed"),
             ("sig_persistence_lost_pct", "hole lost"),
             ("cross_cluster_betweenness_pct", "paths between topics through it")]
    GATE = 0.7  # si >= 70 % de sus caminos entre temas salen o llegan a su propio tema, es una «puerta» de su tema
    PW = CW - 0.32 - 0.28 - 0.55  # ancho del texto de cada par de temas
    n_gate = int((top10.own_topic_share >= GATE).sum())
    pages = {}
    for rank, r in enumerate(top10.itertuples(), 1):
        half = "a" if rank <= 5 else "b"  # dos páginas de 5 papers (una sola quedaba demasiado larga para el móvil)
        if half not in pages:
            lo = 1 if half == "a" else 6
            pages[half] = Page(f"08{half}_bridge_papers_{lo}_to_{lo + 4}.png")
            pages[half].header(f"The papers that hold the structure together most ({lo}–{lo + 4}), and which topics they link",
                               "For each paper: three measures expressed as “beats X% of papers” (how much the holes "
                               "change if it is removed, how much significant hole is lost, and how many paths between "
                               "different topics run through it) and, of those paths, which pairs of topics they connect.")
        p = pages[half]
        t = wrap(f"{r.title} ({int(r.year) if r.year == r.year else 'n.d.'})", 8, CW - 0.32)
        hueco = ("in hole " + r.cycles.replace("full-", "").replace(",", " and ")) if isinstance(r.cycles, str) and r.cycles else "in no hole"
        meta = wrap(f"{names[r.cluster]} · {int(r.citations)} citations · {hueco} · score {dec(r.bridge_score)}", 7, CW - 0.32)
        pairs = json.loads(r.cluster_pairs)[:3]
        ptexts = [wrap(f"{names[a]}  ↔  {names[b]}", 7, PW) for a, b, _ in pairs]
        gate = r.own_topic_share >= GATE
        role = wrap((f"Role: gateway of its own topic. {pct(r.own_topic_share)} of those paths start or end in its topic."
                     if gate else
                     f"Role: also links topics other than its own. Only {pct(r.own_topic_share)} of those paths start or end in its topic."),
                    7 * (1.0 if gate else 1.12), CW - 0.32)
        sub = "Which topics it links (share of the paths between topics that run through it):"
        h = (text_h(t, 8) + text_h(meta, 7) + 0.05 + 0.42 + 0.06 + text_h(sub, 6.8) + 0.04
             + sum(text_h(pt, 7) + 0.11 for pt in ptexts) + text_h(role, 7) + 0.2)

        def item(fig, top, H, rank=rank, r=r, t=t, meta=meta, pairs=pairs, ptexts=ptexts, role=role, gate=gate, sub=sub):
            colr = CLUSTER_COLORS[r.cluster]
            fig.text(M / W, 1 - top / H, str(rank), fontsize=10, fontweight="bold", color=colr, va="top")
            x = M + 0.32
            fig.text(x / W, 1 - top / H, t, fontsize=8, va="top", linespacing=1.35)
            y = top + text_h(t, 8)
            fig.text(x / W, 1 - y / H, meta, fontsize=7, va="top", color=colr, linespacing=1.35)
            y += text_h(meta, 7) + 0.05
            bw = (CW - 0.32 - 0.3) / 3
            for k, (colname, lab_) in enumerate(comps):
                bx = x + k * (bw + 0.15)
                v = float(getattr(r, colname))
                fig.text(bx / W, 1 - y / H, wrap(f"{lab_}: {pct(v)}", 6.3, bw), fontsize=6.3, va="top", color=DARK["muted"],
                         linespacing=1.25)
                by = y + 0.27
                fig.patches.append(Rectangle((bx / W, 1 - (by + 0.07) / H), bw / W, 0.07 / H, transform=fig.transFigure,
                                             facecolor=DARK["panel"], edgecolor="none"))
                fig.patches.append(Rectangle((bx / W, 1 - (by + 0.07) / H), bw * v / W, 0.07 / H, transform=fig.transFigure,
                                             facecolor=colr, edgecolor="none"))
            # qué pares de temas une
            y += 0.42 + 0.06
            fig.text(x / W, 1 - y / H, sub, fontsize=6.8, va="top", color=DARK["muted"])
            y += text_h(sub, 6.8) + 0.04
            for (a, b, share), pt in zip(pairs, ptexts):
                fig.text(x / W, 1 - y / H, "●", fontsize=8, color=CLUSTER_COLORS[a], va="top")
                fig.text((x + 0.12) / W, 1 - y / H, "●", fontsize=8, color=CLUSTER_COLORS[b], va="top")
                fig.text((x + 0.28) / W, 1 - y / H, pt, fontsize=7, va="top", linespacing=1.3)
                fig.text((M + CW) / W, 1 - y / H, pct(share), fontsize=7.5, fontweight="bold", va="top", ha="right")
                ph = text_h(pt, 7)
                bx0, bwid = x + 0.28, PW
                fig.patches.append(Rectangle((bx0 / W, 1 - (y + ph + 0.06) / H), bwid / W, 0.04 / H,
                                             transform=fig.transFigure, facecolor=DARK["panel"], edgecolor="none"))
                fig.patches.append(Rectangle((bx0 / W, 1 - (y + ph + 0.06) / H), bwid * share / W, 0.04 / H,
                                             transform=fig.transFigure, facecolor=DARK["muted"], edgecolor="none"))
                y += ph + 0.11
            fig.text(x / W, 1 - y / H, role, fontsize=7, va="top", color=DARK["fg"] if gate else DARK["warn"],
                     fontweight="normal" if gate else "bold", linespacing=1.35)
        p.add(h, item)
    for p in pages.values():
        p.footer(f"How to read it: the number and colour show the paper's topic; the two dots on each row, the two topics that "
                 f"path links. {n_gate} of the top 10 are “gateways” of their own topic: through them their topic reaches the "
                 "others, above all immunity, infection and long COVID, the largest topic. The others (in yellow) also connect "
                 "topics that are not their own. Paths computed between every pair of papers from different topics, on the "
                 "graph of the 15 most similar neighbours.")
        p.render()

    # el antiguo gráfico único de los ciclos se sustituye por los dos de tres huecos
    # ---------------------------------------------------------------- 9. los huecos sobre el mapa 2D del atlas
    span = np.ptp(xy, axis=0)
    p = Page("09_holes_on_the_map.png")
    p.header("The 6 holes on the atlas's 2D map",
             "Top: where each hole falls on the map. Below: each hole enlarged; the numbers match the lists in charts 05a "
             "and 05b, and the colour is each paper's topic.")

    # en la vista general cada hueco tiene su color de línea (en los paneles ampliados los nodos llevan el color del tema)
    CYC_COLORS = ["#F0F6FC", "#F2CC60", "#58A6FF", "#FF7B72", "#A371F7", "#3FB950"]

    def overview(fig, top, H):
        ax = axes_in(fig, H, M, top, CW, map_h)
        scatter_map(ax, ["#30363D"] * len(P))
        used = []
        for k, cid in enumerate(ids, 1):
            nodes = cyc[cid]["nodes"]
            pts = xy[nodes + nodes[:1]]
            colr = CYC_COLORS[k - 1]
            ax.plot(pts[:, 0], pts[:, 1], color=colr, lw=1.3, zorder=2)
            ax.scatter(xy[nodes, 0], xy[nodes, 1], s=12, color=colr, zorder=3, lw=0)
            # número junto al paper más alto del hueco; si ese sitio ya lo usa otro hueco (comparten paper), el siguiente
            order = sorted(nodes, key=lambda v: -xy[v, 1])
            hi = next((v for v in order if all(np.linalg.norm(xy[v] - u) > 0.45 for u in used)), order[0])
            used.append(xy[hi])
            ax.annotate(str(k), xy[hi], xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=DARK["bg"], zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.2", facecolor=colr, edgecolor="none"))
    p.add(map_h + 0.3, overview)

    fig_dpi_scale = plt.rcParams["figure.dpi"] / 72  # puntos -> píxeles de pantalla (las coordenadas de display)

    def place_labels(ax, pts, gap=10.0, sep=11.0, avoid=6.5, iters=200):
        """Posiciones (en datos) para los números de los nodos: hacia fuera del lazo y separadas entre sí y de los nodos."""
        ax.apply_aspect()
        T = ax.transData
        P_ = T.transform(pts)
        c = P_.mean(0)
        d = P_ - c
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(nrm > 1e-6, d / np.maximum(nrm, 1e-6), np.array([[0.0, 1.0]]))
        L = P_ + d * gap * fig_dpi_scale
        for _ in range(iters):
            moved = False
            for i in range(len(L)):
                for j in range(len(L)):
                    if i == j:
                        continue
                    v = L[i] - L[j]
                    dist = np.linalg.norm(v)
                    if dist < sep * fig_dpi_scale:
                        L[i] += (v / max(dist, 1e-6)) * 0.6 * fig_dpi_scale
                        moved = True
                    w = L[i] - P_[j]
                    dist = np.linalg.norm(w)
                    if dist < avoid * fig_dpi_scale:
                        L[i] += (w / max(dist, 1e-6)) * 0.6 * fig_dpi_scale
                        moved = True
            if not moved:
                break
        return T.inverted().transform(L)
    PANEL = (CW - 0.3) / 2
    for row in range(3):
        pair = ids[row * 2: row * 2 + 2]
        caps = []
        for cid in pair:
            k = ids.index(cid) + 1
            nodes = cyc[cid]["nodes"]
            ext = np.ptp(xy[nodes], axis=0) / span
            ntop = len(set(cl[nodes]))
            look = "compact on the map" if ext.max() < 0.2 else f"stretched on the map: spans {pct(ext.max())} of it"
            caps.append((wrap(f"Hole {k} · {len(nodes)} papers · {ntop} topic{'s' if ntop > 1 else ''}", 8 * 1.12, PANEL),
                         wrap(look, 7, PANEL)))
        cap_h = max(text_h(a, 8) + text_h(b, 7) for a, b in caps) + 0.08

        def panels(fig, top, H, pair=pair, caps=caps):
            for c_, (cid, (a, b)) in enumerate(zip(pair, caps)):
                x0 = M + c_ * (PANEL + 0.3)
                nodes = cyc[cid]["nodes"]
                fig.text(x0 / W, 1 - top / H, a, fontsize=8, fontweight="bold", va="top", linespacing=1.3)
                fig.text(x0 / W, 1 - (top + text_h(a, 8)) / H, b, fontsize=7, va="top", color=DARK["muted"], linespacing=1.3)
                ax = axes_in(fig, H, x0, top + cap_h, PANEL, PANEL)
                pts = xy[nodes]
                cx, cy = (pts.max(0) + pts.min(0)) / 2
                half = max(np.ptp(pts, axis=0)) / 2 * 1.35 + 0.15
                inview = (np.abs(xy[:, 0] - cx) < half) & (np.abs(xy[:, 1] - cy) < half)
                ax.scatter(xy[inview, 0], xy[inview, 1], s=5, color="#30363D", lw=0)
                loop = xy[nodes + nodes[:1]]
                ax.plot(loop[:, 0], loop[:, 1], color=DARK["fg"], lw=1.2, alpha=0.8, zorder=2)
                ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                labs = place_labels(ax, pts)
                for i, (v, (lx, ly)) in enumerate(zip(nodes, labs)):
                    colr = CLUSTER_COLORS[cl[v]]
                    ax.scatter(*xy[v], s=34, color=colr, zorder=3, edgecolor=DARK["bg"], lw=0.6)
                    ax.plot([xy[v][0], lx], [xy[v][1], ly], color=colr, lw=0.6, alpha=0.7, zorder=2)
                    ax.text(lx, ly, str(i + 1), fontsize=7, fontweight="bold", ha="center", va="center", color=colr, zorder=4,
                            bbox=dict(boxstyle="circle,pad=0.12", facecolor=DARK["bg"], edgecolor="none"))
                for s_ in ax.spines.values():
                    s_.set_visible(True); s_.set_color(DARK["grid"])
        p.add(cap_h + PANEL + 0.25, panels)
    present = sorted({int(cl[v]) for cid in ids for v in cyc[cid]["nodes"]})
    topic_list(p, present, names)
    p.footer("How to read it: the holes are computed in the full embedding space (768 dimensions), where all of them are "
             "small. The 2D map is a projection that keeps who is near whom, but not the distances: that is why hole 2 looks "
             "compact while others (3, 4 and 5) stretch across half the map. Even so, none crosses itself: on the map they "
             "appear as closed loops. The “hole” lives in the full space: no paper directly connects opposite sides of the "
             "loop. On the 2D map the inside of a loop may contain papers, because the projection flattens them onto it.")
    p.render()



if __name__ == "__main__":
    main()
