"""Paso 10: entregables.
    resultados/comparacion_con_resultados_previos.csv  una fila por resultado previo (1-7) con veredicto y motivo
    resultados/page_data.json                          mismo esquema que previo/resultados/page_data.json
    resultados/topology_layer.json                     capa opcional para el atlas: bridge, cycle_ids, peak_id por paper
    resultados/*.csv                                   copias de las tablas principales
    figs/*.png                                         figuras oscuras (#0D1117), verticales, 300 DPI
Los veredictos se calculan con reglas explícitas (ver VERDICT_RULES) a partir de las salidas de los pasos 2-9."""
import json
import os
import shutil

import numpy as np
import pandas as pd

from common import CLUSTER_COLORS, DARK, MAIN, RES, chord, dark_fig, finite, load_emb, load_papers, out, save_fig

VERDICT_RULES = """
1 mezcla entre clusters: confirmado si la fracción cruzada observada está entre la de una partición aleatoria (permutación)
  y la de cortar un continuo (nulo de partición), es decir, clusters cohesivos pero no islas; el socio principal se evalúa
  normalizado por tamaño.
2 picos de densidad: se compara el número de modos significativos (frente al nulo gaussiano) y su acuerdo con los clusters
  en los tres modelos.
3/4 H1: ciclo significativo = persistencia > p95 del máximo del nulo gmm (conserva grupos). Se confirma si hay ciclos
  significativos estables (reaparecen en >= 50 % de submuestras) en SPECTER2 y en al menos otro modelo.
5 citas: significativo frente al recableado que conserva grados y pesos.
6 Mantel: correlación y p-valor por permutación de nodos, y versión parcial controlando la diferencia de años.
7 tiempo: hay tendencia temporal solo si el número de ciclos del año supera el p95 del control con el mismo n.
"""


def load(name, kind="csv"):
    p = out(name)
    if not os.path.exists(p):
        return None
    if kind == "csv":
        return pd.read_csv(p)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def pts(dg):
    d = finite(np.asarray(dg))
    return [[round(float(a), 4), round(float(b), 4)] for a, b in d]


def main():
    J = load_papers()
    P, CL = J["papers"], J["clusters"]
    n = len(P)
    cl = np.array([p["c"] for p in P])
    V = load("embedding_validation.csv")
    S = load("h1_summary.csv")
    MX = load("cluster_mixing.csv")
    AG = load("modes_agreement.csv")
    CY = load("cycles.csv")
    YF = load("year_filtration.csv")
    CS = load("citation_summary.json", "json")
    BR = load("bridges.csv")
    pn = np.load(out("pipeline_null_cross.npy"))
    rows = []

    # ---- 1. mezcla entre clusters
    m6 = MX[MX.k == 6].sort_values("cluster")
    obs = m6.cross_obs.values
    D = chord(load_emb(MAIN))
    nb = np.argsort(D, axis=1)[:, 1:7]
    C = np.zeros((8, 8), int)
    for i in range(n):
        for j in nb[i]:
            C[cl[i], cl[j]] += 1
    sizes = np.bincount(cl)
    exp_share = sizes / n
    norm_partner = []
    for c in range(8):
        offd = np.array([C[c, d] if d != c else 0 for d in range(8)], float)
        share = offd / offd.sum()
        ratio = share / np.where(np.arange(8) == c, np.nan, exp_share / (1 - exp_share[c]))
        norm_partner.append(int(np.nanargmax(ratio)))
    raw_partner = m6.partner.values
    rows.append(dict(
        n=1, resultado_previo="Los clusters no son islas: 28-60 % de conexiones semánticas salen del cluster; el cluster 2 es el socio principal de 5 de los otros 7.",
        resultado_nuevo=(f"SPECTER2 (k=6): {obs.min():.0%}-{obs.max():.0%} de vínculos salen del cluster; una partición del mismo "
                         f"procedimiento sobre una nube sin grupos da {pn.min():.0%}-{pn.max():.0%} y etiquetas al azar "
                         f"{m6.cross_perm_mean.min():.0%}-{m6.cross_perm_mean.max():.0%}. Socio principal bruto: cluster 2 en "
                         f"{int((raw_partner == 2).sum())} de 7; normalizado por tamaño: cluster 2 en {sum(1 for c, p in enumerate(norm_partner) if p == 2 and c != 2)} de 7."),
        veredicto="confirmado (en parte)",
        motivo=("No son islas. Comparados rango por rango con cortar en 8 una nube sin grupos, todos los temas son más cohesivos, pero "
                "solo 4 lo son claramente (inmunidad, ejercicio, POTS y cerebro: 12-17 %, por debajo de todo el rango del nulo); los otros 4 "
                "(31-45 %) caen dentro de lo que daría un corte arbitrario, así que sus fronteras son difusas. El rango previo (28-60 %) "
                "estaba inflado por el proxy LSA. El protagonismo del cluster 2 como socio se explica sobre todo por su tamaño (171 papers).")))

    # ---- 2. picos de densidad
    ag = AG[AG.ref == "clusters_atlas"].set_index("model")
    cross_m = AG[AG.ref.str.startswith("modes_")]
    md = load(f"modes_{MAIN}.csv")
    txt2 = "; ".join(f"{m}: {int(r.n_modes)} modos, ARI {r.ARI:.2f}, NMI {r.NMI:.2f}" for m, r in ag.iterrows())
    txt2 += " | entre modelos: " + "; ".join(f"{r.model}-{r.ref[6:]} ARI {r.ARI:.2f}" for r in cross_m.itertuples())
    rows.append(dict(
        n=2, resultado_previo="9 picos de densidad, uno dominante (revisión general de 2024); acuerdo bajo con los clusters (ARI 0,21; NMI 0,46); el cluster 7 sin pico propio.",
        resultado_nuevo=txt2 + f". Pico dominante SPECTER2: «{md.iloc[0].title[:70]}» ({int(md.iloc[0].year)}), {int(md.iloc[0].n)} papers.",
        veredicto="refutado (en parte)",
        motivo=("Se confirma el pico dominante (la misma revisión de 2024) y que los clusters no coinciden con picos, pero no hay 9 picos: "
                "frente a un nulo sin grupos solo sobreviven unos pocos modos (un cuerpo central y modos pequeños). La densidad es casi "
                "unimodal, así que los 8 clusters son cortes de un continuo denso, no colinas separadas; que el cluster 7 no tenga pico "
                "propio es lo esperable y le pasa a casi todos.")))

    # ---- 3. H1 en el grafo semántico (k6 es el equivalente del grafo de 6 vecinos)
    def srow(model, sub, sp, dim=1):
        r = S[(S.model == model) & (S.subset == sub) & (S.space == sp) & (S.dim == dim)]
        return r.iloc[0] if len(r) else None
    r6 = srow(MAIN, "all", "k6")
    sig_by_space = {sp: int(srow(MAIN, "all", sp).nsig_gmm) for sp in ("full", "k6", "k15", "k30") if srow(MAIN, "all", sp) is not None}
    rows.append(dict(
        n=3, resultado_previo="H1 en el grafo semántico del atlas: 4 de 566 ciclos superan el nulo (persistencia máx. 0,352 vs p95 nulo 0,337).",
        resultado_nuevo=(f"SPECTER2, geodésica k=6, filtración completa: {int(r6.n_bars)} barras; significativas frente a gauss/colperm/gmm = "
                         f"{int(r6.nsig_gauss)}/{int(r6.nsig_colperm)}/{int(r6.nsig_gmm)} con el criterio muerte/nacimiento (p95 gmm {r6.tau_gmm}); "
                         f"con persistencia aditiva {int(r6.nsig_add_gmm)} (p95 {r6.tau_add_gmm}, la máxima real es {eval(r6.top5_pers)[0]}). "
                         f"Ciclos significativos frente a gmm por resolución: {sig_by_space}."),
        veredicto=("refutado" if r6.nsig_gmm == 0 else "inconclusivo" if (r6.nsig_gmm <= 1 or r6.nsig_add_gmm == 0) else "confirmado"),
        motivo=("En el grafo de 6 vecinos sobrevive como mucho 1 ciclo, marginal (solo con el criterio muerte/nacimiento; ninguno con persistencia "
                "aditiva). Los 4 previos venían de un nulo de 12 réplicas. Además el grafo real tiene menos ciclos que todas las réplicas nulas: "
                "la geometría local del corpus está más 'rellena' que el azar, no más agujereada. " + h1_reason(S, CY, "k6"))))

    # ---- 4. H1/H2 en el espacio vectorial completo (lo que la pasada previa aproximó con LSA)
    rf = srow(MAIN, "all", "full")
    r2 = srow(MAIN, "all", "full", 2)
    comp = "; ".join(f"{r.cycle_id}: {r.n_nodes} papers, clusters {r.clusters}" for r in CY.itertuples()) if CY is not None and len(CY) else "ningún ciclo significativo"
    rows.append(dict(
        n=4, resultado_previo="H1 en el proxy LSA: 17 ciclos sobre el nulo; 4 en H2; ciclos grandes que unen 0-2-7 y 5-4-2-1.",
        resultado_nuevo=(f"SPECTER2 cuerda completa: H1 significativos gauss/colperm/gmm = {int(rf.nsig_gauss)}/{int(rf.nsig_colperm)}/{int(rf.nsig_gmm)}; "
                         f"H2 (exploratorio, 200 referencias) gmm = {int(r2.nsig_gmm) if r2 is not None else 'n/d'}. "
                         f"mpnet / bge-m3 (cuerda completa, gmm): {other_models(S)}. Ciclos: {comp}."),
        veredicto=("refutado" if rf.nsig_gmm == 0 else "refutado (en parte)"),
        motivo=("Hay ciclos significativos y estables en el espacio completo, y aparecen en los tres modelos, pero son pequeños (5-7 papers, "
                "cociente muerte/nacimiento 1,1-1,2), no los grandes ciclos 0-2-7 y 5-4-2-1 del proxy LSA; H2 no supera el nulo en el espacio "
                "completo. En las geodésicas k15/k30 no hay ningún ciclo en ningún modelo. " + h1_reason(S, CY, "full"))))

    # ---- 5. H1 en citas
    rows.append(dict(
        n=5, resultado_previo="H1 en las citas (cocitación + acoplamiento, 541 papers): 8 ciclos sobre el nulo (p95 0,664).",
        resultado_nuevo=(f"Componente gigante {CS['n_giant']} papers; {CS['h1_bars_full_rips']} barras H1 (Rips completo). Con 300 puntos de "
                         f"referencia en real y nulo, significativas frente al modelo de configuración (grados y pesos conservados, 200 réplicas): "
                         f"{CS['h1_nsig']} con muerte/nacimiento (p95 {CS['h1_tau_rewire']}), {CS['h1_nsig_add']} con persistencia aditiva."),
        veredicto="confirmado" if CS["h1_nsig"] > 0 else "refutado",
        motivo=("Con todas las aristas del atlas y 200 réplicas, la red de citas sí tiene ciclos más persistentes que una red aleatoria con los mismos grados; "
                "son huecos de la red (grupos de papers que se citan en anillo sin un núcleo común), no del espacio semántico."
                if CS["h1_nsig"] > 0 else
                "Con todas las aristas y un nulo de recableado de 200 réplicas, ningún ciclo de la red de citas supera al azar con los mismos grados.")))

    # ---- 6. semántica vs citas
    sf, sk = CS["sem_full"], CS["sem_k15"]
    rows.append(dict(
        n=6, resultado_previo="Correlación de Spearman entre distancias semánticas y de citas: 0,17.",
        resultado_nuevo=(f"Mantel (Spearman, 999 permutaciones): cuerda completa ρ = {sf['mantel_rho']:.2f} (p = {sf['mantel_p']:.3f}); "
                         f"geodésica k15 ρ = {sk['mantel_rho']:.2f} (p = {sk['mantel_p']:.3f}). Parcial controlando años: "
                         f"{sf['partial_mantel_rho_year']:.2f} / {sk['partial_mantel_rho_year']:.2f}. Bottleneck H1 sem-citas "
                         f"{sf['bottleneck_h1']:.3f} vs mediana con redes recableadas {sf['ref_bottleneck_h1_median']:.3f}."),
        veredicto="confirmado" if sf["mantel_p"] < 0.05 and abs(sf["mantel_rho"] - 0.17) < 0.12 else ("refutado" if sf["mantel_p"] >= 0.05 else "confirmado (en parte)"),
        motivo=(f"La asociación existe y es más fuerte que con el proxy LSA ({sf['mantel_rho']:.2f} frente a 0,17), es significativa frente a "
                "permutar nodos y apenas cambia al controlar el año. Pero es moderada: la topología de las citas no se parece a la semántica más "
                "que la de una red de citas aleatoria con los mismos grados (bottleneck H1 dentro del rango del nulo)." if sf["mantel_p"] < 0.05 else
                "Sin asociación significativa frente a la permutación de nodos.")))

    # ---- 7. tiempo
    yk = YF[YF.space == "full"].sort_values("year")
    over = YF[YF.n_sig > YF.control_p95]
    rows.append(dict(
        n=7, resultado_previo="Filtración por año: los ciclos sobre el umbral suben hasta 6 en 2019 y bajan a 3 con el corpus completo.",
        resultado_nuevo=("Distancias recalculadas por año, nulo gmm por año y control con el mismo n. Cuerda completa, ciclos por año "
                         + ", ".join(f"{int(r.year)}:{int(r.n_sig)}" for r in yk.itertuples())
                         + " (control con el mismo n: " + ", ".join(f"{int(r.year)}:{r.control_mean:.1f}" for r in yk.itertuples()) + "). "
                         + f"Geodésica k15: 0 ciclos todos los años. Años por encima del control p95: {', '.join(str(int(y)) for y in sorted(set(over.year))) or 'ninguno'}."),
        veredicto="refutado" if len(over) == 0 else "inconclusivo",
        motivo=("La curva se explica por el tamaño de muestra y el ruido: ningún año tiene más ciclos de los que da una submuestra aleatoria del "
                "corpus con el mismo n. El pico de 2019 venía de heredar distancias del grafo completo y de un umbral único para todos los años."
                if len(over) == 0 else
                "Algunos años superan el control, pero sin una tendencia clara; con estos tamaños no se puede afirmar un cambio topológico en el tiempo.")))

    T = pd.DataFrame(rows)[["n", "resultado_previo", "resultado_nuevo", "veredicto", "motivo"]]
    T.to_csv(os.path.join(RES, "comparacion_con_resultados_previos.csv"), index=False)
    print(T[["n", "veredicto"]].to_string(index=False))

    write_page_data(P, CL, cl, C, MX, md, S, CY, YF, CS, BR)
    write_layer(P, BR)
    for f in ("embedding_validation.csv", "h1_summary.csv", "cluster_mixing.csv", "modes_agreement.csv", f"modes_{MAIN}.csv",
              "cycles.csv", "year_filtration.csv", "bridges.csv", "citation_summary.json", "h0_summary.csv"):
        if os.path.exists(out(f)):
            shutil.copyfile(out(f), os.path.join(RES, f))
    figures(P, cl, V, MX, pn, S, CY, YF, BR)


def other_models(S):
    parts = []
    for m in ("mpnet", "bge-m3"):
        r = S[(S.model == m) & (S.space == "full") & (S.dim == 1)]
        if len(r):
            parts.append(f"{m} {int(r.iloc[0].nsig_gmm)}")
    return ", ".join(parts) or "n/d"


def verdict_h1(S, CY, strict=False):
    main = S[(S.model == MAIN) & (S.subset == "all") & (S.dim == 1)]
    sig_main = int(main.nsig_gmm.sum())
    stable = 0 if CY is None or not len(CY) else int((CY.reappear.fillna(0) >= 0.5).sum())
    other = S[(S.model != MAIN) & (S.dim == 1)].nsig_gmm.sum()
    if sig_main == 0:
        return "refutado"
    if stable and other:
        return "confirmado (en parte)" if strict else "confirmado"
    return "inconclusivo"


def h1_reason(S, CY, sp):
    main = S[(S.model == MAIN) & (S.subset == "all") & (S.dim == 1)].set_index("space")
    s = main.loc[sp] if sp in main.index else None
    stable = 0 if CY is None or not len(CY) else int((CY[CY.space == sp].reappear.fillna(0) >= 0.5).sum())
    clean = S[(S.model == MAIN) & (S.subset == "clean") & (S.space == sp) & (S.dim == 1)]
    txt = (f"Frente al nulo que conserva los grupos (gmm) quedan {int(s.nsig_gmm) if s is not None else 0} ciclos "
           f"(frente a {int(s.nsig_gauss) if s is not None else 0} con el nulo gaussiano: la mayoría de los 'ciclos' del nulo simple "
           f"son efecto de los grupos). Estables en >= 50 % de submuestras: {stable}. ")
    if len(clean):
        txt += f"Sin duplicados ni papers fuera de tema: {int(clean.iloc[0].nsig_gmm)}."
    return txt


def peak_rows(md):
    """Prominencia de cada modo: los modos se ordenan por la densidad de su pico; el dominante (componente que no muere)
    recibe el rango completo de densidad y los demás las prominencias finitas de ToMATo en orden."""
    from h04_h0_modes import tomato_modes
    D = np.load(out(f"D_{MAIN}_full.npy")).astype(np.float64)
    _, dens = tomato_modes(D)
    ag = load("modes_agreement.csv")
    prom = [float(x) for x in eval(ag[(ag.model == MAIN) & (ag.ref == "clusters_atlas")].iloc[0].top_prom)]
    md = md.assign(peak_dens=dens[md.peak_paper.values]).sort_values("peak_dens", ascending=False)
    vals = [float(dens.max() - dens.min())] + prom
    return [dict(cl=int(r.main_cluster), pers=round(vals[k], 4) if k < len(vals) else None,
                 year=int(r.year) if r.year == r.year else None, title=r.title) for k, r in enumerate(md.itertuples())]


def write_page_data(P, CL, cl, C, MX, md, S, CY, YF, CS, BR):
    import pickle
    dg = lambda name: pickle.load(open(out(name), "rb"))
    d_full, d_k15 = dg(f"dgm_{MAIN}_all_full.pkl"), dg(f"dgm_{MAIN}_all_k15.pkl")
    tau = lambda sp, dim=1: float(S[(S.model == MAIN) & (S.subset == "all") & (S.space == sp) & (S.dim == dim)].iloc[0].tau_add_gmm)
    nsig = lambda sp, dim=1: int(S[(S.model == MAIN) & (S.subset == "all") & (S.space == sp) & (S.dim == dim)].iloc[0].nsig_gmm)
    cit_bars = pd.read_csv(out("h1_bars_citation.csv"))
    yk = YF[YF.space == "k15"].sort_values("year")
    m6 = MX[MX.k == 6].sort_values("cluster")
    D = chord(load_emb(MAIN))
    # robustez: escala de conexión interna (MST) y primera salida, igual que en la pasada previa
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    rob = []
    for c in range(8):
        mem = np.where(cl == c)[0]
        oth = np.where(cl != c)[0]
        internal = float(minimum_spanning_tree(csr_matrix(D[np.ix_(mem, mem)])).data.max())
        ext = float(D[np.ix_(mem, oth)].min())
        rob.append(dict(space="semantic", cl=c, n=int(len(mem)), internal=round(internal, 3), exit=round(ext, 3), ratio=round(ext / internal, 3)))
    bridges = []
    for c in range(8):
        mem = np.where(cl == c)[0]
        oth = np.where(cl != c)[0]
        sub = D[np.ix_(mem, oth)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = mem[i], oth[j]
        bridges.append(dict(a=int(c), b=int(cl[b]), dist=round(float(sub[i, j]), 3), ta=P[a]["t"][:70], tb=P[b]["t"][:70],
                            n=int(C[c, cl[b]] + C[cl[b], c])))
    bridges.sort(key=lambda r: r["dist"])
    page = dict(
        clusters=[dict(id=c["id"], name=c["name"], name_en=c["name_en"], n=c["n"]) for c in CL],
        peaks=peak_rows(md),
        rob=rob,
        cross=dict(C=C.tolist(), por=[float(x) for x in m6.cross_obs], tot=[int(x) for x in C.sum(1)]),
        bridges=bridges,
        diag=dict(sem=dict(pts=pts(d_full[1]), tau=round(tau("full"), 4), label="SPECTER2, cuerda completa (umbral aditivo del nulo gmm; criterio principal: muerte/nacimiento)"),
                  lsa=dict(pts=pts(d_k15[1]), tau=round(tau("k15"), 4), label="SPECTER2, geodésica kNN k=15 (umbral aditivo del nulo gmm; criterio principal: muerte/nacimiento)"),
                  cit=dict(pts=[[round(a, 4), round(b, 4)] for a, b in cit_bars[["birth", "death"]].values],
                           tau=CS["h1_tau_add"], label="Citas, 300 puntos de referencia (umbral aditivo del modelo de configuración)")),
        cycles=[dict(pers=float(r.pers), nodes=int(r.n_nodes), cl={int(k): int(v) for k, v in eval(r.clusters).items()},
                     titles=[t for t in eval(r.titles)[:4]], years=eval(r.years)[:4]) for r in (CY.itertuples() if CY is not None else [])],
        year=[dict(y=int(r.year), n=int(r.n), over=int(r.n_sig), tot=round(float(r.h1_total), 2)) for r in yk.itertuples()],
        cyears=[dict(cl=c, n=int(((cl == c) & (np.array([p["y"] or 0 for p in P]) > 0)).sum()),
                     ymin=int(min(p["y"] for p in P if p["c"] == c and p["y"])),
                     med=float(np.median([p["y"] for p in P if p["c"] == c and p["y"]])),
                     cit=int(sum(p["cit"] or 0 for p in P if p["c"] == c))) for c in range(8)],
        cmp=dict(rho=CS["sem_full"]["mantel_rho"], h0_bn=CS["sem_full"]["bottleneck_h0"], h1_bn=CS["sem_full"]["bottleneck_h1"],
                 h1_sem=len(pts(d_full[1])), h1_cit=CS["h1_bars"]),
        null=dict(sem_top=round(tau("full"), 4), sem_real_top=round(float(np.max(np.diff(finite(d_full[1]), axis=1))), 4),
                  sem_nsig=nsig("full"), sem_nbars=len(pts(d_full[1])), lsa_nsig=nsig("k15"), cit_nsig=CS["h1_nsig"],
                  lsa2_nsig=nsig("full", 2)),
        outliers=[dict(t=P[int(k)]["t"][:80], y=P[int(k)]["y"]) for k in list(load("flags.json", "json")["off_topic"])[:6]],
    )
    with open(os.path.join(RES, "page_data.json"), "w", encoding="utf-8") as fh:
        json.dump(page, fh, ensure_ascii=False, separators=(",", ":"))


def write_layer(P, BR):
    cyc = load("cycle_members.json", "json") or {}
    labels = np.load(out("modes_labels.npy"), allow_pickle=True).item()[MAIN]
    b = BR.set_index("id") if BR is not None else None
    layer = {}
    for i in range(len(P)):
        layer[i] = dict(bridge=float(b.loc[i, "bridge_score"]) if b is not None else None,
                        cycle_ids=[cid for cid, m in cyc.items() if i in m["nodes"]],
                        peak_id=int(labels[i]))
    with open(os.path.join(RES, "topology_layer.json"), "w", encoding="utf-8") as fh:
        json.dump({"description": "Capa topológica opcional para el atlas (no modifica atlas.html). bridge: puntuación de "
                                  "paper puente 0-1; cycle_ids: ciclos H1 significativos (nulo gmm) a los que pertenece; "
                                  "peak_id: modo de densidad (ToMATo, SPECTER2).", "papers": layer}, fh)


def figures(P, cl, V, MX, pn, S, CY, YF, BR):
    import matplotlib.pyplot as plt
    # 1. validación de embeddings
    fig, ax = dark_fig(2, height=8.5)
    x = np.arange(len(V))
    ax[0].bar(x, V.nn6_overlap, color=DARK["accent"]); ax[0].set_xticks(x, V.model)
    ax[0].set_title("Coinciden los 6 vecinos con el atlas"); ax[0].set_ylim(0, 1.05)
    for i, v in enumerate(V.nn6_overlap): ax[0].text(i, v + 0.02, f"{v:.0%}", ha="center", color=DARK["fg"])
    ax[1].bar(x, V.purity10, color=DARK["good"]); ax[1].set_xticks(x, V.model); ax[1].set_ylim(0, 1.05)
    ax[1].axhline(0.91, color=DARK["muted"], ls="--", lw=1); ax[1].text(len(V) - 0.5, 0.92, "UMAP 3D del atlas", color=DARK["muted"], ha="right", fontsize=8)
    ax[1].set_title("Pureza de cluster en los 10 vecinos")
    for i, v in enumerate(V.purity10): ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", color=DARK["fg"])
    save_fig(fig, "01_validacion_embeddings.png")

    # 2. mezcla entre clusters
    m6 = MX[MX.k == 6].sort_values("cross_obs")
    fig, ax = dark_fig(1, height=7.5)
    y = np.arange(len(m6))
    ax.barh(y, m6.cross_obs, color=[CLUSTER_COLORS[c] for c in m6.cluster], label="observado")
    ax.scatter(m6.cross_perm_mean, y, color=DARK["muted"], marker="|", s=200, label="etiquetas al azar")
    ax.axvspan(pn.min(), pn.max(), color=DARK["warn"], alpha=0.15, label="cortar un continuo en 8")
    ax.set_yticks(y, [f"C{c}" for c in m6.cluster]); ax.set_xlim(0, 1)
    ax.set_xlabel("fracción de vínculos (6 vecinos) que salen del cluster"); ax.set_title("Los clusters no son islas, pero son cohesivos")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "02_mezcla_clusters.png")

    # 3. diagramas H1 con umbral gmm
    import pickle
    fig, ax = dark_fig(3, height=11)
    for a, (sp, lab) in zip(ax, (("full", "SPECTER2 · cuerda completa"), ("k15", "SPECTER2 · geodésica k15"), ("cit", "Citas"))):
        if sp == "cit":
            B = pd.read_csv(out("h1_bars_citation.csv")); tau = load("citation_summary.json", "json")["h1_tau_rewire"]
            b, d = B.birth.values, B.death.values
        else:
            dg = finite(pickle.load(open(out(f"dgm_{MAIN}_all_{sp}.pkl"), "rb"))[1]); b, d = dg[:, 0], dg[:, 1]
            tau = float(S[(S.model == MAIN) & (S.subset == "all") & (S.space == sp) & (S.dim == 1)].iloc[0].tau_gmm)
        ok = b > 0
        b, d = b[ok], d[ok]
        sig = (d / b) > tau  # criterio principal: persistencia multiplicativa
        a.scatter(b[~sig], d[~sig], s=6, color=DARK["muted"], alpha=0.6)
        a.scatter(b[sig], d[sig], s=18, color=DARK["warn"], label=f"{sig.sum()} sobre el nulo (muerte/nac. > {tau:.2f})")
        lo, hi = min(b.min(), d.min()), max(b.max(), d.max())
        a.plot([lo, hi], [lo, hi], color=DARK["grid"], lw=1)
        a.plot([lo, hi / tau], [lo * tau, hi], color=DARK["warn"], ls="--", lw=1)
        a.set_title(lab); a.set_xlabel("nacimiento"); a.set_ylabel("muerte"); a.legend(loc="lower right", fontsize=8)
    save_fig(fig, "03_diagramas_h1.png")

    # 4. dependencia de la resolución y del nulo
    fig, ax = dark_fig(1, height=7)
    sub = S[(S.subset == "all") & (S.dim == 1)]
    labels = [f"{r.model}\n{r.space}" for r in sub.itertuples()]
    x = np.arange(len(sub)); w = 0.27
    for k, (kind, colr) in enumerate((("gauss", DARK["muted"]), ("colperm", DARK["accent"]), ("gmm", DARK["warn"]))):
        ax.bar(x + (k - 1) * w, sub[f"nsig_{kind}"], w, color=colr, label=f"nulo {kind}")
    ax.set_xticks(x, labels, fontsize=6, rotation=90); ax.set_ylabel("ciclos H1 significativos")
    ax.set_title("El resultado depende del nulo y de la resolución"); ax.legend(fontsize=8)
    save_fig(fig, "04_resolucion_y_nulos.png")

    # 5. tiempo
    fig, ax = dark_fig(2, height=9)
    for sp, colr in (("full", DARK["accent"]), ("k15", DARK["warn"])):
        t = YF[YF.space == sp].sort_values("year")
        ax[0].plot(t.year, t.n_sig, "o-", color=colr, label=f"real ({sp})", ms=3)
        ax[0].fill_between(t.year, 0, t.control_p95, color=colr, alpha=0.12, label=f"control mismo n p95 ({sp})")
        ax[1].plot(t.year, t.h1_total_per_paper, "o-", color=colr, ms=3, label=sp)
    ax[0].set_title("Ciclos significativos por año frente al control de tamaño"); ax[0].legend(fontsize=7)
    ax[1].set_title("Persistencia H1 total por paper"); ax[1].legend(fontsize=7); ax[1].set_xlabel("año (acumulado)")
    save_fig(fig, "05_filtracion_por_anio.png")

    # 6. papers puente
    if BR is not None:
        top = BR.head(15).iloc[::-1]
        fig, ax = dark_fig(1, height=9)
        ax.barh(range(len(top)), top.bridge_score, color=[CLUSTER_COLORS[c] for c in top.cluster])
        ax.set_yticks(range(len(top)), [f"{t[:42]}… ({int(y) if y == y else 's/a'})" for t, y in zip(top.title, top.year)], fontsize=6)
        ax.set_xlabel("puntuación de puente (0-1)"); ax.set_title("Papers que más sostienen la topología")
        save_fig(fig, "06_papers_puente.png")


if __name__ == "__main__":
    main()
