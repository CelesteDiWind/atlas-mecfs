"""Paso o22: análisis temporales del corpus OpenAlex -> out/o22_temporal.json (05_build.py lo incrusta en la página
«Evolución» del visor, #e; o23_temporal_figs.py lo dibuja). Todo se calcula una vez por filtro de relevancia del visor:
all = todos; me = centrales y secundarios; core = solo centrales (estos dos, sin retractados).
  growth  trabajos por tema y año; crecimiento de la cuota de cada tema (GLM binomial con cuasi-verosimilitud, 2010-2025)
  cites   citas directas internas por tema y año del trabajo que cita: cuántas se quedan en el tema (frente a repartirlas al
          azar entre lo citado ese año), cuántas recibe de otros temas; y qué parte de todas las referencias apunta al corpus
  bursts  estallidos de términos (Kleinberg, dos estados, por año) en título + resumen + palabras clave
  sem     deriva semántica (SPECTER2) por ventana de 3 años: similitud media entre pares de papers dentro y entre temas
          (estimador sin sesgo por tamaño), similitud corregida entre centroides, deriva de cada tema, términos por periodo
  path    ruta principal de citas (SPC: búsqueda global y key-route con las 15 aristas de más tráfico)
  geo     países (conteo completo), especialización país x tema, autores por trabajo, colaboración internacional, autores nuevos
  rel     relevancia por año (solo en all)
  pub     acceso abierto, revisiones, revistas por periodo, concentración
  impact  citas normalizadas por año de publicación (MNCS) y fracción en el 10 % más citado de su año
  age     edad de las referencias e índice de Price (años de las referencias externas: o21)"""
import collections
import json
import os
import re

import numpy as np
import statsmodels.api as sm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(OA))
OUT = os.path.join(OA, "out")
K, Y0, Y1 = 8, 2010, 2026
YRS = list(range(Y0, Y1 + 1))
NY = len(YRS)
WINDOWS = [(y, y + 2) for y in range(Y0, Y1 - 1)]
PERIODS = [(2010, 2013), (2014, 2017), (2018, 2021), (2022, 2026)]
SEED = 20261021
GENERIC = set("""myalgic encephalomyelitis encephalopathy chronic fatigue syndrome syndromes cfs mecfs cfsme patients patient study
studies results result conclusion conclusions methods method background objective objectives aim aims purpose using used use
associated association compared comparison significant significantly group groups participants data disease diseases symptoms
symptom including included include based analysis analyses performed showed shown found findings research review evidence
health clinical treatment treatments condition conditions illness individuals people subjects controls control healthy level
levels effect effects factors factor risk increased higher lower high low reported report years year potential role
present possible new different however may also total number related identified investigated observed et al copyright rights
reserved elsevier published authors article sample samples case cases mean months time measures measured assessed http www
university author human examine examined aimed aim abstract subsequent conflict interest source journal required selection
observe difference treated treating useful agent activity percent consecutive aged collected period received various issue
outcomes outcome specific structured improved type order abundance regarding reduction known open modern introduction plan
referred seven tool science like future conducted impact completed paper materials measurements pattern problems
advertisement van och att med und der die london america large hypothesis day women men versus vs particular looking
age sex doi org https scored""".split())
UK_US = [(r"behaviour", "behavior"), (r"paediatric", "pediatric"), (r"randomis", "randomiz"), (r"haematolog", "hematolog"),
         (r"oestrogen", "estrogen"), (r"programme", "program"), (r"centre", "center")]


def normalize(t):
    t = re.sub(r"covid-19|covid 19|sars-cov-2", "covid", t.lower())
    for a, b in UK_US:
        t = re.sub(a, b, t)
    return t
# palabras que crecen en los resúmenes biomédicos desde 2023 sin relación con el contenido, asociadas a la redacción con LLM
# (Kobak et al. 2024, «Delving into ChatGPT usage in academic writing through excess vocabulary»). Se excluyen de los
# estallidos de contenido y se miden aparte (índice de estilo).
STYLE = set("""delve delves delving underscore underscores underscoring showcase showcases showcasing comprehensive insights
insight highlight highlights highlighting crucial notably pivotal intricate intricacies multifaceted landscape nuanced
emphasizing emphasize emphasizes enhancing framework frameworks challenges remain remains characterized shared elucidate
elucidating foster fostering paving valuable unveiling holistic robust additionally particularly offering strategies
interplay""".split())
STYLE_STRONG = ["delv", "underscor", "showcas", "intricate", "pivotal", "multifaceted", "nuanced", "notably", "comprehensive",
                "insight", "crucial", "interplay"]
TOKEN = r"(?u)\b[a-z][a-z]+(?:-[a-z0-9]+)*\b"


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


# ---------------------------------------------------------------- crecimiento
def growth(sel, lab, yr):
    n = np.zeros((K, NY), int)
    np.add.at(n, (lab[sel], yr[sel] - Y0), 1)
    N = n.sum(0)
    trend = []
    full = np.array(YRS) <= 2025
    x = sm.add_constant(np.array(YRS)[full] - 2018.0)
    for c in range(K):
        y_ = np.column_stack([n[c, full], N[full] - n[c, full]])
        try:
            # cuasi-binomial: errores inflados por la sobredispersión de Pearson (scale="X2" de statsmodels la calcula
            # sobre proporciones cuando la respuesta es éxitos/fracasos, y da intervalos demasiado estrechos)
            f = sm.GLM(y_, x, family=sm.families.Binomial()).fit()
            phi = max(1.0, f.pearson_chi2 / f.df_resid)
            b, se = float(f.params[1]), float(f.bse[1]) * np.sqrt(phi)
            trend.append(dict(b=r3(b), lo=r3(b - 1.96 * se), hi=r3(b + 1.96 * se)))
        except Exception:
            trend.append(None)
    return dict(n=n.tolist(), trend=trend)


# ---------------------------------------------------------------- citas dentro / fuera
def cites(sel, lab, yr, a, b, refs_n, refs_in, refs_post):
    e = sel[a] & sel[b]
    a, b = a[e], b[e]
    ya = yr[a] - Y0
    out_n = np.zeros((K, NY), int); self_n = np.zeros((K, NY), int)
    np.add.at(out_n, (lab[a], ya), 1)
    np.add.at(self_n, (lab[a], ya), (lab[a] == lab[b]).astype(int))
    tgt = np.zeros((K, NY), float)                      # reparto de lo citado en cada año
    np.add.at(tgt, (lab[b], ya), 1)
    share = tgt / np.maximum(tgt.sum(0), 1)
    exp_self = out_n * share
    in_n = np.zeros((K, NY), int); in_other = np.zeros((K, NY), int)
    np.add.at(in_n, (lab[b], ya), 1)
    np.add.at(in_other, (lab[b], ya), (lab[a] != lab[b]).astype(int))
    # referencias al propio corpus (a trabajos del mismo filtro) frente a todas las referencias, por año y tema
    tot = np.zeros((K, NY), int); inn = np.zeros((K, NY), int); post = np.zeros((K, NY), int)
    s = sel & (refs_n > 0)
    np.add.at(tot, (lab[s], yr[s] - Y0), refs_n[s])
    np.add.at(inn, (lab[s], yr[s] - Y0), refs_in[s])
    np.add.at(post, (lab[s], yr[s] - Y0), refs_post[s])
    return dict(out=out_n.tolist(), self=self_n.tolist(), exp=np.round(exp_self, 1).tolist(), inn=in_n.tolist(),
                in_other=in_other.tolist(), refs=tot.tolist(), refs_in=inn.tolist(), refs_post=post.tolist())


# ---------------------------------------------------------------- estallidos de términos (Kleinberg)
def kleinberg(r, d, s=2.0, gamma=1.0):
    """Dos estados (base y estallido) sobre lotes por año. Devuelve [(inicio, fin, peso)] en índices de año."""
    R, D = r.sum(), d.sum()
    p0 = R / D
    p1 = min(s * p0, 0.9999)
    ok = d > 0
    def cost(p):
        c = np.zeros(len(r))
        c[ok] = -(r[ok] * np.log(p) + (d[ok] - r[ok]) * np.log(1 - p))
        return c
    c0, c1 = cost(p0), cost(p1)
    T = len(r); up = gamma * np.log(T)
    # Viterbi
    C = np.zeros((T, 2)); B = np.zeros((T, 2), int)
    C[0] = [c0[0], c1[0] + up]
    for t in range(1, T):
        C[t, 0] = min(C[t - 1, 0], C[t - 1, 1]) + c0[t]; B[t, 0] = int(C[t - 1, 1] < C[t - 1, 0])
        st = [C[t - 1, 0] + up, C[t - 1, 1]]
        C[t, 1] = min(st) + c1[t]; B[t, 1] = int(np.argmin(st))
    q = np.zeros(T, int); q[-1] = int(np.argmin(C[-1]))
    for t in range(T - 1, 0, -1):
        q[t - 1] = B[t, q[t]]
    out, t = [], 0
    while t < T:
        if q[t] == 1:
            t0 = t
            while t < T and q[t] == 1:
                t += 1
            out.append((t0, t - 1, float((c0[t0:t] - c1[t0:t]).sum())))
        else:
            t += 1
    return out


BUCKETS = [(2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024), (2025, 2026)]


def bursts(sel, docs, yr, per_bucket=6):
    idx = np.where(sel)[0]
    cv = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=25, stop_words=sorted(ENGLISH_STOP_WORDS | GENERIC | STYLE),
                         token_pattern=TOKEN)
    X = cv.fit_transform([docs[i] for i in idx]).tocsc()
    terms = cv.get_feature_names_out()
    Y = np.zeros((len(idx), NY), bool); Y[np.arange(len(idx)), yr[idx] - Y0] = True
    d = Y.sum(0).astype(float)
    cand = []
    for j in range(X.shape[1]):
        rows = X.indices[X.indptr[j]:X.indptr[j + 1]]
        r = np.bincount(yr[idx[rows]] - Y0, minlength=NY).astype(float)
        for t0, t1, w in kleinberg(r, d):
            if t1 - t0 >= 0 and r[t0:t1 + 1].sum() >= 15:
                cand.append((w, j, t0, t1, rows))
    cand.sort(key=lambda x: -x[0])
    # un mismo estallido aparece con varias formas («long covid», «covid», «post covid»): se agrupan las que comparten
    # casi los mismos papers y se queda la de más peso, con las demás como alias
    kept, per = [], collections.Counter()
    bucket = lambda t0: next(k for k, (a, b) in enumerate(BUCKETS) if a <= YRS[t0] <= b)
    for w, j, t0, t1, rows in cand:
        S = set(rows.tolist())
        dup = None
        for k in kept:
            inter = len(S & k["rows"])
            if inter / min(len(S), len(k["rows"])) >= 0.7 and not (t1 < k["t0"] - 1 or t0 > k["t1"] + 1):
                dup = k; break
        if dup is not None:
            if len(dup["alias"]) < 4 and terms[j] not in dup["alias"]:
                dup["alias"].append(terms[j])
            continue
        if per[bucket(t0)] >= per_bucket:
            continue
        per[bucket(t0)] += 1
        kept.append(dict(term=terms[j], w=w, t0=t0, t1=t1, rows=S, alias=[]))
    res = []
    for k in kept:
        r = np.bincount(yr[idx[list(k["rows"])]] - Y0, minlength=NY)
        res.append(dict(term=k["term"], alias=k["alias"], y0=YRS[k["t0"]], y1=YRS[k["t1"]], w=round(k["w"], 1), n=int(r.sum()),
                        f=[round(float(v), 4) for v in r / np.maximum(d, 1)]))
    res.sort(key=lambda x: (x["y0"], -x["w"]))
    return res


# ---------------------------------------------------------------- estilo de redacción
def style(sel, docs, yr):
    """Fracción de resúmenes en inglés con al menos una palabra marcadora de redacción con LLM, por año, y frecuencia de cada una."""
    n = np.bincount(yr[sel] - Y0, minlength=NY)
    hit = np.zeros(NY, int)
    per = {w: np.zeros(NY, int) for w in STYLE_STRONG}
    for i in np.where(sel)[0]:
        found = [w for w in STYLE_STRONG if re.search(r"\b" + w, docs[i])]
        if found:
            hit[yr[i] - Y0] += 1
        for w in found:
            per[w][yr[i] - Y0] += 1
    base = hit[:13].sum() / n[:13].sum()                     # 2010-2022: antes de ChatGPT
    return dict(n=n.tolist(), hit=hit.tolist(), base=r3(base), words={w: v.tolist() for w, v in per.items()})


# ---------------------------------------------------------------- deriva semántica
def pair_sim(Sa, na, Sb=None, nb=None):
    """Similitud media (coseno) entre pares de papers: dentro de un grupo sin contar cada paper consigo mismo, o entre dos."""
    if Sb is None:
        return (Sa @ Sa - na) / (na * (na - 1)) if na > 1 else np.nan
    return (Sa @ Sb) / (na * nb)


def semantic(sel, lab, yr, E, tdocs, txt, MIN=15):
    """Todo con distancias entre centroides estimadas sin sesgo a partir de las similitudes medias entre pares
    (|mu_s - mu_t|^2 = M_ss + M_tt - 2 M_st), que no dependen de dónde esté el origen del espacio ni del tamaño de cada grupo.
    spread_s = 1 - M_ss es la dispersión interna de un tema (media de |x - x'|^2 / 2 entre sus pares)."""
    res = dict(windows=[f"{a}-{b}" for a, b in WINDOWS], spread=[], d2=[], sep=[], n=[], field=[], drift=[[] for _ in range(K)])
    sums, ns, Ms = {}, {}, {}
    for w, (a, b) in enumerate(WINDOWS):
        m = sel & (yr >= a) & (yr <= b)
        S = np.zeros((K, E.shape[1])); n = np.zeros(K, int)
        for c in range(K):
            mc = m & (lab == c)
            n[c] = mc.sum(); S[c] = E[mc].sum(0)
        sums[w], ns[w] = S, n
        M = np.full((K, K), np.nan)
        for s in range(K):
            if n[s] < MIN:
                continue
            M[s, s] = pair_sim(S[s], n[s])
            for t in range(s + 1, K):
                if n[t] >= MIN:
                    M[s, t] = M[t, s] = pair_sim(S[s], n[s], S[t], n[t])
        dg = np.diag(M)
        D2 = dg[:, None] + dg[None, :] - 2 * M                  # distancia al cuadrado entre centroides
        sp = 1 - dg
        # separación entre temas: distancia media entre centroides (sin XMRV, que solo existe en 2010-12) frente a la
        # dispersión media dentro de los temas. Más alta = temas más distintos en contenido.
        ok = [c for c in range(K) if c != 1 and n[c] >= MIN]
        pairs = [D2[s, t] for i, s in enumerate(ok) for t in ok[i + 1:]]
        res["sep"].append(r3(np.mean(pairs) / np.mean(2 * sp[ok])))
        res["spread"].append([r3(v) for v in sp])
        res["d2"].append([[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in D2])
        res["n"].append(n.tolist())
        res["field"].append(r3(1 - pair_sim(S.sum(0), n.sum())))   # dispersión de todo el campo en la ventana
    # deriva de cada tema: distancia de su centroide al de su primera ventana con tamaño suficiente, en unidades de su
    # dispersión interna en esa ventana (0 = no se ha movido; 1 = tanto como dista un paper típico de otro del mismo tema / 2)
    for c in range(K):
        ref = next((w for w in range(len(WINDOWS)) if ns[w][c] >= 30), None)
        for w in range(len(WINDOWS)):
            if ref is None or ns[w][c] < MIN:
                res["drift"][c].append(None); continue
            Sa, na, Sb, nb = sums[ref][c], ns[ref][c], sums[w][c], ns[w][c]
            d2 = pair_sim(Sa, na) + pair_sim(Sb, nb) - 2 * pair_sim(Sa, na, Sb, nb) if w != ref else 0.0
            res["drift"][c].append(r3(d2 / (1 - pair_sim(Sa, na))))
        res.setdefault("drift_ref", []).append(None if ref is None else res["windows"][ref])
    # términos propios de cada periodo dentro de cada tema (log-odds con prior de Dirichlet informativo, Monroe et al. 2008),
    # con título y palabras clave: el resumen añade sobre todo cambios de estilo
    cv = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=5, stop_words=sorted(ENGLISH_STOP_WORDS | GENERIC | STYLE),
                         token_pattern=TOKEN)
    idx = np.where(sel & txt)[0]
    X = cv.fit_transform([tdocs[i] for i in idx]).tocsr()
    terms = cv.get_feature_names_out()
    prior = np.asarray(X.sum(0)).ravel() + 0.01
    a0 = prior.sum()
    prior_w = prior / a0 * 500                                   # fuerza del prior
    words = [[None] * len(PERIODS) for _ in range(K)]
    for c in range(K):
        mc = lab[idx] == c
        for p, (a, b) in enumerate(PERIODS):
            mp = mc & (yr[idx] >= a) & (yr[idx] <= b)
            mr = mc & ~((yr[idx] >= a) & (yr[idx] <= b))
            if mp.sum() < 15 or mr.sum() < 15:
                continue
            yi = np.asarray(X[mp].sum(0)).ravel(); yj = np.asarray(X[mr].sum(0)).ravel()
            ni, nj, a_ = yi.sum(), yj.sum(), prior_w.sum()
            d = np.log((yi + prior_w) / (ni + a_ - yi - prior_w)) - np.log((yj + prior_w) / (nj + a_ - yj - prior_w))
            z = d / np.sqrt(1 / (yi + prior_w) + 1 / (yj + prior_w))
            order = [j for j in np.argsort(-z) if yi[j] >= max(6, 0.04 * mp.sum())]
            pick = []
            for j in order:                                       # sin repetir un unigrama dentro de un bigrama ya elegido
                t = terms[j]
                if any(t in q.split() or q in t.split() for q in pick):
                    continue
                pick.append(t)
                if len(pick) == 6:
                    break
            words[c][p] = dict(n=int(mp.sum()), terms=pick)
    res["periods"] = [f"{a}-{b}" for a, b in PERIODS]
    res["terms"] = words
    return res


# ---------------------------------------------------------------- ruta principal de citas
def main_path(sel, a, b, W, lab, yr, cit, n_key=15):
    """Red cita -> citado convertida en flujo de conocimiento citado -> citante (j -> i). Los índices siguen el orden temporal
    de works.json, así que se quitan las citas a trabajos con fecha posterior (versiones en línea antes de la definitiva)
    y la red queda acíclica. SPC (search path count): nº de caminos fuente-sumidero que pasan por cada arista."""
    e = sel[a] & sel[b]
    src, dst = b[e], a[e]
    fwd = src < dst
    dropped = int((~fwd).sum())
    src, dst = src[fwd], dst[fwd]
    nodes = np.unique(np.r_[src, dst])
    succ, pred = collections.defaultdict(list), collections.defaultdict(list)
    for u, v in zip(src.tolist(), dst.tolist()):
        succ[u].append(v); pred[v].append(u)
    order = sorted(nodes.tolist())
    Nm, Np = {}, {}
    for v in order:
        Nm[v] = 1 if not pred[v] else sum(Nm[u] for u in pred[v])
    for v in reversed(order):
        Np[v] = 1 if not succ[v] else sum(Np[w] for w in succ[v])
    spc = {(u, v): Nm[u] * Np[v] for u, v in zip(src.tolist(), dst.tolist())}
    total = sum(Np[v] for v in order if not pred[v])
    # búsqueda global: el camino fuente-sumidero de mayor SPC total
    best, back = {}, {}
    for v in order:
        if not pred[v]:
            best[v] = 0; back[v] = None
        else:
            u = max(pred[v], key=lambda u: best[u] + spc[(u, v)])
            best[v] = best[u] + spc[(u, v)]; back[v] = u
    end = max((v for v in order if not succ[v]), key=lambda v: best[v])
    gpath = []
    v = end
    while v is not None:
        gpath.append(v); v = back[v]
    gpath = gpath[::-1]
    # key-route: las n_key aristas de más tráfico, prolongadas hacia atrás y hacia delante por la arista local de más SPC
    top = sorted(spc, key=lambda k: -spc[k])[:n_key]
    kr = set()
    for u, v in top:
        kr.add((u, v))
        x = u
        while pred[x]:
            y = max(pred[x], key=lambda y: spc[(y, x)]); kr.add((y, x)); x = y
        x = v
        while succ[x]:
            y = max(succ[x], key=lambda y: spc[(x, y)]); kr.add((x, y)); x = y
    gset = set(zip(gpath[:-1], gpath[1:]))
    kr |= gset
    kn = sorted({u for e_ in kr for u in e_})
    lg = lambda v: round(float(np.log10(float(v))), 2)
    def node(i):
        au = W[i]["authors"] or []
        return dict(id=int(i), y=int(yr[i]), c=int(lab[i]), t=W[i]["title"][:160], a=(au[0] if au else "") + (" et al." if len(au) > 1 else ""),
                    cit=int(cit[i]), main=i in gpath)
    return dict(dropped=dropped, edges=int(len(src)), total=lg(total), nodes=[node(i) for i in kn], main=[int(i) for i in gpath],
                links=[[int(u), int(v), lg(spc[(u, v)]), int((u, v) in gset)] for u, v in sorted(kr)])


# ---------------------------------------------------------------- geografía y colaboración
def geo(sel, W, lab, yr, topn=12):
    idx = [i for i in np.where(sel)[0] if W[i]["countries"]]
    cnt = collections.Counter(c for i in idx for c in W[i]["countries"])
    top = [c for c, _ in cnt.most_common(topn)]
    with_c = np.zeros(NY, int); by = {c: np.zeros(NY, int) for c in top}
    for i in idx:
        with_c[yr[i] - Y0] += 1
        for c in W[i]["countries"]:
            if c in by:
                by[c][yr[i] - Y0] += 1
    # especialización: cuota del tema entre los trabajos del país / cuota del tema en el corpus (RCA, >1 = especializado)
    nt = np.bincount(lab[idx], minlength=K).astype(float)
    rca = []
    for c in top:
        m = [i for i in idx if c in W[i]["countries"]]
        nc = np.bincount(lab[m], minlength=K).astype(float)
        rca.append([r3(v) for v in (nc / nc.sum()) / (nt / nt.sum())])
    # colaboración por año
    au_mean, au_med, single, intl, newc = [], [], [], [], []
    first = {}
    for i in sorted(np.where(sel)[0], key=lambda i: yr[i]):
        for a in W[i]["author_ids"] or []:
            first.setdefault(a, yr[i])
    for y in YRS:
        m = [i for i in np.where(sel & (yr == y))[0] if W[i]["author_ids"]]
        k = np.array([len(W[i]["author_ids"]) for i in m]) if m else np.array([0])
        au_mean.append(r3(k.mean())); au_med.append(r3(np.median(k))); single.append(r3((k == 1).mean()))
        mc = [i for i in m if W[i]["countries"]]
        intl.append(r3(np.mean([len(W[i]["countries"]) > 1 for i in mc])) if mc else None)
        A = {a for i in m for a in W[i]["author_ids"]}
        newc.append(r3(np.mean([first[a] == y for a in A])) if A and y >= 2012 else None)
    return dict(top=top, total={c: cnt[c] for c in top}, papers_with=with_c.tolist(), by={c: v.tolist() for c, v in by.items()},
                rca=rca, authors_mean=au_mean, authors_median=au_med, single=single, intl=intl, newcomers=newc,
                coverage=r3(len(idx) / max(1, sel.sum())))


# ---------------------------------------------------------------- publicación: acceso abierto, revisiones, revistas
def pub(sel, W, extra, lab, yr):
    oa = np.array([bool(extra.get(w["key"], {}).get("oa")) for w in W])
    rev = np.array([w["type"] == "review" for w in W])
    ven = [extra.get(w["key"], {}).get("v", "") for w in W]
    n_y = np.bincount(yr[sel] - Y0, minlength=NY)
    oa_y = np.bincount(yr[sel & oa] - Y0, minlength=NY)
    rev_y = np.bincount(yr[sel & rev] - Y0, minlength=NY)
    oa_t = [r3((sel & oa & (lab == c)).sum() / max(1, (sel & (lab == c)).sum())) for c in range(K)]
    rev_t = [r3((sel & rev & (lab == c)).sum() / max(1, (sel & (lab == c)).sum())) for c in range(K)]
    venues = []
    for a, b in PERIODS:
        cc = collections.Counter(ven[i] for i in np.where(sel & (yr >= a) & (yr <= b))[0] if ven[i])
        tot = sum(cc.values())
        venues.append(dict(p=f"{a}-{b}", n=tot, top=[[v, k] for v, k in cc.most_common(8)]))
    distinct, top10 = [], []
    for y in YRS:
        cc = collections.Counter(ven[i] for i in np.where(sel & (yr == y))[0] if ven[i])
        tot = sum(cc.values())
        distinct.append(len(cc)); top10.append(r3(sum(k for _, k in cc.most_common(10)) / tot) if tot else None)
    return dict(n=n_y.tolist(), oa=oa_y.tolist(), review=rev_y.tolist(), oa_topic=oa_t, review_topic=rev_t, venues=venues,
                distinct=distinct, top10=top10)


# ---------------------------------------------------------------- impacto normalizado por cohorte
def impact(sel, lab, yr, cit, rng, last=2023):
    cit = cit.astype(float)
    mncs = np.full(len(cit), np.nan); pct = np.full(len(cit), np.nan)
    for y in YRS:
        m = np.where(sel & (yr == y))[0]
        if len(m) < 20:
            continue
        mu = cit[m].mean()
        mncs[m] = cit[m] / mu if mu > 0 else np.nan
        # percentil dentro del año (empates repartidos al azar)
        o = np.lexsort((rng.random(len(m)), cit[m]))
        rk = np.empty(len(m)); rk[o] = np.arange(len(m)) / (len(m) - 1)
        pct[m] = rk
    by = np.full((K, NY), np.nan); n_by = np.zeros((K, NY), int); top_by = np.full((K, NY), np.nan)
    for c in range(K):
        for t, y in enumerate(YRS):
            m = sel & (lab == c) & (yr == y) & np.isfinite(mncs)
            n_by[c, t] = m.sum()
            if m.sum():
                by[c, t] = mncs[m].mean(); top_by[c, t] = (pct[m] >= 0.9).mean()
    overall = []
    for c in range(K):
        m = np.where(sel & (lab == c) & (yr <= last) & np.isfinite(mncs))[0]
        if len(m) < 20:
            overall.append(None); continue
        v = mncs[m]
        bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
        overall.append(dict(n=int(len(m)), mncs=r3(v.mean()), lo=r3(np.percentile(bs, 2.5)), hi=r3(np.percentile(bs, 97.5)),
                            median=r3(np.median(v)), top10=r3((pct[m] >= 0.9).mean())))
    return dict(by=[[r3(v) for v in row] for row in by], n=n_by.tolist(), top=[[r3(v) for v in row] for row in top_by],
                overall=overall, last=last)


# ---------------------------------------------------------------- edad de las referencias
def ref_age(sel, W, lab, yr, ry):
    ages = [[] for _ in range(NY)]; t_ages = [[[] for _ in range(NY)] for _ in range(K)]
    nref = np.zeros(NY, int); ndated = np.zeros(NY, int)
    for i in np.where(sel)[0]:
        rs = W[i]["refs"]
        if not rs:
            continue
        t = yr[i] - Y0
        nref[t] += len(rs)
        for r in rs:
            y = ry.get(r)
            if y is None:
                continue
            g = yr[i] - y
            if g < -1:                      # año de la referencia posterior al del trabajo: error de metadatos
                continue
            ndated[t] += 1
            ages[t].append(max(g, 0)); t_ages[lab[i]][t].append(max(g, 0))
    st = lambda v: dict(median=r3(np.median(v)), mean=r3(np.mean(v)), price=r3(np.mean(np.array(v) <= 5)),
                        old=r3(np.mean(np.array(v) > 20)), n=len(v)) if len(v) >= 50 else None
    return dict(year=[st(v) for v in ages], coverage=[r3(d / n) if n else None for d, n in zip(ndated, nref)],
                topic=[[(st(v) or {}).get("price") for v in row] for row in t_ages],
                topic_median=[[(st(v) or {}).get("median") for v in row] for row in t_ages],
                topic_all=[st([g for v in row for g in v]) for row in t_ages])


def main():
    rng = np.random.default_rng(SEED)
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    m = json.load(open(os.path.join(ROOT, "viewer", "oa", "oa_map.json"), encoding="utf-8"))
    extra = json.load(open(os.path.join(OUT, "works_extra.json"), encoding="utf-8"))
    P = m["papers"]
    assert len(P) == len(W)
    lab = np.array([p["c"] for p in P]); yr = np.array([p["y"] for p in P]); cit = np.array([w["cited_by"] or 0 for w in W])
    rel = np.array([p.get("r", 0) for p in P]); rt = np.array([bool(p.get("rt")) for p in P])
    f = m["edges"]["di"]; a, b = np.array(f[0::4], int), np.array(f[1::4], int)   # a cita a b
    E = np.load(os.path.join(OUT, "E_specter2.npy")).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    docs = [normalize(" ".join([w["title"] or "", w["abstract"] or "", " ".join(extra.get(w["key"], {}).get("kw", []))])) for w in W]
    tdocs = [normalize(" ".join([w["title"] or "", " ; ".join(extra.get(w["key"], {}).get("kw", []))])) for w in W]
    txt = np.array([(w["lang"] or "en") == "en" for w in W])          # análisis de términos: solo textos en inglés
    has_abs = np.array([len(w["abstract"] or "") > 300 for w in W])
    key = {w["key"]: i for i, w in enumerate(W)}
    refs_n = np.array([len(w["refs"]) for w in W])
    ry_path = os.path.join(OUT, "ref_years.json")
    ry = json.load(open(ry_path)) if os.path.exists(ry_path) else {}
    for w in W:
        ry[w["key"]] = w["year"]
    modes = dict(all=np.ones(len(W), bool), me=(rel <= 1) & ~rt, core=(rel == 0) & ~rt)
    from o13_sys import SHORT
    R = dict(years=YRS, partial=dict(year=2026, through="2026-10"), short=SHORT, modes={})
    for name, sel in modes.items():
        print(name, int(sel.sum()), flush=True)
        refs_in = np.array([sum(1 for r in w["refs"] if r in key and sel[key[r]]) for w in W])
        # refs_post: referencias con año conocido >= 2010 (internas o externas); es el denominador justo para la cuota interna,
        # porque un trabajo de 2011 apenas tiene corpus anterior al que citar
        refs_post = np.array([sum(1 for r in w["refs"] if ry.get(r, 0) >= Y0 and (r not in key or sel[key[r]])) for w in W])
        D = dict(n=int(sel.sum()), growth=growth(sel, lab, yr), cites=cites(sel, lab, yr, a, b, refs_n, refs_in, refs_post))
        D["bursts"] = bursts(sel & txt, docs, yr); print("  bursts", len(D["bursts"]), flush=True)
        D["sem"] = semantic(sel, lab, yr, E, tdocs, txt); print("  sem", flush=True)
        D["style"] = style(sel & txt & has_abs, docs, yr)
        D["path"] = main_path(sel, a, b, W, lab, yr, cit); print("  path", len(D["path"]["nodes"]), "nodos", flush=True)
        D["geo"] = geo(sel, W, lab, yr)
        D["pub"] = pub(sel, W, extra, lab, yr)
        D["impact"] = impact(sel, lab, yr, cit, rng)
        D["age"] = ref_age(sel, W, lab, yr, ry)
        R["modes"][name] = D
    # relevancia por año (sobre todos los trabajos)
    R["rel"] = dict(classes=["central", "secundaria", "periferica", "modelo_animal", "otro_concepto", "incierto"],
                    n=[np.bincount(yr[rel == k] - Y0, minlength=NY).tolist() for k in range(6)],
                    retracted=np.bincount(yr[rt] - Y0, minlength=NY).tolist(),
                    topic_periph=[[r3(((rel >= 2) & (lab == c) & (yr >= p0) & (yr <= p1)).sum() / max(1, ((lab == c) & (yr >= p0) & (yr <= p1)).sum()))
                                   for p0, p1 in PERIODS] for c in range(K)], periods=[f"{p0}-{p1}" for p0, p1 in PERIODS])
    ext = collections.Counter(r for w in W for r in w["refs"] if r not in key)
    R["age_coverage"] = dict(ext_distinct=len(ext), ext_dated=sum(1 for k in ext if k in ry),
                             weighted=r3(sum(v for k, v in ext.items() if k in ry) / max(1, sum(ext.values()))))
    json.dump(R, open(os.path.join(OUT, "o22_temporal.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(len(json.dumps(R, ensure_ascii=False)) / 1e3, "KB; cobertura de años de referencias externas:", R["age_coverage"])


if __name__ == "__main__":
    main()
