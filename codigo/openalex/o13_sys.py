"""Paso o13: datos de la pestaña "Análisis sistemático" del visor -> out/sys_data.json (pocos KB; 05_build.py lo incrusta).
Reúne los resultados de o03 (sesgo de selección), o05 (homología), o06 (indicadores), o07/o08 (episodios) y, para
comparar, los indicadores por ventana del corpus manual (homologia/out/h16_windows.csv)."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.dirname(HERE)
OUT = os.path.join(OA, "out")
HOM = os.path.dirname(OA)


def r(x, k=3):
    return None if pd.isna(x) else round(float(x), k)


SHORT = {"es": ["COVID / diagnóstico", "XMRV", "TCC / PACE", "Medicina china", "Ejercicio / autonómico", "Dolor / fibromialgia",
                "Inmunidad / mitocondria", "Revisiones / fatiga"],
         "en": ["Long COVID / diagnosis", "XMRV", "CBT / PACE", "Chinese medicine", "Exercise / autonomic", "Pain / fibromyalgia",
                "Immunity / mitochondria", "Reviews / fatigue"]}


def dialogue():
    """Diálogo entre temas (paso o14): lo que dibuja la sección «Qué temas dialogan entre sí»."""
    path = os.path.join(OUT, "o14_dialogue.json")
    if not os.path.exists(path):
        return None
    R = json.load(open(path, encoding="utf-8"))
    pick = lambda d, ks: {k: d[k] for k in ks}
    trends = [t for t in R["trends"] if t["net"] == "cp"][:4]
    series = []
    for t in trends:
        rows = [w for w in R["windows"] if w["net"] == "cp" and w["a"] == t["a"] and w["b"] == t["b"] and w["oe"] is not None
                and w["n_a"] >= 30 and w["n_b"] >= 30 and w["exp"] >= 5]
        series.append(dict(a=t["a"], b=t["b"], first=t["first"], last=t["last"], pts=[[w["ventana"], w["oe"]] for w in rows]))
    return dict(short=SHORT, names=R["names"], n=R["n_topic"],
                cp=pick(R["cp"], ["oe", "q", "obs", "exp", "ei", "ei_null"]), di=pick(R["di"], ["oe", "q", "obs", "exp", "ei", "ei_null"]),
                cp_ext=pick(R["cp_ext"], ["oe", "q"]), di_ext=pick(R["di_ext"], ["oe", "q"]),
                balance=R["flow"]["balance"], balance_null=R["flow"]["balance_null"],
                content=[pick(r, ["a", "b", "sim", "oe_cp", "res2_oe_cp"]) for r in R["content"]], fit=R["fit2_oe_cp"],
                trends=series, participation=R["participation"],
                bridges=[pick(b, ["id", "title", "year", "topic", "k", "participation"]) for b in R["bridges"][:10]])


def qc():
    """Control de calidad del corpus (o01b: limpieza y retractados; o18-o19 vía o12: relevancia)."""
    try:
        c = json.load(open(os.path.join(OUT, "o01b_summary.json"), encoding="utf-8"))
        meta = json.load(open(os.path.join(os.path.dirname(HOM), "viewer", "oa", "oa_meta.json"), encoding="utf-8"))
    except FileNotFoundError:
        return None
    W = json.load(open(os.path.join(OUT, "works.json"), encoding="utf-8"))
    return dict(raw=c["raw"], kept=c["kept"], suppl=c["n_suppl"], dup=c["n_dup"], retracted=len(c["retracted"]), rel=meta["relevance"],
                manual_in=meta["manual_in"], with_refs=sum(bool(w["refs"]) for w in W))


def main():
    QC = qc()
    y = pd.read_csv(os.path.join(OUT, "o06_years.csv"))
    w = pd.read_csv(os.path.join(OUT, "o06_windows.csv"))
    ph = json.load(open(os.path.join(OUT, "o05_summary.json")))
    o3 = json.load(open(os.path.join(OUT, "o03_summary.json"), encoding="utf-8"))
    o7 = json.load(open(os.path.join(OUT, "o07_summary.json")))
    o8 = json.load(open(os.path.join(OUT, "o08_summary.json")))
    o1 = json.load(open(os.path.join(OUT, "o01_summary.json")))
    man = pd.read_csv(os.path.join(HOM, "out", "h16_windows.csv"))
    cov = o3["cobertura_equipo_periodo"]
    S = dict(
        n=QC["kept"] if QC else o1["n"], manual_in=QC["manual_in"] if QC else o1["manual_in"], manual_total=o1["manual_total"],
        with_refs=QC["with_refs"] if QC else o1["with_refs"],
        years=[dict(y=int(a.year), n=int(a.n), coh=r(a.cohesion), core=r(a.cohesion_core), nocovid=r(a.cohesion_sin_covid),
                    canon=r(a.canon_medio), covid=r(a.covid), teams=r(a.equipos)) for a in y.itertuples()],
        win=[dict(v=a.ventana.replace("-20", "–").replace("-", "–") if False else a.ventana, mix=r(a.mixing_ratio), mixSub=r(a.mixing_ratio_sub),
                  mixSd=r(a.mixing_ratio_sub_sd), mantel=r(a.mantel_rho), mantelSub=r(a.mantel_rho_sub), mantelSd=r(a.mantel_rho_sub_sd),
                  jac=r(a.continuity_jaccard), covMix=r(a.covid_mix_ratio), covN=int(a.covid_papers), validated=int(a.validated))
             for a in w.itertuples()],
        ph=[dict(v=ph[k]["ventana"], bars=ph[k]["n_bars"], null=ph[k]["null_nbars_median"], edges=ph[k]["validated"], sig=ph[k]["n_sig"])
            for k in ph if k.startswith("win")],
        phFull=dict(strength=dict(bars=ph["full_fuerza"]["n_bars"], null=ph["full_fuerza"]["null_nbars_median"], sig=ph["full_fuerza"]["n_sig"]),
                    date=dict(bars=ph["full_fecha"]["n_bars"], null=ph["full_fecha"]["null_nbars_median"], sig=ph["full_fecha"]["n_sig"])),
        cov=[dict(per=per, team=cov[f"{per}|1"]["cobertura"], rest=cov[f"{per}|0"]["cobertura"], nTeam=cov[f"{per}|1"]["n"], nRest=cov[f"{per}|0"]["n"])
             for per in ("2017-19", "2020-21", "2022-24")],
        coverage_all=o3["cobertura_2017_24"], logit=o3["logit"],
        xmrv=dict(o8, boot=o7["bootstrap"], top=float(pd.read_csv(os.path.join(OUT, "o07_refs_2010_11.csv")).frac_2010_11.iloc[0])),
        manual=[dict(v=a.ventana, mix=r(a.mixing_ratio), mantel=r(a.mantel_rho)) for a in man.itertuples()],
        continuity={k: v for k, v in o7.items() if "->" in k},
        dialogue=dialogue(),
        qc=QC,
    )
    json.dump(S, open(os.path.join(OUT, "sys_data.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(len(json.dumps(S)), "bytes;", list(S))


if __name__ == "__main__":
    main()
