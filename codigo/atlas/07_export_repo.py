"""Paso 7: copia código, resultados y figuras al repositorio público (carpeta site/, que es el repositorio de GitHub
https://github.com/CelesteDiWind/atlas-mecfs; 05_build.py escribe ahí la web).

Se copia:   código (scripts del atlas, de la homología y del corpus OpenAlex, y la plantilla del visor), resultados
            (tablas y JSON pequeños), figuras y los prompts usados.
No se copia (derechos de autor o tamaño; se regeneran con los scripts):
            PDFs (data/raw), texto extraído de los PDF (data/extracted.json, _text de data/papers.json), conclusiones
            copiadas de los artículos (data/analysis/inputs), caché de OpenAlex (data/cache), embeddings, modelos,
            puntos de control (homologia/out, openalex/out/*.npz|npy), corpus descargado (works*.json).
Relanzar tras cambiar código o resultados; después, commit + push desde site/."""
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
HOM = os.path.join(ROOT, "homologia")
OAO = os.path.join(HOM, "openalex", "out")

# carpeta de destino -> lista de (patrón de origen relativo a ROOT)
PLAN = {
    "codigo/atlas": ["scripts/*.py", "viewer/template.html"],
    "codigo/homologia": ["homologia/scripts/*.py", "homologia/LEEME.md", "homologia/PLAN_citas.md"],
    "codigo/openalex": ["homologia/openalex/scripts/*.py", "homologia/openalex/scripts/*.sh"],
    "resultados/atlas": ["data/cluster_labels*.json", "data/citations.json"],
    "resultados/atlas/analisis_por_paper": ["data/analysis/out/*.json"],
    "resultados/atlas/sintesis_por_tema": ["data/analysis/synth/*.json"],
    "resultados/atlas/analisis_por_paper_en": ["data/analysis/en/out/*.jsonl"],
    "resultados/atlas/sintesis_por_tema_en": ["data/analysis/en/synth_out/*"],
    "resultados/homologia": ["homologia/resultados/*"],
    "resultados/openalex": [
        "homologia/openalex/out/o0*_summary.json", "homologia/openalex/out/o05_bars_*.csv",
        "homologia/openalex/out/o06_*.csv", "homologia/openalex/out/o07_*", "homologia/openalex/out/o08_summary.json",
        "homologia/openalex/out/o14_*", "homologia/openalex/out/o20_sensitivity.json", "homologia/openalex/out/o22_temporal.json",
        "homologia/openalex/out/relevance.json", "homologia/openalex/out/relevance_rules.json",
        "homologia/openalex/out/retracted.json", "homologia/openalex/out/sys_data.json",
        "homologia/openalex/out/clusters_oa.json", "data/cluster_labels_oa*.json",
    ],
    "resultados/openalex/sintesis_por_tema": ["homologia/openalex/out/synth/*.json"],
    "figuras": ["homologia/figs/*.png"],
    "figuras/explicativas": ["homologia/figs/explicativas/*.png"],
    "figuras/explicativas_citas": ["homologia/figs/explicativas_citas/*.png"],
    "metodologia/prompts": ["prompt_claude_code.md"],
}


def main():
    n, size = 0, 0
    for dest, pats in PLAN.items():
        d = os.path.join(SITE, dest)
        if os.path.isdir(d):                     # se rehace entera: no quedan archivos que ya no existen en origen
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    os.remove(p)
        os.makedirs(d, exist_ok=True)
        for pat in pats:
            files = [f for f in glob.glob(os.path.join(ROOT, pat)) if os.path.isfile(f)]
            assert files, f"sin archivos para {pat}"
            for f in files:
                shutil.copyfile(f, os.path.join(d, os.path.basename(f)))
                n += 1
                size += os.path.getsize(f)
    print(f"{n} archivos copiados a site/ ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
