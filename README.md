# Atlas EM/SFC · ME/CFS literature atlas

**Web: https://celestediwind.github.io/atlas-mecfs/**

Mapa interactivo de la literatura sobre encefalomielitis miálgica / síndrome de fatiga crónica (EM/SFC), 2010–2026:
una selección manual de 583 artículos y un corpus sistemático de 7169 trabajos de OpenAlex, con análisis por tema,
síntesis por tema y un análisis del campo: qué contiene el corpus, cómo se relacionan sus temas y qué cambió con los años.

Interactive map of the ME/CFS research literature, 2010–2026: a hand-picked selection of 583 papers and a systematic
corpus of 7,169 OpenAlex works, with topic analyses, topic syntheses and an analysis of the field: what the corpus
contains, how its topics connect and what changed over the years.

**Autoría / Author:** Celeste Giraldo Lozano concibió y dirigió el proyecto: las preguntas, la estrategia de análisis,
los flujos de trabajo y la elección de métodos. Claude (Anthropic) implementó el código y los cálculos y redactó los
borradores bajo su dirección. / Celeste Giraldo Lozano conceived and led the project: the questions, the analytical
strategy, the workflows and the choice of methods. Claude (Anthropic) implemented the code and computations and
drafted the texts under her direction. Cómo citarlo / How to cite: botón «Cite this repository» o
[CITATION.cff](CITATION.cff).

## Contenido / Contents

| Carpeta / Folder | Qué hay / What it holds |
|---|---|
| `index.html` y `oa_*.json` | La web (GitHub Pages) / The website |
| [`metodologia/`](metodologia/) | [Metodología](metodologia/METODOLOGIA.md) · [Methods](metodologia/METHODS.md) · prompts usados con la IA / AI prompts |
| [`codigo/atlas/`](codigo/atlas/) | Selección manual: extracción, metadatos, citas, embeddings, mapa, construcción de la web / Hand-picked selection pipeline |
| [`codigo/homologia/`](codigo/homologia/) | Homología persistente y redes de citas validadas de la selección manual / Persistent homology and validated citation networks |
| [`codigo/openalex/`](codigo/openalex/) | Corpus sistemático de OpenAlex: descarga, relevancia, temas, redes, diálogo, análisis temporales / Systematic OpenAlex corpus |
| [`resultados/`](resultados/) | Tablas y JSON de resultados, análisis y síntesis por tema / Result tables, analyses and syntheses |
| [`figuras/`](figuras/) | Figuras (fondo oscuro, 300 DPI) / Figures (dark background, 300 DPI) |

No se incluyen los PDF de los artículos, el texto extraído de ellos ni la caché de OpenAlex (derechos de autor y
tamaño); los scripts regeneran todo salvo los PDF. / Article PDFs, their extracted text and the OpenAlex cache are not
included (copyright and size); the scripts regenerate everything except the PDFs.

Los cálculos y textos se hicieron con ayuda de IA y pueden contener errores: comprueba lo importante en las fuentes antes
de citarlo. / Calculations and texts were produced with AI assistance and may contain errors: check anything important
against the sources before citing it.

## Licencia / License

- **Código / Code** (`codigo/` y el JavaScript/CSS de `index.html`): [MIT](LICENSE-CODE).
- **Todo lo demás / Everything else** (textos, metodología, análisis, resultados, figuras):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Puedes usarlo, compartirlo y adaptarlo citando su autoría.
  / You may use, share and adapt it with attribution.
- Los metadatos de OpenAlex son de dominio público (CC0); los resúmenes de los artículos pertenecen a sus autores o
  editoriales. / OpenAlex metadata are public domain (CC0); article abstracts belong to their authors or publishers.
  Detalles / Details: [LICENSE](LICENSE).
