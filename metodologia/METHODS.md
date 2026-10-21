# Atlas EM/SFC: methods

*Celeste Giraldo Lozano, with the assistance of Claude (Anthropic). October 2026.*
*Versión en español: [METODOLOGIA.md](METODOLOGIA.md).*

The atlas has three layers, each built on the one before:

1. **The hand-picked selection**: 583 papers on ME/CFS and long COVID, gathered by hand, with their map, citation
   networks and a content analysis by topic.
2. **Persistent homology of the hand-picked selection**: does the map contain structure (holes, loops, breaks) that
   chance alone would not produce?
3. **A systematic OpenAlex corpus**: 7,169 works retrieved with a fixed query and no manual selection. It tests whether
   what the hand-picked set shows reflects the field or the way papers were chosen, and it is used to study how research
   changed between 2010 and 2026.

Each section says which scripts reproduce it (folder [`codigo/`](../codigo/)) and where its results are (folder
[`resultados/`](../resultados/)).

---

## 1. Hand-picked selection

**Scripts:** [`codigo/atlas/`](../codigo/atlas/), steps 01 to 06.

### 1.1 Corpus and metadata

- **Source:** a collection of PDFs gathered by hand. The PDFs are not published, because they are under copyright.
- **Extraction (01):** text, DOI and metadata from each PDF.
- **Enrichment (02):** each PDF is looked up in OpenAlex by DOI. Without a DOI, or when it is not found, it is looked up
  in Crossref by title and then in OpenAlex by the DOI Crossref returns (title search in OpenAlex is paid). Duplicates are
  removed, leaving 583 papers with a record.
- **Abstract:** taken from OpenAlex or Crossref, or from the PDF itself when neither has one.

### 1.2 Citation networks (02b)

Three networks between the papers in the collection:

- **Direct citations:** A cites B, both in the collection.
- **Bibliographic coupling:** A and B cite references in common. The link exists from the day the newer one is
  published.
- **Co-citation:** a third work, from anywhere, cites both A and B. It requires downloading the works that cite the
  collection.

### 1.3 Map and topics (03, 04)

- **Embeddings:** SPECTER2 (base model plus the proximity adapter, applied by hand so as not to depend on the
  `adapters` package) on title + abstract. When the abstract is shorter than 200 characters, the start of the PDF fills
  the gap. SPECTER2 is trained on scientific citations, so papers that are close in its space tend to cite each other.
- **Map:** 3D and 2D UMAP (cosine, 15 neighbours) for display.
- **Topics:** KMeans with 8 clusters on a 10-dimensional UMAP, which preserves the original distances better than the 3D
  one. Topic names come from each cluster's most representative terms and titles.
- **Citation view:** a second layout that combines the citation networks with content.

### 1.4 Content analysis by topic (06)

- For each paper we gathered keywords, abstract and conclusions: the "Conclusions" section, or the conclusions part of
  a structured abstract, or the end of the discussion.
- AI agents (Claude) analysed each paper: study type, question, design, findings and limitations. They then wrote a
  synthesis per topic from those analyses, citing the papers.
- The inputs, which contain conclusions copied from the PDFs, are not published. The analyses and syntheses are
  ([`resultados/atlas/`](../resultados/atlas/)).

---

## 2. Persistent homology of the hand-picked selection

**Scripts:** [`codigo/homologia/`](../codigo/homologia/) (detailed guide in Spanish in `LEEME.md`).
**Results:** [`resultados/homologia/`](../resultados/homologia/) and [`figuras/`](../figuras/).

### 2.1 Semantic space (h01–h11)

- **Question:** are there holes or loops in the cloud of papers that chance does not explain? For instance, a set of
  topics joined in a ring with no common centre.
- **Spaces:** SPECTER2 is the main one; mpnet and bge-m3 serve as sensitivity checks.
- **Distances:** chord distance, and geodesics on neighbour graphs (k = 6, 15, 30).
- **H0:** density modes (ToMATo) and mixing between topics.
- **H1/H2:** three nulls (same covariance, permuted columns, and a Gaussian mixture chosen by BIC), 200 replicates each.
  The Gaussian mixture keeps the clusters but destroys loops, which makes it the main null.
- **Significance:** multiplicative persistence (death/birth; Bobrowski, Kahle and Skraba, 2017) against the maximum of
  each null replicate. This controls false positives per whole diagram rather than per bar.
- **Also:** a filtration by year, the topology of the citation networks, and bridge papers.

### 2.2 Validated citation networks (h13–h17)

- **Validated projections:** coupling and co-citation are projected from the bipartite network (paper × reference,
  or citer × paper). A pair is kept only if it shares more than expected under a bipartite configuration model
  (BiCM), tested with a Poisson test and Benjamini–Hochberg correction (q < 0.05). Without this step, a paper with a long
  reference list ends up linked to almost everything.
- **Comparison of networks (h14):** coverage, density, topic mixing, agreement with content, bootstrap stability.
  Result: coupling is the right network for 3-year windows; co-citation is almost empty.
- **Topology of citations (h15):** filtrations by link strength and by date. Nulls: Curveball rewiring of the bipartite
  network (both degree sequences fixed) and permuted weights, 200 replicates.
- **Breaks and convergence (h16, h16b–d):**
  - Merges of components over time, against permuted dates.
  - Indicators per 3-year window, with a same-size control.
  - Continuity between windows, with a bootstrap over references.
  - Cohesion of the cited literature by year.
- **The NICE hypothesis (h17):** the rise in citations to the diagnostic criteria in 2020–21 was tested against
  predictions fixed before looking at the data (timing, co-cited references, content) and against two alternatives.

---

## 3. Systematic OpenAlex corpus

**Scripts:** [`codigo/openalex/`](../codigo/openalex/), steps o01 to o22 (`run_update.sh` chains the full recompute).
**Results:** [`resultados/openalex/`](../resultados/openalex/).

### 3.1 Retrieval and cleaning (o01, o01b, o10)

- **Query:** title or abstract containing `"myalgic encephalomyelitis"`, `"chronic fatigue syndrome"`,
  `"myalgic encephalopathy"` or `"ME/CFS"`; articles and reviews; published from 2010 to October 2026. Downloaded on
  19–21 October 2026.
- **Access:** no API key, page-number pagination, and every response cached.
- **Cleaning:** of 7,245 records, 72 supplementary files indexed as articles and 4 duplicates sharing a DOI were
  removed, leaving **7,169 works**. The 9 retracted works are kept and flagged.
- **Embeddings (o02):** the same SPECTER2 with adapter as the atlas, on title + abstract.

### 3.2 Relevance to ME/CFS (o18, o19)

The query also returns works that only mention ME/CFS. Each work was labelled:

- **Core:** ME/CFS is the subject of the work.
- **Secondary:** ME/CFS is one of several subjects.
- **Passing mention:** the work is about something else and mentions ME/CFS in passing.
- **Animal model:** an animal study of a "chronic fatigue syndrome model".
- **Other notion:** "chronic fatigue" used in another sense (sub-health, burnout, overtraining).
- **Unclear:** no abstract, and a title that does not allow a decision.

Method:

- **A rule settles the clear cases:** ME/CFS in the title (long forms, or the uppercase acronyms CFS and ME, checked by
  hand) means core.
- **AI (Claude) classified the rest**, reading the title and the abstract sentences that mention the term, in batches
  validated by script. This includes the Chinese-medicine topic even when the title names ME/CFS, because there the
  term sometimes refers to animal models or to "sub-health".
- **Result:** 4,599 core, 613 secondary, 1,678 passing mentions, 113 animal models, 100 other notions and 66 unclear.
  Each work's label and the reason for it are in `relevance.json`.
- Nothing is removed for relevance: the viewer and the temporal analyses can filter by it.

### 3.3 Topics and map (o11, o12)

- **Method:** the same recipe as the atlas (10D UMAP + KMeans with 8 topics), giving the corpus its own topics.
- **Fixed topics:** labels were fitted on the 2010–2025 works. Works from 2026 are assigned to the nearest SPECTER2
  centroid, so topics do not change meaning as new works are added.
- **The eight topics:** long COVID and clinical diagnosis; the XMRV retrovirus; CBT, PACE and the patient experience;
  Chinese medicine and complementary treatments; exercise, the autonomic system and the brain; pain, fibromyalgia and
  functional syndromes; immunity, mitochondria and biomarkers; fatigue, sleep, stress and general reviews.

### 3.4 Validated coupling network and homology (o04, o05)

- **Network:** validated coupling (BiCM + Poisson + BH, q < 0.05), for the full period and for 15 three-year windows
  (2010–12 to 2024–26).
- **Homology:** a strength filtration in each window and in the full network, with a Curveball null (100 replicates),
  and a date filtration against permuted dates (200 replicates).

### 3.5 Indicators, episodes and selection bias (o03, o06–o08)

- **Topic mixing:** the share of links between different topics, relative to permuted topics (1 = chance).
- **Content–citation alignment:** Spearman correlation between SPECTER2 similarity and excess shared references.
- **Same-size control:** the same metrics on subsamples with the same number of papers, so that corpus growth is not
  mistaken for a trend.
- **Continuity:** Jaccard index of the links between papers shared by two consecutive windows.
- **Cohesion:** cosine between the reference profiles of two halves of the same year, with equal-sized cohorts.
- **Episodes (o07, o08):**
  - The 2019–21 continuity step, with a reference bootstrap and a variant without long COVID papers.
  - The 2010–11 XMRV episode, with a permutation test and cohesion recomputed without its block of references.
- **Bias of the hand-picked selection (o03):** logistic regression of the probability of being selected on year, team,
  citations, work type and title.

### 3.6 Dialogue between topics (o14, o15, o20)

- **Networks:** direct citations (directed) and coupling.
- **Null:** topics shuffled among papers of the same year and the same degree quintile. This keeps the network and each
  topic's size and age. 1,000 replicates, with BH across cells.
- **Metrics:**
  - Observed/expected for each pair of topics.
  - Quasi-independence (iterative fitting over off-diagonal cells), so that dialogue between topics is read without the
    weight of self-citation.
  - E–I openness index.
  - Balance of citations received and made.
  - Content versus citations.
  - Participation coefficient of bridge papers.
- **Sensitivity (o20):** dialogue, mixing and cohesion were recomputed on core and secondary works only. The conclusions
  hold.

### 3.7 Topic syntheses (o16, o17)

- **Sample:** about 80 works with abstracts per topic: the most central, the most cited, recent ones, and the most
  representative of each subgroup. Only core, secondary and animal-model works, without retracted ones.
- **Quantitative context:** the topic's size, years, journals, and dialogue with the other topics.
- **Writing:** AI (Claude) wrote each synthesis in Spanish and English; a script checked that every cited paper belongs
  to the sample.

### 3.8 Temporal analyses (o21, o22)

Everything is computed three times, once per relevance filter: all works; core and secondary works; core works only.

- **Growth:** works per topic and year. The yearly change in each topic's share comes from an overdispersed binomial
  model (quasi-likelihood), 2010–2025.
- **Citations within and across topics:**
  - The share of a topic's citations that go to itself.
  - The share of the citations made to it that come from other topics.
  - A size-adjusted version: self-citations divided by the number expected if the topic cited according to what was
    cited that year.
  - Rolling 3-year windows; windows with fewer than 30 citations are left out.
  - Also the share of references published since 2010 that point to the corpus itself.
- **Term bursts:** Kleinberg's two-state model by year (s = 2, γ = 1), on the title, abstract and keywords of
  English-language texts. Forms that share almost the same papers are grouped.
- **Writing style:** the share of abstracts containing at least one of 12 marker words of writing with language models
  (Kobak et al., 2024). These words are excluded from the bursts.
- **Semantic drift:**
  - Distances between topic centres, estimated without group-size bias from mean pairwise similarities.
  - Separation: the mean distance between centres divided by the mean spread.
  - Each topic's drift since 2010–12.
  - Distinctive terms per period (log-odds with an informative Dirichlet prior; Monroe et al., 2008).
- **Main citation path:** the number of paths through each link (SPC) in the direct-citation network, with a global
  search and a key-route over the 15 busiest links. Citations to works with a later date are removed so that the network
  is acyclic.
- **Geography and collaboration:** affiliation country (full counting), country specialisation (a topic's share in the
  country over its share in the corpus), authors per work, international collaboration and new authors.
- **Open access and journals:** current status in OpenAlex, not the status at publication.
- **Normalised impact:** each work's citations divided by the mean for its year (MNCS), with a bootstrap interval;
  works from 2010–2023.
- **Age of references:** median and Price index (the share of references 5 years old or less). The years of external
  references were downloaded from OpenAlex (o21). In October 2026 they cover 84% of citations to works outside the corpus.

---

## 4. Use of artificial intelligence

- **Claude (Anthropic)** wrote the code, ran the analyses, drafted the website texts, and classified and summarised
  papers. That covers the per-paper analyses and syntheses of the hand-picked set, the relevance labels for unclear
  cases, and the OpenAlex topic syntheses.
- **Celeste Giraldo Lozano** set the questions, chose between methodological options, reviewed the results and decided
  what to publish.
- **Checks:** AI classifications and syntheses were validated by script (format, every citation present in the input,
  every work labelled). The website texts that quote figures are generated from the data.
- **Even so**, the interpretation may contain errors: check anything important against the sources before citing it.
- The main prompts are in [`prompts/`](prompts/).

---

## 5. Reproducibility

- **Dependencies:** Python 3 with numpy, scipy, pandas, scikit-learn, statsmodels, umap-learn, networkx, ripser, persim,
  gudhi, transformers and torch (SPECTER2), sentence-transformers (mpnet, bge-m3), matplotlib and requests.
- **Order:**
  1. `codigo/atlas/` 01 to 06, and 05 to build the website.
  2. `codigo/homologia/` h01 to h17.
  3. `codigo/openalex/` o01 to o22. Then `05_build.py` for the website and `07_export_repo.py` for this repository.
- **OpenAlex without a key:** the free daily quota is limited; the download steps can be rerun and resume where they
  stopped.
- **Data not in the repository:** the PDFs, the text extracted from them, the OpenAlex cache, the embeddings and the
  intermediate checkpoints. The scripts regenerate all of them except the PDFs of the hand-picked selection.
- **Randomness:** every script fixes its seed. Permutation nulls may differ in the last decimal across library versions.

## 6. Main limitations

- **Keyword query:** the OpenAlex corpus misses works that do not use the search terms, and it includes works where
  ME/CFS is secondary. The relevance labels mitigate this, but AI assigned them in unclear cases.
- **Topics:** they come from clustering embeddings, so their number (8) and boundaries are a choice. Trends are more
  robust than absolute values.
- **References:** only 73% of works have a reference list in OpenAlex; the rest are left out of the citation networks.
- **Metadata:** countries, open-access status and citations depend on OpenAlex, which has gaps and errors.
- **Current year:** 2026 is incomplete (up to October).
- **Hand-picked selection:** it is biased towards certain teams and towards highly cited papers (section 3.5), which is
  why it was checked against the systematic corpus.

## Method references

- Bobrowski, O., Kahle, M., & Skraba, P. (2017). Maximally persistent cycles in random geometric complexes. *Annals of
  Applied Probability*, 27(4).
- Kleinberg, J. (2003). Bursty and hierarchical structure in streams. *Data Mining and Knowledge Discovery*, 7(4).
- Kobak, D., González-Márquez, R., Horvát, E.-Á., & Lause, J. (2024). Delving into ChatGPT usage in academic writing
  through excess vocabulary. *arXiv:2406.07016*.
- Liu, J. S., & Lu, L. Y. Y. (2012). An integrated approach for main path analysis. *Journal of the American Society for
  Information Science and Technology*, 63(3).
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words. *Political Analysis*, 16(4).
- Saracco, F., et al. (2017). Inferring monopartite projections of bipartite networks: an entropy-based approach.
  *New Journal of Physics*, 19.
- Singh, A., et al. (2023). SciRepEval: a multi-format benchmark for scientific document representations (SPECTER2).
  *EMNLP 2023*.
