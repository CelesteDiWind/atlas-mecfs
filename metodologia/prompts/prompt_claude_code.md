# Prompt para Claude Code: homología persistente del Atlas EM/SFC con embeddings reales

## Contexto

Tengo un atlas interactivo de 583 artículos sobre ME/CFS y COVID largo (`atlas.html`, un solo archivo). Dentro hay un objeto `DATA = {...}` (JSON, se extrae buscando `DATA = ` dentro del segundo `<script>` y decodificando con `json.JSONDecoder().raw_decode`). Estructura:

- `clusters`: 8 clusters (id, name, name_en, terms, n).
- `papers`: 583 registros con `id, t` (título), `y` (año; 25 sin año), `a` (autores), `v` (revista), `doi`, `cit` (citas recibidas), `abs` (abstract; 529 de 583 lo tienen), `kw` (palabras clave), `c` (cluster), `p` / `p2` (coordenadas 3D / 2D del layout semántico, que es una proyección), `nn` / `nn_s` (6 vecinos semánticos y su similitud coseno), `pc` / `pc2` (layout por citas).
- `edges`: `co` (cocitación), `cp` (acoplamiento bibliográfico) y `di` (citas directas), como arrays planos de 4 números por arista: `[a, b, peso 0-1, recuento]`.

No hay embeddings crudos en el atlas. Los originales vienen de un modelo neuronal que desconozco; la similitud coseno de `nn_s` va de 0,85 a 0,99.

## Qué ya hice (resultados de entrada)

Una primera pasada la hice con Claude en la nube, sin acceso a modelos de embeddings. Está en la carpeta `resultados/` (CSV y `page_data.json`) y en `scripts/` (pipeline 01 a 10, con checkpoints en `out/`). Usé un proxy TF-IDF/LSA de 30 dimensiones y el grafo de 6 vecinos del atlas; el proxy coincide solo en el 27 % de los vecinos con el atlas. Los resultados que quiero **poner a prueba**, no dar por buenos:

1. Los 8 clusters no son islas: entre el 28 % y el 60 % de las conexiones semánticas de cada cluster van a otro cluster (`cluster_robustness.csv`, `cluster_bridges_semantic.csv`). El cluster 2 (inmunidad, infección y COVID largo) es el socio principal de 5 de los otros 7.
2. Hay 9 picos de densidad, uno dominante (revisión general de 2024). El acuerdo con los 8 clusters es bajo (ARI 0,21; NMI 0,46) (`density_peaks_semantic.csv`). El cluster 7 no tiene pico propio.
3. H1 en el grafo semántico del atlas: solo 4 de 566 ciclos superan el nulo (persistencia máxima 0,352 frente a un p95 nulo de 0,337).
4. H1 en el proxy LSA: 17 ciclos sobre el nulo gaussiano (p95 0,198); 4 en H2. Los ciclos grandes unen los clusters 0-2-7 y 5-4-2-1 (`h1_cycles.csv`).
5. H1 en las citas (cocitación + acoplamiento, 541 papers en el componente gigante): 8 ciclos sobre el nulo (p95 0,664).
6. Correlación de Spearman entre distancias semánticas y de citas: 0,17.
7. Filtración por año (`year_filtration.csv`): los ciclos sobre el umbral suben hasta 6 en 2019 y bajan a 3 con el corpus completo.

## Qué quiero que hagas

Rehacer el análisis con embeddings reales y decirme qué de lo anterior se confirma, se refuta o queda inconcluso.

**Antes de implementar nada**, revisa el plan y señálame cualquier inconsistencia metodológica. Los puntos que ya sospecho: los clusters se definieron en el espacio semántico del atlas, así que compararlos con su propia geometría es circular; los nulos que usé tenían pocas réplicas; y la filtración por año heredó distancias del grafo completo.

### Pasos

0. **Embeddings.** Calcula embeddings locales de `título + palabras clave + abstract` con un modelo de sentence-transformers (prueba al menos dos, por ejemplo `all-mpnet-base-v2` y `BAAI/bge-m3`). Para los papers sin abstract usa título + palabras clave y márcalos. Valida contra el atlas: solapamiento de los 6 vecinos con `nn` y pureza de cluster en los 10 vecinos (referencia: 0,91 en las coordenadas 3D del atlas y 0,61 en mi proxy LSA). Elige el modelo que mejor reproduzca el atlas y documenta por qué.
1. **Distancias.** Distancia coseno completa (583 x 583) sin pasar por un grafo de 6 vecinos. Repite con grafos kNN de k = 6, 15 y 30 para medir cuánto depende todo de la resolución.
2. **H0 y modos.** Persistencia de picos de densidad (ToMATo o superlevel set sobre una densidad kNN con k = 15). Compara con los 8 clusters (ARI, NMI) y con los resultados 1 y 2 de arriba.
3. **H1 y H2.** Rips sobre la distancia coseno completa (`ripser`), con un nulo adecuado y **al menos 100 réplicas** (nube gaussiana con la misma covarianza, y permutación de coordenadas por columna). Añade estabilidad por submuestreo (80 %, 50 réplicas): qué barras reaparecen. Contrasta con los resultados 3 y 4.
4. **Representantes de ciclos.** Con `gudhi` o cociclos de `ripser`, extrae los papers de cada ciclo significativo y su composición por cluster. Evita mi aproximación (arista de nacimiento más camino mínimo), que falla con distancias empatadas.
5. **Tiempo.** Filtración por año recalculando las distancias solo con los papers de cada subconjunto (sin heredarlas del grafo completo). Curvas de Betti y persistencia total por año.
6. **Citas.** Repite H0/H1 sobre las redes de citas y compáralas con la red semántica (bottleneck y Wasserstein, escala normalizada por la distancia mediana; Spearman de distancias). Contrasta con los resultados 5 y 6.
7. **Papers puente.** Lista de papers cuya eliminación cambia más la topología (puente entre clusters y miembros de ciclos significativos), con título, año, cluster y citas.

### Formato de entrega

- Pipeline modular y reanudable, con un script o celda por paso y salidas en `out/` (CSV y `.npy`). Si algo falla, debe poder continuar desde el último punto de control.
- Si trabajas en Jupyter, dame cada celda completa, no fragmentos.
- Figuras en modo oscuro (fondo `#0D1117`), 300 DPI, formato vertical optimizado para móvil.
- Una tabla final `comparacion_con_resultados_previos.csv` con una fila por resultado (1 a 7), columnas: resultado previo, resultado nuevo, veredicto (confirmado / refutado / inconclusivo) y motivo.
- Un `page_data.json` actualizado con el mismo esquema que el mío (`resultados/page_data.json`) para poder regenerar la página de resultados.
- Una capa opcional para el atlas: campos nuevos por paper (`bridge`, `cycle_ids`, `peak_id`) en un archivo aparte (`topology_layer.json`). No modifiques `atlas.html` original; trabaja siempre sobre una copia.

Sé conciso en los informes: primero el veredicto, después el detalle.
