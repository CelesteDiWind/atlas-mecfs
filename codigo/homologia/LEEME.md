# Homología persistente del Atlas EM/SFC — segunda pasada (embeddings reales)

Rehace la pasada previa (`previo/`, proxy TF-IDF/LSA) con los embeddings reales del atlas y pone a prueba sus 7 resultados.
Veredictos: `resultados/comparacion_con_resultados_previos.csv`.

## Orden de ejecución (cada paso reutiliza lo que ya está en `out/`, así que se puede relanzar tras un fallo)

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `h01_data.py` | Datos del atlas (`data/map.json`) y embeddings SPECTER2 originales; marca duplicados, papers fuera de tema y sin abstract |
| 2 | `h02_embeddings.py` | mpnet y bge-m3 (título + palabras clave + abstract) y validación de los tres frente al atlas |
| 3 | `h03_distances.py` | Distancia de cuerda completa y geodésicas kNN (k = 6, 15, 30) por modelo; subconjuntos de sensibilidad |
| 4 | `h04_h0_modes.py` | H0, modos de densidad (ToMATo, densidad kNN k = 15) y mezcla entre clusters frente a dos nulos |
| 5 | `h05_h1_h2.py` | H1 (filtración completa) y H2 (exploratorio), tres nulos × 200 réplicas, estabilidad por submuestreo |
| 6 | `h06_cycles.py` | Representantes de los ciclos significativos (guiados por el cociclo) y su estabilidad |
| 7 | `h07_time.py` | Filtración por año con distancias recalculadas, nulo por año y control con el mismo n |
| 8 | `h08_citations.py` | Topología de las citas, nulo de modelo de configuración, Mantel, comunidades de Louvain |
| 9 | `h09_bridges.py` | Papers puente (eliminación uno a uno, intermediación entre clusters, pertenencia a ciclos) |
| 10 | `h10_report.py` | Tabla de veredictos, `page_data.json`, `topology_layer.json` y figuras técnicas |
| 11 | `h11_explain.py` | Gráficos explicativos para no especialistas (`figs/explicativas/`): uno por pregunta, con el título como respuesta |

Ejecutar desde `homologia/scripts/` con `python -X utf8 hNN_....py`. Dependencias: ripser, persim, gudhi, umap-learn,
sentence-transformers, scikit-learn, networkx. bge-m3 se guarda en `models/` (2,2 GB; se puede borrar tras el paso 2).

## Decisiones de método (y por qué)

- **Embeddings:** el atlas usa SPECTER2 (base + adaptador de proximidad), guardado en `data/embeddings.npy`; reproduce el 100 %
  de los vecinos. Es el espacio principal; mpnet y bge-m3 son sensibilidad, no candidatos a "reproducir el atlas".
- **Criterio de significación:** persistencia multiplicativa muerte/nacimiento (Bobrowski, Kahle y Skraba, 2017) frente al
  máximo de cada réplica nula (control del falso positivo por diagrama). La aditiva se reporta como secundaria: los nulos tienen
  la misma mediana de distancias que el corpus pero no sus vecindarios cerrados, y en las geodésicas inflan todas las barras.
- **Nulos:** `gauss` (misma covarianza), `colperm` (columnas permutadas) y `gmm` (mezcla gaussiana con componentes por BIC en
  PCA-50 + residuo). `gmm` conserva los grupos y destruye los ciclos; es el criterio principal. `colperm` es muy permisivo con
  embeddings anisótropos (destruye la concentración de distancias) y sirve solo de referencia.
- **Circularidad de los clusters:** la mezcla se compara con aplicar el mismo procedimiento (UMAP 10D + KMeans 8) a una nube sin
  grupos; los modos se comparan entre modelos y con las comunidades de citas.
- **Citas:** Rips con 300 puntos de referencia (permutación voraz) en real, nulos y comparación; Rips completo en las redes nulas
  tardaba ~3 min por réplica por los empates de las geodésicas.
- **Año:** distancias y nulo recalculados por subconjunto (desde 2015, ≥ 40 papers) y control de tamaño con submuestras del
  corpus completo.

## Salidas

- `resultados/`: tabla de veredictos, `page_data.json` (mismo esquema que `previo/resultados/page_data.json`; los umbrales del
  diagrama son los aditivos para que la página previa los dibuje igual, los recuentos `nsig` usan el criterio principal),
  `topology_layer.json` (por paper: `bridge`, `cycle_ids`, `peak_id`) y las tablas de cada paso.
- `figs/`: figuras en modo oscuro (#0D1117), verticales, 300 DPI.
- `out/`: puntos de control (`.npy`, `.pkl`, `.csv`). `out/v1_aditivo/` guarda la primera corrida con el criterio aditivo.
