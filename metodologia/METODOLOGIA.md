# Metodología del Atlas EM/SFC

*Celeste Giraldo Lozano (concepción, diseño y dirección), con la asistencia de Claude (Anthropic). Octubre de 2026.*
*English version: [METHODS.md](METHODS.md).*

El atlas tiene tres capas, cada una construida sobre la anterior:

1. **Selección manual**: 583 artículos sobre EM/SFC y COVID persistente reunidos a mano, con su mapa, sus redes de
   citas y un análisis de contenido por tema.
2. **Homología persistente de la selección manual**: ¿tiene el mapa estructuras (huecos, ciclos, quiebres) que no
   produzca el azar?
3. **Corpus sistemático de OpenAlex**: 7169 trabajos recuperados con una consulta fija, sin selección manual, que
   sirven para comprobar si lo visto en la selección manual se debe al campo o a cómo se eligieron los artículos, y para
   estudiar cómo ha cambiado la investigación entre 2010 y 2026.

Cada sección indica qué scripts la reproducen (carpeta [`codigo/`](../codigo/)) y dónde están sus resultados
(carpeta [`resultados/`](../resultados/)).

---

## 1. Selección manual

**Scripts:** [`codigo/atlas/`](../codigo/atlas/), pasos 01 a 06.

### 1.1 Corpus y metadatos

- **Origen:** una colección de PDF reunida a mano. Los PDF no se publican (tienen derechos de autor).
- **Extracción (01):** texto, DOI y metadatos de cada PDF.
- **Enriquecimiento (02):** cada PDF se busca en OpenAlex por DOI y, si no tiene DOI o no aparece, en Crossref por
  título, y después en OpenAlex por el DOI encontrado (la búsqueda por título en OpenAlex tiene coste). Se eliminan
  duplicados. Quedan 583 artículos con registro.
- **Resumen:** el de OpenAlex o Crossref; si no hay, el del propio PDF.

### 1.2 Redes de citas (02b)

Tres redes entre los artículos de la colección:

- **Citas directas:** A cita a B, los dos en la colección.
- **Acoplamiento bibliográfico:** A y B citan referencias en común. Existe desde que se publica el más reciente.
- **Cocitación:** un tercer trabajo, de cualquier parte, cita a A y a B a la vez. Requiere descargar los trabajos que
  citan a la colección.

### 1.3 Mapa y temas (03, 04)

- **Embeddings:** SPECTER2 (modelo base más el adaptador de proximidad, aplicado a mano para no depender del paquete
  `adapters`) sobre título + resumen. Si el resumen tiene menos de 200 caracteres, se completa con el comienzo del PDF.
  SPECTER2 está entrenado con citas científicas: dos artículos cercanos en ese espacio tienden a citarse entre sí.
- **Mapa:** UMAP 3D y 2D (coseno, 15 vecinos) para la visualización.
- **Temas:** KMeans con 8 grupos sobre un UMAP de 10 dimensiones (más fiel a las distancias originales que el de 3D).
  Los nombres se pusieron a partir de los términos y títulos más representativos de cada grupo.
- **Vista por citas:** una segunda disposición que combina las redes de citas con el contenido.

### 1.4 Análisis de contenido por tema (06)

- Para cada artículo se reunieron palabras clave, resumen y conclusiones (sección «Conclusiones», o la parte de
  conclusiones de un resumen estructurado, o el final de la discusión).
- Agentes de IA (Claude) analizaron cada artículo: tipo de estudio, pregunta, diseño, hallazgos, limitaciones.
  Después escribieron una síntesis por tema a partir de esos análisis, con citas a los artículos.
- Las entradas con conclusiones copiadas de los PDF no se publican; los análisis y las síntesis, sí
  ([`resultados/atlas/`](../resultados/atlas/)).

---

## 2. Homología persistente de la selección manual

**Scripts:** [`codigo/homologia/`](../codigo/homologia/) (guía detallada en `LEEME.md`).
**Resultados:** [`resultados/homologia/`](../resultados/homologia/) y [`figuras/`](../figuras/).

### 2.1 Espacio semántico (h01–h11)

- **Pregunta:** ¿hay huecos o ciclos en la nube de artículos que no se expliquen por el azar? Por ejemplo, un conjunto
  de temas conectados en anillo sin un centro común.
- **Espacios:** SPECTER2 (principal), y mpnet y bge-m3 como análisis de sensibilidad.
- **Distancias:** de cuerda y geodésicas por grafo de vecinos (k = 6, 15, 30).
- **H0:** modos de densidad (ToMATo) y mezcla entre temas.
- **H1/H2:** con tres nulos (misma covarianza, columnas permutadas y mezcla gaussiana por BIC). La mezcla gaussiana
  conserva los grupos pero no los ciclos, así que es el nulo principal. 200 réplicas cada uno.
- **Criterio de significación:** persistencia multiplicativa (muerte/nacimiento; Bobrowski, Kahle y Skraba, 2017)
  frente al máximo de cada réplica nula. Así se controla el falso positivo por diagrama completo, no por barra.
- **También:** filtración por año, topología de las redes de citas y artículos puente.

### 2.2 Redes de citas validadas (h13–h17)

- **Proyecciones validadas:** el acoplamiento y la cocitación se proyectan desde la red bipartita (artículo ×
  referencia, o citante × artículo). Solo se conservan los pares que comparten más de lo esperado: modelo de
  configuración bipartito (BiCM), prueba de Poisson y corrección de Benjamini-Hochberg (q < 0,05). Sin esta validación,
  un artículo con muchas referencias se conecta con casi todos.
- **Comparación de redes (h14):** cobertura, densidad, mezcla entre temas, alineamiento con el contenido, estabilidad
  por bootstrap. Resultado: el acoplamiento es la red adecuada para ventanas de 3 años; la cocitación queda casi vacía.
- **Homología de las citas (h15):** filtraciones por fuerza de los vínculos y por fecha. Nulos: recableado Curveball
  de la bipartita (conserva los dos grados) y permutación de pesos, 200 réplicas.
- **Quiebres y convergencia (h16, h16b–d):** fusiones de componentes en el tiempo frente a fechas permutadas, e
  indicadores por ventana de 3 años con control del mismo tamaño. Continuidad entre ventanas con bootstrap de las
  referencias. Cohesión de la literatura citada por año.
- **Hipótesis NICE (h17):** el aumento de citas a los criterios diagnósticos en 2020–21 se puso a prueba con
  predicciones fijadas antes de mirar los datos (momento, compañía de citas, contenido) y dos alternativas.

---

## 3. Corpus sistemático de OpenAlex

**Scripts:** [`codigo/openalex/`](../codigo/openalex/), pasos o01 a o22 (`run_update.sh` encadena el recálculo).
**Resultados:** [`resultados/openalex/`](../resultados/openalex/).

### 3.1 Recuperación y limpieza (o01, o01b, o10)

- **Consulta:** título o resumen con `"myalgic encephalomyelitis"`, `"chronic fatigue syndrome"`,
  `"myalgic encephalopathy"` o `"ME/CFS"`; artículos y revisiones; publicados entre 2010 y octubre de 2026.
  Descarga del 19 al 21 de octubre de 2026.
- **Acceso:** sin clave de API, con paginación por número de página; todas las respuestas se guardan en caché.
- **Limpieza:** de 7245 registros se quitaron 72 archivos suplementarios indexados como artículos y 4 duplicados con el
  mismo DOI. Quedan **7169 trabajos**. Los 9 retractados se conservan marcados.
- **Embeddings (o02):** el mismo SPECTER2 con adaptador que el atlas, sobre título + resumen.

### 3.2 Relevancia para la EM/SFC (o18, o19)

La consulta recupera también trabajos que solo mencionan la EM/SFC. Cada trabajo se clasificó en:

- **Central:** la EM/SFC es el objeto del trabajo.
- **Secundaria:** es uno de varios objetos.
- **Mención periférica:** el trabajo trata de otra cosa y la nombra de pasada.
- **Modelo animal:** estudio en animales de un «modelo de síndrome de fatiga crónica».
- **Otro concepto:** «fatiga crónica» en otro sentido (subsalud, burnout, sobreentrenamiento).
- **Incierto:** sin resumen y con un título que no permite decidir.

Método:

- **Regla para los casos claros:** EM/SFC en el título (formas largas, o las siglas CFS y ME en mayúsculas,
  comprobadas a mano) equivale a central.
- **El resto lo clasificó la IA** (Claude) leyendo el título y las frases del resumen que la mencionan, en lotes
  validados por script. Incluye los del tema de medicina china aunque la nombren en el título, porque ahí el término
  designa a veces modelos animales o la «subsalud».
- **Resultado:** 4599 centrales, 613 secundarios, 1678 menciones periféricas, 113 modelos animales, 100 de otro
  concepto y 66 inciertos. La etiqueta y su motivo de cada trabajo están en `relevance.json`.
- Nada se elimina por relevancia: el visor y los análisis temporales permiten filtrar.

### 3.3 Temas y mapa (o11, o12)

- **Método:** la misma receta que el atlas (UMAP 10D + KMeans con 8 temas), con sus propios temas.
- **Temas fijos:** las etiquetas se calcularon con los trabajos de 2010–2025. Los de 2026 se asignan al centroide
  SPECTER2 más cercano, para que los temas no cambien de significado al añadir trabajos.
- **Los 8 temas:** COVID persistente y diagnóstico clínico; retrovirus XMRV; TCC, PACE y vivencia del paciente;
  medicina china y tratamientos complementarios; ejercicio, sistema autonómico y cerebro; dolor, fibromialgia y
  síndromes funcionales; inmunidad, mitocondria y biomarcadores; fatiga, sueño, estrés y revisiones generales.

### 3.4 Red de acoplamiento validada y homología (o04, o05)

- **Red:** acoplamiento validado (BiCM + Poisson + BH, q < 0,05), completa y en 15 ventanas de 3 años (2010–12 a
  2024–26).
- **Homología:** filtración por fuerza en cada ventana y en la red completa, con nulo Curveball (100 réplicas), y
  filtración por fecha con fechas permutadas (200 réplicas).

### 3.5 Indicadores, episodios y sesgo de selección (o03, o06–o08)

- **Mezcla entre temas:** fracción de vínculos entre temas distintos, frente a permutar los temas (1 = azar).
- **Alineamiento contenido–citas:** Spearman entre similitud SPECTER2 y exceso de referencias compartidas.
- **Control del mismo tamaño:** las mismas métricas en submuestras de igual número de artículos, para que el
  crecimiento del corpus no se confunda con una tendencia.
- **Continuidad:** Jaccard de los vínculos entre artículos compartidos por dos ventanas consecutivas.
- **Cohesión:** coseno entre los perfiles de referencias de dos mitades del mismo año, con cohortes del mismo tamaño.
- **Episodios (o07, o08):** el escalón de continuidad de 2019–21, con bootstrap de referencias y la variante sin
  COVID persistente; y el episodio XMRV de 2010–11, con prueba de permutación y la cohesión sin su bloque de
  referencias.
- **Sesgo de la selección manual (o03):** regresión logística de la probabilidad de estar en la selección según año,
  equipo, citas, tipo y título.

### 3.6 Diálogo entre temas (o14, o15, o20)

- **Redes:** citas directas (con dirección) y acoplamiento.
- **Nulo:** los temas se permutan entre artículos del mismo año y del mismo quintil de grado, lo que conserva la red, el
  tamaño y la antigüedad de cada tema. 1000 réplicas y BH sobre las celdas.
- **Métricas:**
  - Observado/esperado por par de temas.
  - Cuasi-independencia (ajuste iterativo sobre las celdas fuera de la diagonal) para el diálogo entre temas sin el
    peso de la autocita.
  - Índice E–I de apertura.
  - Balance de citas recibidas y hechas.
  - Contenido frente a citas.
  - Coeficiente de participación de los artículos puente.
- **Sensibilidad (o20):** el diálogo, la mezcla y la cohesión se repiten solo con los trabajos centrales y secundarios.
  Las conclusiones se mantienen.

### 3.7 Síntesis por tema (o16, o17)

- **Muestra:** unos 80 trabajos con resumen por tema: los más centrales, los más citados, los recientes y los más
  representativos de cada subgrupo. Solo centrales, secundarios y modelos animales, sin retractados.
- **Contexto cuantitativo:** tamaño del tema, años, revistas, diálogo con los demás temas.
- **Redacción:** la IA (Claude) escribió cada síntesis en español e inglés, y un script comprobó que todo artículo
  citado esté en la muestra.

### 3.8 Análisis temporales (o21, o22)

Todos se calculan tres veces, una por filtro de relevancia: todos los trabajos; centrales y secundarios; solo
centrales.

- **Crecimiento:** trabajos por tema y año. El cambio anual de la cuota de cada tema sale de un modelo binomial con
  sobredispersión (cuasi-verosimilitud), 2010–2025.
- **Citas dentro y fuera de cada tema:**
  - Qué parte de las citas que hace un tema va a sí mismo.
  - Qué parte de las que le hacen viene de otros temas.
  - Una versión que descuenta el tamaño: citas a sí mismo divididas por las que tendría si citara según el reparto de
    lo citado ese año.
  - Ventanas móviles de 3 años; se omiten las de menos de 30 citas.
  - Además, la parte de las referencias publicadas desde 2010 que apunta al propio corpus.
- **Estallidos de términos:** modelo de Kleinberg de dos estados por año (s = 2, γ = 1), sobre título, resumen y
  palabras clave de los textos en inglés. Las formas que comparten casi los mismos artículos se agrupan.
- **Estilo de redacción:** fracción de resúmenes con alguna de 12 palabras marcadoras de la redacción con modelos de
  lenguaje (Kobak et al., 2024). Se excluyen de los estallidos.
- **Deriva semántica:**
  - Distancias entre centros de temas estimadas sin sesgo por el tamaño de cada grupo, a partir de las similitudes
    medias entre pares.
  - Separación: distancia media entre centros dividida por la dispersión media.
  - Deriva de cada tema desde 2010–12.
  - Términos propios de cada periodo (log-odds con prior de Dirichlet informativo; Monroe et al., 2008).
- **Ruta principal de citas:** número de caminos que pasan por cada arista (SPC) en la red de citas directas, con búsqueda
  global y key-route de las 15 aristas de más tráfico. Se quitan las citas a trabajos de fecha posterior, para que la red
  sea acíclica.
- **Geografía y colaboración:** país de afiliación (conteo completo), especialización por país (cuota del tema en el
  país / cuota en el corpus), autores por trabajo, colaboración internacional y autores nuevos.
- **Acceso abierto y revistas:** estado actual en OpenAlex, no el que tenía cada trabajo al publicarse.
- **Impacto normalizado:** citas de cada trabajo divididas por la media de su año (MNCS), con intervalo bootstrap;
  trabajos de 2010–2023.
- **Edad de las referencias:** mediana e índice de Price (fracción de referencias de 5 años o menos). Los años de las
  referencias externas se descargaron de OpenAlex (o21); en octubre de 2026 cubren el 84 % de las citas a trabajos de
  fuera del corpus.

---

## 4. Uso de inteligencia artificial

- **Celeste Giraldo Lozano** concibió y dirigió el proyecto. En concreto:
  - Reunió la selección manual de artículos.
  - Planteó las preguntas de investigación.
  - Diseñó las estrategias generales de análisis y los flujos de trabajo: del mapa a la homología persistente, de ahí
    a las redes de citas validadas, al corpus sistemático y a los análisis temporales.
  - Decidió qué metodologías priorizar en cada etapa y qué hipótesis contrastar.
  - Evaluó los resultados y orientó cada nueva fase según lo que iban mostrando.
  - Decidió qué publicar y cómo presentarlo.
- **Claude (Anthropic)**, bajo esa dirección, implementó el código, ejecutó los cálculos, propuso alternativas técnicas
  cuando había varias opciones y redactó los borradores de los textos. También clasificó y resumió artículos: los
  análisis por artículo y las síntesis de la selección manual, la relevancia de los casos dudosos y las síntesis por
  tema del corpus OpenAlex.
- **Controles:** las clasificaciones y síntesis de la IA se validaron con scripts (formato, que cada cita exista en la
  entrada, que cada trabajo tenga etiqueta). Los textos con cifras de la web se generan a partir de los datos.
- **Aun así**, la interpretación puede contener errores: comprueba lo importante en las fuentes antes de citarlo.
- Los prompts principales están en [`prompts/`](prompts/).

---

## 5. Reproducibilidad

- **Dependencias:** Python 3 con numpy, scipy, pandas, scikit-learn, statsmodels, umap-learn, networkx, ripser,
  persim, gudhi, transformers y torch (SPECTER2), sentence-transformers (mpnet, bge-m3), matplotlib y requests.
- **Orden:**
  1. `codigo/atlas/` de 01 a 06, y 05 para construir la web.
  2. `codigo/homologia/` de h01 a h17.
  3. `codigo/openalex/` de o01 a o22. Después `05_build.py` para la web y `07_export_repo.py` para este repositorio.
- **OpenAlex sin clave:** la cuota gratuita diaria es limitada; los pasos de descarga se pueden relanzar y
  continúan donde quedaron.
- **Datos que no están en el repositorio:** los PDF, el texto extraído de ellos, la caché de OpenAlex, los embeddings y
  los puntos de control intermedios. Se regeneran con los scripts, salvo los PDF de la selección manual.
- **Azar:** cada script fija su semilla. Los nulos con permutaciones pueden variar en el último decimal entre versiones
  de las bibliotecas.

## 6. Limitaciones principales

- **Consulta por palabras:** el corpus OpenAlex depende de una consulta en el título o el resumen. Pierde trabajos que
  no usan esos términos e incluye otros donde la EM/SFC es secundaria. La clasificación de relevancia lo mitiga, pero
  la hizo una IA en los casos dudosos.
- **Temas:** salen de agrupar embeddings. Su número (8) y sus límites son una elección; las tendencias son más robustas
  que los valores absolutos.
- **Referencias:** solo el 73 % de los trabajos tiene lista de referencias en OpenAlex; los demás quedan fuera de las
  redes de citas.
- **Metadatos:** los países, el acceso abierto y las citas dependen de OpenAlex, que tiene huecos y errores.
- **Año en curso:** 2026 está incompleto (hasta octubre).
- **Selección manual:** está sesgada hacia ciertos equipos y hacia los artículos más citados (sección 3.5); por eso
  se contrastó con el corpus sistemático.

## Referencias de método

- Bobrowski, O., Kahle, M., y Skraba, P. (2017). Maximally persistent cycles in random geometric complexes. *Annals of
  Applied Probability*, 27(4).
- Kleinberg, J. (2003). Bursty and hierarchical structure in streams. *Data Mining and Knowledge Discovery*, 7(4).
- Kobak, D., González-Márquez, R., Horvát, E.-Á., y Lause, J. (2024). Delving into ChatGPT usage in academic writing
  through excess vocabulary. *arXiv:2406.07016*.
- Liu, J. S., y Lu, L. Y. Y. (2012). An integrated approach for main path analysis. *Journal of the American Society
  for Information Science and Technology*, 63(3).
- Monroe, B. L., Colaresi, M. P., y Quinn, K. M. (2008). Fightin' words. *Political Analysis*, 16(4).
- Saracco, F., et al. (2017). Inferring monopartite projections of bipartite networks: an entropy-based approach.
  *New Journal of Physics*, 19.
- Singh, A., et al. (2023). SciRepEval: a multi-format benchmark for scientific document representations (SPECTER2).
  *EMNLP 2023*.
