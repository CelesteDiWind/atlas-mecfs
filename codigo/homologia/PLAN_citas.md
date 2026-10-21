# Homología persistente de las redes de citas: plan (2026-10-19)

Preguntas del usuario:
- **a.** Qué red sirve mejor para ver la homología del estado del arte por años: cocitación o acoplamiento.
- **b.** Según a, buscar huecos significativos.
- **c.** Cruzar con la semántica.
- **Pregunta de fondo:** ¿se ven momentos de quiebre y de convergencia disciplinar?

Decisiones confirmadas: ventanas deslizantes de 3 años, descarga de los citantes con fecha, y entregar primero las fases 0 y 1.

## Por qué h08 no sirve como respuesta

- Mezcló cocitación y acoplamiento con el peso máximo.
- Usó Rips sobre geodésicas de una red muy densa, con muchos empates.
- El nulo eran redes aleatorias. Una proyección bipartita crea triángulos por construcción, así que esas redes tenían el doble de bucles que la real y ningún hueco podía destacar.

## Fases

**0. Datos (h13).**
- Citantes con `publication_date`.
- Redes bipartitas paper × referencia y citante × paper.
- Fechas exactas de los papers.

**0. Proyección validada (h14).**
- Se ajusta el modelo de configuración bipartito (BiCM), que conserva el número de referencias de cada paper y la popularidad de cada referencia.
- Se conserva un vínculo si comparte más de lo esperado (Poisson y Benjamini-Hochberg, q < 0,05).

**1. Comparación de redes (h14-h15).**
- **Cobertura:** papers con datos, componente gigante y papers aislados por ventana.
- **Vínculos validados y densidad.**
- **Fecha:** el acoplamiento se fecha por el paper más reciente del par; la cocitación, por el segundo citante común.
- **Estabilidad:** bootstrap de referencias o citantes, comparando la coincidencia de vínculos (Jaccard).
- **Bucles frente al nulo bipartito (Curveball):** en la filtración por fuerza y en la filtración por fecha.
- **Relación con los temas.**

**2. Huecos (b).**
- **Filtración por fuerza:** rango de significación del vínculo, sobre la red completa y por ventana.
- **Filtración por fecha:** una laguna nace cuando un anillo de papers queda sin unir y muere cuando un paper la cierra, que es el evento de convergencia. En H0, que dos componentes se fusionen indica que dos líneas confluyen.
- **Nulos:** Curveball con 200 réplicas; permutación de fechas, también dentro de cada tema.
- **Representantes:** guiados por el cociclo, con estabilidad.

**3. Cruce semántico (c).**
- Distancia de contenido dentro de cada hueco: si es baja, es una conexión perdida; si es alta, es un anillo interdisciplinar.
- Si los huecos semánticos aparecen también en las citas.
- Si los papers que cierran huecos son los puentes semánticos de h09.
- Mantel contenido–citas por ventana.

**4. Quiebres y convergencia.** Ventanas 2017-19 … 2022-24, cada indicador con un control del mismo tamaño:
- Continuidad de comunidades (ARI entre ventanas consecutivas).
- Mezcla entre temas.
- Escala de fusión de los temas en H0.
- Cronología de lagunas que se abren y se cierran.
- Mantel contenido–citas.

Hipótesis a contrastar: la llegada de la COVID persistente (2020-21) reorganizó la red.
