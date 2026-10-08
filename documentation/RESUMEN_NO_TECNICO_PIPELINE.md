# Pipeline de Estrategias — Explicación en lenguaje sencillo

Este documento explica, sin tecnicismos, el proceso descrito en `PIPELINE_OPTIMIZACION_CLUSTERING_VALIDACION_FINAL.md`.
Idea central: **no queremos la estrategia que mejor explica el pasado, sino la que tiene más probabilidad de ganar en el futuro.**

Todo se controla desde un único archivo de recetas: `pipeline_master.yaml`. Si se cambia la receta, se empieza una prueba nueva. No se cambian las reglas a mitad del partido.

---

## Etapa 0 — Preparación: dejar todo claro antes de probar

Antes de probar nada, se escribe en papel:
- Qué hace la estrategia y por qué debería funcionar (su hipótesis).
- Qué parámetros se pueden tocar y cuáles no.
- Con qué datos históricos se prueba, de qué broker vienen y en qué horario.
- Cuánto cuesta cada operación (comisión, spread, deslizamiento).
- Cuántas pruebas se van a hacer, con qué presupuesto.

Además se comprueba:
- Que los datos son correctos y no tienen trampas (huecos, horas duplicadas por cambio de hora, usar información del futuro sin querer).
- Que hay suficientes operaciones para que las estadísticas tengan sentido. Si hay muy pocas, el resultado es "no concluyente", no se aprueba ni se suspende.

Analogía: es como decidir las reglas del examen antes de hacerlo, no después de ver las preguntas.

## Etapa 1 — Búsqueda de parámetros: probar muchas variantes

Se prueban cientos de combinaciones de parámetros (por ejemplo, periodo 10, 12, 14... desviación 1.8, 2.0...) siempre con los mismos datos y mismos costes.

Se guarda el resultado de TODAS las pruebas, no solo de las buenas. Las malas también son evidencia.

Se busca la zona estable y rentable, no el pico perfecto que solo funcionó una vez. Es como buscar un barrio bueno para vivir, no la casa más bonita de una calle mala.

## Etapa 2 — Agrupar parecidos: quedarse con pocos representantes

De cientos de variantes buenas, muchas son casi iguales entre sí. Se agrupan las parecidas y se elige un representante por grupo.

Así pasamos de, por ejemplo, 300 variantes a menos de 250 realmente distintas. También se calcula cuántas ideas independientes hay de verdad, porque muchas pruebas copian la misma idea.

Analogía: si 50 personas proponen casi lo mismo, elegimos a un portavoz por grupo.

## Etapa 3 — Mercados imaginarios: ¿es suerte o es de verdad?

Aquí vienen tres preguntas separadas:

1. **Consistencia:** se crean 20-50 mercados falsos pero realistas (con la misma volatilidad que el real) y se comprueba si el resultado real parece normal dentro de esos mundos. Si en el mundo real rinde mucho peor que en todos los falsos, algo falla.
2. **Trampa:** se prueba la estrategia en datos donde no hay nada que ganar (ruido puro). Si ahí también gana, es que el simulador tiene un error o hace trampa. Se para todo y se arregla.
3. **Suerte:** como hemos probado cientos de variantes, alguna ganará por pura suerte. Se aplican pruebas estadísticas que corrigen por haber hecho tantos intentos. Solo pasa quien sigue ganando después de ese descuento.

Además se vuelve a probar con costes más altos (doble spread, más deslizamiento). Si solo gana con costes perfectos, no sirve.

## Etapa 3B — Probar en distintos periodos: que no dependa de dos años concretos

No vale que funcione solo en 2015-2021 y falle en el resto. Se parte la historia en trozos y se comprueba:
- Que eligiendo al mejor del pasado, ese mejor también funciona en el trozo siguiente.
- Que re-optimizando en cada periodo (como se haría en la vida real) el método sigue ganando.
- Que en la mayoría de los trozos el resultado es positivo.

Si el éxito depende de un periodo concreto, se marca como dudoso.

## Etapa 4 — Selección final, prueba ciega y tamaño de apuesta

1. **Quitar duplicados:** entre los supervivientes, si dos ganan y pierden a la vez, nos quedamos con el mejor. Queremos pocos y diferentes, no 10 iguales.
2. **Mover un poco los parámetros:** si cambiando un 10% todo se rompe, era un pico de suerte. Solo vale la zona robusta.
3. **Prueba con ticks reales:** se repite la prueba con datos tick a tick del broker y con peores condiciones (más spread, retrasos). El simulador por velas suele ser optimista.
4. **Puertas de calidad:** rentabilidad mínima, pérdidas máximas limitadas, que gane en distintos años y horarios, que no dependa de un solo mes milagroso.
5. **Prueba ciega (lockbox):** solo los que pasaron todo lo anterior ven por primera y única vez los datos reservados de 2022-2023. Una sola oportunidad, sin reintentar. Si falla aquí, fuera. Cada uso queda registrado.
6. **Tamaño de apuesta:** se calcula cuánto arriesgar por operación y dónde poner stop y objetivo, con criterios prudentes, nunca con el mejor caso.

Salida: `promoted.csv`, la lista corta de candidatos finales con todas sus pruebas trazables.

## Meta-etapa M — Calibrar el examen: ¿el filtro funciona?

Antes de fiarse de los filtros, se comprueba que los filtros funcionan:
- Se pasan 200 estrategias tontas (al azar) por todo el proceso. Casi ninguna debería llegar al final (menos del 5%). Si llegan muchas, los filtros son blandos.
- Se crean estrategias con una ventaja conocida y se comprueba que el proceso sí las deja pasar (al menos el 70% de las buenas).

Solo con esto se fijan los umbrales definitivos. Sin calibrar, los números son opiniones.

## Etapa 5 — Cartera: no apostar todo a una carta

La unidad importante no es una estrategia, es el conjunto. Cinco variantes del mismo sistema no diversifican.

Se hace:
- Reparto prudente entre estrategias distintas (ninguna pesa demasiado).
- Control de que no estamos apostando 5 veces a lo mismo sin darnos cuenta (por ejemplo, 5 pares que todos apuestan contra el dólar).
- Simulación de miles de futuros posibles para saber qué pérdida máxima es normal y cuál es señal de alarma.
- Prueba de capacidad: cuánto dinero aguanta la estrategia antes de que los costes se la coman.

Solo entra una estrategia nueva si mejora la cartera completa, no solo porque ella sola sea buena.

## Etapa 6 — Dinero real: salir a operar con red de seguridad

Salida gradual y vigilada:
1. **Paper:** opera sin dinero real 60 días, se compara que hace lo mismo que en el simulador.
2. **Micro:** dinero real muy pequeño.
3. **Escalado:** se sube el tamaño solo si todo cuadra.
4. **Vigilancia continua:** se mira si el resultado se sale de lo esperado, si tarda mucho en recuperarse, si opera mucho más o menos de lo normal, si los costes suben.
5. **Frenos automáticos:** límites por operación, por día, por estrategia y por cartera. Si algo se rompe (se cae la conexión, los datos no llegan, el spread se dispara), el sistema para solo y avisa. Hay botón de pánico manual.
6. **Retirada:** regla escrita de antemano para quitar una estrategia (por ejemplo, pérdida mayor que el peor caso simulado). No se arregla en caliente; si hay que re-optimizar, es un proceso nuevo completo.

## Gobernanza — Quién aprobó qué

Cada estrategia tiene un pasaporte: qué versión es, con qué datos se probó, qué pruebas pasó, quién la aprobó y qué versión exacta está en real. Sin ese pasaporte completo, no se opera.

> Resumen en una frase: probar mucho, desconfiar de lo que brilla demasiado, comprobar que sobrevive a costes altos, a otros periodos y a datos nunca vistos, calibrar los filtros, combinar en cartera y salir a real poco a poco con frenos automáticos.
