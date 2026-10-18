# Mapa de probabilidades de Attrition = Yes

1. Carga el CSV IBM sintético y abre **Mapa de probabilidades**.
2. Elige las **Variables del mapa (columnas)**; se ofrecen los 30 campos de segmentación disponibles en el archivo IBM completo.
3. Pulsa **Generar mapa de probabilidades**. Si ya entrenaste el modelo general, se reutiliza.
4. Elige los **Segmentos del mapa (filas)**. Las selecciones cambian el mapa y la tabla de detalle.
5. Pasa el cursor por cada celda para ver su rango/categoría, probabilidad media y número de registros.
6. Abre **Detalle de segmentos, condiciones y probabilidades** para ver las condiciones ordenadas por probabilidad descendente.

Cada celda promedia las probabilidades estimadas por el modelo general para los registros del 20% de prueba que pertenecen al segmento indicado. No son predicciones de personas reales, porcentajes de correlación ni probabilidades de salida en un plazo futuro. La tabla separa la probabilidad media predicha de la proporción de Attrition=Yes observada.

Las variables categóricas usan sus categorías. Las numéricas se dividen con los terciles aprendidos únicamente del 80% de entrenamiento, en rangos bajo, medio y alto. Si hay valores repetidos pueden quedar menos rangos o tamaños distintos. Cada variable tiene sus propios límites, visibles al pasar el cursor y en la tabla. Los valores ausentes se agrupan como Sin dato.

Cada columna segmenta por separado: no se aplica una intersección entre todas las filas o columnas seleccionadas. Por ejemplo, una celda Age/Rango bajo resume personas de ese rango de edad; otra OverTime/Categoría: Yes resume quienes tienen horas extra. No se estima el efecto causal de una variable. Para intersecciones específicas se utiliza una consulta con filtros en el agente.

El mínimo de registros válidos de la barra lateral se aplica a cada celda. Las celdas insuficientes o no aplicables quedan vacías; no equivalen a 0%. Gender y MaritalStatus se pueden usar para segmentar y examinar resultados, pero no son predictores del modelo general. El mapa no usa los modelos parciales del simulador.

La pestaña incluye la evaluación 80/20 del modelo utilizado. Se verificaron 93 pruebas automáticas y el flujo de interfaz para generación, selección de columnas/filas y selecciones vacías, con datos controlados; no se han calculado aquí métricas del CSV del usuario.

## Top 5 de combinaciones (versión 3.3)

Después del mapa, activa **Mostrar las cinco combinaciones principales**. Selecciona al menos dos variables en las columnas. El apartado examina todos los pares de variables seleccionadas y cruza los segmentos permitidos en las filas. Cada combinación es una intersección: los mismos registros deben cumplir ambas condiciones, por ejemplo Department=Sales Y OverTime=Yes.

El ranking usa la media de probabilidades del modelo general sobre el 20% de prueba, no la proporción observada de salidas. Muestra ambas medidas por separado, las dos condiciones y el número de registros. En empate de probabilidades se prioriza el mayor tamaño del grupo; después, el orden de las condiciones. Se muestran hasta cinco combinaciones entre las válidas; no se inventan filas si hay menos. El umbral de la barra lateral se aplica a cada intersección.

Los rangos numéricos son los mismos del mapa, definidos solo con entrenamiento. El modelo se reutiliza sin ajustarlo por combinación. Se informa cuántos pares y combinaciones se examinaron y cuántas se omitieron por tamaño.

Este orden es exploratorio. Los grupos pueden compartir registros y sus conteos no deben sumarse. Examinar muchas combinaciones puede destacar diferencias fortuitas; no demuestra causalidad, interacción estadística ni estabilidad futura. Confirmar hallazgos con datos independientes antes de generalizar. No seleccionar modelos o hiperparámetros a partir de este ranking ni usarlo para decisiones sobre personas reales.

Se verificaron cálculos de intersecciones, orden descendente, umbrales, selección de segmentos y que la etiqueta observada no determina el ranking, además del flujo de interfaz con datos controlados.
