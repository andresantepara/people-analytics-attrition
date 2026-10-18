# Agente analítico ampliado versión 2

El agente puede trabajar con las columnas presentes en el CSV y combinar filtros, agrupaciones, indicadores y operaciones sobre agregados. Python realiza los cálculos; Ollama traduce la pregunta a un plan. Python redacta las cifras y su explicación básica con plantillas verificables, sin una segunda llamada al modelo que pueda cambiar los resultados. No es un sistema capaz de resolver cualquier pregunta: el catálogo y los tipos de datos delimitan lo que puede calcular.

## Qué puedes preguntar

| Capacidad | Ejemplo en el chat de Streamlit |
|---|---|
| Media y mediana | ¿Cuál es la edad promedio y la mediana del dataset? |
| Mínimo y máximo | ¿Cuál es el mínimo y el máximo de MonthlyIncome? |
| Suma | ¿Cuál es la suma del ingreso mensual registrado en el dataset? |
| Dispersión | Calcula desviación estándar, varianza e IQR de MonthlyIncome. |
| Percentiles | Calcula los percentiles 25, 75 y 90 de Age. |
| Moda y categorías | ¿Cuál es la moda de JobRole y cuántos roles distintos hay? |
| Coeficiente de variación | Calcula el CV de MonthlyIncome. |
| Forma de la distribución | Calcula asimetría y curtosis de MonthlyIncome. |
| Frecuencias | Distribución de registros por Department y Gender. |
| Proporciones | ¿Qué porcentaje tiene Attrition = Yes por Department? |
| Eventos combinados | ¿Qué porcentaje del dataset tiene más de 35 años y OverTime = Yes? |
| Filtros | Media y percentil 90 de MonthlyIncome entre quienes tienen OverTime = Yes. |
| Dos agrupaciones | Edad promedio por Department y Gender. |
| Comparación de poblaciones | Resta al ingreso mensual promedio de quienes salen el de quienes permanecen. |
| Cocientes y porcentajes | ¿Cuánto representa la mediana de MonthlyIncome respecto de su media, en porcentaje? |
| Cambio relativo | ¿En qué porcentaje difiere la media de MonthlyIncome de Attrition = Yes respecto de Attrition = No? |
| Ranking | Los tres JobRole con mayor ingreso mensual promedio. |
| Correlaciones múltiples | Correlación de Pearson entre Age, MonthlyIncome y TotalWorkingYears. |
| Variables ordinales | Correlación de Spearman entre JobSatisfaction y WorkLifeBalance. |
| Variable binaria | Correlación entre Age y Attrition, indicando la codificación Yes/No. |
| Cruce categórico | Cruza Gender con Attrition e indica porcentajes por fila y V de Cramér. |
| Nulos | ¿Qué columnas tienen valores ausentes? |

Usa nombres exactos de columnas cuando exista ambigüedad. «Sueldo» puede referirse a ingreso mensual o a tarifas diarias/horarias: precisa MonthlyIncome si ese es el dato que necesitas. El modelo debe pedir aclaración si no puede representar fielmente la solicitud. Hay un intento de corrección de planes inválidos. El agente comprueba también ciertas condiciones literales (por ejemplo OverTime = Yes), agrupaciones explícitas y diferencias que requieren una operación; estas reglas no constituyen un intérprete completo del español.

## Cómo comprobar una respuesta

1. Carga el CSV y selecciona el modelo descargado (preferiblemente el 3b de los dos ya admitidos para planes complejos).
2. Autoriza la consulta local en Agente de IA.
3. Formula una pregunta; evita pedir muchas operaciones no relacionadas en un único mensaje.
4. Abre «Evidencia calculada y herramienta ejecutada».
5. Revisa el plan: columnas, filtros, grupos, evento del numerador y operaciones.
6. Contrasta el resultado escrito por el modelo con la tabla calculada. Si el plan no representa tu pregunta, reformúlala; un JSON válido no garantiza que el modelo haya entendido bien.

El selector de los gráficos no filtra el chat. Los filtros deben estar en la pregunta. Cambiar CSV, modelo, umbral o versión del agente reinicia la memoria de sesión.

## Convenciones de cálculo

- Filtros unidos con AND. `in` admite varias categorías de una misma columna. No se admite una expresión lógica OR arbitraria entre columnas.
- Medias y estadísticas omiten nulos. `count` de una columna cuenta valores presentes; `count` de `*` cuenta filas.
- Desviación y varianza son muestrales, con ddof=1; cuantiles usan interpolación lineal; CV=100×std/abs(media). Curtosis es la de Pandas (exceso de curtosis).
- Una suma sin valores válidos produce null, no un cero inventado. Dividir por cero produce null.
- Proporción: numerador = filas que cumplen el evento; denominador = filas del grupo/filtro con observaciones válidas en las variables del evento. Los eventos is_null/not_null incluyen los nulos en el denominador. Se muestran ambos conteos.
- Frecuencias: incluyen valores ausentes como categoría y usan todas las filas del grupo como denominador.
- Pearson usa pares completos; Spearman calcula Pearson sobre rangos medios de esos mismos pares, incluyendo tratamiento de empates. Se informa el número de pares. Las constantes o pares insuficientes producen null.
- En correlaciones, Yes=1 y No=0. La codificación se informa. Las categorías nominales restantes se analizan mediante cruce, no mediante etiquetas numéricas inventadas.
- V de Cramér es descriptiva, sin corrección de sesgo; no indica dirección ni significancia estadística. No se calculan p-valores, intervalos de confianza ni pruebas inferenciales en esta versión.
- Operaciones matemáticas disponibles: suma, resta, multiplicación, división, porcentaje y cambio porcentual entre métricas agregadas o constantes. La referencia de cambio porcentual es el segundo operando. No hay fórmulas libres ni cálculos fila a fila de columnas derivadas.
- EmployeeNumber se excluye de consultas. El umbral de grupos es una regla de estabilidad; no garantiza anonimización.

## Límites explícitos

Hasta 8 condiciones por lista de filtros, 2 variables de agrupación, 10 métricas y 8 operaciones encadenadas por consulta. Correlación de 2 a 12 variables, agrupaciones de hasta 50 categorías por columna, y cruces de hasta 30 categorías por variable. La salida contiene como máximo 100 resultados y marca la truncación. La respuesta resume hasta 12 filas; la interfaz conserva la evidencia calculada dentro de esos límites. No se atribuyen al modelo las explicaciones generadas por Python. Los cruces demasiado grandes piden acotar con filtros.

Los cálculos no demuestran causalidad, no son tasas anuales y no evalúan empleados reales. La respuesta en lenguaje natural puede contener errores: revisa la evidencia. El contexto del modelo aumentó a 8192 tokens para planes más completos; puede consumir más memoria y tardar más que la versión inicial.

## Archivos que cambian

- `src/analytics.py`: motor nuevo de cálculos y validación.
- `src/agent.py`: catálogo dinámico, plan estructurado y coordinación.
- `src/reporting.py`: redacción determinista de resultados, denominadores y limitaciones.
- `src/llm_client.py`: mayor espacio para el plan y contexto.
- `app.py`: ejemplos, plan visible y tabla de resultados.
- `tests/test_analytics.py` y `tests/test_core.py`: 40 pruebas en total.
- `requirements.txt`: NumPy declarado explícitamente (ya instalado como dependencia de Pandas en el entorno probado).

## Instalación de la actualización

Si ya se aplicó directamente al proyecto del Escritorio, basta reiniciar Streamlit. Para usar el ZIP en otro equipo, descomprímelo en una carpeta separada; conserva el CSV y tu entorno anterior. Desde esa carpeta:

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

En el Mac donde el entorno ya funciona, no necesitas recrearlo. Detén Streamlit con Control+C y vuelve a ejecutar `python -m streamlit run app.py` desde el proyecto con `.venv314` activado. Ollama debe estar abierto y el modelo instalado.

## Verificación y documentación académica

Las 40 pruebas verifican cálculos con datos controlados, nulos, filtros, denominadores, planes inválidos y entrega de evidencia mediante un cliente simulado. Se comprobó también la interfaz con carga y evidencia de chat mediante Streamlit AppTest. Estas pruebas no acreditan que cualquier pregunta en español será interpretada correctamente por el modelo local. Las proporciones simples y explícitas de Attrition y el patrón «Cruza Gender con Attrition» (con las columnas que indiques) usan reglas deterministas para proteger la población analizada; el resto se planifica con Ollama. Los planes que no pasan la validación se corrigen una vez y, si siguen siendo inválidos, se pide reformular. Los filtros y agrupaciones siguen visibles para revisión.

Además, se probaron seis consultas de extremo a extremo con el agente y Ollama llama3.2:3b, sobre un conjunto controlado de 40 filas. Cuatro consultas usaron planificación del modelo y dos reglas deterministas. Se contrastaron edad media/mediana, proporciones por departamento, correlaciones, percentil con filtro, diferencia de medias y V de Cramér con resultados esperados. Estas cifras de prueba no son resultados del CSV IBM.

El informe Word y el cuaderno de reconstrucción anteriores describen la versión inicial del agente. Para documentar esta versión, agrega los módulos analytics y reporting, explica que Python redacta la respuesta calculada, sustituye la descripción del catálogo de tres herramientas por las capacidades anteriores y actualiza el número de pruebas y las capturas. No presentes las pruebas simuladas como evaluaciones reales de Ollama.

Fuentes técnicas: [Pandas corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) y [salidas estructuradas de Ollama](https://docs.ollama.com/capabilities/structured-outputs). Las validaciones y límites detallados aquí corresponden al código del proyecto.
