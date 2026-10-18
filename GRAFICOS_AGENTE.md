# Gráficos en el agente — versión 4.1

Reinicia Streamlit, carga el CSV y abre **Agente de IA**. Autoriza la consulta local como hasta ahora. El gráfico aparece debajo de la respuesta y permanece en el historial de las últimas tres consultas. El botón Borrar conversación elimina también los gráficos del chat.

## Preguntas de ejemplo

- Histograma de Age.
- Grafica la distribución de edad.
- Grafica el ingreso mensual promedio por departamento.
- Grafica el porcentaje de Attrition = Yes por Department.
- Gráfico de barras de Gender.
- Gráfico de dispersión de Age y MonthlyIncome.
- Mapa de calor de correlación entre Age, MonthlyIncome y YearsAtCompany.
- Muéstrame en barras la mediana de MonthlyIncome según Department.
- Histograma de Age entre OverTime = Yes.
- Grafica el resultado anterior.

Las preguntas exactas de los ejemplos básicos usan reglas deterministas. Para otras formulaciones, Ollama propone un plan estructurado de gráfico, variables, filtros y medida. Python valida el plan, calcula los valores y dibuja con Plotly. Nunca se ejecuta código generado por el modelo. Si el plan falla, se solicita una corrección una vez; si sigue siendo inválido se pide reformular.

## Qué representan

- **Barras:** conteo, media, mediana, suma o porcentaje observado de Attrition=Yes por categoría. Hasta 50 categorías.
- **Histograma:** frecuencia de una variable numérica, 15 intervalos por defecto, de 2 a 50 a petición. Excluye nulos y muestra todos los intervalos. El umbral mínimo se aplica al total válido, no a cada intervalo.
- **Dispersión:** dos variables numéricas; excluye pares incompletos. Hasta 5000 puntos; si hay más, muestra fija con semilla 42. Solo incluye los dos ejes, sin identificadores.
- **Correlación:** Pearson o Spearman entre 2 y 12 variables, con la misma validación del motor analítico. La diagonal queda vacía. Las variables ordinales de los ejemplos exactos usan Spearman.
- **Cruce:** mapa de porcentajes por fila de dos categorías, calculado por el motor de contingencia.

El mínimo de registros de la barra lateral se respeta en las estadísticas agrupadas. Los grupos o resultados insuficientes no se inventan. La evidencia informa filtros, denominadores, ausentes y resultados calculados. El usuario debe comprobar que el plan interpretado coincide con la pregunta.

«Grafica el resultado anterior» reutiliza la evidencia o el gráfico anterior sin recalcular con otros filtros. Si el resultado no admite una visualización de las soportadas, el agente pide que se especifiquen las variables y medida. Las predicciones tienen sus tablas y mapas propios: no se convierte silenciosamente una solicitud predictiva en un porcentaje observado. No se generan series temporales sin una fecha válida ni gráficos de tipos no soportados.

Los datos se procesan localmente. Ollama recibe el catálogo y la pregunta para las consultas que requieren interpretación; no recibe imágenes para inventar los resultados. Los gráficos son descriptivos del dataset sintético y no prueban causalidad.

## Verificación

93 pruebas automáticas, incluidas nueve de gráficos: conteos, medias, denominadores, filtros, nulos, rechazo de identificadores y código, autorización e historial. Se comprobó que el gráfico aparece en el chat y se borra con la conversación. Se verificó con Ollama local una consulta libre de histograma de Age filtrado por OverTime=Yes; la mediana por departamento usa una regla directa; las pruebas usan datos controlados, no métricas atribuidas al CSV del usuario.
