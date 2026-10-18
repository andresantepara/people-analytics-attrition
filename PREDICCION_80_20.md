# Predicción de Attrition con aprendizaje 80/20

La versión 4.0 compara regresión lineal, árbol de decisión y random forest para Attrition. Ollama interpreta las preguntas; scikit-learn entrena los modelos y Python calcula y presenta los resultados. Los análisis descriptivos de la versión anterior se conservan.

## Uso en Streamlit

1. Carga el CSV sintético IBM.
2. Abre **Predicción 80/20** y pulsa **Entrenar y evaluar (80/20)**. Esta operación no necesita Ollama.
3. Revisa tamaño de entrenamiento/prueba, métricas, baseline y matriz de confusión.
4. Para preguntas por segmento, abre **Agente de IA**, comprueba Ollama y autoriza la consulta.
5. Para un perfil ficticio, utiliza el formulario de la pestaña predictiva. El formulario ofrece los 30 predictores del CSV IBM y todos empiezan vacíos. Completa al menos uno; los campos vacíos se excluyen del modelo y no se imputan. Puedes ver todos los valores usados en la evidencia.

También puedes preguntar:

- «Entrena y evalúa el modelo de attrition con 80/20».
- «¿Cuál es el ROC-AUC y el recall del modelo predictivo?».
- «Predice attrition por Department en el conjunto de prueba».
- «Predice attrition para OverTime = Yes en el conjunto de prueba».
- «Estima la probabilidad para un perfil ficticio: Age=30, MonthlyIncome=3500, YearsAtCompany=3, OverTime=Yes».

Las cifras se muestran junto con el plan y su alcance. Una pregunta ambigua puede requerir reformulación. Las predicciones por grupos no son las proporciones descriptivas del archivo completo: se calculan solo sobre la partición de prueba.

## Qué significa 80/20

Se excluyen del aprendizaje las filas sin etiqueta Attrition, mostrando su número. Con 1470 etiquetas válidas, la división produce **1176 registros de entrenamiento y 294 de prueba**. Si cambia el número de filas válidas, cambian los tamaños, con el redondeo propio de train_test_split.

La división es aleatoria, estratificada por Attrition y reproducible con semilla 42. Cada fila etiquetada pertenece a una única partición. No se reentrena con los filtros de las preguntas, no se incorpora el 20% al entrenamiento posterior y no se utiliza ese 20% para seleccionar hiperparámetros o umbrales. Cambiar el CSV reinicia el modelo almacenado en la sesión.

El conjunto de prueba sirve para medir desempeño fuera del entrenamiento. No representa un período futuro: este CSV no aporta una secuencia temporal adecuada para validar predicciones de rotación futura.

## Modelo y prevención de fuga de datos

Se comparan LinearRegression (mínimos cuadrados sobre Yes=1/No=0), DecisionTreeClassifier (max_depth=5, min_samples_leaf=5) y RandomForestClassifier (100 árboles, max_depth=8, min_samples_leaf=3). El umbral de clasificación es 0.5 para los tres. Se recomienda la mayor AP media en validación cruzada estratificada de tres particiones dentro del entrenamiento; el 20% de prueba se usa después para evaluar. Consulta COMPARACION_MODELOS.md para el criterio y las limitaciones. La clase positiva es Attrition=Yes. No se ponderan las clases automáticamente ni se busca mejorar la métrica seleccionando configuraciones sobre el conjunto de prueba.

Las variables numéricas se imputan con la mediana y se escalan; las categóricas se imputan con la moda y se codifican mediante one-hot. Estos pasos forman una Pipeline que se ajusta **únicamente con el entrenamiento**. Una categoría que solo aparece en prueba se ignora en la codificación, sin añadirla al entrenamiento.

El modelo general de evaluación y cohortes incluye las variables laborales disponibles en el CSV, como Age, MonthlyIncome, OverTime, Department, JobRole, antigüedad, experiencia y satisfacción. Se excluyen Attrition como predictor, EmployeeNumber, EmployeeCount, StandardHours, Over18, Gender y MaritalStatus. Las variables totalmente vacías en entrenamiento también se excluyen y se identifican en la evidencia. La simulación tiene su propio catálogo ampliado y usa solo los campos completados, como se explica abajo.

Excluir atributos demográficos no elimina todos los sesgos ni los proxies. El modelo no se valida para usarlo en empleados reales. Se informa si hay predictores idénticos en filas de entrenamiento y prueba; esa coincidencia puede reducir la independencia efectiva de la evaluación.

## Cómo leer los indicadores

| Indicador | Lectura |
|---|---|
| Accuracy o exactitud | Fracción total de clasificaciones correctas. Puede parecer alta por la clase mayoritaria. |
| Balanced accuracy | Promedio de la sensibilidad de ambas clases. |
| Precision de Yes | Entre los Yes predichos, fracción realmente etiquetada Yes. |
| Recall de Yes | Entre los Yes observados, fracción detectada. |
| F1 de Yes | Media armónica de precisión y recall. |
| ROC-AUC | Capacidad de ordenar positivos por encima de negativos; no mide directamente calibración. |
| Average precision | Resumen precisión-recall ponderado por incrementos de recall; no es un AUC trapezoidal. |
| Brier | Error cuadrático de probabilidades; menor es mejor, se compara con el baseline. |
| Matriz de confusión | TN y TP son aciertos; FP y FN son errores. Yes es la clase positiva. |

Se compara el modelo con un baseline que asigna a todos la prevalencia aprendida en entrenamiento y clasifica con el mismo umbral. Si el modelo no supera una referencia, el informe debe decirlo. No se prometen métricas específicas antes de ejecutar el CSV.

## Tres tipos de consulta predictiva

**Evaluación:** devuelve las métricas del 20% reservado. No admite filtros que cambien silenciosamente esa evaluación global.

**Cohorte o grupo:** mantiene el modelo ya entrenado y aplica filtros/agrupaciones sobre el 20% de prueba. Presenta tamaño del grupo, media de probabilidades estimadas, número de Yes predichos con umbral 0.5 y número de Yes observados. La media de probabilidades y la proporción de clases Yes predichas son indicadores distintos. Los grupos pequeños se omiten conforme al umbral de la interfaz. No se filtra por Attrition ni se exponen identificadores o predicciones fila a fila.

**Perfil ficticio (versión 3.1):** el modelo se entrena exclusivamente con las columnas completadas. No requiere los cuatro campos anteriores ni rellena los omitidos. Cero es un valor informado, no un campo vacío. Ofrece las 23 variables numéricas predictoras y las siete categóricas, incluidas Gender y MaritalStatus para análisis académico sintético. Se excluyen Attrition, EmployeeNumber, EmployeeCount, StandardHours y Over18. Las categorías no observadas en entrenamiento se rechazan.

Cada combinación tiene su evaluación y baseline propios sobre la misma partición estratificada 80/20. La evidencia indica variables utilizadas y omitidas, probabilidades, métricas y valores fuera del rango de entrenamiento. Cambiar las variables cambia el modelo; no selecciones la combinación ganadora por las métricas del conjunto de prueba, pues eso lo convertiría en un conjunto de ajuste. Se conserva una caché de hasta ocho combinaciones durante la sesión.

Los campos omitidos del perfil no se imputan. Si el CSV tiene datos ausentes en alguna columna seleccionada, su preprocesado sigue usando mediana/moda aprendidas exclusivamente del entrenamiento. Incluir atributos demográficos puede introducir sesgos: no existe validación de equidad ni autorización para decisiones sobre empleados reales. La probabilidad no tiene calibración externa.

## Requisitos y archivos

Nueva dependencia: `scikit-learn>=1.6,<2`, que instala sus dependencias científicas. Se mantiene el entorno Python 3.14 usado en el proyecto.

- `src/prediction.py`: partición, Pipeline, evaluación, cohortes y perfiles.
- `src/prediction_agent.py`: planificación de preguntas y redacción de respuestas predictivas.
- `src/agent.py`: deriva las preguntas predictivas al nuevo módulo.
- `app.py`: pestaña Predicción 80/20 y simulación de perfiles.
- `tests/test_prediction.py`: pruebas de partición, fuga de datos, métricas, entradas y rutas del agente.

Para iniciar después de actualizar:

```bash
cd ~/Desktop/proyecto_final
source .venv314/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si las dependencias ya se instalaron durante la actualización, basta reiniciar Streamlit. Mantén Ollama abierto para preguntas que requieren planificación en lenguaje natural.

## Documentación académica

El informe y el guion previos describen una aplicación descriptiva. Si presentas esta ampliación, agrega el objetivo predictivo, la división estratificada 80/20, el preprocesado ajustado solo con entrenamiento, la comparación con baseline y las métricas reales obtenidas del CSV. Actualiza las capturas y explica que la estimación es académica sobre datos sintéticos. No copies métricas de las pruebas controladas como si fueran resultados del dataset IBM.

Fuentes técnicas verificables: [partición train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), [prevención de fuga de datos](https://scikit-learn.org/stable/common_pitfalls.html), [regresión lineal](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

## Verificación de la versión 3

93 pruebas automáticas con datos controlados verifican cálculos, partición, prevención de fuga de datos, validación y rutas del agente. También se comprobó la interfaz de carga, entrenamiento y simulación. Se probaron cuatro consultas completas: una interpretada por Ollama local y tres mediante reglas para preguntas explícitas. Estas comprobaciones no equivalen a una evaluación exhaustiva del lenguaje natural. Las métricas del dataset IBM se obtienen al cargar su CSV y entrenar en la aplicación.

En la versión 3.1 se verificó también el formulario con 30 campos inicialmente vacíos y simulaciones de una sola variable. Las pruebas confirman que modificar columnas omitidas no afecta la predicción, que cero es un dato válido y que el subconjunto conserva la partición 80/20 sin fuga de preprocesado.

La simulación parcial compara también los tres modelos con las variables completadas. Devuelve las tres estimaciones y recomienda por CV dentro de ese entrenamiento. El mapa, las cohortes y el Top 5 usan el modelo general recomendado, cuyo nombre se muestra. Si gana la regresión lineal, sus valores recortados son puntuaciones de referencia sin calibración, no probabilidades probabilísticamente validadas.
