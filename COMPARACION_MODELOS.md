# Comparación de tres modelos — versión 4.0

## Uso

Reinicia completamente Streamlit y vuelve a cargar el CSV. En **Predicción 80/20**, pulsa **Entrenar y evaluar (80/20)**. Aparece una tabla con tres filas y una recomendación. El simulador muestra también las tres estimaciones para el perfil y la comparación de desempeño con sus variables. Los campos sin completar siguen excluidos.

## Métodos y selección

- **Regresión lineal:** LinearRegression, mínimos cuadrados sobre Attrition Yes=1 y No=0. No se sustituye por regresión logística.
- **Árbol de decisión:** DecisionTreeClassifier, profundidad máxima 5, mínimo 5 registros por hoja, semilla 42.
- **Random forest:** RandomForestClassifier, 100 árboles, profundidad máxima 8, mínimo 3 registros por hoja, semilla 42. Sin ponderación de clases; demás parámetros por defecto.

1. Se mantiene la partición estratificada 80/20, semilla 42.
2. Dentro del 80% se aplica validación cruzada estratificada de 3 particiones, semilla 42. Cada modelo usa las mismas particiones. La imputación, el escalado y la codificación one-hot se ajustan de nuevo solo en el entrenamiento de cada partición. Los datos de validación no ajustan el preprocesado.
3. Se recomienda la mayor **average precision (AP) media** de validación. AP resume precisión/recuperación de la clase positiva a distintos umbrales, útil cuando hay desbalance. Se informa su desviación estándar entre particiones. Desempates: F1 y luego ROC-AUC de validación; si persiste, orden fijo lineal, árbol, bosque.
4. Cada modelo se ajusta con todo el 80% y se evalúa en el mismo 20%. Esa evaluación no cambia el ganador, aunque otro modelo tenga una métrica de prueba mayor. No hay búsqueda de hiperparámetros ni ajuste del umbral con prueba.

La recomendación significa «mejor según el criterio predefinido entre estos tres modelos y configuraciones». No implica significación estadística ni superioridad universal; diferencias pequeñas o alta variación entre particiones requieren cautela. La AP de validación utilizada para seleccionar es optimista como estimación del ganador; el 20% reservado aporta la evaluación independiente. No se deben repetir cambios de modelos mirando ese 20% para optimizarlo.

## Tabla e interpretación

AP y F1 de validación se calculan dentro del 80%. Exactitud, precisión Yes, recall Yes, F1 Yes, ROC-AUC, AP y Brier de prueba se calculan sobre el 20%. Mayor es mejor salvo Brier (menor es mejor). La recomendación no se basa solo en exactitud. También se mantiene el baseline de prevalencia y la matriz de confusión del modelo elegido.

La regresión lineal puede dar valores negativos o mayores de uno. Se conserva la salida original de cada perfil y se informa cuántas salidas de prueba están fuera de [0,1]. Para comparar con los clasificadores y usar la interfaz, sus salidas se **recortan a [0,1]**; todas las métricas se calculan sobre esa puntuación recortada, con umbral 0.5 para la clase. Recortar no calibra probabilidades. Esta es una referencia de mínimos cuadrados para un objetivo binario, no un modelo probabilístico validado. Árbol y bosque producen probabilidades estimadas pero tampoco tienen calibración externa. Los valores mostrados no son certezas ni tasas anuales.

## Integración

El modelo general recomendado alimenta cohortes, mapa de calor y Top 5, y su nombre aparece en la interfaz. Para cada combinación de campos del simulador se comparan de nuevo los tres modelos, usando únicamente las columnas completadas y la misma partición 80/20. La recomendación del perfil puede ser distinta de la general. No selecciones un modelo por producir la probabilidad de Attrition más alta.

No se requiere una dependencia nueva: se usa scikit-learn ya instalado. El entrenamiento tarda más porque se entrenan y validan tres modelos. El resultado se conserva en memoria durante la sesión.

## Verificación

93 pruebas automáticas con datos controlados, incluidas las regresiones existentes y cuatro pruebas específicas de comparación. Se verificó que modificar el conjunto de prueba no cambia la selección ni las métricas CV, el recorte explícito de salidas lineales y el uso exclusivo de las variables completadas. También se comprobó la tabla, el simulador y la consulta «Compara los tres modelos». No se han calculado métricas del CSV del usuario: aparecerán al entrenar con su archivo.

## Documentación oficial

- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Validación cruzada y separación de prueba](https://scikit-learn.org/stable/modules/cross_validation.html)

Uso académico con datos sintéticos; no se valida para decisiones sobre empleados reales.
