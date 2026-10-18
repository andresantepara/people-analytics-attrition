# Corrección 3.0.1: edad promedio con Attrition=Yes

La consulta «Cual es la edad promedio de los que tuvieron atrittion=yes?» ahora normaliza el nombre Attrition y utiliza una regla determinista. Python filtra Attrition=Yes y calcula la media de Age sobre sus valores no ausentes. Se informa el número de registros válidos y ausentes. No usa la partición de predicción 80/20: es un análisis descriptivo de todas las filas que cumplen el filtro.

También admite Attrition=No. Las preguntas con condiciones adicionales siguen la ruta de planificación y validación para no descartar filtros. Se verificaron 60 pruebas con datos controlados; no se dispone del CSV del usuario para informar su resultado numérico.

Reinicia Streamlit y vuelve a cargar el CSV. Repite la pregunta y comprueba en la evidencia: variable Age, estadístico mean, filtro Attrition eq Yes.
