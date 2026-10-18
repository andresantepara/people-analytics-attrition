# Versión 4.3: interpretación local y reproducción

Python produce y conserva la respuesta numérica y la evidencia. Después, `src/interpretation.py` solicita a Ollama una interpretación breve basada en la respuesta calculada, los resultados agregados disponibles y el alcance validado. La interfaz muestra ambos bloques separados. Esto se aplica a consultas analíticas, predictivas y gráficos, incluidas las preguntas resueltas por reglas exactas. Una aclaración no necesita interpretación.

La autorización del usuario precede a ambas fases. No se envía el dataframe ni los puntos individuales de los gráficos. No se ejecuta texto del modelo. Un error de Ollama no elimina el cálculo: aparece una advertencia explícita. Los contextos demasiado grandes requieren reducir variables o segmentos.

Las instrucciones piden conservar unidades, filtros y denominadores, evitar causas o decisiones individuales y reconocer ausencia de evidencia. Estos controles no garantizan que cada interpretación sea correcta; el texto generado debe contrastarse con la evidencia. Una interpretación no modifica las cifras ni representa aprobación humana.

Reproducción: seguir el README desde obtención del CSV, entorno Python, dependencias fijadas, instalación de Ollama, descarga del modelo y ejecución. `scripts/environment.py` registra versiones y modelos instalados; `entorno_ejecucion.json` corresponde a la comprobación del entorno y distingue un servicio desconectado. `.gitignore` excluye `.venv314/`, `.ven314/` y variantes `.venv*/`.

El informe y el guion anteriores deben actualizarse con la demostración de esta versión. La accesibilidad del repositorio GitHub y del video YouTube se verifica mediante sus enlaces reales; no queda acreditada por crear el paquete local.

Fuentes oficiales consultadas: https://docs.ollama.com/macos y https://docs.ollama.com/api/chat (17 de octubre de 2026).

La interpretación es cualitativa: las cifras permanecen en el resultado Python. Si el modelo incluye dígitos, se solicita una reformulación una vez; si persiste, se muestra como no disponible. Este control no garantiza la corrección semántica de toda afirmación: sigue siendo necesaria la revisión humana.

Además de los dígitos, el filtro de salida rechaza expresiones de comparación, tendencias y significancia para evitar que el comentario introduzca análisis nuevos. Es una comprobación textual conservadora, no una garantía semántica.
