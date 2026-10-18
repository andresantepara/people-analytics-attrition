# People Analytics — versión 4.3

Aplicación académica local en Python, Streamlit y Ollama para explorar IBM HR Analytics Employee Attrition. Incluye estadísticas, cruces, correlaciones, gráficos, comparación de modelos con partición 80/20, simulación, mapa y top 5 de combinaciones.

## Problema y dataset

El objetivo es describir cómo varía Attrition entre segmentos y explorar asociaciones con edad, ingresos, horas extra, departamento y antigüedad desde People Analytics. Se elige el dataset público sintético IBM por su tamaño manejable localmente (referencia: 1470 filas y 35 variables), variables mixtas y objetivo Yes/No. No permite atribuir causas ni decidir sobre empleados reales.

1. Abre https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset.
2. Descarga el dataset (Kaggle puede solicitar cuenta), descomprime el ZIP y localiza `WA_Fn-UseC_-HR-Employee-Attrition.csv`.
3. Conserva el CSV en tu equipo o en `data/` y selecciónalo desde Streamlit. No está incluido en el repositorio; consulta las condiciones de su fuente antes de redistribuirlo.

## 1. Obtener el proyecto y preparar Python

Descarga y descomprime este proyecto desde el repositorio o el ZIP de entrega; abre Terminal dentro de su carpeta, donde está `app.py`. Instala Python 3.14 desde https://www.python.org/downloads/ si no lo tienes. La configuración documentada es macOS con Python 3.14.

```bash
python3.14 --version
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` registra las versiones del entorno local verificado. En otro sistema, si alguna dependencia fijada no tiene distribución compatible, utiliza `python -m pip install -r requirements.txt`, ejecuta las pruebas y registra las versiones de ese nuevo entorno. No se garantiza compatibilidad idéntica en cualquier plataforma. Para una instalación ya existente, activa el entorno y actualiza dependencias sin borrarlo.

## 2. Instalar y ejecutar Ollama

1. Descarga Ollama para Mac: https://ollama.com/download/mac.
2. Abre el archivo descargado y coloca Ollama en **Aplicaciones**. Abre la aplicación y completa su configuración de Terminal si la solicita.
3. Inicia Ollama y descarga un modelo (la descarga requiere Internet y espacio en disco):

```bash
open -a Ollama
ollama --version
ollama pull llama3.2:3b
ollama list
```

Si Terminal no encuentra el comando pero Ollama está instalado:

```bash
export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"
ollama --version
```

En equipos con recursos limitados puedes descargar `ollama pull llama3.2:1b`; selecciona ese mismo nombre en Streamlit. No es necesario descargar ambos modelos. El servicio local utiliza `http://127.0.0.1:11434`; no publiques ese puerto. Si utilizas únicamente la CLI, `ollama serve` inicia el servidor en una terminal aparte; no lo ejecutes simultáneamente si la aplicación ya sirve ese puerto.

Instalación oficial y requisitos del sistema: https://docs.ollama.com/macos.

## 3. Probar e iniciar Streamlit

Con el entorno activado y desde la carpeta del proyecto:

```bash
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Abre la dirección local indicada por Streamlit. Carga el CSV, elige el modelo instalado y utiliza la comprobación de conexión con Ollama. Autoriza el envío local de pregunta, catálogo y resultados calculados. Pregunta, por ejemplo: **Edad promedio entre Attrition=Yes**. Verás el resultado de Python y, después, la interpretación de Ollama. Despliega la evidencia y contrasta valores, filtros y denominadores.

Para detener Streamlit, pulsa `Ctrl+C` en su terminal. Tras actualizar archivos, detén y vuelve a iniciar la aplicación para evitar importaciones antiguas.

## Arquitectura y agente

```text
app.py                       interfaz y controles Streamlit
requirements.txt             dependencias compatibles
requirements-lock.txt        versiones del entorno verificado
.gitignore                   exclusiones de publicación
README.md                    reconstrucción y demostración
data/                        instrucciones de obtención del CSV
assets/                      recursos auxiliares
src/__init__.py
src/data_loader.py           carga y validación
src/eda.py, analytics.py      funciones de análisis
src/visualizations.py        gráficos Plotly
src/chart_agent.py           planificación y gráficos del chat
src/agent.py                 autorización, planificación y coordinación
src/llm_client.py            conexión HTTP local
src/reporting.py             resultado numérico redactado por Python
src/interpretation.py        interpretación de resultados mediante Ollama
src/model_comparison.py      comparación y selección de modelos
src/prediction.py            entrenamiento, evaluación y perfiles
src/prediction_agent.py      consultas predictivas
src/risk_heatmap.py          segmentos, mapa y combinaciones
tests/                       pruebas automatizadas
scripts/environment.py       registro de entorno y versión/modelos Ollama
```

Flujo: usuario → Streamlit → agente Python → plan validado → herramientas analíticas → resultado calculado → Ollama interpreta → respuesta y evidencia para revisión humana. Las consultas exactas pueden omitir la planificación LLM, pero también pasan por la interpretación de Ollama. Las solicitudes de aclaración no generan una interpretación sin resultados.

Python conserva la autoridad sobre los cálculos. Ollama recibe el texto numérico calculado y su alcance, no el dataframe completo ni puntos individuales de gráficos. La interpretación se etiqueta separadamente y puede contener errores: el usuario debe contrastarla. Si falla la conexión, se conserva el resultado y se identifica la interpretación como no disponible. Los contextos excesivos se rechazan explícitamente sin recortarlos silenciosamente.

El modelo no ejecuta código generado ni modifica datos. Hay autorización previa, catálogo restringido, validación de planes, umbral de grupos y memoria de tres intercambios durante la sesión. No hay autenticación empresarial; el umbral no garantiza anonimato.

## Evidencia de reproducibilidad

```bash
python scripts/environment.py > entorno_ejecucion.json
```

Este comando registra Python, plataforma, paquetes, versión del servicio Ollama y nombres/identificadores de modelos disponibles. No incluye directorios personales ni CSV. Revisa el registro y documenta el modelo seleccionado para la demostración. El registro no instala ni descarga nada y señala si Ollama está desconectado. Conserva el registro usado en la entrega junto con el código.

Las pruebas usan datos controlados y clientes simulados; no reemplazan una comprobación real con Ollama y el CSV de la demostración. La selección de modelos usa validación cruzada dentro del 80% de entrenamiento; el 20% se reserva para evaluación. No se promete un porcentaje de exactitud.

## GitHub y entrega

Publica estos archivos en un repositorio accesible al evaluador. Antes de subir, revisa `git status --short` y comprueba que no incluyes entornos, CSV, secretos, pesos ni copias innecesarias. `.gitignore` excluye `.venv314/`, variantes `.venv*/`, `.ven314/`, archivos temporales y modelos. Si ya estaban versionados, ignorarlos no elimina su seguimiento: retíralos del índice sin borrar tu copia local.

Incluye el enlace real y accesible en el Anexo B del informe. El código preparado no acredita por sí solo que GitHub sea público o accesible. Anexo A: capturas reales; Anexo C: YouTube accesible, máximo cinco minutos. Informe final: PDF máximo 15 páginas incluidos anexos. No se incluyen pesos de Ollama ni entornos en el paquete.

El notebook v4.3 reconstruye los archivos, incluido `.gitignore`, como material complementario. Para el flujo completo usa ejecución local: el localhost de Colab no corresponde al Ollama del Mac. Las guías de versiones previas se conservan como historial; este README y `INTERPRETACION_OLLAMA.md` describen el comportamiento actual.

La interpretación es cualitativa: las cifras permanecen en el resultado Python. Si el modelo incluye dígitos, se solicita una reformulación una vez; si persiste, se muestra como no disponible. Este control no garantiza la corrección semántica de toda afirmación: sigue siendo necesaria la revisión humana.
