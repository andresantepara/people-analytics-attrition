"""Interfaz: ejecutar con python -m streamlit run app.py."""
import hashlib
import streamlit as st
import plotly.io as pio
from src.data_loader import load_dataset
from src.eda import profile, summary, segment, numeric_summary, SEGMENTS
from src.visualizations import attrition_bars, distribution, correlation_heatmap
from src.llm_client import OllamaClient, ALLOWED_MODELS
from src.agent import AnalyticsAgent, VERSION
from src.prediction import AttritionPredictor, SIMULATION_FEATURES, NUMERIC_FEATURES
from src.prediction_agent import render_prediction
from src.risk_heatmap import segmented_probabilities, probability_heatmap, top_combinations

st.set_page_config(page_title='People Analytics IBM', page_icon='📊', layout='wide')
st.title('People Analytics con IA local')
st.caption('Proyecto académico • IBM HR Analytics Employee Attrition • Python, Streamlit y Ollama')
st.info('Usa únicamente el dataset sintético IBM. Las asociaciones no prueban causalidad ni justifican decisiones sobre empleados reales.')

with st.sidebar:
    st.header('1 · Dataset')
    uploaded = st.file_uploader('CSV de IBM HR Analytics', type=['csv'])
    st.markdown('[Descargar dataset en Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)')
    st.header('2 · Modelo local')
    model = st.selectbox('Modelo', ALLOWED_MODELS)
    client = OllamaClient(model)
    if st.button('Comprobar Ollama'):
        try:
            status = client.status()
            st.json(status)
            if status['disponible']:
                st.success('Servidor accesible y modelo instalado.')
            else:
                st.warning(f'Descarga el modelo: ollama pull {model}')
        except RuntimeError as exc:
            st.error(str(exc))
    min_group = st.number_input('Mínimo de registros válidos por grupo', 1, 100, 5)
    st.caption('Umbral de estabilidad descriptiva; no garantiza anonimato.')

if uploaded is None:
    st.write('Carga el archivo WA_Fn-UseC_-HR-Employee-Attrition.csv para comenzar.')
    st.stop()
content = uploaded.getvalue()
try:
    df = load_dataset(content)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

fingerprint = hashlib.sha256(content).hexdigest() + model + str(min_group) + VERSION
if st.session_state.get('context_id') != fingerprint:
    st.session_state.context_id = fingerprint
    st.session_state.history = []
    st.session_state.predictor = AttritionPredictor(df)
    st.session_state.prediction_report = None
if df.shape != (1470, 35):
    st.warning(f'El archivo tiene {df.shape[0]} filas y {df.shape[1]} columnas. La versión de referencia contiene 1470 × 35. Verifica su procedencia.')

metrics = summary(df)
a, b, c, e = st.columns(4)
a.metric('Registros', metrics['filas']); b.metric('Variables', metrics['variables'])
c.metric('Attrition = Yes', metrics['salidas'])
e.metric('Proporción de Attrition', 'Sin datos' if metrics['proporcion_pct'] is None else f"{metrics['proporcion_pct']} %")
st.caption(f"Denominador: {metrics['attrition_validos']} valores válidos. Attrition ausente: {metrics['attrition_nulos']}. No es una tasa anual.")

exploration, charts, chat, prediction_tab, heatmap_tab = st.tabs(['Exploración', 'Visualizaciones', 'Agente de IA', 'Predicción 80/20', 'Mapa de probabilidades'])
with exploration:
    st.subheader('Perfilado y valores nulos')
    st.dataframe(profile(df), use_container_width=True)
    st.subheader('Resumen numérico')
    numeric_table = numeric_summary(df).T.rename_axis('Variable')
    st.caption(f'{len(numeric_table)} variables numéricas detectadas en el CSV. count indica valores no ausentes; std, desviación estándar; 25%, 50% y 75%, cuartiles. Los identificadores y códigos numéricos también aparecen: su promedio puede no tener interpretación sustantiva.')
    numeric_columns = list(numeric_table.index)
    selected_numeric = st.multiselect('Variables numéricas del resumen', numeric_columns, default=numeric_columns, key='numeric_summary_' + hashlib.sha256(content).hexdigest())
    if selected_numeric:
        st.dataframe(numeric_table.loc[selected_numeric], use_container_width=True)
    else:
        st.info('Selecciona al menos una variable numérica para mostrar el resumen.')
    st.caption('No se imputan ni eliminan filas automáticamente; los cálculos indican su denominador.')
with charts:
    st.subheader('Comparación entre grupos')
    column = st.selectbox('Segmentar por', SEGMENTS)
    table = segment(df, column, int(min_group))
    if table.empty:
        st.warning('No hay grupos que cumplan el mínimo de valores válidos.')
    else:
        st.plotly_chart(attrition_bars(table, column), use_container_width=True)
        st.dataframe(table, use_container_width=True)
    st.subheader('Explorar variables numéricas')
    st.caption('Busca y selecciona las variables que quieres visualizar. Estas selecciones cambian las columnas mostradas; no filtran registros ni las consultas del agente.')
    if numeric_columns:
        number = st.selectbox('Distribución numérica', numeric_columns)
        st.plotly_chart(distribution(df, number), use_container_width=True)
    correlation_columns = st.multiselect('Variables numéricas para correlacionar', numeric_columns, default=numeric_columns, key='numeric_correlation_all_' + hashlib.sha256(content).hexdigest())
    if len(correlation_columns) >= 2:
        matrix = df[correlation_columns].corr(method='pearson', min_periods=3)
        st.plotly_chart(correlation_heatmap(matrix), use_container_width=True)
    else:
        st.info('Selecciona al menos dos variables para mostrar la matriz de correlación.')
    st.caption('Pearson resume asociación lineal, no causalidad; usa pares completos. JobSatisfaction es ordinal: interpretar con cautela. Las celdas vacías indican información insuficiente o varianza nula.')
    approved = st.checkbox('He revisado esta tabla y autorizo descargar los resultados agregados.')
    st.download_button('Descargar tabla revisada', table.to_csv(index=False).encode('utf-8-sig'),
                       'segmentacion.csv', 'text/csv', disabled=not approved)
with chat:
    st.subheader('Consulta sobre resultados calculados')
    st.write('Consulta medias, percentiles, proporciones, comparaciones y correlaciones de las variables del CSV.')
    with st.expander('Ejemplos de preguntas y alcance'):
        st.markdown('''- Histograma de Age.
- Grafica el ingreso mensual promedio por departamento.
- Grafica el porcentaje de Attrition = Yes por Department.
- Gráfico de dispersión de Age y MonthlyIncome.
- Mapa de calor de correlación entre Age, MonthlyIncome y YearsAtCompany.
- Grafica el resultado anterior.
- ¿Cuál es la edad promedio y la mediana?
- ¿Qué porcentaje tiene Attrition = Yes por departamento?
- Promedio y percentil 90 de MonthlyIncome entre quienes tienen OverTime = Yes.
- Correlación de Pearson entre Age, MonthlyIncome y TotalWorkingYears.
- Correlación de Spearman entre JobSatisfaction y WorkLifeBalance.
- Diferencia entre el ingreso mensual promedio de quienes salen y quienes permanecen.
- Cruza Gender con Attrition e indica porcentajes por fila.
- Entrena y evalúa el modelo de attrition con 80/20.
- Predice attrition por Department en el conjunto de prueba.
- Estima la probabilidad para un perfil ficticio: Age=30, MonthlyIncome=3500, YearsAtCompany=3, OverTime=Yes.

Las operaciones se aplican a agregados. No se ejecuta código libre. La evidencia muestra el plan interpretado: verifica que coincide con tu pregunta.''')
    st.caption('El chat parte del archivo completo y aplica los filtros que pidas en la pregunta. La selección de gráficos no filtra el chat. Memoria: últimas tres consultas, solo durante esta sesión.')
    authorize = st.checkbox('Autorizo enviar mi pregunta, el catálogo, el alcance y los resultados calculados al modelo Ollama de este equipo para planificar e interpretar.')
    if st.button('Borrar conversación'):
        st.session_state.history = []
        st.rerun()
    for history_index,item in enumerate(st.session_state.history):
        with st.chat_message('user'):
            st.write(item['pregunta'])
        with st.chat_message('assistant'):
            st.markdown('**Resultado calculado por Python**')
            st.write(item['respuesta'])
            if item.get('estado_interpretacion') == 'generada':
                st.markdown('**Interpretación de Ollama · ' + item['modelo'] + '**')
                st.write(item['interpretacion'])
                st.caption('Interpretación generada por IA: contrástala con las cifras y la evidencia antes de usarla.')
            elif item.get('estado_interpretacion') == 'no_disponible':
                st.warning(item['aviso_interpretacion'])
            if item.get('grafico_json'):
                st.plotly_chart(pio.from_json(item['grafico_json']),use_container_width=True,key=f'chat_chart_{history_index}')
            with st.expander('Evidencia calculada y herramienta ejecutada'):
                st.json(item['plan'])
                if item['evidencia'].get('resultados'):
                    st.dataframe(item['evidencia']['resultados'], use_container_width=True)
                st.json(item['evidencia'])
                st.caption('Modelo: ' + item['modelo'])
    question = st.chat_input('Pregunta sobre el dataset', disabled=not authorize)
    if question:
        try:
            with st.spinner('Python calcula los resultados y Ollama prepara su interpretación…'):
                item = AnalyticsAgent(client, st.session_state.predictor).run(question, df, int(min_group), authorize,
                                                  st.session_state.history)
            st.session_state.history = (st.session_state.history + [item])[-3:]
            st.rerun()
        except (ValueError, RuntimeError, PermissionError) as exc:
            st.error(str(exc))
    st.warning('Revisa que las variables, filtros y grupos del plan correspondan a tu pregunta antes de utilizar los resultados.')


with prediction_tab:
    st.subheader('Aprendizaje supervisado de Attrition')
    st.write('El 80% se usa para aprender y el 20% queda reservado para evaluar. Se comparan regresión lineal, árbol de decisión y random forest. División estratificada fija, semilla 42.')
    st.caption('La imputación, el escalado y la codificación se ajustan solo con entrenamiento. Las consultas por segmentos se calculan sobre el 20% de prueba, sin reentrenar por filtro.')
    if st.button('Entrenar y evaluar (80/20)'):
        try:
            with st.spinner('Entrenando y evaluando en la partición reservada…'):
                st.session_state.prediction_report = st.session_state.predictor.evaluate()
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    report = st.session_state.prediction_report
    if report:
        st.subheader('Comparación de los tres modelos')
        st.dataframe(report['comparacion_modelos'],use_container_width=True,hide_index=True)
        st.success(report['recomendacion'])
        st.caption('AP y F1 de validación corresponden a validación cruzada dentro del 80%. Las columnas de prueba usan el 20% reservado. AP, F1 y ROC-AUC: mayor es mejor; Brier: menor es mejor. Regresión lineal: puntuación recortada a [0,1], sin calibración probabilística.')
        st.write(render_prediction(report))
        scores = report['metricas_prueba']
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Exactitud de prueba', f"{scores['accuracy']:.1%}")
        c2.metric('Recall de Attrition Yes', f"{scores['recall_Yes']:.1%}")
        c3.metric('F1 de Attrition Yes', f"{scores['f1_Yes']:.3f}")
        c4.metric('ROC-AUC', f"{scores['roc_auc']:.3f}")
        with st.expander('Partición, variables, baseline y matriz de confusión'):
            st.json(report)
    st.subheader('Simular un perfil ficticio')
    st.caption('Completa únicamente los campos que quieras usar. Los campos vacíos se excluyen del entrenamiento y la predicción; no se rellenan. Cada combinación de variables tiene su propia evaluación 80/20.')
    st.caption('Se ofrecen todos los predictores del dataset IBM, incluidos Gender y MaritalStatus para esta simulación sintética. Se excluyen Attrition (resultado), EmployeeNumber (identificador), EmployeeCount, StandardHours y Over18 (constantes administrativas).')
    predictor = st.session_state.predictor
    profile_key_prefix = 'profile_field_' + hashlib.sha256(content).hexdigest() + '_'
    profile_field_keys = [profile_key_prefix + c for c in SIMULATION_FEATURES if c in df]

    def clear_profile_fields():
        for field_key in profile_field_keys:
            st.session_state[field_key] = None

    with st.form('synthetic_profile_optional_' + hashlib.sha256(content).hexdigest()):
        profile_values={}
        fields=st.columns(3)
        for index,name in enumerate(c for c in SIMULATION_FEATURES if c in df):
            with fields[index % 3]:
                if name in NUMERIC_FEATURES:
                    value=st.number_input(name,min_value=0.0,value=None,step=1.0,placeholder='Sin completar: no usar',key=profile_key_prefix + name)
                else:
                    value=st.selectbox(name,sorted(df[name].dropna().unique()),index=None,placeholder='Sin completar: no usar',key=profile_key_prefix + name)
                if value is not None: profile_values[name]=value
        simulate=st.form_submit_button('Estimar probabilidad del perfil ficticio')
        st.form_submit_button('Limpiar todos los campos', on_click=clear_profile_fields)
    if simulate:
        try:
            with st.spinner('Entrenando y evaluando solo con las variables completadas…'):
                evidence=predictor.profile(profile_values)
            st.subheader('Estimación del perfil con los tres modelos')
            st.dataframe(evidence['comparacion_perfil'],use_container_width=True,hide_index=True)
            st.success(evidence['recomendacion'])
            st.write(render_prediction(evidence))
            with st.expander('Comparación de desempeño con estas variables'):
                st.dataframe(evidence['comparacion_modelos'],use_container_width=True,hide_index=True)
            with st.expander('Variables utilizadas, omitidas y evaluación del modelo'):
                st.json(evidence)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    st.info('Uso académico con el dataset sintético IBM. Las probabilidades no están calibradas externamente, no son certezas ni pronósticos anuales y no deben usarse para decisiones sobre empleados reales.')


with heatmap_tab:
    st.subheader('Mapa de probabilidades por variable y segmento')
    st.write('Columnas: variables que seleccionas. Filas: categorías o rangos. Cada celda contiene la probabilidad media estimada de Attrition = Yes para ese segmento del 20% de prueba.')
    st.caption('Se usa el modelo general 80/20. Cada columna segmenta por separado: el mapa no representa combinaciones simultáneas ni el efecto causal de cambiar una variable. Los rangos bajo, medio y alto son propios de cada variable y se definen con los terciles del entrenamiento; pueden tener tamaños diferentes si hay valores repetidos.')
    available=[c for c in SIMULATION_FEATURES if c in df]
    chosen=st.multiselect('Variables del mapa (columnas)',available,
        default=[c for c in ['Age','MonthlyIncome','YearsAtCompany','Department','OverTime'] if c in available],
        key='heatmap_variables_'+hashlib.sha256(content).hexdigest())
    if st.button('Generar mapa de probabilidades'):
        try:
            with st.spinner('Preparando el modelo general 80/20…'):
                st.session_state.predictor.fit()
        except (ValueError,RuntimeError) as exc:
            st.error(str(exc))
    if not chosen:
        st.info('Selecciona al menos una variable para las columnas.')
    elif st.session_state.predictor.pipeline is None:
        st.info('Pulsa «Generar mapa de probabilidades». También puedes entrenar desde la pestaña Predicción 80/20.')
    else:
        try:
            result=segmented_probabilities(st.session_state.predictor,chosen,int(min_group))
            heat_table=result['tabla']
            st.caption('Modelo usado en el mapa y Top 5: '+result['configuracion']['modelo']+'. '+result['configuracion'].get('interpretacion_salida',''))
            st.caption(f"Celdas omitidas por tener menos de {int(min_group)} registros de prueba: {result['celdas_omitidas']}. Una celda vacía no equivale a 0%.")
            if heat_table.empty:
                st.info('No hay segmentos que cumplan el mínimo de registros. Ajusta las variables o el mínimo de la barra lateral.')
            else:
                options=list(dict.fromkeys(heat_table['Segmento']))
                segment_key='heatmap_segments_'+hashlib.sha256((fingerprint+repr(chosen)+repr(options)).encode()).hexdigest()
                chosen_segments=st.multiselect('Segmentos del mapa (filas)',options,default=options,key=segment_key)
                if chosen_segments:
                    st.plotly_chart(probability_heatmap(heat_table,chosen,chosen_segments),use_container_width=True)
                    st.caption('Pasa el cursor sobre una celda para ver el rango o categoría exactos y el número de registros. Los espacios en blanco corresponden a segmentos no aplicables o insuficientes.')
                    with st.expander('Detalle de segmentos, condiciones y probabilidades'):
                        st.dataframe(heat_table[heat_table['Segmento'].isin(chosen_segments)].sort_values('Probabilidad media Attrition Yes (%)',ascending=False),use_container_width=True)
                    st.subheader('Top 5 de combinaciones con mayor probabilidad de Attrition')
                    st.caption('Cruza de dos en dos las variables y los segmentos seleccionados arriba. Cada fila reúne registros que cumplen ambas condiciones simultáneamente. El orden usa la probabilidad media predicha, no la tasa observada de salidas.')
                    if len(chosen)<2:
                        st.info('Selecciona al menos dos variables en las columnas del mapa.')
                    else:
                        ranking_key='top_combinations_'+hashlib.sha256((fingerprint+repr(chosen)+repr(chosen_segments)).encode()).hexdigest()
                        show_ranking=st.checkbox('Mostrar las cinco combinaciones principales',key=ranking_key)
                        if show_ranking:
                            ranking=top_combinations(st.session_state.predictor,chosen,int(min_group),chosen_segments)
                            st.caption(f"Pares de variables examinados: {ranking['pares_evaluados']}. Combinaciones con suficientes registros: {ranking['combinaciones_validas']}. Omitidas por tamaño: {ranking['combinaciones_omitidas']}. Probabilidad media de todo el conjunto de prueba: {ranking['probabilidad_media_prueba_pct']:.1f}%.")
                            if ranking['tabla'].empty:
                                st.info('No hay combinaciones que cumplan los segmentos y el mínimo de registros seleccionados.')
                            else:
                                st.dataframe(ranking['tabla'][['Condición 1','Condición 2','Probabilidad media Attrition Yes (%)','Attrition Yes observado (%)']],use_container_width=True,hide_index=True,
                                    column_config={'Probabilidad media Attrition Yes (%)':st.column_config.NumberColumn(format='%.2f %%'),
                                        'Attrition Yes observado (%)':st.column_config.NumberColumn(format='%.2f %%')})
                                for _,row in ranking['tabla'].iterrows():
                                    st.write(f"{row['Puesto']}. **{row['Condición 1']} Y {row['Condición 2']}**: probabilidad media estimada **{row['Probabilidad media Attrition Yes (%)']:.1f}%**, con {row['Registros de prueba']} registros de prueba.")
                            st.caption('Ranking exploratorio de segmentos sintéticos, no recomendaciones sobre empleados. Los grupos pueden solaparse; sus conteos no se suman. Examinar muchas combinaciones puede destacar diferencias inestables: confirma los patrones con datos independientes antes de generalizar. No se reentrena ni se elige el modelo usando este ranking.')
                    with st.expander('Evaluación del modelo usado en el mapa'):
                        st.write(render_prediction(st.session_state.predictor.evaluate()))
                else:
                    st.info('Selecciona al menos un segmento para las filas.')
        except (ValueError,RuntimeError) as exc:
            st.error(str(exc))
    st.caption('Probabilidades del modelo sobre datos sintéticos, no tasas futuras ni decisiones sobre empleados reales. Gender y MaritalStatus están disponibles para segmentar; no son predictores del modelo general.')
