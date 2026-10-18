"""Interpretación acotada de preguntas predictivas para el dataset sintético."""
import json
import re
from .prediction import AttritionPredictor, SIMULATION_FEATURES, NUMERIC_FEATURES
from .analytics import OPS


def is_predictive(question):
    return bool(re.search(r'compara.*model|mejor.*model|predi|pron[oó]st|probabilidad|entrena|80\s*[/–-]\s*20|roc.?auc|\bauc\b|\bf1\b|\brecall\b|matriz de confusi|eval[uú]a.*modelo|desempe[nñ]o.*modelo|simula.*perfil',question,re.I))


def schema(columns, mode_hint=None):
    def obj(props,required): return {'type':'object','properties':props,'required':required,'additionalProperties':False}
    var={'type':'string','enum':columns}
    scalar={'anyOf':[{'type':'string'},{'type':'number'},{'type':'null'}]}
    condition=obj({'variable':var,'op':{'type':'string','enum':OPS},
        'valor':{'anyOf':[scalar,{'type':'array','items':scalar,'maxItems':20}]}},['variable','op'])
    fixed={'accion':{'type':'string','enum':['prediccion']}}
    result = {'oneOf':[
        obj({**fixed,'modo':{'type':'string','enum':['evaluar']}},['accion','modo']),
        obj({**fixed,'modo':{'type':'string','enum':['cohorte']},
            'filtros':{'type':'array','items':condition,'maxItems':8},
            'grupos':{'type':'array','items':var,'maxItems':2}},['accion','modo','filtros','grupos']),
        obj({**fixed,'modo':{'type':'string','enum':['perfil']},
            'perfil':obj({c:{'type':'number' if c in NUMERIC_FEATURES else 'string'} for c in columns},[]),
            'imputar':{'type':'boolean','enum':[False]}},['accion','modo','perfil','imputar']),
        obj({'accion':{'type':'string','enum':['aclarar']},'aclaracion':{'type':'string'}},['accion','aclaracion'])]}
    if mode_hint:
        result['oneOf']=[v for v in result['oneOf'] if 'modo' not in v['properties'] or v['properties']['modo']['enum']==[mode_hint]]
    return result


INSTRUCTIONS='''Planifica una consulta predictiva académica del dataset sintético IBM; solo JSON.
Python compara regresión lineal, árbol de decisión y random forest; selecciona por validación cruzada dentro del entrenamiento, no por el LLM. Entrena con80% y evalúa con20%,
estratificado y semilla42; el split y umbral0.5 no pueden cambiarse desde el chat.
modo evaluar: entrenar/evaluar, ver métricas, AUC, F1, matriz de confusión o comparar con baseline.
modo cohorte: resumir las probabilidades y clases predichas SOLO del20% de prueba, opcionalmente
filtros [{variable,op,valor}] unidos conAND y grupos [hasta2columnas]. No filtres por Attrition.
modo perfil: estimación para un perfil FICTICIO explícito; perfil es diccionario de valores
proporcionados, nunca inventes campos ni valores. imputar siempre false: las variables omitidas
no participan en el modelo. Basta un predictor; no exijas campos básicos.
Edad=Age, ingreso mensual=MonthlyIncome, antigüedad=YearsAtCompany, horas extra=OverTime.
No mezcles evaluar con filtros: si se piden predicciones para un segmento usa cohorte.
No inventes filtros; usa listas vacías cuando no se pidan. El resultado no es probabilidad anual,
causalidad ni una recomendación sobre empleados reales. Para nombres, identificadores, decisiones
reales o solicitudes ambiguas usa accion aclarar y explica la limitación.
Ejemplos:
{"accion":"prediccion","modo":"evaluar"}
{"accion":"prediccion","modo":"cohorte","filtros":[{"variable":"OverTime","op":"eq","valor":"Yes"}],"grupos":[]}
{"accion":"prediccion","modo":"perfil","perfil":{"Age":30,"MonthlyIncome":3500,"YearsAtCompany":3,"OverTime":"Yes"},"imputar":false}
'''


def run_prediction(client, question, df, predictor=None, min_group=5):
    predictor=predictor or AttritionPredictor(df)
    columns=[c for c in SIMULATION_FEATURES if c in df]
    # Exact assignment syntax makes a fictitious profile reproducible without model-invented fields.
    standard=None
    if re.search(r'perfil|fictici|simula',question,re.I):
        assignments=re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\s*=',question)
        unknown=[c for c in assignments if c.lower() not in {x.lower() for x in columns}]
        if unknown: raise ValueError('Variables del perfil no admitidas: '+', '.join(unknown))
        values={}
        for c in columns:
            pattern=r'\b'+re.escape(c)+r'\s*=\s*'
            if c in NUMERIC_FEATURES:
                match=re.search(pattern+r'(-?\d+(?:\.\d+)?)\b',question,re.I)
                if match: values[c]=float(match.group(1))
            else:
                for value in df[c].dropna().unique():
                    if re.search(pattern+re.escape(str(value))+r'(?=\s*[,;.]|\s*$)',question,re.I): values[c]=str(value)
        # Only use this path when every assignment is recognized.
        if values and len(values)==question.count('='):
            standard={'accion':'prediccion','modo':'perfil','perfil':values,
                      'imputar':bool(re.search(r'\bimputa\b|imputar|completa.*falt|completa.*resto',question,re.I))}
    elif re.fullmatch(r'(?:compara (?:los )?(?:tres|3) modelos|cu[aá]l es el mejor modelo)[.!?]*',question.strip(' ¿?'),re.I):
        standard={'accion':'prediccion','modo':'evaluar'}
    elif re.fullmatch(r'(?:entrena y eval[uú]a|eval[uú]a|entrena) (?:el )?modelo(?: de attrition)?(?: con (?:divisi[oó]n )?80/20)?[.!?]*',question.strip(),re.I):
        standard={'accion':'prediccion','modo':'evaluar'}
    if standard is None:
        names='|'.join(re.escape(c) for c in columns)
        group_match=re.fullmatch(r'predice attrition por ('+names+r')(?: y ('+names+r'))?(?: en el conjunto de prueba)?[.!?]*',question.strip(),re.I)
        if group_match:
            canonical={c.lower():c for c in columns}
            standard={'accion':'prediccion','modo':'cohorte','filtros':[],
                      'grupos':[canonical[c.lower()] for c in group_match.groups() if c]}
        else:
            filter_match=re.fullmatch(r'predice attrition para ('+names+r')\s*(>=|<=|>|<|=)\s*(.+?)(?: en el conjunto de prueba)?[.!?]*',question.strip(),re.I)
            if filter_match:
                c,op,value=filter_match.groups();c=next(x for x in columns if x.lower()==c.lower())
                if c in NUMERIC_FEATURES:
                    try:value=float(value)
                    except ValueError:value=None
                else:
                    value=next((str(v) for v in df[c].dropna().unique() if str(v).lower()==value.lower()),None)
                if value is not None:
                    standard={'accion':'prediccion','modo':'cohorte','grupos':[],
                              'filtros':[{'variable':c,'op':{'=':'eq','>':'gt','<':'lt','>=':'ge','<=':'le'}[op],'valor':value}]}
    mode_hint='perfil' if re.search(r'perfil|fictici|simula',question,re.I) else 'evaluar' if re.search(r'compara.*model|mejor.*model|auc|recall|precision|precisi[oó]n|f1|matriz|entrena|eval[uú]a|desempe[nñ]o',question,re.I) else 'cohorte'
    metadata=[{'variable':c,'tipo':'numero' if c in NUMERIC_FEATURES else 'categoria',
        **({'valores':[str(x) for x in df[c].dropna().unique()][:30]} if c not in NUMERIC_FEATURES else {})} for c in columns]
    messages=[{'role':'system','content':INSTRUCTIONS},
        {'role':'user','content':json.dumps({'pregunta':question,'catalogo':metadata},ensure_ascii=False)}]
    for attempt in range(2):
        try:
            text=json.dumps(standard) if standard else client.chat(messages,schema=schema(columns,mode_hint))
            plan=json.loads(text)
            if not isinstance(plan,dict):raise ValueError('Plan inválido.')
            if plan.get('accion')=='aclarar':
                if set(plan)!= {'accion','aclaracion'} or not isinstance(plan['aclaracion'],str):raise ValueError('Aclaración inválida.')
                return plan,{'accion':'prediccion','modo':'aclarar','aclaracion':plan['aclaracion']}
            if plan.get('modo')!=mode_hint and standard is None:
                raise ValueError('Esta pregunta requiere modo '+mode_hint+'. No omitas su alcance.')
            if plan.get('modo')=='cohorte':
                from .agent import check_explicit_constraints
                check_explicit_constraints(question,df,plan)
            if plan.get('modo')=='perfil':
                if plan.get('imputar') and not re.search(r'imputa|completa.*falt|completa.*resto',question,re.I):
                    raise ValueError('No se autorizó imputar el perfil. Usa imputar=false.')
                for c,v in plan.get('perfil',{}).items():
                    if c in NUMERIC_FEATURES and not re.search(r'(?<!\d)'+re.escape(str(float(v)).removesuffix('.0'))+r'(?!\d)',question):
                        raise ValueError('Valor del perfil no indicado en la pregunta: '+c)
                    if c not in NUMERIC_FEATURES:
                        pattern=r'\b'+re.escape(str(v))+r'\b'
                        if c=='OverTime' and v=='Yes': pattern+=r'|con horas extra|hace horas extra'
                        if c=='OverTime' and v=='No': pattern+=r'|sin horas extra'
                        if v=='Sales': pattern+=r'|ventas'
                        if not re.search(pattern,question,re.I):
                            raise ValueError('Categoría del perfil no indicada en la pregunta: '+c)

            # Model errors are corrected; validation/training errors are reported, not rewritten by the LLM.
            break
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            if attempt or standard:raise ValueError('No se pudo interpretar la predicción: '+str(exc)) from exc
            messages.append({'role':'user','content':'Corrige el plan sin inventar valores. Error: '+str(exc)})
    result=predictor.execute(plan,min_group)
    return plan,result


def render_prediction(evidence):
    if evidence.get('aclaracion'):return evidence['aclaracion']
    meta=evidence['configuracion']
    lines=[f"**Modelo predictivo de Attrition — demostración sintética.** Entrenamiento: {meta['entrenamiento']} registros; prueba: {meta['prueba']}. División estratificada 80/20, semilla42, umbral0.5."]
    lines.append('Modelo seleccionado: **'+meta['modelo']+'**. '+meta.get('interpretacion_salida',''))
    if evidence.get('recomendacion'): lines.append(evidence['recomendacion'])
    if evidence['modo']=='evaluar':
        m=evidence['metricas_prueba'];b=evidence['baseline_prior']
        lines.append(f"Resultados sobre el **20% reservado**: exactitud {m['accuracy']:.1%}; precisión para Yes {m['precision_Yes']:.1%}; recall para Yes {m['recall_Yes']:.1%}; F1 {m['f1_Yes']:.3f}; ROC-AUC {m['roc_auc']:.3f}; average precision {m['average_precision']:.3f}; Brier {m['brier']:.3f}.")
        lines.append(f"Baseline de probabilidad según prevalencia de entrenamiento: exactitud {b['accuracy']:.1%}; F1 {b['f1_Yes']:.3f}; ROC-AUC {b['roc_auc']:.3f}. No basta evaluar solo exactitud si las clases están desbalanceadas.")
        cm=m['matriz_confusion'];lines.append(f"Matriz de confusión: TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']} (positivo=Attrition Yes).")
    elif evidence['modo']=='cohorte':
        lines.append(evidence['alcance'])
        lines.append('Filtros: '+json.dumps(evidence['filtros'],ensure_ascii=False)+'; grupos: '+json.dumps(evidence['grupos'],ensure_ascii=False)+'.')
        for row in evidence['resultados'][:12]:
            group=', '.join(f'{g}={row[g]}' for g in evidence['grupos']) or 'Conjunto consultado'
            lines.append(f"- {group}: n={row['n_prueba']}; probabilidad media estimada **{row['probabilidad_media_Yes']:.1%}**; Yes predichos al umbral0.5: **{row['Yes_estimados_umbral_05']}** ({row['porcentaje_Yes_estimado']:.1f}%); Yes observados: {row['Yes_observados']} ({row['porcentaje_Yes_observado']:.1f}%).")
        if not evidence['resultados']:lines.append('Sin grupos con suficientes registros de prueba. No significa probabilidad cero.')
        if len(evidence['resultados'])>12 or evidence['truncado']:lines.append('Resumen parcial; revisa la evidencia.')
    else:
        lines.append(f"Perfil ficticio: probabilidad estimada de Attrition=Yes **{evidence['probabilidad_estimada_Yes']:.1%}**; clase al umbral0.5: {evidence['clase_al_umbral_05']}.")
        lines.append('Valores proporcionados: '+json.dumps(evidence['perfil_sintetico_proporcionado'],ensure_ascii=False)+'.')
        lines.append('Variables usadas para entrenar y simular: '+', '.join(evidence['variables_utilizadas'])+'. Los campos omitidos no intervienen ni se rellenan.')
        m=evidence['metricas_prueba']
        lines.append(f"Evaluación del modelo con estas variables sobre el 20% reservado: ROC-AUC {m['roc_auc']:.3f}; recall {m['recall_Yes']:.1%}; F1 {m['f1_Yes']:.3f}.")
        lines.append(evidence['advertencia'])
        if evidence['fuera_de_rango_entrenamiento']:lines.append('Fuera del rango observado en entrenamiento: '+', '.join(evidence['fuera_de_rango_entrenamiento'])+'. La estimación implica extrapolación.')
    if evidence.get('comparacion_modelos'):
        lines.append('Comparación de modelos (AP validación / F1 prueba / ROC-AUC prueba):')
        for row in evidence['comparacion_modelos']:
            lines.append(f"- {row['Modelo']}: {row['AP validación (80%)']:.3f} / {row['F1 Yes prueba']:.3f} / {row['ROC-AUC prueba']:.3f}.")
    if evidence.get('comparacion_perfil'):
        for row in evidence['comparacion_perfil']:
            lines.append(f"- Perfil — {row['Modelo']}: estimación Yes {row['Estimación Yes (%)']:.1f}%; salida original {row['Salida original']:.4f}.")
    lines.append('Probabilidades sin calibración externa, no certezas ni tasas anuales. Resultado académico sobre datos sintéticos; no usar para decisiones sobre empleados reales.')
    return '\n\n'.join(lines)
