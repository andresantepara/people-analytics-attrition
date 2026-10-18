"""Oraciones naturales construidas con evidencia calculada, sin alterar cifras."""
import json

LABELS={'count':'conteo válido','missing':'cantidad de valores ausentes','unique':'cantidad de valores distintos',
    'mean':'promedio','median':'mediana','mode':'moda','sum':'suma','min':'mínimo','max':'máximo',
    'std':'desviación estándar muestral','var':'varianza muestral','quantile':'cuantil',
    'iqr':'rango intercuartílico','cv':'coeficiente de variación (%)','skew':'asimetría','kurt':'exceso de curtosis'}
NAMES={'Age':'edad','MonthlyIncome':'ingreso mensual','YearsAtCompany':'antigüedad en la empresa',
    'Department':'departamento','JobRole':'cargo','OverTime':'horas extra','Gender':'género',
    'Attrition':'salida registrada','TotalWorkingYears':'experiencia laboral total','DistanceFromHome':'distancia al trabajo',
    'JobSatisfaction':'satisfacción laboral','WorkLifeBalance':'equilibrio entre trabajo y vida personal',
    'Education':'nivel educativo','MaritalStatus':'estado civil','DailyRate':'tarifa diaria','HourlyRate':'tarifa por hora',
    'MonthlyRate':'tarifa mensual','NumCompaniesWorked':'número de empresas anteriores','PercentSalaryHike':'incremento salarial porcentual',
    'YearsInCurrentRole':'antigüedad en el cargo','YearsSinceLastPromotion':'años desde el último ascenso',
    'YearsWithCurrManager':'años con el responsable actual','EnvironmentSatisfaction':'satisfacción con el entorno laboral',
    'JobInvolvement':'involucramiento laboral','JobLevel':'nivel del cargo','PerformanceRating':'evaluación del desempeño',
    'RelationshipSatisfaction':'satisfacción con las relaciones','StockOptionLevel':'nivel de opciones sobre acciones',
    'TrainingTimesLastYear':'capacitaciones del último año','BusinessTravel':'viajes de trabajo','EducationField':'área de estudios'}
YEARS={'Age','YearsAtCompany','TotalWorkingYears','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'}


def fmt(value):
    if value is None:return 'sin resultado válido'
    if isinstance(value,float):return f'{value:.4f}'.rstrip('0').rstrip('.')
    if isinstance(value,(dict,list)):return json.dumps(value,ensure_ascii=False)
    return str(value)


def name(variable):return NAMES.get(variable,variable)


def condition_text(f):
    variable=f['variable'];op=f['op'];value=f.get('valor')
    if variable=='Attrition' and op=='eq' and value in ('Yes','No'):
        return 'salida registrada (Attrition = Yes)' if value=='Yes' else 'permanencia registrada (Attrition = No)'
    operators={'eq':'igual a','ne':'distinto de','gt':'mayor que','ge':'mayor o igual que','lt':'menor que',
        'le':'menor o igual que','in':'entre los valores','not_in':'fuera de los valores','is_null':'sin dato','not_null':'con dato'}
    return name(variable)+' '+operators.get(op,op)+(' '+fmt(value) if op not in ('is_null','not_null') else '')


def scope(filters):
    if filters==[{'variable':'Attrition','op':'eq','valor':'Yes'}]:return 'entre quienes salieron'
    if filters==[{'variable':'Attrition','op':'eq','valor':'No'}]:return 'entre quienes permanecen'
    return 'en los registros con '+' y '.join(condition_text(f) for f in filters) if filters else 'en el dataset'


def metric_sentence(item,value,filters):
    stat=item['estadistico'];variable=item['variable'];context=scope(filters)
    if value is None:return f'No hay suficientes datos válidos para calcular {LABELS[stat]} de {name(variable)} {context}.'
    unit=' años' if variable in YEARS and stat in ('mean','median','min','max','std','iqr','quantile') else ''
    result='**'+fmt(value)+unit+'**'
    if stat=='mean':return f'{("La edad promedio" if variable=="Age" else "El promedio de "+name(variable))} {context} es de {result}.'
    if stat=='count':return f'Hay {result} '+('registros' if variable=='*' else 'valores válidos de '+name(variable))+f' {context}.'
    if stat=='missing':return f'Hay {result} valores ausentes de {name(variable)} {context}.'
    if stat=='unique':return f'{name(variable).capitalize()} presenta {result} valores distintos {context}.'
    label=f"percentil {fmt(100*item.get('q',.5))}" if stat=='quantile' else LABELS[stat]
    article='La' if stat in ('median','mode','sum','std','var','skew') else 'El'
    return f'{article} {label} de {name(variable)} {context} es {result}.'


def render_answer(evidence):
    if 'aclaracion' in evidence:return evidence['aclaracion']
    rows=evidence['resultados'];plan=evidence['plan_ejecutado'];action=plan['accion'];lines=[]
    if not rows:return 'No encontré resultados que cumplan los filtros y el mínimo de registros seleccionado. Esto no significa que el valor sea cero.'
    for row in rows[:12]:
        groups=[f'{name(k[6:])}: {fmt(v)}' for k,v in row.items() if k.startswith('grupo_')]
        prefix='Para el grupo '+', '.join(groups)+', ' if groups else ''
        parts=[]
        if action=='estadisticas':
            for item in plan['metricas']:
                filters=evidence.get('filtros',[])+item.get('filtros',[])
                sentence=metric_sentence(item,row[item['id']],filters)
                detail=row['detalle'][item['id']]
                if detail['nulos']:sentence+=f" Se excluyeron {detail['nulos']} valores ausentes del cálculo." if item['estadistico'] not in ('missing','count','unique','mode') else f" Hay {detail['nulos']} valores ausentes."
                parts.append(sentence)
            metrics={m['id']:m for m in plan['metricas']}
            for calc in plan.get('calculos',[]):
                val=row[calc['id']]
                if val is None:parts.append('No se pudo calcular '+calc['id']+' con los datos disponibles.');continue
                def operand(ref):
                    if isinstance(ref,str) and ref in metrics:
                        m=metrics[ref];return LABELS[m['estadistico']]+' de '+name(m['variable'])+' '+scope(evidence.get('filtros',[])+m.get('filtros',[]))
                    return fmt(ref)
                if calc['op']=='subtract':parts.append(f"Al restar el {operand(calc['b'])} del {operand(calc['a'])}, la diferencia es **{fmt(val)}**.")
                else:parts.append(f"El resultado de {calc['id']} es **{fmt(val)}**; la operación y sus valores están en la evidencia.")
        elif action=='proporcion':
            event=' y '.join(condition_text(f) for f in plan['evento'])
            if row['proporcion_pct'] is None:
                parts.append('No hay registros con datos válidos suficientes para calcular el porcentaje de '+event+'.')
            else:
                parts.append(f"El **{fmt(row['proporcion_pct'])}%** cumple la condición de {event}: son {row['numerador']} de {row['denominador']} registros con datos válidos.")
            if row['nulos_excluidos']:parts.append(f"Se excluyeron {row['nulos_excluidos']} registros sin los datos necesarios para el denominador.")
        elif action=='correlacion':
            value=row['coeficiente']
            if value is None:parts.append(f"No se pudo calcular la correlación entre {name(row['variable_x'])} y {name(row['variable_y'])}. {row['motivo_sin_resultado'] or ''}")
            else:
                direction='positiva' if value>0 else 'negativa' if value<0 else 'nula'
                parts.append(f"La correlación de {row['metodo'].capitalize()} entre {name(row['variable_x'])} y {name(row['variable_y'])} es **{fmt(value)}**, una asociación {direction} calculada con {row['pares_validos']} pares válidos. Esto no demuestra causalidad.")
            if row['codificacion']:parts.append('La codificación utilizada se detalla en la evidencia.')
        elif action=='cruce':
            parts.append(f"Al cruzar {name(row['variable_fila'])} con {name(row['variable_columna'])}, la V de Cramér es **{fmt(row['cramers_v'])}**, calculada con {row['validos']} registros válidos.")
            for cell in row['celdas'][:12]:parts.append(f"Dentro de {row['variable_fila']} = {cell['fila']}, {cell['conteo']} registros tienen {row['variable_columna']} = {cell['columna']}, lo que representa el **{fmt(cell['pct_fila'])}%** de esa fila.")
            if len(row['celdas'])>12:parts.append('El resto de las combinaciones está en la evidencia.')
        elif action=='resumen':
            parts.append(f"El dataset contiene **{row['filas']} registros** y {row['variables']} variables.")
            if row['proporcion_pct'] is None:parts.append('No hay valores válidos de Attrition para calcular el porcentaje de salidas.')
            else:parts.append(f"Hay {row['salidas']} salidas registradas, equivalentes al **{fmt(row['proporcion_pct'])}%** de los {row['attrition_validos']} registros con Attrition válido.")
        elif action=='frecuencias':
            category=' y '.join(f'{name(v)} igual a {fmt(row[v])}' for v in plan['variables'])
            parts.append(f"Hay **{row['conteo']} registros** con {category}, que representan el **{fmt(row['porcentaje'])}%** de {row['denominador']} registros.")
        else:parts.append(f"La variable {name(row['variable'])} tiene {row['nulos']} valores ausentes y {row['unicos']} valores distintos.")
        paragraph=' '.join(parts)
        lines.append(prefix+(paragraph[0].lower()+paragraph[1:] if prefix and paragraph else paragraph))
    if evidence.get('filtros') and action!='estadisticas':lines.append('El análisis corresponde a los registros con '+' y '.join(condition_text(f) for f in evidence['filtros'])+'.')
    if len(rows)>12 or evidence.get('truncado'):lines.append('Esta es una vista parcial; puedes consultar los demás resultados en la evidencia.')
    if evidence.get('grupos_o_celdas_omitidos'):lines.append(f"Se omitieron {evidence['grupos_o_celdas_omitidos']} grupos o celdas por el mínimo de registros seleccionado.")
    return '\n\n'.join(lines)


def render_prediction_sentence(evidence):
    if evidence.get('aclaracion'):return evidence['aclaracion']
    meta=evidence['configuracion'];mode=evidence['modo']
    if mode=='perfil':
        answer=f"Para el perfil ficticio que ingresaste, la estimación de Attrition = Yes es **{100*evidence['probabilidad_estimada_Yes']:.1f}%**, usando {meta['modelo']}. El modelo utiliza únicamente {', '.join(name(v) for v in evidence['variables_utilizadas'])}; los demás campos no participan."
        if evidence['fuera_de_rango_entrenamiento']:answer+=' Hay valores fuera del rango de entrenamiento en '+', '.join(name(v) for v in evidence['fuera_de_rango_entrenamiento'])+'.'
    elif mode=='cohorte':
        parts=[]
        for row in evidence['resultados'][:12]:
            group=', '.join(f'{name(g)}: {row[g]}' for g in evidence['grupos']) or 'el conjunto consultado'
            parts.append(f"Para {group}, la probabilidad media estimada de Attrition = Yes es **{100*row['probabilidad_media_Yes']:.1f}%**, calculada con {row['n_prueba']} registros de prueba. El porcentaje observado de salidas es {row['porcentaje_Yes_observado']:.1f}%.")
        answer=' '.join(parts) or 'No hay grupos que cumplan los filtros y el mínimo de registros solicitado; esto no significa una probabilidad de cero.'
        if evidence.get('filtros'):answer+=' Se aplicaron las condiciones '+', '.join(condition_text(f) for f in evidence['filtros'])+'.'
        if len(evidence['resultados'])>12 or evidence.get('truncado'):answer+=' La respuesta es parcial; consulta la evidencia completa.'
    else:
        m=evidence['metricas_prueba']
        answer=f"El modelo seleccionado es **{meta['modelo']}**. En el 20% reservado para prueba alcanza una exactitud de **{m['accuracy']:.1%}**, una precisión de **{m['precision_Yes']:.1%}**, un recall de **{m['recall_Yes']:.1%}** para las salidas, un F1 de **{m['f1_Yes']:.3f}** y un ROC-AUC de **{m['roc_auc']:.3f}**. La AP de prueba es {m['average_precision']:.3f} y el error Brier es {m['brier']:.3f}."
    if evidence.get('recomendacion'):answer+='\n\n'+evidence['recomendacion']
    if evidence.get('comparacion_modelos'):
        answer+='\n\n'+' '.join(f"{r['Modelo']} obtuvo una AP de validación de {r['AP validación (80%)']:.3f} y una exactitud de prueba de {r['Exactitud prueba']:.1%}." for r in evidence['comparacion_modelos'])
    if evidence.get('comparacion_perfil'):
        answer+='\n\n'+' '.join(f"Para este perfil, {r['Modelo']} estima {r['Estimación Yes (%)']:.1f}% de Attrition = Yes." for r in evidence['comparacion_perfil'])
    if meta.get('modelo_id')=='lineal':answer+=' La regresión lineal usa una puntuación recortada a [0,1], que no es una probabilidad calibrada.'
    return answer+'\n\nSon estimaciones sobre datos sintéticos, no certezas ni predicciones de salida en un plazo determinado.'
