"""Motor analítico: planes declarativos, sin ejecutar código del modelo."""
import itertools
import math
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .eda import summary, profile

STATS = ['count', 'missing', 'unique', 'mean', 'median', 'mode', 'sum', 'min', 'max',
         'std', 'var', 'quantile', 'iqr', 'cv', 'skew', 'kurt']
OPS = ['eq', 'ne', 'gt', 'ge', 'lt', 'le', 'in', 'not_in', 'is_null', 'not_null']
MATH = ['add', 'subtract', 'multiply', 'divide', 'percentage', 'pct_change']
BLOCKED = {'EmployeeNumber'}
ORDINAL = {'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
           'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'StockOptionLevel'}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def column(df, name):
    if not isinstance(name, str) or name not in df.columns or name in BLOCKED:
        raise ValueError(f'Variable no disponible para análisis agregado: {name}.')
    return df[name]


def numeric(df, name):
    s = column(df, name)
    if not is_numeric_dtype(s):
        raise ValueError(f'{name} no es numérica. Usa frecuencias, proporciones o cruce.')
    return s.replace([np.inf, -np.inf], np.nan)


def objects(value, limit, label):
    if not isinstance(value, list) or len(value) > limit or not all(isinstance(x, dict) for x in value):
        raise ValueError(f'{label}: lista de hasta {limit} objetos requerida.')
    return value


def keys(obj, allowed, required=()):
    if not isinstance(obj, dict) or set(obj) - set(allowed) or not set(required) <= set(obj):
        raise ValueError('Plan con campos ausentes o no permitidos.')


def filtered(df, filters, reference=None):
    mask = pd.Series(True, index=df.index)
    equals = {}
    for f in objects(filters, 8, 'Filtros'):
        keys(f, ['variable', 'op', 'valor'], ['variable', 'op'])
        s = column(df, f['variable']); op = f['op']; value = f.get('valor')
        if op == 'eq':
            if f['variable'] in equals and equals[f['variable']] != value:
                raise ValueError('Filtros contradictorios: dos igualdades distintas para la misma columna.')
            equals[f['variable']] = value
        if op not in OPS:
            raise ValueError('Operador de filtro no autorizado.')
        if op in ('is_null', 'not_null'):
            part = s.isna() if op == 'is_null' else s.notna()
        else:
            values = value if isinstance(value, list) else [value]
            if not values or len(values) > 30 or any(v is None or isinstance(v, (dict, list, bool)) for v in values):
                raise ValueError('Valor de filtro inválido; usa is_null para valores ausentes.')
            if is_numeric_dtype(s):
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                    raise ValueError(f'El filtro de {f["variable"]} requiere números.')
            else:
                if any(not isinstance(v, str) for v in values):
                    raise ValueError('El filtro categórico requiere texto.')
                if op not in ('eq', 'ne', 'in', 'not_in'):
                    raise ValueError('No se permiten comparaciones de orden entre categorías de texto.')
                source = column(reference, f['variable']) if reference is not None else s
                unknown = set(values) - set(source.dropna().unique())
                if unknown:
                    raise ValueError(f'Categoría no encontrada en {f["variable"]}: {sorted(unknown)}.')
            if op in ('in', 'not_in'):
                part = s.isin(values)
                if op == 'not_in': part = ~part
            else:
                if isinstance(value, list): raise ValueError('Usa in para una lista de valores.')
                part = {'eq': s.eq, 'ne': s.ne, 'gt': s.gt, 'ge': s.ge, 'lt': s.lt, 'le': s.le}[op](value)
            part = part & s.notna()
        mask &= part.fillna(False)
    return df.loc[mask]


def metric(df, item, reference=None):
    keys(item, ['id', 'variable', 'estadistico', 'q', 'filtros'], ['id', 'variable', 'estadistico'])
    work = filtered(df, item.get('filtros', []), reference); name = item['variable']; stat = item['estadistico']
    if stat not in STATS: raise ValueError('Estadístico no permitido.')
    if name == '*':
        if stat != 'count': raise ValueError('La variable * solo admite count de filas.')
        return len(work), {'filas': len(work), 'validos': len(work), 'nulos': 0}
    s = column(work, name)
    if stat not in ('count', 'missing', 'unique', 'mode'): s = numeric(work, name)
    valid = s.dropna(); n = len(valid)
    info = {'filas': len(work), 'validos': n, 'nulos': len(work)-n}
    if stat == 'count': result = n
    elif stat == 'missing': result = len(work)-n
    elif stat == 'unique': result = valid.nunique()
    elif stat == 'mode':
        modes = valid.mode(); result = modes.iloc[:20].tolist()
        info['modas_totales'] = len(modes)
    elif stat == 'quantile':
        q = item.get('q', .5)
        if type(q) not in (int, float) or not 0 <= q <= 1: raise ValueError('q debe estar entre 0 y 1.')
        result = valid.quantile(q) if n else None
    elif not n: result = None
    elif stat == 'sum': result = valid.sum(min_count=1)
    elif stat == 'iqr': result = valid.quantile(.75)-valid.quantile(.25)
    elif stat == 'cv': result = 100*valid.std(ddof=1)/abs(valid.mean()) if n > 1 and valid.mean() != 0 else None
    else: result = getattr(valid, stat)()
    return clean(result), info


def calculate(df, plan, min_group=5):
    """Execute a validated plan and return JSON-safe aggregate evidence."""
    keys(plan, ['accion', 'filtros', 'grupos', 'metricas', 'calculos', 'variables', 'evento',
                'metodo', 'ordenar', 'descendente', 'limite', 'aclaracion'], ['accion'])
    action = plan['accion']
    if action not in ['resumen', 'nulos', 'estadisticas', 'frecuencias', 'proporcion', 'correlacion', 'cruce', 'aclarar']:
        raise ValueError('Acción no permitida.')
    if type(min_group) is not int or min_group < 1: raise ValueError('Umbral inválido.')
    relevant = {
        'resumen': set(), 'nulos': set(), 'aclarar': {'aclaracion'},
        'estadisticas': {'metricas', 'calculos'}, 'frecuencias': {'variables'},
        'proporcion': {'evento'}, 'correlacion': {'variables', 'metodo'}, 'cruce': {'variables'}}
    common = {'accion', 'filtros', 'grupos', 'ordenar', 'descendente', 'limite'}
    for k, v in plan.items():
        if k not in common | relevant[action] and v not in (None, [], ''):
            raise ValueError(f'El campo {k} no corresponde a {action}.')
    if action == 'aclarar':
        message = plan.get('aclaracion')
        if not isinstance(message, str) or not message.strip(): raise ValueError('Falta la aclaración.')
        return {'aclaracion': message[:800]}
    base = filtered(df, plan.get('filtros', []))
    groups = plan.get('grupos', [])
    if not isinstance(groups, list) or len(groups) > 2 or len(set(groups)) != len(groups):
        raise ValueError('Se permiten hasta dos variables de agrupación distintas.')
    for name in groups:
        column(base, name)
        if base[name].nunique() > 50: raise ValueError('Demasiados grupos; usa filtros o una categoría con hasta 50 valores.')
    variables = plan.get('variables', [])
    if not isinstance(variables, list) or not all(isinstance(x, str) for x in variables): raise ValueError('Variables inválidas.')
    if len(variables) > 12 or len(set(variables)) != len(variables): raise ValueError('Elige hasta 12 variables distintas.')
    for name in variables: column(df, name)
    if action in ('frecuencias', 'cruce') and len(variables) not in ([1, 2] if action == 'frecuencias' else [2]):
        raise ValueError('Frecuencias requiere una o dos variables; cruce requiere dos.')
    if action == 'correlacion' and len(variables) < 2: raise ValueError('Indica al menos dos variables para correlacionar.')
    limit = plan.get('limite', 50)
    if type(limit) is not int or not 1 <= limit <= 100: raise ValueError('Límite permitido: de 1 a 100 resultados.')
    if type(plan.get('descendente', True)) is not bool: raise ValueError('Orden descendente inválido.')
    rows = []; suppressed = 0
    iterable = base.groupby(groups, dropna=False, observed=True) if groups else [((), base)]
    warnings = ['Resultados descriptivos del dataset sintético; no prueban causalidad ni son tasas anuales.',
                'Filtros combinados con AND; in permite varias categorías. Se excluyen nulos de comparaciones.']
    if any(x in ORDINAL for x in variables): warnings.append('Variables ordinales: considera Spearman y revisa la interpretación de sus escalas.')
    for group, work in iterable:
        group = group if isinstance(group, tuple) else (group,)
        labels = {f'grupo_{k}': clean(v) if pd.notna(v) else None for k, v in zip(groups, group)}
        if groups and len(work) < min_group:
            suppressed += 1; continue
        if action == 'resumen': data = [summary(work)]
        elif action == 'nulos': data = profile(work).to_dict('records')
        elif action == 'estadisticas':
            metrics = objects(plan.get('metricas', []), 10, 'Métricas')
            if not metrics: raise ValueError('Indica al menos una métrica.')
            values = {}; details = {}
            for item in metrics:
                ident = item.get('id')
                if not isinstance(ident, str) or not re.fullmatch('[a-z][a-z0-9_]{0,30}', ident) or ident in values:
                    raise ValueError('Identificador de métrica inválido o repetido.')
                value, info = metric(work, item, df)
                if groups and info['validos'] < min_group and item['estadistico'] not in ('missing',):
                    value = None; info['suprimido_por_umbral'] = True
                values[ident] = value; details[ident] = info
            for expr in objects(plan.get('calculos', []), 8, 'Cálculos'):
                keys(expr, ['id', 'op', 'a', 'b'], ['id', 'op', 'a', 'b'])
                ident = expr['id']; op = expr['op']
                if not isinstance(ident, str) or not re.fullmatch('[a-z][a-z0-9_]{0,30}', ident) or ident in values or op not in MATH:
                    raise ValueError('Cálculo no permitido o identificador repetido.')
                operands = []
                for ref in (expr['a'], expr['b']):
                    if isinstance(ref, str):
                        if ref not in values: raise ValueError('Referencia de cálculo inexistente.')
                        val = values[ref]
                    else: val = ref
                    if val is not None and (type(val) not in (int, float) or not math.isfinite(val)):
                        raise ValueError('Solo se admiten números o referencias a métricas numéricas.')
                    operands.append(val)
                a, b = operands
                if a is None or b is None or (op in ('divide', 'percentage', 'pct_change') and b == 0): value = None
                elif op == 'add': value = a+b
                elif op == 'subtract': value = a-b
                elif op == 'multiply': value = a*b
                elif op == 'divide': value = a/b
                elif op == 'percentage': value = 100*a/b
                else: value = 100*(a-b)/b
                values[ident] = clean(value)
            data = [{**values, 'detalle': details}]
        elif action == 'proporcion':
            event = plan.get('evento', [])
            if not event: raise ValueError('Define el evento del numerador.')
            objects(event, 8, 'Evento')
            if any(c in plan.get('filtros', []) for c in event):
                raise ValueError('El filtro del denominador repite el evento del numerador. Para calcular una proporción, conserva el evento solo en evento; pide aclaración si la población ya está restringida a ese mismo evento.')
            # Only rows with observed event variables form the denominator, unless explicitly asking nulls.
            denom = work
            for condition in event:
                keys(condition, ['variable','op','valor'], ['variable','op'])
                column(work, condition['variable'])
                if condition['op'] not in ('is_null', 'not_null'):
                    denom = denom[denom[condition['variable']].notna()]
            numerator = len(filtered(denom, event, df)); n = len(denom)
            data = [{'numerador': numerator, 'denominador': n, 'nulos_excluidos': len(work)-n,
                     'proporcion_pct': 100*numerator/n if n and (not groups or n >= min_group) else None}]
        elif action == 'frecuencias':
            counts = work.groupby(variables, dropna=False, observed=True).size()
            if len(counts) > 250: raise ValueError('Demasiadas categorías; acota con filtros antes de consultar frecuencias.')
            data = []
            for category, n in counts.items():
                category = category if isinstance(category, tuple) else (category,)
                if n < min_group: suppressed += 1; continue
                data.append({**{k: v if pd.notna(v) else None for k, v in zip(variables, category)},
                             'conteo': int(n), 'denominador': len(work), 'porcentaje': 100*n/len(work)})
        elif action == 'correlacion':
            method = plan.get('metodo', 'pearson')
            if method not in ('pearson', 'spearman'): raise ValueError('Métodos disponibles: pearson y spearman.')
            converted = pd.DataFrame(index=work.index); mapping = {}
            for name in variables:
                s = column(work, name)
                if is_numeric_dtype(s): converted[name] = numeric(work, name)
                elif set(s.dropna().unique()) <= {'Yes', 'No'}:
                    converted[name] = s.map({'Yes': 1, 'No': 0}); mapping[name] = {'Yes': 1, 'No': 0}
                else: raise ValueError(f'{name} es categórica no binaria Yes/No: usa cruce, no Pearson/Spearman.')
            data = []
            for a, b in itertools.combinations(variables, 2):
                pair = converted[[a,b]].dropna(); n = len(pair); reason = None
                if n < max(3, min_group): reason = 'Pares válidos insuficientes'
                elif pair[a].nunique() < 2 or pair[b].nunique() < 2: reason = 'Variable constante'
                ranked = pair.rank(method='average') if method == 'spearman' else pair
                coeff = None if reason else ranked[a].corr(ranked[b], method='pearson')
                data.append({'variable_x': a, 'variable_y': b, 'metodo': method, 'coeficiente': coeff,
                             'pares_validos': n, 'pares_excluidos': len(work)-n,
                             'motivo_sin_resultado': reason, 'codificacion': mapping})
        else:  # contingency and descriptive Cramer's V
            a,b = variables; complete = work[[a,b]].dropna()
            if any(complete[v].nunique() > 30 for v in variables): raise ValueError('Cruce: hasta 30 categorías por variable.')
            table = pd.crosstab(complete[a], complete[b]); n = int(table.to_numpy().sum())
            r,k = table.shape; v = None
            if n >= max(3,min_group) and min(r,k) > 1:
                obs = table.to_numpy(dtype=float); expected = np.outer(obs.sum(axis=1),obs.sum(axis=0))/n
                chi = (((obs-expected)**2)/expected).sum(); v = math.sqrt(chi/(n*min(r-1,k-1)))
            cells = []
            for x in table.index:
                for y in table.columns:
                    count = int(table.loc[x,y]); total_row = int(table.loc[x].sum()); total_col = int(table[y].sum())
                    if count < min_group: suppressed += 1; continue
                    cells.append({'fila': x, 'columna': y, 'conteo': count, 'pct_fila': 100*count/total_row,
                                  'pct_columna': 100*count/total_col, 'pct_total': 100*count/n})
            if len(cells) > 100: raise ValueError('Cruce demasiado extenso; acota categorías con filtros.')
            data = [{'variable_fila': a, 'variable_columna': b, 'validos': n, 'nulos_excluidos': len(work)-n,
                     'cramers_v': v, 'celdas': cells,
                     'nota': 'V descriptiva sin corrección de sesgo, sin dirección ni prueba de significancia; usa todas las celdas.'}]
        rows.extend({**labels, **x} for x in data)
    order = plan.get('ordenar', '')
    if order:
        if not isinstance(order, str) or not rows or not all(order in x for x in rows): raise ValueError('Campo para ordenar no disponible.')
        nonnull = [r for r in rows if r[order] is not None]; nulls = [r for r in rows if r[order] is None]
        if any(not isinstance(r[order], (int,float,str)) for r in nonnull): raise ValueError('Ordena por una medida simple.')
        rows = sorted(nonnull, key=lambda r:r[order], reverse=plan.get('descendente',True))+nulls
    return clean({'accion': action, 'filas_originales': len(df), 'filas_filtradas': len(base),
                  'filtros': plan.get('filtros', []), 'grupos': groups, 'plan_ejecutado': plan,
                  'resultados': rows[:limit], 'resultados_totales': len(rows), 'truncado': len(rows)>limit,
                  'grupos_o_celdas_omitidos': suppressed, 'advertencias': warnings,
                  'nota_umbral': 'Umbral de estabilidad, no anonimización. Los totales pueden incluir grupos omitidos.'})
