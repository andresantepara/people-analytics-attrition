"""Funciones analíticas: devuelven resultados sin modificar el original."""
import pandas as pd

SEGMENTS = ['Department', 'JobRole', 'OverTime']
NUMERIC = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']


def profile(df):
    return pd.DataFrame({'variable': df.columns,
        'tipo': [str(t) for t in df.dtypes],
        'nulos': df.isna().sum().to_numpy(),
        'unicos': df.nunique().to_numpy()})


def summary(df):
    valid = df['Attrition'].isin(['Yes', 'No'])
    denominator = int(valid.sum())
    exits = int(df['Attrition'].eq('Yes').sum())
    return {'filas': len(df), 'variables': len(df.columns),
            'attrition_validos': denominator, 'attrition_nulos': len(df)-denominator,
            'salidas': exits,
            'proporcion_pct': round(100*exits/denominator, 2) if denominator else None}


def segment(df, column, min_group=5):
    if column not in SEGMENTS:
        raise ValueError('La variable de segmentación no está autorizada.')
    if not isinstance(min_group, int) or min_group < 1:
        raise ValueError('El tamaño mínimo debe ser un entero positivo.')
    work = df[[column, 'Attrition']].copy()
    work['grupo'] = work[column].fillna('(sin dato)')
    work['validos'] = work['Attrition'].isin(['Yes', 'No']).astype(int)
    work['salidas'] = work['Attrition'].eq('Yes').fillna(False).astype(int)
    table = work.groupby('grupo', dropna=False).agg(
        filas=('validos', 'size'), validos=('validos', 'sum'), salidas=('salidas', 'sum')).reset_index()
    table['nulos_attrition'] = table['filas'] - table['validos']
    table['proporcion_pct'] = (100*table['salidas']/table['validos'].replace(0, float('nan'))).round(2)
    # El umbral evita interpretaciones inestables; no garantiza anonimización.
    table = table[table['validos'] >= min_group]
    return table.sort_values('proporcion_pct', ascending=False).reset_index(drop=True)


def correlations(df):
    return df[NUMERIC].corr(method='pearson', min_periods=3)


def numeric_summary(df):
    numeric = df.select_dtypes(include='number')
    if numeric.empty: return pd.DataFrame()
    return numeric.describe().round(2)
