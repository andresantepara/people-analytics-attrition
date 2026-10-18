"""Carga y validación del archivo público IBM HR Analytics."""
from io import BytesIO
import pandas as pd

REQUIRED = {'Attrition', 'Department', 'JobRole', 'OverTime', 'Age',
            'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction'}
NUMERIC = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']
MAX_BYTES = 10 * 1024 * 1024


def load_dataset(content: bytes) -> pd.DataFrame:
    if not content or len(content) > MAX_BYTES:
        raise ValueError('Selecciona un CSV no vacío de hasta 10 MB.')
    try:
        df = pd.read_csv(BytesIO(content), encoding='utf-8-sig')
    except (ValueError, UnicodeError) as exc:
        raise ValueError('No se pudo leer el CSV. Usa UTF-8 y separador coma.') from exc
    if df.empty:
        raise ValueError('El archivo no contiene registros.')
    if len(df) > 100000:
        raise ValueError('El límite del ejercicio es 100000 filas.')
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('Hay nombres de columnas duplicados.')
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError('Faltan columnas: ' + ', '.join(sorted(missing)))
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].astype('string').str.strip().replace('', pd.NA)
    for col in ['Attrition', 'OverTime']:
        invalid = set(df[col].dropna()) - {'Yes', 'No'}
        if invalid:
            raise ValueError(f'{col} debe contener Yes, No o valores ausentes.')
    for col in NUMERIC:
        converted = pd.to_numeric(df[col], errors='coerce')
        if (df[col].notna() & converted.isna()).any():
            raise ValueError(f'{col} contiene valores no numéricos.')
        if converted.isin([float('inf'), float('-inf')]).any():
            raise ValueError(f'{col} contiene valores infinitos.')
        if (converted.dropna() < 0).any():
            raise ValueError(f'{col} contiene valores negativos.')
        df[col] = converted
    return df
