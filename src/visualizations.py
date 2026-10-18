"""Gráficos Plotly adecuados a variables y objetivos analíticos."""
import plotly.express as px


def attrition_bars(table, column):
    fig = px.bar(table, x='grupo', y='proporcion_pct',
                 hover_data=['filas', 'validos', 'salidas', 'nulos_attrition'],
                 labels={'grupo': column, 'proporcion_pct': 'Attrition (%)'},
                 title=f'Proporción de Attrition por {column}')
    fig.update_yaxes(range=[0, 100])
    return fig


def distribution(df, column):
    return px.histogram(df, x=column, nbins=25,
                        title=f'Distribución de {column}', labels={'count': 'Registros'})


def correlation_heatmap(matrix):
    return px.imshow(matrix, text_auto='.2f', zmin=-1, zmax=1,
                     color_continuous_scale='RdBu_r',
                     title='Correlación de Pearson entre variables numéricas')
