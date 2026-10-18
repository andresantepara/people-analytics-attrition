import json
import unittest
import pandas as pd
from test_core import FakeClient, sample
from src.agent import AnalyticsAgent
from src.chart_agent import execute_chart, exact_chart, wants_chart


class ChartAgentTests(unittest.TestCase):
    def setUp(self):
        self.df=sample();self.df['Gender']=['Female','Female','Male','Male']
        self.df['Department']=['Sales','Sales','Research','Research']

    def run_chart(self,question):
        client=FakeClient('{}')
        result=AnalyticsAgent(client).run(question,self.df,authorized=True,min_group=1)
        self.assertEqual(len(client.calls),1)
        self.assertIsNotNone(result['grafico_json'])
        return result

    def test_exact_histogram(self):
        result=self.run_chart('Histograma de Age.')
        self.assertEqual(sum(r['conteo'] for r in result['evidencia']['resultados']),4)

    def test_exact_mean_bars(self):
        result=self.run_chart('Grafica el ingreso mensual promedio por departamento.')
        values={r['grupo_Department']:r['valor'] for r in result['evidencia']['resultados']}
        self.assertEqual(values,{'Research':3500,'Sales':1500})

    def test_exact_proportion_correct_denominator(self):
        result=self.run_chart('Grafica el porcentaje de Attrition = Yes por Department.')
        rows=result['evidencia']['resultados']
        sales=next(r for r in rows if r['grupo_Department']=='Sales')
        self.assertEqual(sales['proporcion_pct'],50)
        self.assertEqual(sales['denominador'],2)
        self.assertEqual(result['evidencia']['filtros'],[])

    def test_exact_scatter_count_and_correlation(self):
        result=self.run_chart('Gráfico de dispersión de Age y MonthlyIncome.')
        self.assertEqual(result['evidencia']['pares_o_valores_validos'],4)
        result=self.run_chart('Mapa de calor de correlación entre Age, MonthlyIncome y YearsAtCompany.')
        self.assertEqual(len(result['evidencia']['resultados']),3)
        self.run_chart('Gráfico de barras de Gender.')

    def test_filtered_chart_and_missing_values(self):
        self.df.loc[0,'Age']=None
        plan={'tipo':'histograma','variables':['Age'],'filtros':[{'variable':'OverTime','op':'eq','valor':'Yes'}]}
        client=FakeClient(json.dumps(plan))
        result=AnalyticsAgent(client).run('Histograma de Age entre OverTime = Yes',self.df,authorized=True,min_group=1)
        self.assertEqual(result['evidencia']['filas_filtradas'],2)
        self.assertEqual(result['evidencia']['pares_o_valores_validos'],1)
        self.assertEqual(result['evidencia']['nulos_excluidos'],1)

    def test_omitted_filter_rejected(self):
        client=FakeClient(json.dumps({'tipo':'histograma','variables':['Age'],'filtros':[]}))
        with self.assertRaisesRegex(ValueError,'condición explícita'):
            AnalyticsAgent(client).run('Histograma de Age entre OverTime = Yes',self.df,authorized=True,min_group=1)
        self.assertEqual(len(client.calls),2)

    def test_blocked_columns_and_code_rejected(self):
        frame=self.df.assign(EmployeeNumber=[1,2,3,4])
        for plan in [{'tipo':'histograma','variables':['EmployeeNumber'],'filtros':[]},
                     {'tipo':'histograma','variables':['Age'],'filtros':[],'codigo':'print(1)'},
                     {'tipo':'histograma','variables':['Age'],'filtros':[],'bins':999}]:
            with self.assertRaises(ValueError):execute_chart(plan,'Histograma',frame,1)

    def test_previous_result_and_authorization(self):
        prior=self.run_chart('Histograma de Age')
        client=FakeClient('{}')
        result=AnalyticsAgent(client).run('Grafica el resultado anterior',self.df,min_group=1,authorized=True,history=[prior])
        self.assertIsNotNone(result['grafico_json']);self.assertEqual(len(client.calls),1)
        result=AnalyticsAgent(client).run('Grafica eso',self.df,min_group=1,authorized=True)
        self.assertIn('Primero',result['respuesta'])
        with self.assertRaises(PermissionError):AnalyticsAgent(client).run('Histograma de Age',self.df)

    def test_extra_scope_not_silently_discarded(self):
        self.assertIsNone(exact_chart('Histograma de Age para OverTime=Yes',self.df))
        self.assertFalse(wants_chart('Cuál es la edad promedio'))
        self.assertTrue(wants_chart('Haz un gráfico de barras'))
        self.assertTrue(wants_chart('Muéstrame en barras la mediana de MonthlyIncome según Department'))
