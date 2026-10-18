import json
import unittest
from src.agent import AnalyticsAgent
from src.interpretation import interpret_result
from test_core import FakeClient, sample

class InterpretationTests(unittest.TestCase):
    def test_computed_context_reaches_interpreter(self):
        client=FakeClient('{}')
        result=AnalyticsAgent(client).run('Edad promedio entre Attrition=Yes',sample(),min_group=1,authorized=True)
        self.assertEqual(len(client.calls),1)
        context=json.loads(client.calls[-1][-1]['content'])
        self.assertEqual(context['respuesta_calculada'],result['respuesta'])
        self.assertIn('20',context['respuesta_calculada'])
        self.assertEqual(context['alcance'],result['plan'])
        self.assertEqual(result['estado_interpretacion'],'generada')
        self.assertIn('Interpretación simulada',result['interpretacion'])

    def test_failure_preserves_calculations(self):
        class Offline(FakeClient):
            def chat(self,messages,schema=None): raise RuntimeError('Offline')
        result=AnalyticsAgent(Offline('{}')).run('Edad promedio entre Attrition=Yes',sample(),min_group=1,authorized=True)
        self.assertEqual(result['estado_interpretacion'],'no_disponible')
        self.assertIsNone(result['interpretacion'])
        self.assertEqual(result['evidencia']['resultados'][0]['edad_media'],20)
        self.assertIn('20',result['respuesta'])

    def test_generated_text_cannot_overwrite_numeric_answer(self):
        class Incorrect(FakeClient):
            def chat(self,messages,schema=None): return 'La media es 999999.'
        result=AnalyticsAgent(Incorrect('{}')).run('Edad promedio entre Attrition=Yes',sample(),min_group=1,authorized=True)
        self.assertNotIn('999999',result['respuesta'])
        self.assertEqual(result['evidencia']['resultados'][0]['edad_media'],20)
        self.assertIsNone(result['interpretacion'])
        self.assertEqual(result['estado_interpretacion'],'no_disponible')

    def test_empty_and_oversized_context(self):
        class Empty(FakeClient):
            def chat(self,messages,schema=None): return ' '
        base={'pregunta':'Resumen','plan':{},'evidencia':{},'respuesta':'Sin valores válidos.'}
        self.assertEqual(interpret_result(Empty('{}'),base)['estado_interpretacion'],'no_disponible')
        client=FakeClient('{}')
        self.assertEqual(interpret_result(client,{**base,'respuesta':'x'*15000})['estado_interpretacion'],'no_disponible')
        self.assertEqual(client.calls,[])

    def test_raw_chart_data_excluded(self):
        client=FakeClient('{}')
        interpret_result(client,{'pregunta':'Gráfico','plan':{'tipo':'dispersion'},'respuesta':'Hay cuatro pares válidos.',
                                'evidencia':{'puntos_secretos':[1,2,3]},'grafico_json':'raw_rows'})
        payload=client.calls[-1][-1]['content']
        self.assertNotIn('puntos_secretos',payload)
        self.assertNotIn('raw_rows',payload)

    def test_unrequested_comparison_is_retried(self):
        class Comparison(FakeClient):
            def chat(self,messages,schema=None):
                self.calls.append(messages)
                return 'Es significativamente más joven.' if len(self.calls)==1 else 'La media describe la edad del grupo consultado en el dataset sintético.'
        client=Comparison('{}')
        result=AnalyticsAgent(client).run('Edad promedio entre Attrition=Yes',sample(),min_group=1,authorized=True)
        self.assertEqual(len(client.calls),2)
        self.assertEqual(result['estado_interpretacion'],'generada')
        self.assertNotIn('significativamente',result['interpretacion'])
