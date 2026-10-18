import unittest
import pandas as pd
from src.data_loader import load_dataset
from src.eda import summary, segment
from src.agent import AnalyticsAgent


def sample():
    return pd.DataFrame({'Attrition': ['Yes','No',None,'No'],
        'Department': ['Sales']*4, 'JobRole': ['Representative']*4,
        'OverTime': ['Yes','Yes','No','No'], 'Age': [20,30,40,50],
        'MonthlyIncome': [1000,2000,3000,4000], 'YearsAtCompany': [1,2,3,4],
        'JobSatisfaction': [1,2,3,4]})


class FakeClient:
    model = 'doble_de_prueba_sin_llm'
    def __init__(self, plan):
        self.plan = plan
        self.calls = []
    def chat(self, messages, schema=None):
        self.calls.append(messages)
        return self.plan if schema else 'Interpretación simulada solo para prueba de integración.'


class CoreTests(unittest.TestCase):
    def test_missing_denominator(self):
        result = summary(sample())
        self.assertEqual(result['attrition_validos'], 3)
        self.assertEqual(result['proporcion_pct'], 33.33)
        self.assertEqual(result['attrition_nulos'], 1)

    def test_segment_counts_and_no_mutation(self):
        frame = sample(); before = frame.copy(deep=True)
        result = segment(frame, 'OverTime', 1).set_index('grupo')
        self.assertEqual(result.loc['Yes', 'proporcion_pct'], 50)
        self.assertEqual(result.loc['No', 'validos'], 1)
        self.assertEqual(result.loc['No', 'nulos_attrition'], 1)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(len(segment(frame, 'OverTime', 3)), 0)

    def test_all_missing_target(self):
        frame = sample(); frame['Attrition'] = None
        self.assertIsNone(summary(frame)['proporcion_pct'])
        self.assertTrue(segment(frame, 'OverTime', 1).empty)

    def test_loader_missing_column_and_invalid_target(self):
        with self.assertRaises(ValueError):
            load_dataset(b'Attrition\nYes\n')
        frame = sample(); frame.loc[0, 'Attrition'] = 'Maybe'
        with self.assertRaises(ValueError):
            load_dataset(frame.to_csv(index=False).encode())

    def test_loader_valid_roundtrip(self):
        frame = load_dataset(sample().to_csv(index=False).encode())
        self.assertEqual(summary(frame)['attrition_validos'], 3)

    def test_authorization_before_model_call(self):
        client = FakeClient('{}')
        with self.assertRaises(PermissionError):
            AnalyticsAgent(client).run('Resumen', sample())
        self.assertEqual(len(client.calls), 0)

    def test_rejects_unauthorized_or_malformed_plans(self):
        for plan in ['{"herramienta":"exec","variable":"ninguna"}',
                     '{"herramienta":"segmentacion","variable":"Age"}',
                     '{"herramienta":"resumen","variable":"Department"}',
                     '[]', 'texto libre']:
            with self.subTest(plan=plan):
                with self.assertRaises(ValueError):
                    AnalyticsAgent(FakeClient(plan)).run('Resumen', sample(), authorized=True)

    def test_tool_evidence_reaches_response(self):
        client = FakeClient('{"accion":"resumen"}')
        result = AnalyticsAgent(client).run('Resume el dataset', sample(), authorized=True)
        self.assertEqual(result['evidencia']['resultados'][0]['salidas'], 1)
        self.assertEqual(len(client.calls), 2)
        self.assertIn('33.33', result['respuesta'])

    def test_age_average_reaches_interpreter(self):
        client = FakeClient('{"accion":"estadisticas","metricas":[{"id":"edad_media","variable":"Age","estadistico":"mean"}]}')
        result = AnalyticsAgent(client).run('Edad promedio', sample(), authorized=True)
        data = result['evidencia']['resultados'][0]
        self.assertEqual(data['edad_media'], 35.0)
        self.assertEqual(data['detalle']['edad_media']['validos'], 4)
        self.assertIn('35', result['respuesta'])

    def test_numeric_missing_and_empty(self):
        frame = sample(); frame.loc[0, 'Age'] = float('nan')
        plan = '{"accion":"estadisticas","metricas":[{"id":"edad_media","variable":"Age","estadistico":"mean"}]}'
        result = AnalyticsAgent(FakeClient(plan)).run('Edad promedio', frame, authorized=True)
        self.assertEqual(result['evidencia']['resultados'][0]['edad_media'], 40.0)
        self.assertEqual(result['evidencia']['resultados'][0]['detalle']['edad_media']['nulos'], 1)
        frame['Age'] = float('nan')
        result = AnalyticsAgent(FakeClient(plan)).run('Edad promedio', frame, authorized=True)
        self.assertIsNone(result['evidencia']['resultados'][0]['edad_media'])

    def test_numeric_rejects_categorical_variable(self):
        client = FakeClient('{"accion":"estadisticas","metricas":[{"id":"media","variable":"Department","estadistico":"mean"}]}')
        with self.assertRaises(ValueError):
            AnalyticsAgent(client).run('Promedio', sample(), authorized=True)
        self.assertEqual(len(client.calls), 2)


if __name__ == '__main__':
    unittest.main()
