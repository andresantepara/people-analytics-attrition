import json
import unittest
import pandas as pd
from src.analytics import calculate
from src.agent import AnalyticsAgent, plan_schema
from test_core import sample, FakeClient


def m(ident='media', variable='Age', stat='mean', **kw):
    return dict(id=ident, variable=variable, estadistico=stat, **kw)


def f(variable, op, valor=None):
    return dict(variable=variable, op=op, valor=valor)


def stats(*metrics, **kw):
    return dict(accion='estadisticas', metricas=list(metrics), **kw)


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.df = sample()
        self.df['DistanceFromHome'] = [1,2,4,8]
        self.df['Gender'] = ['Female','Female','Male','Male']
        self.df['EmployeeNumber'] = [101,102,103,104]
        self.df['Constant'] = 1

    def run_plan(self, plan, min_group=1):
        return calculate(self.df, plan, min_group)

    def first(self, plan): return self.run_plan(plan)['resultados'][0]

    def test_all_numeric_columns_and_percentile(self):
        row = self.first(stats(m(variable='DistanceFromHome'),m('p75','Age','quantile',q=.75),m('desv','Age','std'),m('iqr','Age','iqr')))
        self.assertEqual(row['media'], 3.75)
        self.assertEqual(row['p75'], 42.5)
        self.assertAlmostEqual(row['desv'], (500/3)**.5)
        self.assertEqual(row['iqr'],15)

    def test_filters_and_in(self):
        row=self.first(stats(m(), filtros=[f('Age','ge',30),f('Gender','in',['Male'])]))
        self.assertEqual(row['media'],45)
        self.assertEqual(row['detalle']['media']['validos'],2)

    def test_negative_filters_exclude_null(self):
        row=self.first(stats(m('n','*','count'),filtros=[f('Attrition','ne','Yes')]))
        self.assertEqual(row['n'],2)

    def test_null_filters(self):
        row=self.first(stats(m('n','*','count'),filtros=[f('Attrition','is_null')]))
        self.assertEqual(row['n'],1)

    def test_grouped_metrics_and_sort(self):
        result=self.run_plan(stats(m(),grupos=['Gender'],ordenar='media',limite=1))
        self.assertEqual(result['resultados'][0]['grupo_Gender'],'Male')
        self.assertTrue(result['truncado'])

    def test_group_minimum(self):
        result=self.run_plan(stats(m(),grupos=['Gender']),3)
        self.assertEqual(result['resultados'],[])
        self.assertEqual(result['grupos_o_celdas_omitidos'],2)

    def test_proportion_valid_denominator(self):
        row=self.first({'accion':'proporcion','evento':[f('Attrition','eq','Yes')]})
        self.assertEqual(row['numerador'],1)
        self.assertEqual(row['denominador'],3)
        self.assertAlmostEqual(row['proporcion_pct'],100/3)

    def test_zero_events_in_group_are_not_unknown_categories(self):
        result=self.run_plan({'accion':'proporcion','grupos':['Gender'],'evento':[f('Attrition','eq','Yes')]})
        male=next(x for x in result['resultados'] if x['grupo_Gender']=='Male')
        self.assertEqual(male['proporcion_pct'],0)

    def test_filtered_proportion(self):
        row=self.first({'accion':'proporcion','filtros':[f('Age','ge',30)],'evento':[f('OverTime','eq','Yes')]})
        self.assertEqual(row['denominador'],3)
        self.assertEqual(row['numerador'],1)

    def test_null_proportion_includes_all_rows(self):
        row=self.first({'accion':'proporcion','evento':[f('Attrition','is_null')]})
        self.assertEqual(row['proporcion_pct'],25)

    def test_arithmetic_of_cohorts(self):
        plan=stats(m('salidas','MonthlyIncome',filtros=[f('Attrition','eq','Yes')]),
                   m('permanecen','MonthlyIncome',filtros=[f('Attrition','eq','No')]),
                   calculos=[dict(id='diferencia',op='subtract',a='salidas',b='permanecen'),
                             dict(id='brecha_pct',op='pct_change',a='salidas',b='permanecen')])
        row=self.first(plan)
        self.assertEqual(row['diferencia'],-2000)
        self.assertAlmostEqual(row['brecha_pct'],-200/3)

    def test_zero_division_and_missing_propagate(self):
        row=self.first(stats(m(),calculos=[dict(id='cero',op='divide',a='media',b=0),dict(id='otro',op='add',a='cero',b=2)]))
        self.assertIsNone(row['cero']);self.assertIsNone(row['otro'])

    def test_frequency_includes_missing_denominator(self):
        rows=self.run_plan({'accion':'frecuencias','variables':['Attrition']})['resultados']
        self.assertEqual(sum(r['porcentaje'] for r in rows),100)
        self.assertEqual(sum(r['conteo'] for r in rows),4)

    def test_pearson_spearman_pair_counts(self):
        self.df.loc[0,'MonthlyIncome']=float('nan')
        for method in ['pearson','spearman']:
            row=self.first({'accion':'correlacion','variables':['Age','MonthlyIncome'],'metodo':method})
            self.assertAlmostEqual(row['coeficiente'],1)
            self.assertEqual(row['pares_validos'],3)

    def test_spearman_ties(self):
        self.df['DistanceFromHome']=[1,1,2,3]
        row=self.first({'accion':'correlacion','variables':['Age','DistanceFromHome'],'metodo':'spearman'})
        self.assertAlmostEqual(row['coeficiente'],.9486832980505138)

    def test_binary_encoding_and_constant(self):
        row=self.first({'accion':'correlacion','variables':['Age','Attrition']})
        self.assertEqual(row['codificacion']['Attrition'],{'Yes':1,'No':0})
        self.assertEqual(row['pares_validos'],3)
        row=self.first({'accion':'correlacion','variables':['Age','Constant']})
        self.assertIsNone(row['coeficiente'])
        self.assertEqual(row['motivo_sin_resultado'],'Variable constante')

    def test_categorical_correlation_rejected(self):
        with self.assertRaises(ValueError):
            self.first({'accion':'correlacion','variables':['Gender','Age']})

    def test_cramers_v_and_percentages(self):
        row=self.first({'accion':'cruce','variables':['Gender','OverTime']})
        self.assertEqual(row['cramers_v'],1)
        self.assertEqual(row['validos'],4)
        self.assertTrue(all(x['pct_fila']==100 for x in row['celdas']))

    def test_reject_code_identifiers_invalid_category_and_fields(self):
        plans=[stats(m(variable='EmployeeNumber')),stats(m(variable='missing_column')),
               stats(m(),filtros=[f('Gender','eq','INVALID')]),
               stats(m(),calculos=[dict(id='x',op='exec',a='media',b=1)]),
               {'accion':'resumen','codigo':'print(1)'},
               {'accion':'correlacion','variables':['Age','Age']},
               {'accion':'resumen','metricas':[m()]},
               stats(m(),calculos=[dict(id='x',op='add',a='df.Age.mean()',b=1)])]
        for plan in plans:
            with self.subTest(plan=plan),self.assertRaises(ValueError): self.first(plan)

    def test_no_mutation_and_json_finite(self):
        before=self.df.copy(deep=True)
        self.df['Age']=float('nan')
        result=self.run_plan(stats(m(),m('total','Age','sum')))
        self.assertIsNone(result['resultados'][0]['media'])
        self.assertIsNone(result['resultados'][0]['total'])
        json.dumps(result,allow_nan=False)
        self.df=before.copy(deep=True)
        self.run_plan(stats(m(),grupos=['Gender']))
        pd.testing.assert_frame_equal(self.df,before)

    def test_clarification_no_interpretation_call(self):
        client=FakeClient('{"accion":"aclarar","aclaracion":"¿Ingreso mensual o tarifa diaria?"}')
        result=AnalyticsAgent(client).run('Sueldo promedio',self.df,authorized=True)
        self.assertEqual(len(client.calls),1)
        self.assertIn('mensual',result['respuesta'])

    def test_dynamic_schema_and_agent_evidence(self):
        client=FakeClient(json.dumps({'accion':'correlacion','variables':['Age','DistanceFromHome']}))
        result=AnalyticsAgent(client).run('Correlación de edad y distancia',self.df,min_group=1,authorized=True)
        self.assertEqual(result['evidencia']['resultados'][0]['pares_validos'],4)
        self.assertIn('DistanceFromHome',client.calls[0][1]['content'])
        json.dumps(plan_schema(list(self.df.columns)))


class SemanticGuardTests(unittest.TestCase):
    def test_omitted_literal_filter_repaired(self):
        class RepairClient(FakeClient):
            def chat(self, messages, schema=None):
                self.calls.append(messages)
                if not schema: return 'Interpretación de prueba.'
                plan=stats(m())
                if len(self.calls)>1: plan['filtros']=[f('OverTime','eq','Yes')]
                return json.dumps(plan)
        client=RepairClient('')
        result=AnalyticsAgent(client).run('Edad promedio entre OverTime = Yes',sample(),authorized=True)
        self.assertEqual(result['evidencia']['resultados'][0]['media'],25)
        self.assertEqual(len(client.calls),3)

    def test_contradictory_cohort_filters_rejected(self):
        with self.assertRaises(ValueError):
            calculate(sample(),stats(m(filtros=[f('Attrition','eq','Yes'),f('Attrition','eq','No')])) )

    def test_uncomputed_difference_rejected(self):
        client=FakeClient(json.dumps(stats(m())))
        with self.assertRaises(ValueError):
            AnalyticsAgent(client).run('Diferencia de edad promedio entre grupos',sample(),authorized=True)

    def test_event_cannot_filter_own_denominator(self):
        condition=f('Attrition','eq','Yes')
        with self.assertRaises(ValueError):
            calculate(sample(),{'accion':'proporcion','filtros':[condition],'evento':[condition]})

    def test_unrequested_category_filter_rejected(self):
        client=FakeClient(json.dumps({'accion':'cruce','variables':['Gender','Attrition'],
            'filtros':[f('Attrition','eq','Yes')]}))
        frame=sample();frame['Gender']=['Female']*4
        with self.assertRaises(ValueError):
            AnalyticsAgent(client).run('Analiza el cruce de Gender y Attrition',frame,authorized=True)

    def test_standard_attrition_proportion_only_requires_interpretation(self):
        client=FakeClient('{}')
        result=AnalyticsAgent(client).run('¿Qué porcentaje tiene Attrition = Yes por Department?',sample(),min_group=1,authorized=True)
        self.assertEqual(len(client.calls),1)
        self.assertAlmostEqual(result['evidencia']['resultados'][0]['proporcion_pct'],100/3)
        self.assertIn('de 3 registros con datos válidos',result['respuesta'])

    def test_exact_cross_has_no_invented_filter(self):
        frame=sample();frame['Gender']=['Female','Female','Male','Male']
        client=FakeClient('{}')
        result=AnalyticsAgent(client).run('Cruza Gender con Attrition e indica porcentajes por fila y V de Cramér.',frame,min_group=1,authorized=True)
        self.assertEqual(result['plan']['filtros'],[])
        self.assertEqual(result['evidencia']['resultados'][0]['validos'],3)
        self.assertEqual(len(client.calls),1)

    def test_age_mean_attrition_spelling_variants(self):
        frame=sample()
        frame['Age']=[20,80,90,40]
        for name in ['Attrition','atrittion','atrition']:
            for category,expected in [('yes',20),('no',60)]:
                client=FakeClient('{}')
                result=AnalyticsAgent(client).run(f'Cual es la edad promedio de los que tuvieron {name}={category}?',frame,min_group=1,authorized=True)
                self.assertEqual(result['evidencia']['resultados'][0]['edad_media'],expected)
                self.assertEqual(len(client.calls),1)
                self.assertEqual(result['plan']['metricas'][0]['variable'],'Age')

    def test_age_mean_excludes_missing_ages_and_unlabeled_rows(self):
        frame=sample();frame['Attrition']=['Yes','Yes',None,'No'];frame['Age']=[20,None,90,40]
        result=AnalyticsAgent(FakeClient('{}')).run('Edad promedio entre Attrition = Yes',frame,min_group=1,authorized=True)
        row=result['evidencia']['resultados'][0]
        self.assertEqual(row['edad_media'],20)
        self.assertEqual(row['detalle']['edad_media']['validos'],1)
        self.assertEqual(row['detalle']['edad_media']['nulos'],1)

    def test_exact_age_mean_never_discards_additional_constraints(self):
        from src.agent import standard_age_mean
        for suffix in [' y OverTime = Yes',' por Department',' y compara con Attrition=No',' mayores de 30']:
            self.assertIsNone(standard_age_mean('Edad promedio entre Attrition=Yes'+suffix,sample()))

    def test_exact_cross_row_percentages_without_cramers_suffix(self):
        frame=sample();frame['Gender']=['Female','Female','Male','Male']
        client=FakeClient('{}')
        result=AnalyticsAgent(client).run('Cruza Gender con Attrition e indica porcentajes por fila.',frame,min_group=1,authorized=True)
        self.assertEqual(len(client.calls),1)
        self.assertEqual(result['plan']['variables'],['Gender','Attrition'])
        self.assertEqual(result['plan']['filtros'],[])
        row=result['evidencia']['resultados'][0]
        cells={(c['fila'],c['columna']):c for c in row['celdas']}
        self.assertEqual(cells['Female','Yes']['pct_fila'],50)
        self.assertEqual(cells['Female','No']['pct_fila'],50)
        self.assertEqual(cells['Male','No']['pct_fila'],100)
        self.assertEqual(row['nulos_excluidos'],1)

    def test_cross_row_percentages_does_not_drop_extra_filter(self):
        from src.agent import standard_cross
        frame=sample();frame['Gender']=['Female']*len(frame)
        self.assertIsNone(standard_cross('Cruza Gender con Attrition e indica porcentajes por fila para OverTime=Yes',frame))

    def test_income_difference_exact_query_and_reverse_order(self):
        for first,second,expected in [('salen','permanecen',-2000),('permanecen','salen',2000)]:
            client=FakeClient('{}')
            question=f'Diferencia entre el ingreso mensual promedio de quienes {first} y quienes {second}.'
            result=AnalyticsAgent(client).run(question,sample(),min_group=1,authorized=True)
            row=result['evidencia']['resultados'][0]
            self.assertEqual(row['salen'],1000)
            self.assertEqual(row['permanecen'],3000)
            self.assertEqual(row['diferencia'],expected)
            self.assertEqual(len(client.calls),1)
            self.assertEqual(result['plan']['filtros'],[])

    def test_income_difference_keeps_additional_scope_for_planner(self):
        from src.agent import standard_income_difference
        for suffix in [' por Department',' con OverTime=Yes',' en porcentaje']:
            self.assertIsNone(standard_income_difference('Diferencia entre el ingreso mensual promedio de quienes salen y quienes permanecen'+suffix,sample()))
