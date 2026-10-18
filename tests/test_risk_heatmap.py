import unittest
import numpy as np
from test_prediction import fixture
from src.prediction import AttritionPredictor
from src.risk_heatmap import segmented_probabilities, probability_heatmap


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.model=AttritionPredictor(fixture()).fit()

    def test_probabilities_use_holdout_and_correct_segment(self):
        result=segmented_probabilities(self.model,['OverTime','Age'],1)
        table=result['tabla']
        self.assertEqual(table.groupby('Variable')['Registros de prueba'].sum().to_dict(),{'Age':40,'OverTime':40})
        row=table[(table.Variable=='OverTime') & (table.Segmento=='Categoría: Yes')].iloc[0]
        mask=self.model.test_frame.OverTime.eq('Yes').to_numpy()
        self.assertAlmostEqual(row['Probabilidad media Attrition Yes (%)'],100*self.model.probabilities[mask].mean())
        self.assertTrue(table['Probabilidad media Attrition Yes (%)'].between(0,100).all())

    def test_small_segments_are_suppressed_not_zero(self):
        result=segmented_probabilities(self.model,['OverTime'],100)
        self.assertTrue(result['tabla'].empty)
        self.assertEqual(result['celdas_omitidas'],2)

    def test_bounds_depend_only_on_training(self):
        original=segmented_probabilities(self.model,['Age'],1)['tabla']
        self.model.df.loc[self.model.test_indices,'Age']=999
        again=segmented_probabilities(self.model,['Age'],1)['tabla']
        self.assertEqual(original['Condición'].tolist(),again['Condición'].tolist())

    def test_missing_and_constant_segments(self):
        self.model.test_frame.loc[self.model.test_indices[0],'Gender']=None
        table=segmented_probabilities(self.model,['Gender','MaritalStatus'],1)['tabla']
        self.assertIn('Sin dato',table.Segmento.tolist())
        self.assertEqual(table[table.Variable=='MaritalStatus']['Registros de prueba'].sum(),40)
        frame=fixture();frame['DailyRate']=100
        table=segmented_probabilities(AttritionPredictor(frame),['DailyRate'],1)['tabla']
        self.assertEqual(table.Segmento.tolist(),['Valor único en entrenamiento'])

    def test_chart_selection_and_invalid_variables(self):
        table=segmented_probabilities(self.model,['Age','OverTime'],1)['tabla']
        fig=probability_heatmap(table,['Age','OverTime'],['Rango bajo','Categoría: Yes'])
        self.assertEqual(list(fig.data[0].x),['Age','OverTime'])
        self.assertEqual(list(fig.data[0].y),['Rango bajo','Categoría: Yes'])
        self.assertTrue(np.isnan(fig.data[0].z[0][1]))
        for variables in [[],['Attrition'],['EmployeeNumber']]:
            with self.assertRaises(ValueError):segmented_probabilities(self.model,variables)


class TopCombinationTests(unittest.TestCase):
    def setUp(self):
        self.model=AttritionPredictor(fixture()).fit()

    def test_pair_means_and_counts_are_actual_intersections(self):
        from src.risk_heatmap import top_combinations
        result=top_combinations(self.model,['Department','OverTime'],1)
        self.assertEqual(result['pares_evaluados'],1)
        self.assertEqual(result['tabla']['Registros de prueba'].sum(),40)
        for _,row in result['tabla'].iterrows():
            mask=self.model.test_frame.Department.eq(row['Segmento 1'].removeprefix('Categoría: ')) & self.model.test_frame.OverTime.eq(row['Segmento 2'].removeprefix('Categoría: '))
            self.assertEqual(row['Registros de prueba'],mask.sum())
            self.assertAlmostEqual(row['Probabilidad media Attrition Yes (%)'],100*self.model.probabilities[mask.to_numpy()].mean())

    def test_top_five_sorted_without_changing_model(self):
        from src.risk_heatmap import top_combinations
        pipe=self.model.pipeline
        result=top_combinations(self.model,['Age','Department','OverTime'],1)
        table=result['tabla']
        self.assertEqual(len(table),5)
        self.assertEqual(table.Puesto.tolist(),[1,2,3,4,5])
        self.assertTrue(table['Probabilidad media Attrition Yes (%)'].is_monotonic_decreasing)
        self.assertEqual(result['pares_evaluados'],3)
        self.assertGreater(result['combinaciones_validas'],5)
        self.assertIs(self.model.pipeline,pipe)

    def test_filters_and_threshold_do_not_invent_results(self):
        from src.risk_heatmap import top_combinations
        result=top_combinations(self.model,['Department','OverTime'],1,['Categoría: Sales','Categoría: Yes'])
        self.assertEqual(len(result['tabla']),1)
        self.assertEqual(result['tabla'].iloc[0]['Segmento 2'],'Categoría: Yes')
        self.assertTrue(top_combinations(self.model,['Department','OverTime'],100)['tabla'].empty)
        self.assertTrue(top_combinations(self.model,['Department','OverTime'],1,[])['tabla'].empty)
        for variables in [['Age'],['Attrition','Age'],['Age','Age']]:
            with self.assertRaises(ValueError):top_combinations(self.model,variables)

    def test_ranking_does_not_use_observed_target(self):
        from src.risk_heatmap import top_combinations
        a=top_combinations(self.model,['Department','OverTime'],1)['tabla']
        self.model.test_frame['Attrition']='Yes'
        b=top_combinations(self.model,['Department','OverTime'],1)['tabla']
        self.assertEqual(a['Probabilidad media Attrition Yes (%)'].tolist(),b['Probabilidad media Attrition Yes (%)'].tolist())
        self.assertEqual(a['Condición 1'].tolist(),b['Condición 1'].tolist())
