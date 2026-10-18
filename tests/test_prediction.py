import unittest
import numpy as np
import pandas as pd
from src.prediction import AttritionPredictor
from src.prediction_agent import is_predictive, run_prediction, render_prediction
from src.agent import AnalyticsAgent
from test_core import FakeClient


def fixture(n=200):
    rng=np.random.default_rng(29)
    age=rng.integers(20,60,n);overtime=rng.choice(['No','Yes'],n)
    chance=1/(1+np.exp(-(-1+.9*(overtime=='Yes')-.035*(age-35))))
    return pd.DataFrame({'Age':age,'MonthlyIncome':rng.integers(2000,18000,n),
        'YearsAtCompany':rng.integers(0,15,n),'TotalWorkingYears':rng.integers(15,30,n),
        'OverTime':overtime,'Department':rng.choice(['Sales','Research & Development'],n),
        'JobRole':rng.choice(['Manager','Research Scientist'],n),'JobSatisfaction':rng.integers(1,5,n),
        'Attrition':np.where(rng.random(n)<chance,'Yes','No'),
        'Gender':rng.choice(['Female','Male'],n),'MaritalStatus':['Single']*n,
        'EmployeeNumber':np.arange(n),'Over18':['Y']*n,'EmployeeCount':np.ones(n)})


class PredictionTests(unittest.TestCase):
    def setUp(self): self.frame=fixture()

    def test_split_is_stratified_disjoint_and_reproducible(self):
        a=AttritionPredictor(self.frame).fit();b=AttritionPredictor(self.frame).fit()
        self.assertEqual(len(a.train_indices),160);self.assertEqual(len(a.test_indices),40)
        self.assertFalse(set(a.train_indices)&set(a.test_indices))
        self.assertEqual(set(a.train_indices)|set(a.test_indices),set(range(200)))
        self.assertLess(abs(a.y_train.mean()-a.y_test.mean()),.03)
        np.testing.assert_array_equal(a.test_indices,b.test_indices)
        np.testing.assert_allclose(a.probabilities,b.probabilities)

    def test_original_ibm_size_and_missing_targets(self):
        frame=fixture(1470);model=AttritionPredictor(frame).fit()
        self.assertEqual(model.metadata['entrenamiento'],1176)
        self.assertEqual(model.metadata['prueba'],294)
        frame.loc[0,'Attrition']=None
        model=AttritionPredictor(frame).fit()
        self.assertEqual(model.metadata['objetivo_ausente_excluido'],1)
        self.assertNotIn(0,set(model.train_indices)|set(model.test_indices))

    def test_preprocessing_and_fit_do_not_leak_test_values(self):
        a=AttritionPredictor(self.frame).fit();changed=self.frame.copy()
        changed.loc[a.test_indices,'MonthlyIncome']=1e9
        changed.loc[a.test_indices,'JobRole']='OnlyInTest'
        b=AttritionPredictor(changed).fit()
        np.testing.assert_allclose(a.models['lineal']['classifier'].coef_,b.models['lineal']['classifier'].coef_)
        np.testing.assert_allclose(a.pipeline['preprocess'].named_transformers_['numeric']['imputer'].statistics_,
                                   b.pipeline['preprocess'].named_transformers_['numeric']['imputer'].statistics_)
        self.assertNotIn('OnlyInTest',b.categories['JobRole'])

    def test_target_identifiers_sensitive_columns_excluded(self):
        model=AttritionPredictor(self.frame).fit()
        for name in ['Attrition','EmployeeNumber','Gender','MaritalStatus','EmployeeCount','Over18']:
            self.assertNotIn(name,model.features)

    def test_metrics_baseline_and_confusion_counts(self):
        evidence=AttritionPredictor(self.frame).evaluate();m=evidence['metricas_prueba']
        self.assertEqual(sum(m['matriz_confusion'].values()),40)
        for key in ['accuracy','precision_Yes','recall_Yes','f1_Yes','roc_auc','average_precision','brier']:
            self.assertTrue(0<=m[key]<=1)
        self.assertEqual(evidence['baseline_prior']['roc_auc'],.5)
        self.assertIn('20% reservado',render_prediction(evidence))

    def test_single_class_or_too_few_records_rejected(self):
        self.frame['Attrition']='No'
        for frame in [self.frame,fixture(20)]:
            with self.assertRaises(ValueError):AttritionPredictor(frame).fit()

    def test_cohort_uses_holdout_only_and_preserves_fit(self):
        model=AttritionPredictor(self.frame).fit();pipe=model.pipeline
        evidence=model.cohort([{'variable':'OverTime','op':'eq','valor':'Yes'}],['Department'],1)
        expected=model.test_frame['OverTime'].eq('Yes').sum()
        self.assertEqual(sum(r['n_prueba'] for r in evidence['resultados']),expected)
        self.assertIs(pipe,model.pipeline)
        for r in evidence['resultados']:
            self.assertTrue(0<=r['probabilidad_media_Yes']<=1)
            self.assertLessEqual(r['Yes_estimados_umbral_05'],r['n_prueba'])

    def test_target_filter_and_identifier_group_rejected(self):
        model=AttritionPredictor(self.frame)
        with self.assertRaises(ValueError):model.cohort([{'variable':'Attrition','op':'eq','valor':'Yes'}])
        with self.assertRaises(ValueError):model.cohort(groups=['EmployeeNumber'])

    def test_profile_uses_only_completed_features(self):
        model=AttritionPredictor(self.frame)
        values={'Age':30,'MonthlyIncome':3500,'YearsAtCompany':None,'OverTime':''}
        result=model.profile(values)
        self.assertEqual(result['variables_utilizadas'],['Age','MonthlyIncome'])
        self.assertEqual(result['campos_imputados_desde_entrenamiento'],[])
        self.assertEqual(result['perfil_completado'],{'Age':30,'MonthlyIncome':3500})
        self.assertIn('OverTime',result['variables_omitidas'])
        self.assertEqual(result['configuracion']['prueba'],40)
        self.assertTrue(0<=result['metricas_prueba']['roc_auc']<=1)
        with self.assertRaises(ValueError):model.profile({'Age':30},impute=True)

    def test_omitted_features_cannot_affect_prediction(self):
        a=AttritionPredictor(self.frame);ra=a.profile({'Age':30})
        changed=self.frame.copy();changed['MonthlyIncome']=1e20;changed['OverTime']='InvalidButUnused'
        b=AttritionPredictor(changed);rb=b.profile({'Age':30})
        self.assertEqual(ra['probabilidad_estimada_Yes'],rb['probabilidad_estimada_Yes'])
        np.testing.assert_allclose(a.profile_models[('Age',)].models['lineal']['classifier'].coef_,b.profile_models[('Age',)].models['lineal']['classifier'].coef_)
        base=AttritionPredictor(self.frame).fit()
        np.testing.assert_array_equal(base.test_indices,a.profile_models[('Age',)].test_indices)

    def test_partial_categorical_only_and_zero_values(self):
        model=AttritionPredictor(self.frame)
        for values in [{'Gender':'Female'},{'MaritalStatus':'Single'},{'YearsAtCompany':0}]:
            result=model.profile(values)
            self.assertEqual(result['perfil_completado'],values)
            self.assertTrue(0<=result['probabilidad_estimada_Yes']<=1)
        self.assertEqual(model.profile({'Gender':'Female'})['variables_utilizadas'],['Gender'])

    def test_blank_profile_and_unobserved_training_feature_rejected(self):
        model=AttritionPredictor(self.frame)
        for values in [{},{'Age':None},{'Age':''},{'Age':float('nan')}]:
            with self.assertRaises(ValueError):model.profile(values)
        frame=self.frame.copy();frame['Age']=np.nan
        with self.assertRaisesRegex(ValueError,'Sin datos de entrenamiento'):
            AttritionPredictor(frame).profile({'Age':30})

    def test_subset_training_does_not_leak_holdout_values(self):
        a=AttritionPredictor(self.frame,features=['Age']).fit()
        frame=self.frame.copy();frame.loc[a.test_indices,'Age']=999
        b=AttritionPredictor(frame,features=['Age']).fit()
        np.testing.assert_allclose(a.models['lineal']['classifier'].coef_,b.models['lineal']['classifier'].coef_)

    def test_full_profile_and_invalid_inputs(self):
        model=AttritionPredictor(self.frame).fit()
        full=dict(model.defaults)
        evidence=model.profile(full)
        self.assertEqual(evidence['campos_imputados_desde_entrenamiento'],[])
        for overrides in [{'EmployeeNumber':8},{'Attrition':'Yes'},{'Age':float('inf')},
                          {'OverTime':'Maybe'},{'Age':2,'YearsAtCompany':20}]:
            with self.subTest(overrides=overrides),self.assertRaises(ValueError):model.profile({**full,**overrides})

    def test_plan_modes_do_not_silently_ignore_filters(self):
        model=AttritionPredictor(self.frame)
        with self.assertRaises(ValueError):model.execute({'accion':'prediccion','modo':'evaluar','filtros':[{'variable':'Age','op':'gt','valor':30}]})

    def test_agent_authentication_before_fit_and_model_call(self):
        client=FakeClient('{}');predictor=AttritionPredictor(self.frame)
        with self.assertRaises(PermissionError):AnalyticsAgent(client,predictor).run('Entrena el modelo',self.frame)
        self.assertIsNone(predictor.pipeline);self.assertEqual(client.calls,[])

    def test_exact_evaluation_and_profile_routes(self):
        client=FakeClient('{}');predictor=AttritionPredictor(self.frame)
        result=AnalyticsAgent(client,predictor).run('Entrena y evalúa el modelo de attrition con 80/20',self.frame,authorized=True)
        self.assertEqual(result['evidencia']['modo'],'evaluar')
        result=AnalyticsAgent(client,predictor).run('Estima la probabilidad para un perfil ficticio: Age=30, MonthlyIncome=3500, YearsAtCompany=3, OverTime=Yes.',self.frame,authorized=True)
        self.assertEqual(result['evidencia']['modo'],'perfil')
        self.assertEqual(len(client.calls),2)

    def test_predictive_intent(self):
        for q in ['Predice attrition por Department','¿Cuál es el ROC-AUC?','Evalúa el modelo','Entrena con 80/20','¿Cuál es el recall?']:
            self.assertTrue(is_predictive(q),q)
        self.assertFalse(is_predictive('¿Cuál es la edad promedio?'))

    def test_no_dataframe_mutation(self):
        before=self.frame.copy(deep=True)
        AttritionPredictor(self.frame).evaluate()
        pd.testing.assert_frame_equal(self.frame,before)

    def test_exact_cohort_routes_preserve_scope(self):
        client=FakeClient('{}');predictor=AttritionPredictor(self.frame)
        plan,evidence=run_prediction(client,'Predice attrition por Department en el conjunto de prueba.',self.frame,predictor)
        self.assertEqual(plan['grupos'],['Department'])
        self.assertEqual(evidence['modo'],'cohorte')
        plan,evidence=run_prediction(client,'Predice attrition para OverTime = Yes en el conjunto de prueba.',self.frame,predictor)
        self.assertEqual(plan['filtros'],[{'variable':'OverTime','op':'eq','valor':'Yes'}])
        self.assertEqual(client.calls,[])

    def test_model_cannot_replace_cohort_request_with_evaluation(self):
        client=FakeClient('{"accion":"prediccion","modo":"evaluar"}')
        with self.assertRaisesRegex(ValueError,'modo cohorte'):
            run_prediction(client,'Calcula la probabilidad de rotación según departamento',self.frame)
        self.assertEqual(len(client.calls),2)
