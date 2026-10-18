import unittest
import numpy as np
from test_prediction import fixture
from src.prediction import AttritionPredictor, imports
from src.model_comparison import compare_models, predict_scores


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=AttritionPredictor(fixture()).fit()

    def test_three_models_table_and_selection_rule(self):
        model=self.model
        self.assertEqual(set(model.models),{'lineal','arbol','bosque'})
        rows=model.evaluate()['comparacion_modelos']
        self.assertEqual(len(rows),3)
        self.assertEqual(sum(r['Recomendado'] for r in rows),1)
        expected=max(model.cv_metrics,key=lambda k:(model.cv_metrics[k]['average_precision'],model.cv_metrics[k]['f1_Yes'],model.cv_metrics[k]['roc_auc']))
        self.assertEqual(model.selected_model,expected)
        for key in model.models:
            self.assertEqual(sum(model.model_metrics[key]['matriz_confusion'].values()),40)
            self.assertTrue(0<=model.cv_metrics[key]['average_precision']<=1)
        np.testing.assert_allclose(model.probabilities,predict_scores(model.pipeline,model.X_test)[1])

    def test_changed_holdout_cannot_change_selection_or_cv(self):
        m=self.model
        changed=m.X_test.copy();changed['Age']=999
        models,metrics,cv,chosen,rows,why=compare_models(m.pipeline['preprocess'],m.X_train,m.y_train,changed,1-m.y_test,m._metrics,imports())
        self.assertEqual(cv,m.cv_metrics)
        self.assertEqual(chosen,m.selected_model)
        for key in models:
            np.testing.assert_allclose(predict_scores(models[key],m.X_train)[1],predict_scores(m.models[key],m.X_train)[1])

    def test_linear_clipping_is_explicit(self):
        class LinearStub:
            def predict(self,X): return np.array([-0.4,0.4,1.4])
        raw,score=predict_scores(LinearStub(),None)
        np.testing.assert_allclose(raw,[-.4,.4,1.4])
        np.testing.assert_allclose(score,[0,.4,1])
        self.assertIn('recorta',self.model.metadata['interpretacion_salida'])

    def test_profile_has_three_predictions_with_selected_features_only(self):
        report=self.model.profile({'Age':30})
        self.assertEqual(len(report['comparacion_perfil']),3)
        self.assertEqual(report['variables_utilizadas'],['Age'])
        selected=next(r for r in report['comparacion_perfil'] if r['Recomendado'])
        self.assertAlmostEqual(selected['Estimación Yes (%)']/100,report['probabilidad_estimada_Yes'])
        self.assertEqual(report['campos_imputados_desde_entrenamiento'],[])
