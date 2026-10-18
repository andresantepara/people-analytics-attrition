import unittest
import pandas as pd
from src.analytics import calculate
from src.reporting import render_answer

class NaturalAnswersTests(unittest.TestCase):
    def test_filtered_mean_includes_scope_and_units(self):
        frame=pd.DataFrame({'Age':[20,40,80],'Attrition':['Yes','Yes','No']})
        evidence=calculate(frame,{'accion':'estadisticas','filtros':[{'variable':'Attrition','op':'eq','valor':'Yes'}],
            'metricas':[{'id':'edad','variable':'Age','estadistico':'mean'}]},1)
        answer=render_answer(evidence)
        self.assertTrue(answer.startswith('La edad promedio entre quienes salieron es de **30 años**.'))
        self.assertNotIn('80',answer)

    def test_no_data_is_not_reported_as_zero(self):
        plan={'accion':'estadisticas','metricas':[{'id':'promedio','variable':'Age','estadistico':'mean'}]}
        empty=render_answer(calculate(pd.DataFrame({'Age':pd.Series([None],dtype='float64')}),plan,1))
        zero=render_answer(calculate(pd.DataFrame({'Age':[0]}),plan,1))
        self.assertIn('No hay suficientes datos',empty)
        self.assertIn('**0 años**',zero)

    def test_difference_keeps_cohort_order_and_sign(self):
        frame=pd.DataFrame({'MonthlyIncome':[1000,3000],'Attrition':['Yes','No']})
        metrics=[{'id':key,'variable':'MonthlyIncome','estadistico':'mean','filtros':[{'variable':'Attrition','op':'eq','valor':val}]} for key,val in [('salen','Yes'),('permanecen','No')]]
        answer=render_answer(calculate(frame,{'accion':'estadisticas','metricas':metrics,
            'calculos':[{'id':'diferencia','op':'subtract','a':'salen','b':'permanecen'}]},1))
        self.assertIn('**-2000**',answer)
        self.assertIn('Al restar el promedio de ingreso mensual entre quienes permanecen del promedio de ingreso mensual entre quienes salieron',answer)
