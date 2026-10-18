"""Predicción académica de Attrition: split 80/20 fijo, sin fuga de preprocesado."""
import math
import numpy as np
import pandas as pd
from .analytics import clean, filtered, keys
from .model_comparison import compare_models, predict_scores, MODEL_NAMES

NUMERIC_FEATURES = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
    'HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate',
    'NumCompaniesWorked','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',
    'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance',
    'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
CATEGORY_FEATURES = ['BusinessTravel','Department','EducationField','JobRole','OverTime']
FEATURES = NUMERIC_FEATURES + CATEGORY_FEATURES
SIMULATION_FEATURES = FEATURES + ['Gender','MaritalStatus']
CORE_PROFILE = ['Age','MonthlyIncome','YearsAtCompany','OverTime']
SEED = 42
THRESHOLD = .5


def imports():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,
            recall_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix)
    except ImportError as exc:
        raise ValueError('Falta scikit-learn. Instala las dependencias actualizadas con python -m pip install -r requirements.txt.') from exc
    return locals()


class AttritionPredictor:
    def __init__(self, df, features=None):
        self.df = df.copy(deep=True).reset_index(drop=True)
        self.pipeline = None
        self.selected_features = features
        self.profile_models = {}

    def fit(self):
        if self.pipeline is not None: return self
        sk = imports()
        if 'Attrition' not in self.df: raise ValueError('Falta Attrition para entrenamiento supervisado.')
        if set(self.df['Attrition'].dropna().unique()) - {'Yes','No'}: raise ValueError('Attrition debe ser Yes/No.')
        labelled = self.df[self.df['Attrition'].isin(['Yes','No'])].copy()
        y = labelled['Attrition'].eq('Yes').astype(int)
        if len(labelled)<50 or len(y.unique())!=2 or y.value_counts().min()<10:
            raise ValueError('Se requieren al menos 50 registros etiquetados y 10 de cada clase para la demostración 80/20.')
        features = [c for c in (FEATURES if self.selected_features is None else self.selected_features) if c in labelled]
        if not features or set(features)-set(SIMULATION_FEATURES): raise ValueError('Selecciona predictores válidos.')
        if self.selected_features is None and not set(CORE_PROFILE)<=set(features): raise ValueError('Faltan variables básicas para el modelo: '+str(CORE_PROFILE))
        X=labelled[features].copy()
        for name in features:
            if name in NUMERIC_FEATURES:
                converted=pd.to_numeric(X[name],errors='coerce')
                if (X[name].notna() & converted.isna()).any() or converted.isin([np.inf,-np.inf]).any():
                    raise ValueError(f'Valores numéricos inválidos en {name}.')
                if (converted.dropna()<0).any(): raise ValueError(f'{name} contiene valores negativos.')
                X[name]=converted.astype(float)
            else:
                X[name]=X[name].astype(object).where(X[name].notna(),np.nan)
        train_idx,test_idx=sk['train_test_split'](X.index.to_numpy(),test_size=.2,random_state=SEED,stratify=y)
        train=X.loc[train_idx];test=X.loc[test_idx]
        # Feature exclusion is decided on the training subset only.
        self.features=[c for c in features if train[c].notna().any()]
        self.dropped_empty=[c for c in features if c not in self.features]
        if self.selected_features is None and not set(CORE_PROFILE)<=set(self.features):
            raise ValueError('Una variable básica del modelo está completamente vacía en entrenamiento.')
        if self.selected_features is not None and self.dropped_empty:
            raise ValueError('Sin datos de entrenamiento para: '+', '.join(self.dropped_empty))
        self.num=[c for c in self.features if c in NUMERIC_FEATURES]
        self.cat=[c for c in self.features if c not in NUMERIC_FEATURES]
        if not self.features: raise ValueError('No hay predictores observados en entrenamiento.')
        stages=[]
        if self.num:
            stages.append(('numeric',sk['Pipeline']([
                ('imputer',sk['SimpleImputer'](strategy='median',keep_empty_features=True)),
                ('scale',sk['StandardScaler']())]),self.num))
        if self.cat:
            stages.append(('categorical',sk['Pipeline']([
                ('imputer',sk['SimpleImputer'](strategy='most_frequent',keep_empty_features=True)),
                ('encode',sk['OneHotEncoder'](handle_unknown='ignore'))]),self.cat))
        self.train_indices=train_idx;self.test_indices=test_idx
        self.X_train=train[self.features].copy();self.X_test=test[self.features].copy()
        self.y_train=y.loc[train_idx];self.y_test=y.loc[test_idx]
        self.test_frame=labelled.loc[test_idx].copy()
        self.models,self.model_metrics,self.cv_metrics,self.selected_model,self.comparison,self.recommendation=compare_models(
            sk['ColumnTransformer'](stages),self.X_train,self.y_train,self.X_test,self.y_test,self._metrics,sk,SEED)
        model=self.models[self.selected_model]
        _,self.probabilities=predict_scores(model,self.X_test)
        self.predictions=(self.probabilities>=THRESHOLD).astype(int)
        dummy=sk['DummyClassifier'](strategy='prior').fit(self.X_train,self.y_train)
        baseline_prob=dummy.predict_proba(self.X_test)[:,1]
        self.baseline=self._metrics(self.y_test,baseline_prob,sk)
        self.metrics=self._metrics(self.y_test,self.probabilities,sk)
        self.model=model
        self.defaults={}
        for c in self.num: self.defaults[c]=float(self.X_train[c].median())
        for c in self.cat: self.defaults[c]=self.X_train[c].mode().iloc[0]
        self.ranges={c:(float(self.X_train[c].min()),float(self.X_train[c].max())) for c in self.num}
        self.categories={c:set(self.X_train[c].dropna().unique()) for c in self.cat}
        train_hash=set(pd.util.hash_pandas_object(self.X_train,index=False))
        duplicate_overlap=int(pd.util.hash_pandas_object(self.X_test,index=False).isin(train_hash).sum())
        self.metadata={'modelo':MODEL_NAMES[self.selected_model],'modelo_id':self.selected_model,
            'seleccion':'Mayor average precision media en CV estratificada de 3 folds del 80%; desempates F1 y ROC-AUC.',
            'interpretacion_salida':'La regresión lineal ajusta Yes=1/No=0 y recorta sus salidas a [0,1]; es una puntuación de referencia, no una probabilidad calibrada. Árbol y bosque tampoco tienen calibración externa.',
            'semilla':SEED,'umbral':THRESHOLD,'objetivo':'Attrition: Yes=1, No=0',
            'registros_originales':len(self.df),'registros_etiquetados':len(labelled),
            'objetivo_ausente_excluido':len(self.df)-len(labelled),
            'entrenamiento':len(train_idx),'prueba':len(test_idx),
            'reparto':'80% entrenamiento / 20% prueba, estratificado; redondeo si es necesario',
            'clases_entrenamiento':{'Yes':int(self.y_train.sum()),'No':int((1-self.y_train).sum())},
            'clases_prueba':{'Yes':int(self.y_test.sum()),'No':int((1-self.y_test).sum())},
            'variables':self.features,'columnas_vacias_excluidas':self.dropped_empty,
            'exclusiones':[c for c in self.df if c not in self.features],
            'coincidencias_predictores_train_test':duplicate_overlap,
            'preprocesado':'Imputación, escalado y one-hot ajustados únicamente con el 80% de entrenamiento.',
            'limites':['Dataset sintético; no pronóstico temporal ni tasa anual.',
                'Probabilidades sin calibración externa; estimaciones, no certezas.',
                'No se seleccionan hiperparámetros ni umbral usando el 20% de prueba.',
                'Una sola partición no garantiza estabilidad ni desempeño en otras empresas.',
                'El modelo no ha sido auditado para equidad; atributos demográficos y proxies pueden introducir sesgos.',
                'Uso académico: no ranking ni decisiones sobre trabajadores reales.']}
        self.pipeline=model  # Publish cache only after the full fit/evaluation succeeds.
        return self

    @staticmethod
    def _metrics(y,prob,sk):
        pred=(prob>=THRESHOLD).astype(int)
        tn,fp,fn,tp=sk['confusion_matrix'](y,pred,labels=[0,1]).ravel()
        return clean({'accuracy':sk['accuracy_score'](y,pred),
            'balanced_accuracy':sk['balanced_accuracy_score'](y,pred),
            'precision_Yes':sk['precision_score'](y,pred,zero_division=0),
            'recall_Yes':sk['recall_score'](y,pred,zero_division=0),
            'f1_Yes':sk['f1_score'](y,pred,zero_division=0),
            'roc_auc':sk['roc_auc_score'](y,prob),
            'average_precision':sk['average_precision_score'](y,prob),
            'brier':sk['brier_score_loss'](y,prob),
            'matriz_confusion':{'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}})

    def evaluate(self):
        self.fit()
        return clean({'accion':'prediccion','modo':'evaluar','configuracion':self.metadata,
                      'metricas_prueba':self.metrics,'baseline_prior':self.baseline,
                      'comparacion_modelos':self.comparison,'recomendacion':self.recommendation})

    def cohort(self, filters=None, groups=None, min_group=5):
        self.fit();filters=filters or [];groups=groups or []
        if type(min_group) is not int or min_group<1: raise ValueError('Umbral de grupos inválido.')
        if not isinstance(groups,list) or len(groups)>2 or len(set(groups))!=len(groups):
            raise ValueError('Hasta dos agrupaciones distintas.')
        for g in groups:
            if g not in self.features or self.test_frame[g].nunique()>50:
                raise ValueError('Agrupación predictiva no disponible o demasiado extensa.')
        for f in filters:
            if not isinstance(f,dict) or f.get('variable') not in self.features:
                raise ValueError('Filtra con predictores disponibles, nunca con Attrition o identificadores.')
        frame=filtered(self.test_frame,filters,self.df).copy()
        frame['_prob']=pd.Series(self.probabilities,index=self.test_indices).loc[frame.index]
        frame['_pred']=pd.Series(self.predictions,index=self.test_indices).loc[frame.index]
        rows=[];omitted=0
        for group,sub in frame.groupby(groups,dropna=False,observed=True) if groups else [((),frame)]:
            if len(sub)<min_group: omitted+=1;continue
            group=group if isinstance(group,tuple) else (group,)
            rows.append({**{g:None if pd.isna(v) else v for g,v in zip(groups,group)},
                'n_prueba':len(sub),'probabilidad_media_Yes':float(sub['_prob'].mean()),
                'Yes_estimados_umbral_05':int(sub['_pred'].sum()),
                'porcentaje_Yes_estimado':float(100*sub['_pred'].mean()),
                'Yes_observados':int(sub['Attrition'].eq('Yes').sum()),
                'porcentaje_Yes_observado':float(100*sub['Attrition'].eq('Yes').mean())})
        return clean({'accion':'prediccion','modo':'cohorte','configuracion':self.metadata,
            'alcance':'Solo el 20% de prueba; no se reentrena con los filtros ni se devuelven filas individuales.',
            'filtros':filters,'grupos':groups,'filas_prueba_filtradas':len(frame),
            'resultados':rows[:100],'truncado':len(rows)>100,'grupos_omitidos':omitted})

    def profile(self, values, impute=False):
        if type(impute) is not bool: raise ValueError('imputar debe ser booleano.')
        if impute:
            raise ValueError('La simulación ya no rellena campos: omite la solicitud de imputación y proporciona solo las variables que deseas usar.')
        if not isinstance(values,dict): raise ValueError('El perfil debe ser un diccionario de valores.')
        values={c:v for c,v in values.items() if v is not None and v != ''}
        allowed=[c for c in SIMULATION_FEATURES if c in self.df]
        if not values or set(values)-set(allowed):
            raise ValueError('Completa al menos un predictor válido; no uses Attrition, identificadores ni constantes administrativas.')
        for c,v in values.items():
            if c in NUMERIC_FEATURES and (type(v) not in (int,float) or not math.isfinite(v) or v<0):
                raise ValueError(f'{c} requiere un número finito no negativo.')
            if c not in NUMERIC_FEATURES and not isinstance(v,str):
                raise ValueError(f'{c} requiere una categoría válida.')
        if 'Age' in values and 'YearsAtCompany' in values and values['YearsAtCompany']>values['Age']:
            raise ValueError('La antigüedad no puede superar la edad del perfil.')
        if 'TotalWorkingYears' in values and 'YearsAtCompany' in values and values['YearsAtCompany']>values['TotalWorkingYears']:
            raise ValueError('YearsAtCompany no puede superar TotalWorkingYears.')
        selected=tuple(c for c in allowed if c in values)
        if selected not in self.profile_models:
            subset=AttritionPredictor(self.df,features=list(selected)).fit()
            if len(self.profile_models)>=8: self.profile_models.pop(next(iter(self.profile_models)))
            self.profile_models[selected]=subset
        subset=self.profile_models[selected]
        outside=[]
        for c,v in values.items():
            if c in subset.num:
                low,high=subset.ranges[c]
                if not low<=v<=high: outside.append(c)
            elif v not in subset.categories[c]:
                raise ValueError(f'Categoría de {c} no observada en entrenamiento.')
        profile_frame=pd.DataFrame([values],columns=subset.features)
        profile_comparison=[]
        for key,model in subset.models.items():
            raw,scores=predict_scores(model,profile_frame)
            profile_comparison.append({'Modelo':MODEL_NAMES[key],'Recomendado':key==subset.selected_model,
                'Estimación Yes (%)':float(100*scores[0]),'Salida original':float(raw[0]),
                'Clase (umbral 0.5)':'Yes' if scores[0]>=THRESHOLD else 'No'})
        prob=float(predict_scores(subset.pipeline,profile_frame)[1][0])
        return clean({'accion':'prediccion','modo':'perfil','configuracion':subset.metadata,
            'perfil_sintetico_proporcionado':values,'perfil_completado':values,
            'variables_utilizadas':subset.features,'variables_omitidas':[c for c in allowed if c not in values],
            'campos_imputados_desde_entrenamiento':[],'fuera_de_rango_entrenamiento':outside,
            'metricas_prueba':subset.metrics,'baseline_prior':subset.baseline,
            'comparacion_modelos':subset.comparison,'comparacion_perfil':profile_comparison,'recomendacion':subset.recommendation,
            'probabilidad_estimada_Yes':prob,'clase_al_umbral_05':'Yes' if prob>=THRESHOLD else 'No',
            'advertencia':'Modelo ajustado solo con las variables proporcionadas, con su propia evaluación 80/20. Los campos vacíos del perfil no se imputan. Los valores ausentes del CSV en columnas seleccionadas se imputan usando solo entrenamiento. Simulación académica sintética, no decisiones sobre empleados reales.'})

    def execute(self, plan, min_group=5):
        keys(plan,['accion','modo','filtros','grupos','perfil','imputar'],['accion','modo'])
        if plan['accion']!='prediccion':raise ValueError('Acción predictiva inválida.')
        mode=plan['modo']
        if mode not in ['evaluar','cohorte','perfil']: raise ValueError('Modo predictivo inválido.')
        valid={'evaluar':set(),'cohorte':{'filtros','grupos'},'perfil':{'perfil','imputar'}}[mode]
        if any(k not in valid|{'accion','modo'} and v not in ([],{},False,None) for k,v in plan.items()):
            raise ValueError('El plan mezcla modos predictivos incompatibles.')
        if mode=='evaluar': return self.evaluate()
        if mode=='cohorte': return self.cohort(plan.get('filtros',[]),plan.get('grupos',[]),min_group)
        return self.profile(plan.get('perfil',{}),plan.get('imputar',False))
