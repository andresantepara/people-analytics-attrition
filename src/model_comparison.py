"""Tres modelos, selección por CV del entrenamiento y prueba intacta."""
import numpy as np

MODEL_NAMES={'lineal':'Regresión lineal','arbol':'Árbol de decisión','bosque':'Random forest'}


def predict_scores(model, X):
    raw=model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.predict(X)
    raw=np.asarray(raw,dtype=float)
    if not np.isfinite(raw).all(): raise ValueError('El modelo produjo valores no finitos.')
    return raw,np.clip(raw,0,1)


def compare_models(preprocessor, X, y, X_test, y_test, metric_fn, sk, seed=42):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    estimators={'lineal':LinearRegression(),
        'arbol':DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,random_state=seed),
        'bosque':RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=3,random_state=seed,n_jobs=1)}
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=seed).split(X,y))
    models={};cv={};rows=[]
    for key,estimator in estimators.items():
        template=Pipeline([('preprocess',clone(preprocessor)),('classifier',estimator)])
        fold_metrics=[]
        for train,val in folds:
            fitted=clone(template).fit(X.iloc[train],y.iloc[train])
            _,scores=predict_scores(fitted,X.iloc[val])
            fold_metrics.append(metric_fn(y.iloc[val],scores,sk))
        cv[key]={name:float(np.mean([m[name] for m in fold_metrics])) for name in ['average_precision','f1_Yes','roc_auc']}
        cv[key]['ap_std']=float(np.std([m['average_precision'] for m in fold_metrics]))
        models[key]=clone(template).fit(X,y)
    # Decide before evaluating any model on the held-out 20%.
    selected=max(estimators,key=lambda k:(cv[k]['average_precision'],cv[k]['f1_Yes'],cv[k]['roc_auc']))
    metrics={}
    for key,model in models.items():
        raw,scores=predict_scores(model,X_test)
        m=metric_fn(y_test,scores,sk);metrics[key]=m
        rows.append({'Modelo':MODEL_NAMES[key],'Recomendado':key==selected,
            'AP validación (80%)':cv[key]['average_precision'],'DE AP validación':cv[key]['ap_std'],
            'F1 validación (80%)':cv[key]['f1_Yes'],
            'Exactitud prueba':m['accuracy'],'Precisión Yes prueba':m['precision_Yes'],
            'Recall Yes prueba':m['recall_Yes'],'F1 Yes prueba':m['f1_Yes'],
            'ROC-AUC prueba':m['roc_auc'],'AP prueba':m['average_precision'],'Brier prueba':m['brier'],
            'Salidas fuera de [0,1] en prueba':int(((raw<0)|(raw>1)).sum())})
    explanation=(f"Se recomienda {MODEL_NAMES[selected]} entre estos tres modelos por la mayor average precision (AP) media en validación cruzada estratificada de 3 particiones dentro del 80% de entrenamiento: {cv[selected]['average_precision']:.3f}. "
        'AP resume precisión y recuperación de la clase Yes a distintos umbrales. Desempates: F1, luego ROC-AUC de validación; si persiste, orden fijo de modelos. El 20% de prueba no intervino en la elección. La recomendación es provisional para este dataset y estas variables, no una garantía de superioridad.')
    return models,metrics,cv,selected,rows,explanation
