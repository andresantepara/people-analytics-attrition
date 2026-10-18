"""Segmentación descriptiva de probabilidades del 20% de prueba."""
import numpy as np
import pandas as pd
from .prediction import SIMULATION_FEATURES, NUMERIC_FEATURES


def segment_labels(predictor, variable):
    test=predictor.test_frame
    series=test[variable]
    detail={}
    if variable in NUMERIC_FEATURES:
        train=pd.to_numeric(predictor.df.loc[predictor.train_indices,variable],errors='coerce').dropna()
        train=train[np.isfinite(train)]
        numeric=pd.to_numeric(series,errors='coerce')
        labels=pd.Series('Sin dato',index=test.index,dtype=object)
        if train.empty:
            labels.loc[numeric.notna()]='Sin rango de entrenamiento'
            detail['Sin rango de entrenamiento']='No se pueden definir rangos con el 80% de entrenamiento.'
        elif train.nunique()==1:
            val=float(train.iloc[0])
            labels.loc[numeric.eq(val)]='Valor único en entrenamiento'
            labels.loc[numeric.notna() & numeric.ne(val)]='Otro valor fuera del entrenamiento'
            detail['Valor único en entrenamiento']=f'{variable} = {val:g}'
            detail['Otro valor fuera del entrenamiento']=f'{variable} distinto de {val:g}'
        else:
            cuts=sorted(set(float(x) for x in train.quantile([1/3,2/3])))
            names=['Rango bajo','Rango medio','Rango alto'] if len(cuts)==2 else ['Rango bajo','Rango alto']
            edges=[-np.inf]+cuts+[np.inf]
            bins=pd.cut(numeric,bins=edges,labels=names,include_lowest=True)
            labels.loc[numeric.notna()]=bins.loc[numeric.notna()].astype(object)
            for i,name in enumerate(names):
                lo,hi=edges[i:i+2]
                detail[name]=(f'{variable} ≤ {hi:g}' if i==0 else f'{variable} > {lo:g}' if i==len(names)-1 else f'{lo:g} < {variable} ≤ {hi:g}')
    else:
        labels=series.map(lambda x:'Sin dato' if pd.isna(x) else 'Categoría: '+str(x))
        detail={label:f'{variable} = {label.removeprefix("Categoría: ")}' for label in labels.unique() if label!='Sin dato'}
    return labels, detail


def segmented_probabilities(predictor, variables, min_group=5):
    if not isinstance(variables,list) or not variables or len(set(variables))!=len(variables):
        raise ValueError('Selecciona una o más variables distintas.')
    if any(v not in SIMULATION_FEATURES or v not in predictor.df for v in variables):
        raise ValueError('Usa variables de segmentación disponibles; no Attrition ni identificadores.')
    if type(min_group) is not int or min_group<1: raise ValueError('El mínimo de registros debe ser positivo.')
    predictor.fit()
    test=predictor.test_frame
    prob=pd.Series(predictor.probabilities,index=predictor.test_indices)
    rows=[];omitted=0
    for variable in variables:
        labels,detail=segment_labels(predictor,variable)
        for segment in labels.unique():
            mask=labels.eq(segment);n=int(mask.sum())
            if n<min_group:
                omitted+=1;continue
            rows.append({'Variable':variable,'Segmento':segment,
                'Condición':detail.get(segment,f'{variable} ausente'),
                'Registros de prueba':n,
                'Probabilidad media Attrition Yes (%)':float(100*prob.loc[mask[mask].index].mean()),
                'Attrition Yes observado (%)':float(100*test.loc[mask,'Attrition'].eq('Yes').mean())})
    return {'tabla':pd.DataFrame(rows,columns=['Variable','Segmento','Condición','Registros de prueba',
        'Probabilidad media Attrition Yes (%)','Attrition Yes observado (%)']),
        'celdas_omitidas':omitted,'configuracion':predictor.metadata}


def probability_heatmap(table, variables, segments):
    import plotly.graph_objects as go
    data=table[table['Variable'].isin(variables)&table['Segmento'].isin(segments)]
    matrix=data.pivot(index='Segmento',columns='Variable',values='Probabilidad media Attrition Yes (%)').reindex(index=segments,columns=variables)
    counts=data.pivot(index='Segmento',columns='Variable',values='Registros de prueba').reindex(index=segments,columns=variables)
    conditions=data.pivot(index='Segmento',columns='Variable',values='Condición').reindex(index=segments,columns=variables)
    custom=np.empty((len(segments),len(variables),2),dtype=object)
    custom[:,:,0]=counts.to_numpy();custom[:,:,1]=conditions.fillna('').to_numpy()
    fig=go.Figure(go.Heatmap(z=matrix.to_numpy(),x=variables,y=segments,
        zmin=0,zmax=100,colorscale='YlOrRd',colorbar={'title':'P(Yes) %'},
        texttemplate='%{z:.1f}%',customdata=custom,hoverongaps=False,
        hovertemplate='%{x}<br>%{y}<br>%{customdata[1]}<br>Probabilidad media: %{z:.1f}%<br>Registros de prueba: %{customdata[0]:.0f}<extra></extra>'))
    fig.update_layout(title='Probabilidad media estimada de Attrition = Yes',
        xaxis_title='Variables de segmentación',yaxis_title='Segmentos',
        height=max(420,35*len(segments)+180),xaxis={'tickangle':-40},yaxis={'autorange':'reversed'})
    return fig


def top_combinations(predictor, variables, min_group=5, segments=None):
    """Rank observed two-variable intersections by held-out mean probability."""
    from itertools import combinations
    if not isinstance(variables,list) or len(variables)<2 or len(set(variables))!=len(variables):
        raise ValueError('Selecciona al menos dos variables distintas para cruzarlas.')
    if any(v not in SIMULATION_FEATURES or v not in predictor.df for v in variables):
        raise ValueError('Variables de cruce no disponibles; no uses Attrition ni identificadores.')
    if type(min_group) is not int or min_group<1: raise ValueError('Mínimo de registros inválido.')
    if segments is not None and (not isinstance(segments,list) or any(not isinstance(x,str) for x in segments)):
        raise ValueError('Los segmentos deben ser una lista de etiquetas.')
    predictor.fit()
    labels={v:segment_labels(predictor,v) for v in variables}
    probabilities=pd.Series(predictor.probabilities,index=predictor.test_indices)
    observed=predictor.test_frame.Attrition.eq('Yes').astype(int)
    records=[];omitted=0;pair_count=0
    for first,second in combinations(variables,2):
        pair_count+=1
        left,left_detail=labels[first];right,right_detail=labels[second]
        data=pd.DataFrame({'left':left,'right':right,'prob':probabilities,'observed':observed})
        if segments is not None:
            data=data[data.left.isin(segments)&data.right.isin(segments)]
        groups=data.groupby(['left','right'],observed=True,sort=True).agg(n=('prob','size'),prob=('prob','mean'),observed=('observed','mean'))
        for (a,b),row in groups.iterrows():
            if row['n']<min_group:
                omitted+=1;continue
            records.append({'Variable 1':first,'Segmento 1':a,'Variable 2':second,'Segmento 2':b,
                'Condición 1':left_detail.get(a,f'{first} ausente'),
                'Condición 2':right_detail.get(b,f'{second} ausente'),
                'Registros de prueba':int(row['n']),
                'Probabilidad media Attrition Yes (%)':float(100*row['prob']),
                'Attrition Yes observado (%)':float(100*row['observed'])})
    columns=['Variable 1','Segmento 1','Variable 2','Segmento 2','Condición 1','Condición 2',
             'Registros de prueba','Probabilidad media Attrition Yes (%)','Attrition Yes observado (%)']
    table=pd.DataFrame(records,columns=columns)
    table=table.sort_values(['Probabilidad media Attrition Yes (%)','Registros de prueba','Condición 1','Condición 2'],
        ascending=[False,False,True,True],kind='stable').head(5).reset_index(drop=True)
    table.insert(0,'Puesto',range(1,len(table)+1))
    return {'tabla':table,'combinaciones_validas':len(records),'combinaciones_omitidas':omitted,
        'pares_evaluados':pair_count,'probabilidad_media_prueba_pct':float(100*probabilities.mean())}
