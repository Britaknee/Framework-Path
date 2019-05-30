import eel
import pandas as pd
import numpy as np
import json
from pynalytics.regression.linear_regression.lin_regression_num import LinRegressionRes
from pynalytics.regression.linear_regression.lin_regression_visual import LinRegressionVis
from pynalytics.regression.polynomial_regression.poly_regression_num import PolyRegressionRes
from pynalytics.regression.polynomial_regression.poly_regression_visual import PolyRegressionVis
from pynalytics import Preprocessing
from pynalytics.k_means import Centroid_Chart, Scatter_Matrix, Kmeans
from pynalytics.naive_bayes import NaiveBayes, Confusion_Matrix


df = pd.read_csv('regrex1.csv')
prep = Preprocessing()
df[['z']] = prep.bin(df[['z']],3)
nb = NaiveBayes()
nb.naive_bayes(df[['y']],df[['z']])
naive = Confusion_Matrix()
y_true = nb.y_test
y_pred = nb.y_pred
fig = naive.confusion_matrix(y_true,y_pred, classes=['1','2','3'])
naive.fig_show(fig)



eel.init('web')


#Create table
@eel.expose
def table():
    tabledata = df.to_html()
    return(''+ tabledata +'')

@eel.expose
def csvUpload(csvfile):

    #Convert to dictionary
    dicts = {}
    for x in csvfile:
        dicts[x[0]] = x[1:]

    global df

    df = pd.DataFrame.from_dict(dicts,orient='index')
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    
    return (''+ df.to_html() +'')


#Send columns
@eel.expose
def columns():
    columnsList = list(df.columns.values)
    return(columnsList)


#GUI functions
@eel.expose
def kmeans_sil_coef(kdf,c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)
    return str(km.sil_coef())

@eel.expose
def kmeans_centroids(kdf,c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)
    return str(km.centroids())

@eel.expose
def kmeans_centroid_chart(kdf, c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)
    cc = Centroid_Chart()
    fig = cc.centroid_chart(km.centroids(),x_labels=kmdf.columns.values)
    return(''+ cc.fig_to_html(fig) +'')

@eel.expose
def kmeans_cluster_graph(kdf, c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)
    labeled_df = km.labeled_dataset()
    sm = Scatter_Matrix()
    fig = sm.scatter_matrix(labeled_df, clusters_column='clusters')
    return(''+ sm.fig_to_html(fig) +'')

@eel.expose
def naive_classify(nX,ny):
   naive = NaiveBayes()
   X = df[nX]
   y = df[[ny]]
   naive.naive_bayes(X,y)
   return str(naive.classification_report())

@eel.expose
def naive_matrix(nX,ny):
    nb = NaiveBayes()
    X = df[nX]
    y = df[[ny]]
    nb.naive_bayes(X,y)
    naive = Confusion_Matrix()
    y_true = nb.y_test
    y_pred = nb.y_pred
    fig = naive.confusion_matrix(y_true,y_pred)
    return(''+ naive.fig_to_html(fig) +'')

@eel.expose
def lin_num_rsquare(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(lin_res.get_rsquare(y, x))

@eel.expose
def lin_adj_rsquare(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(lin_res.get_adj_rsquare(y, x))

@eel.expose
def lin_pearson(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(lin_res.get_pearsonr(y, x))

@eel.expose
def lin_regression(dv, idv):
    lin_vis = LinRegressionVis()
    x = df[[idv]]
    y = df[[dv]]
    fig = lin_vis.linear_regression(y, x)
    return(''+ lin_vis.fig_to_html(fig)+ '')

@eel.expose
def lin_scatter_matrix(dv, idv):
    lin_vis = LinRegressionVis()
    x = df[[idv]]
    y = df[[dv]]
    fig = lin_vis.scatter_plot(y, x)
    return(''+ lin_vis.fig_to_html(fig)+ '')

@eel.expose
def lin_rtable(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return(''+ lin_res.lin_regression_table(y, x).to_html() +'')

@eel.expose
def lin_rtable_multi(dv, idv):
    lin_res = LinRegressionRes()
    print(idv)
    X = df[idv]
    y = df[[dv]]
    return(''+ lin_res.lin_regression_table(y, X).to_html() +'')

@eel.expose
def simp_lin_num_slope(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(lin_res.get_slope(y, x))

@eel.expose
def simp_lin_num_rslope(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(lin_res.line_eq(y, x))

@eel.expose
def poly_int(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(poly_res.get_poly_intercept(y, x))

@eel.expose
def poly_coefficient(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(poly_res.get_poly_coeff(y, x))

@eel.expose
def poly_rsquared(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(poly_res.get_poly_rsquared(y, x))

@eel.expose
def poly_pearson_r(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(poly_res.get_poly_pearsonr(y, x))

@eel.expose
def poly_equation(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return str(poly_res.poly_eq(y, x))

@eel.expose
def poly_regression(dv, idv):
    poly_vis = PolyRegressionVis()
    x = df[[idv]]
    y = df[[dv]]
    fig = poly_vis.polynomial_reg(y, x)
    return(''+poly_vis.fig_to_html(fig)+'')

@eel.expose
def poly_rtable(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]
    return(''+ poly_res.poly_reg_table(y, x).to_html() +'')

# eel.start('main.html', size=(1920, 1080))
