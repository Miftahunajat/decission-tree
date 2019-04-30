#%%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from IPython.display import Image
from io import StringIO
import pydotplus
from sklearn import preprocessing
from sklearn import tree
import plotly.plotly as py
import plotly.graph_objs as go
py.tools.set_credentials_file(username='amaliawh', api_key='api-key')
iplot(chloromap)
def convert_ya_tidak(txt):
    if 'ya' in txt:
        return 1
    else:
        return 0

def reverse_convert_ya_tidak(txt):
    if txt == 1:
        return 'ya'
    else:
        return 'tidak'

def plot_decision_tree(clf,feature_name,target_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())

df = pd.read_csv('datatraining.csv')
df

df.olahraga = df.olahraga.apply(convert_ya_tidak)
df.jantung = df.jantung.apply(convert_ya_tidak)

df = pd.get_dummies(df)
df
X_train = df.loc[:, df.columns != 'jantung']
Y_train = df.jantung
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train,Y_train)
plot_decision_tree(clf, X_train.columns,["tidak","ya"])

#%%table

df = pd.read_csv('datatraining.csv')
df
predicted = clf.predict(X_train)
predicted_df = pd.DataFrame({'prediksi': predicted})
predicted_df
predicted_df.prediksi = predicted_df.prediksi.apply(reverse_convert_ya_tidak)
predicted_df
df = df.join(predicted_df)
df

label=['True','False']
values=[0,0]
for i, vali in enumerate(df.values):
    if(df.values[i][4]==df.values[i][5]):
        values[0]+=1
    else:
        values[1]+=1

#%%pie

trace = go.Pie(labels=label, values=values)

py.iplot([trace], filename='basic_pie_chart')
