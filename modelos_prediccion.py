# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:48:55 2024

@author: l11057
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

df_fechas = pd.read_pickle("df_fechas.pkl")
df_fechas.columns
X = df_fechas[['Fed Target Rate', 'Effective Fed Funds Rate', 'Tasa 3 Meses']].iloc[:-1].dropna()
y = df_fechas[["Decisión Política Monetaria", 'Effective Fed Funds Rate']].dropna()["Decisión Política Monetaria"]
y_class = pd.Series(np.where(y>0.001, 1, np.where(y<-0.001, -1, 0)), index=df_fechas[['Fed Target Rate', 'Effective Fed Funds Rate', 'Tasa 3 Meses', 'Fecha Siguiente Reunión']].iloc[:-1].dropna()['Fecha Siguiente Reunión'])
y_class.plot()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X, y_class)
y_pred = lr.predict(X)
y_pred
y_class.value_counts()
confusion_matrix(y_class, y_pred)
accuracy_score(y_class, y_pred)


tasas = pd.read_pickle("tasas.pkl")


columns = ['Fecha Reunión', 'Publicacion Minuta', 'Fecha Reunión + 1 BD',
               'Publicacion Minuta + 1 BD', 'Fecha Siguiente Reunión',
               'Fecha Siguiente Reunión + 1 BD']

df_fechas_alt = df_fechas[['Fecha Siguiente Reunión', 'Publicacion Minuta + 1 BD']].dropna()
df_fechas_alt[['Fecha Siguiente Reunión -2D']] = df_fechas_alt[['Fecha Siguiente Reunión']]-pd.offsets.BDay(2)
df_fechas_alt = df_fechas_alt.merge(df_fechas[["Fecha Siguiente Reunión", "Decisión Política Monetaria"]])
df_fechas_alt = df_fechas_alt.merge(tasas, left_on='Fecha Siguiente Reunión -2D', right_index=True)
df_fechas_alt.columns


X = df_fechas_alt[['Fed Target Rate', 'Effective Fed Funds Rate', 'Tasa 3 Meses']].dropna()
y = df_fechas_alt[['Fed Target Rate', 'Effective Fed Funds Rate', 'Tasa 3 Meses', "Decisión Política Monetaria"]].dropna()["Decisión Política Monetaria"]
y_class = pd.Series(np.where(y>0.001, 1, np.where(y<-0.001, -1, 0)), index=df_fechas[['Fed Target Rate', 'Effective Fed Funds Rate', 'Tasa 3 Meses', 'Fecha Siguiente Reunión']].iloc[:-1].dropna()['Fecha Siguiente Reunión'])

lr = LogisticRegression(class_weight='balanced', random_state=42)
lr.fit(X, y_class)
y_pred = pd.Series(lr.predict(X), index=y_class.index)
y_pred
y_class.value_counts()
confusion_matrix(y_class, y_pred)
accuracy_score(y_class, y_pred)

X_y_pred = pd.concat((X.set_index(y_class.index), y_class, y_pred), axis=1)


from transformers import BertTokenizer, BertModel

# Pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example numeric features
num_features = 5  # Replace with the actual number of numeric features



