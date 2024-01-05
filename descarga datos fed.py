# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:49:34 2024

@author: l11057
"""
import pandas_datareader.data as web
import datetime

# Define the start and end dates for the data
start_date = datetime.datetime(1950, 1, 1)

# Fetch Federal Funds Rate data from FRED
ff_rate = web.DataReader(['DFEDTAR',
                "DFEDTARL",
                "DFEDTARU"], 'fred', start_date)
ff_rate_mid = ff_rate.mean(axis=1).ffill()
three_month_rate = web.DataReader(['DTB3',], 'fred', start_date).ffill()
three_month_rate

from FedTools import MonetaryPolicyCommittee

dataset = MonetaryPolicyCommittee().find_statements()

import pandas as pd
fechas_pub_minutas = dataset.index + pd.offsets.BDay(15)
fechas_pub_minutas

fechas = pd.DataFrame({
    "Fecha Reunión" : dataset.index[:],
    "Publicacion Minuta" : fechas_pub_minutas[:],
})
fechas["Fecha Reunión + 1 BD"] = fechas["Fecha Reunión"]+ pd.offsets.BDay()
fechas["Publicacion Minuta + 1 BD"] = fechas["Publicacion Minuta"]+ pd.offsets.BDay()
fechas["Fecha Siguiente Reunión"] = list(dataset.index[1:])+[None]
fechas["Fecha Siguiente Reunión + 1 BD"] = fechas["Fecha Siguiente Reunión"]+ pd.offsets.BDay()
fechas

tasas = pd.concat(
    (ff_rate_mid,
    three_month_rate), axis=1
)
tasas.columns =["Fed Rate", "Tasa 3 Meses"]


df_fechas = fechas[['Fecha Reunión + 1 BD',
 'Publicacion Minuta + 1 BD', 
 'Fecha Siguiente Reunión + 1 BD']]
variacion_tasas = df_fechas.merge(tasas, left_on='Fecha Siguiente Reunión + 1 BD', right_index=True)["Fed Rate"]-df_fechas.merge(tasas, left_on='Fecha Reunión + 1 BD', right_index=True)["Fed Rate"]
df_fechas["Decisión Política Monetaria"] = variacion_tasas
df_fechas = df_fechas.merge(tasas, left_on='Publicacion Minuta + 1 BD', right_index=True)

df_fechas.tail(10)

