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
ff_rate = web.DataReader([
                "EFFR",
                'DFEDTAR',
                "DFEDTARL",
                "DFEDTARU"], 'fred', start_date)
ff_rate_mid = ff_rate[['DFEDTAR',
                "DFEDTARL",
                "DFEDTARU"]].mean(axis=1).ffill()
effective_ff_rate = ff_rate["EFFR"].ffill().rolling(5).mean().dropna()
effective_ff_rate = ff_rate["EFFR"].ffill()

other_short_rates = web.DataReader(["DTB4WK", 'DTB3',], 'fred', start_date).ffill()

from FedTools import MonetaryPolicyCommittee
dataset_statements = MonetaryPolicyCommittee(historical_split = 2017).find_statements()
from FedTools import FederalReserveMins
dataset_minutes = FederalReserveMins(historical_split = 2017).find_minutes()

import pandas as pd
fechas_pub_minutas = dataset_minutes.index + pd.offsets.BDay(15)
fechas_pub_minutas

fechas = pd.DataFrame({
    "Fecha Reunión" : dataset_minutes.index[:],
    "Publicacion Minuta" : fechas_pub_minutas[:],
})
fechas["Fecha Reunión + 1 BD"] = fechas["Fecha Reunión"] + pd.offsets.BDay()
fechas["Publicacion Minuta + 1 BD"] = fechas["Publicacion Minuta"]+ pd.offsets.BDay()
fechas["Fecha Siguiente Reunión"] = list(dataset_minutes.index[1:])+[None]
fechas["Fecha Siguiente Reunión + 1 BD"] = fechas["Fecha Siguiente Reunión"]+ pd.offsets.BDay()
fechas

tasas = pd.concat(
    (ff_rate_mid,
     effective_ff_rate,
    other_short_rates), axis=1
)
tasas.columns =["Fed Target Rate", "Effective Fed Funds Rate", "Tasa 1 Mes", "Tasa 3 Meses"]


df_fechas = fechas[['Fecha Reunión + 1 BD',
 'Publicacion Minuta + 1 BD', 
 'Fecha Siguiente Reunión + 1 BD']]
variacion_tasas = df_fechas.merge(tasas, left_on='Fecha Siguiente Reunión + 1 BD', right_index=True)["Fed Target Rate"]-df_fechas.merge(tasas, left_on='Fecha Reunión + 1 BD', right_index=True)["Fed Target Rate"]
df_fechas["Decisión Política Monetaria"] = variacion_tasas
df_fechas = df_fechas.merge(tasas, left_on='Publicacion Minuta + 1 BD', right_index=True)
df_fechas.tail(10)

df_fechas.to_pickle("df_fechas.pkl")
tasas.to_pickle("tasas.pkl")
dataset_minutes.to_pickle("dataset_minutes.pkl")
dataset_statements.to_pickle("dataset_statements.pkl")
