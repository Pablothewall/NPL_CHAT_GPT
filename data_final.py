import matplotlib.pyplot as plt
from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
import pickle
import pandas as pd
import csv
import pickle
import numpy as np
from remove_stop_words_function import remove_stop_lemma_words
DFF= pd.read_csv(r"C:\Users\pablo\Desktop\NLP\DFF.csv")
DTB3= pd.read_csv(r"C:\Users\pablo\Desktop\NLP\DTB3.csv")
final=DFF.merge(DTB3, on="DATE")
final=final.apply(lambda y: y.replace(to_replace=".", method='ffill'),axis=1)
final['DFF']=final['DFF'].astype(float)
final['DTB3']=final['DTB3'].astype(float)
final['Spread_DFF_DTB3']=final['DFF']-final['DTB3']

final.set_index('DATE', inplace=True)
final.plot()
#plt.show()

final['Avg_DFF_5_days'] = final['DFF'].rolling(window=5).mean()
final['Avg_DTB3_5_days'] = final['DTB3'].rolling(window=5).mean()
final['Spread_Avg_DFF_Avg_DTB3']=final['Avg_DFF_5_days']-final['Avg_DTB3_5_days']



final['Dummies_DDF_DTB3'] = [2 if x > 0.025 else 1 if x < -0.025 else 0 for x in final['Spread_DFF_DTB3']]
final['Dummies_Avg_DDF_Avg_DTB3'] = [2 if x > 0.025 else 1 if x < -0.025 else 0 for x in final['Spread_Avg_DFF_Avg_DTB3']]
final['DFF_+_1'] = final['DFF'].shift(-1)
final['DTB3_+_1'] = final['DTB3'].shift(-1)
final['DFF_+_1-DTB3']=final['DFF_+_1']-final['DTB3']




final.dropna(inplace=True)

print(final['Dummies_DDF_DTB3'].value_counts(),final['Dummies_Avg_DDF_Avg_DTB3'].value_counts())
print(final)


#abro minuta

#Minutes

# dataset = FederalReserveMins().find_minutes()
# FederalReserveMins().pickle_data(r"C:\Users\pablo\Desktop\NLP\minutes.pkl")


with open('minutes.pkl', 'rb') as f:
    minutes = pickle.load(f)

minutes=minutes["Federal_Reserve_Mins"].apply(lambda x: remove_stop_lemma_words(x))
minutes=pd.DataFrame(minutes)
minutes.columns=['Minutes']
minutes.rename_axis("DATE", inplace=True)
# minutes.set_index("DATE")
final.reset_index(inplace=True)
minutes.reset_index(inplace=True)

final['DATE'] = pd.to_datetime(final['DATE'])
result = final.merge(minutes, on='DATE')

result['Spread_DFF_DTB3']=result['DFF_+_1']-result['DTB3']

result.set_index("DATE", inplace=True)

result['Spread_DFF_+_1'] = result['DFF_+_1'].diff()
result['Dummies_DFF_+_1'] = [2 if x > 0.025 else 1 if x < -0.025 else 0 for x in result['Spread_DFF_+_1']]

result['Spread_DTB3_+_1'] = result['DTB3_+_1'].diff()
result['Dummies_DTB3_+_1'] = [2 if x > 0.25 else 1 if x < -0.25 else 0 for x in result['Spread_DTB3_+_1']]





result.dropna(inplace=True)
print(result[['Spread_DFF_+_1', 'DFF_+_1']])
result.to_pickle('merged_result.pkl')



# print(final)
# print(minutes)
# print(result)
# #print(final[final.index>'1995-01-31'])








