import pandas as pd
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier


loaded_result = pd.read_pickle('merged_result.pkl')
print(loaded_result.columns)


X = loaded_result.loc[:, ~loaded_result.columns.isin(["Dummies_DFF_+_1", "Minutes",'Spread_DFF_+_1'])]
y = loaded_result["Dummies_DFF_+_1"]
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(y_train.value_counts())
print(y_test.value_counts())





# ################################################################################################# MODELO FINAL ######################################################################


#X = loaded_result.loc[:, ~loaded_result.columns.isin(["Dummies_DFF_+_1", "Minutes",'Spread_DFF_+_1',"DFF_+_1-DTB3",])]
X = loaded_result[['Spread_DTB3_+_1','Spread_Avg_DFF_Avg_DTB3']]

y = loaded_result["Dummies_DFF_+_1"]


# 'DFF', 'DTB3', 'T10Y3M', 'T3MFF', 'Spread_DFF_DTB3', 'Avg_DFF_5_days',
#        'Avg_DTB3_5_days', 'Spread_Avg_DFF_Avg_DTB3', 'Dummies_DDF_DTB3',
#        'Dummies_Avg_DDF_Avg_DTB3', 'DFF_+_1', 'DTB3_+_1', 'Spread_DTB3_+_1',
#        'Dummies_DTB3_+_1'



oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
print(X.columns)
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(y_train_smote.value_counts())
print(y_test_smote.value_counts())


model = make_pipeline(StandardScaler(),  RandomForestClassifier())

model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)
print("ACCURACY:  ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

feature_importances = model.named_steps['randomforestclassifier'].feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top N features
top_n = 10  # Change this value based on your preference
print(f'Top {top_n} Features:')
print(feature_importance_df.head(top_n))

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:top_n], feature_importance_df['Importance'][:top_n])
plt.xlabel('Importance')
plt.title('Top Features')
plt.show()


##########################################################################      MLPClassifier     #########################################################################################




X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)


model = MLPClassifier(hidden_layer_sizes=150,random_state=1, max_iter=300,warm_start=True, activation="relu",learning_rate="constant",learning_rate_init=0.01).fit(X_train, y_train)
model.predict_proba(X_test)
model.predict(X_test)
y_pred=model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(model.score(X_test, y_test))
