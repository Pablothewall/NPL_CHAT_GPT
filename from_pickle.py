import pandas as pd
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import numpy as np
loaded_result = pd.read_pickle('merged_result.pkl')
print(loaded_result.columns)
# print(loaded_result[['Spread_DFF_+_1','Dummies_DFF_+_1']])
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
# Assuming df is your DataFrame with columns 'Dummies' and 'Spread_Dummies'

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

X = loaded_result.loc[:, ~loaded_result.columns.isin(["Dummies_DFF_+_1", "Minutes",'Spread_DFF_+_1'])]
y = loaded_result["Dummies_DFF_+_1"]
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(y_train.value_counts())
print(y_test.value_counts())











#RandomForestClassifier()
model = make_pipeline(StandardScaler(),  RandomForestClassifier())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
#importance = model.feature_importances_
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Decision Tree Regressor: {accuracy * 100:.6f}%')
#conf_matrix = confusion_matrix(y_test, y_pred)
#print(conf_matrix)


# Plot the regression line
# plt.scatter(X_test, y_test, color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)
# plt.xlabel('Spread')
# plt.ylabel('Dummies')
# plt.title('Linear Regression Model')
# plt.show()



# from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix,ConfusionMatrixDisplay,accuracy_score

# y_train.value_counts()

# # Make predictions on the training data
# y_train_pred = model.predict(X_train)
# y_train_pred_proba = model.predict_proba(X_train)

# # Evaluate the model on training data
# train_mse = mean_squared_error(y_train, y_train_pred)
# train_r2 = r2_score(y_train, y_train_pred)
# print(f'Training Mean Squared Error: {train_mse}')
# print(f'Training R-squared: {train_r2}')

# # Evaluate the model on testing data
# test_mse = mean_squared_error(y_test, y_pred)
# test_r2 = r2_score(y_test, y_pred)
# print(f'Testing Mean Squared Error: {test_mse}')
# print(f'Testing R-squared: {test_r2}')

# (y_train_pred_proba>1/3).sum(axis=0)

# np.unique(y_train_pred_proba.argmax(axis=1), return_counts=True)

# print(accuracy_score(y_test, y_pred))
# confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
#                                display_labels=model.classes_)
# disp.plot()
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Assuming X and y are your feature matrix and target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier model
model = make_pipeline(StandardScaler(), XGBClassifier())
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Print the result
print(f'Mean Absolute Error: {mae:.6f}')




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from imblearn.over_sampling import SMOTE
X = loaded_result.loc[:, ~loaded_result.columns.isin(["Dummies_DFF_+_1", "Minutes",'Spread_DFF_+_1'])]
y = loaded_result["Dummies_DFF_+_1"]
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(y_train.value_counts())
print(y_test.value_counts())



model = make_pipeline(StandardScaler(),  RandomForestClassifier())
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores = cross_val_score(model, X, y, cv=4,scoring='accuracy')

print(scores.mean())