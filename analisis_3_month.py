import pandas as pd
import numpy as np
df=pd.read_csv('minutes_2.csv')
df = pd.DataFrame(df)


three_month=pd.read_csv('DTB3.csv')
three_month = pd.DataFrame(three_month)
three_month['shifted_value'] = three_month['DTB3'].shift(-1)
three_month.columns=['Date',"DTB3","shifted_value"]
print(three_month)

merged_df= df.merge(three_month, on="Date", how="left")
#print(merged_df.columns)

# Convert relevant columns to numeric, handling errors
merged_df['DFF'] = pd.to_numeric(merged_df['DFF'], errors='coerce')
merged_df['shifted_value'] = pd.to_numeric(merged_df['shifted_value'], errors='coerce')

# Create the 'Spread' column
merged_df['Spread'] = merged_df['shifted_value'] - merged_df['DFF']


merged_df.reset_index(drop=True, inplace=True)
merged_df.dropna(inplace=True)

#merged_df['Spread_Dummies'] = [1 if x > 0.025 else 2 if x < -0.025 else 0 for x in merged_df['Spread']]

#merged_df=merged_df[["Dummies", "Spread_Dummies"]]
#print(merged_df[["Dummies", "Spread_Dummies"]])


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with columns 'Dummies' and 'Spread_Dummies'

X = merged_df[['Spread']] 
y = merged_df['Dummies']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train.value_counts()
y_test.value_counts()

model = LogisticRegression(class_weight='balanced')


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Spread')
plt.ylabel('Dummies')
plt.title('Linear Regression Model')
#plt.show()



from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix,ConfusionMatrixDisplay,accuracy_score

y_train.value_counts()

# Make predictions on the training data
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)

# Evaluate the model on training data
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Training Mean Squared Error: {train_mse}')
print(f'Training R-squared: {train_r2}')

# Evaluate the model on testing data
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f'Testing Mean Squared Error: {test_mse}')
print(f'Testing R-squared: {test_r2}')

(y_train_pred_proba>1/3).sum(axis=0)

np.unique(y_train_pred_proba.argmax(axis=1), return_counts=True)

print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                               display_labels=model.classes_)
disp.plot()



##############################################################
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm




from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print(sum(scores)/5)




from sklearn import metrics
scores = cross_val_score(
    clf, X, y, cv=5, scoring='f1_macro')
from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
print("n_sample: ", n_samples )
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)   
def custom_cv_2folds(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
        yield idx, idx
        i += 1
custom_cv = custom_cv_2folds(X)
data=cross_val_score(clf, X, y, cv=custom_cv)
print(data.mean())





from sklearn.model_selection import TimeSeriesSplit

#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
#y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=10)

for train, test in tscv.split(X):
    print("%s %s" % (train, test))
    
    
    
    
    
    





import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

# Pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Example numeric features
num_features = 5  # Replace with the actual number of numeric features

# Define input layers
text_input = Input(shape=(2500000,), dtype=tf.int32, name='text_input')
numeric_input = Input(shape=(num_features,), dtype=tf.float32, name='numeric_input')

# BERT input
text_encoded = bert_model(text_input)[1]  # Getting pooled output

# Concatenate BERT output with numeric features
combined = Concatenate()([text_encoded, numeric_input])

# Additional layers for further processing
# Example:
combined = Dense(128, activation='relu')(combined)
output = Dense(3, activation='softmax')(combined)  # Replace num_classes with your output classes

# Combine text and numeric inputs into a single model
model = Model(inputs=[text_input, numeric_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([text_data, numeric_data], labels, epochs=5, batch_size=32)  # Replace with your data