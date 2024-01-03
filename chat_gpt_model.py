from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
import pickle
import pandas as pd
import csv
import pickle
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from remove_stop_words_function import remove_stop_lemma_words,tokenIze,lemma,join_lists_to_string
from keras.utils import to_categorical

with open('minutes.pkl', 'rb') as f:
    minutes = pickle.load(f)
#minutes['Minutes']=minutes['Minutes'].apply(lambda x: remove_stop_lemma_words(x))
#minutes['Minutes']=minutes['Minutes'].apply(lambda x: tokenIze(x))
#minutes['Minutes']=minutes['Minutes'].apply(lambda x: lemma(x))
#minutes['Minutes']=minutes['Minutes'].apply(lambda x: join_lists_to_string(x))

minutes_minutes_tensor = tf.constant(minutes['Minutes'].values.reshape(-1, 206), dtype=tf.string)
minutes_dummies_tensor = tf.constant(minutes['Dummies'].values.reshape(-1, 206), dtype=tf.int64)


minutes_values = minutes['Minutes'].values
dummies_values = minutes['Dummies'].values
minutes_values = np.array(minutes['Minutes'])
dummies_values = np.array(minutes['Dummies'])
target= tf.one_hot(dummies_values, depth=3, dtype=tf.int64)
num_classes = 3
labels = tf.keras.utils.to_categorical(dummies_values, num_classes=num_classes)
print(minutes_dummies_tensor.shape)
print(dummies_values.shape)


dummies_values=[dummies_values]
# Assuming you have labels for your training samples
labels = np.random.randint(0, 3, size=(len(minutes),))  
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=3)



#print(len(minutes_minutes_tensor))
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

# Pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# Example numeric features
num_features = 1  # Replace with the actual number of numeric features

# Define input layers
text_input = Input(shape=(206,), dtype=tf.int32, name='text_input')
numeric_input = Input(shape=(num_features,), dtype=tf.float32, name='numeric_input')
# BERT input
text_encoded = bert_model(text_input)[1]  # Getting pooled output
print(text_encoded.shape)
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





model.fit([minutes_values,dummies_values],labels,epochs=5, batch_size=32)  # Replace with your data
