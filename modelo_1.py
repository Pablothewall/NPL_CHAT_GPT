import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

# Load and preprocess data
loaded_result = pd.read_pickle('merged_result.pkl')
loaded_result["Minutes"] = loaded_result['Minutes'].apply(lambda x: ' '.join(x))  # Join the list of strings

X_numeric = loaded_result[['Spread_DTB3_+_1', 'Spread_Avg_DFF_Avg_DTB3']]
X_text = loaded_result[['Minutes']]
y = loaded_result["Dummies_DFF_+_1"]

max_seq_length = max(len(tokens) for tokens in loaded_result["Minutes"])
print("Maximum sequence length:", max_seq_length) #4087621

# Tokenizing the text data
max_len = 4087621
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_texts = [tokenizer.tokenize(text)[:max_len] for text in X_text['Minutes']]
input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]

# Padding to ensure all sequences have the same length
input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")

# Convert to NumPy arrays
X_text_array = np.array(input_ids)
X_numeric_array = X_numeric.values
y_array = np.array(y)

# Pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Example numeric features
num_features = 2

# Define input layers
text_input_ids = Input(shape=(max_len,), dtype=tf.int32, name='text_input_ids')
text_attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='text_attention_mask')
text_token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name='text_token_type_ids')
numeric_input = Input(shape=(num_features,), dtype=tf.float32, name='numeric_input')

# BERT input
text_encoded = bert_model([text_input_ids, text_attention_mask, text_token_type_ids])[1]  # Getting pooled output

# Concatenate BERT output with numeric features
combined = Concatenate()([text_encoded, numeric_input])

# Additional layers for further processing
combined = Dense(128, activation='relu')(combined)
output = Dense(3, activation='softmax')(combined)  # Replace num_classes with your output classes

# Combine text and numeric inputs into a single model
model = Model(inputs=[text_input_ids, text_attention_mask, text_token_type_ids, numeric_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_text_array, np.ones_like(X_text_array), np.zeros_like(X_text_array), X_numeric_array], y_array, epochs=5, batch_size=32)