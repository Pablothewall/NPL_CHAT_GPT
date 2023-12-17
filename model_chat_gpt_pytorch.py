

import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Preprocess the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_inputs = [tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512) for text in minutes['Minutes']]

# Extract labels
labels = torch.tensor(minutes['Dummies'].values, dtype=torch.long)

# Split the data into train and test sets
text_inputs_train, text_inputs_test, labels_train, labels_test = train_test_split(text_inputs, labels, test_size=0.2, random_state=42)

numeric_features = minutes['Dummies']

# Split numeric features into train and test sets
numeric_features_train, numeric_features_test = train_test_split(numeric_features, test_size=0.2, random_state=42)

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, text_inputs, numeric_features, labels):
        self.text_inputs = text_inputs
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_inputs[idx], self.numeric_features[idx], self.labels[idx]

# Create DataLoader instances
train_dataset = MyDataset(text_inputs_train, numeric_features_train, labels_train)
test_dataset = MyDataset(text_inputs_test, numeric_features_test, labels_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example numeric features
num_features = 3  # Replace with the actual number of numeric features

class CombinedModel(nn.Module):
    def __init__(self, bert_model, num_features):
        super(CombinedModel, self).__init__()
        self.bert_model = bert_model
        self.num_features = num_features
        self.fc_numeric = nn.Linear(self.num_features, 64)  # Example numeric layers
        self.fc_combined = nn.Linear(768 + 64, 256)  # BERT output size is 768

        self.output_layer = nn.Linear(256, 3)  # Replace num_classes with your output classes

    def forward(self, text_input, numeric_input):
        # BERT input
        encoded_layers = self.bert_model(**text_input)[1]  # Getting pooled output

        # Numeric input
        numeric_output = torch.relu(self.fc_numeric(numeric_input))

        # Concatenate BERT output with numeric features
        combined = torch.cat((encoded_layers, numeric_output), dim=1)

        # Additional layers for further processing
        combined = torch.relu(self.fc_combined(combined))
        output = self.output_layer(combined)
        return output

# Example usage
model = CombinedModel(bert_model, num_features)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(5):  # Replace with the desired number of epochs
    for text_input_batch, numeric_input_batch, label_batch in train_dataloader:
        optimizer.zero_grad()
        output = model(text_input_batch, numeric_input_batch)
        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for text_input_batch, numeric_input_batch, label_batch in test_dataloader:
        outputs = model(text_input_batch, numeric_input_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += label_batch.size(0)
        correct += (predicted == label_batch).sum().item()

accuracy = correct / total
print('Accuracy on the test set: {:.2%}'.format(accuracy))
