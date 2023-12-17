from datasets import DatasetDict, Dataset, Value, ClassLabel
import pandas as pd
import datasets
from datasets import DatasetDict, Dataset
df=pd.read_csv('analidid.csv')
df = pd.DataFrame(df)
df=df[['Dummies', "Minutes"]]
#print(df)
df["Dummies"].replace(1, 2, inplace=True)
df["Dummies"].replace(-1, 1, inplace=True)

dict_possible = datasets.DatasetDict({"labels":df['Dummies'].tolist(), 
               "text": df["Minutes"].tolist()
               })



hf_dataset = Dataset.from_dict(dict_possible)


# Assuming you have two splits: train and test
train_size = int(len(df) * 0.9)  # Adjust the split ratio as needed
train_df = df[:train_size]
test_df = df[train_size:]

# Define features for the dataset
features = {
    "label": ClassLabel(names=df['Dummies'].unique().tolist()),
    "text": Value("string"),
    "input_ids": Value(dtype="int32"),
    "token_type_ids": Value(dtype="int32"),
    "attention_mask": Value(dtype="int32"),
}


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Create a function to tokenize your text data
def tokenize_function(examples):
    # Implement your tokenization logic here
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Create train and test datasets with features


# Tokenize the datasets (you may need to implement your own tokenization logic)
train_dataset = hf_dataset.map(tokenize_function)
test_dataset = hf_dataset.map(tokenize_function)

# Create DatasetDict
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# Print information about the created DatasetDict
print(dataset_dict)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
#print(tokenized_datasets.column_names)

small_train_dataset = dataset_dict["train"].map(tokenize_function, batched=True)
small_eval_dataset = dataset_dict["test"].map(tokenize_function, batched=True)



from transformers import AutoModelForSequenceClassification

#model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3, from_tf=False)


from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")



import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()





# DatasetDict({
#     train: Dataset({
#         features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
#         num_rows: 650000
#     })
#     test: Dataset({
#         features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
#         num_rows: 50000
#     })
# })